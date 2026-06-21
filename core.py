"""
core.py — Chemistry and computation backend for the Analog Designer application.

Contains:
  - Fragment library (Frag dataclass + LIBRARY)
  - Property calculators (SA score, ESOL, Morgan FP)
  - Analog generation (attach, attach_to_sites, generate_analogs)
  - Pocket analysis (distance-shell, fpocket alpha-spheres)
  - Docking helpers (PDB splitting, ACD command builder, score parsing)
  - PLIP / cIFP interaction fingerprints
  - 3D ligand file generation
"""

from __future__ import annotations

import glob
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    QED,
    rdMolDescriptors,
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.DataStructs import TanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

_SA_OK = False
try:
    from rdkit.Chem import RDConfig
    import sys as _sys
    _sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore
    _SA_OK = True
except Exception:
    pass


def sa_score(mol: Chem.Mol) -> float:
    """Synthetic-accessibility score 1 (easy) .. 10 (hard)."""
    if _SA_OK:
        try:
            return float(sascorer.calculateScore(mol))
        except Exception:
            pass
    nr = rdMolDescriptors.CalcNumRings(mol)
    nst = len(
        Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    )
    mw = Descriptors.MolWt(mol)
    sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    return max(1.0, min(10.0, 1.5 + 0.6 * nr + 0.7 * nst + mw / 250.0 + sp3))


def esol_logS(mol: Chem.Mol) -> float:
    """Delaney ESOL aqueous logS estimate."""
    clogp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ap = (
        len(mol.GetAromaticAtoms()) / mol.GetNumHeavyAtoms()
        if mol.GetNumHeavyAtoms()
        else 0.0
    )
    return 0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rb - 0.74 * ap


def morgan(mol: Chem.Mol, r: int = 2, n: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=n)


# ---------------------------------------------------------------------------
# Fragment library
# ---------------------------------------------------------------------------

G = lambda **k: {
    "potency": 0,
    "selectivity": 0,
    "solubility": 0,
    "metabolic": 0,
    "synthesis": 0,
    "novelty": 0,
    **k,
}

CATEGORY_BASE_GOALS: Dict[str, Dict] = {
    "hydrophobic": G(potency=1, selectivity=1, solubility=-1, synthesis=1),
    "polar": G(potency=1, selectivity=1, solubility=1, synthesis=1),
    "basic": G(potency=1, selectivity=1, solubility=2, synthesis=0),
    "acidic": G(potency=1, selectivity=1, solubility=2, synthesis=0),
    "halogen": G(potency=1, selectivity=1, metabolic=2, solubility=-1, synthesis=1),
    "aromatic": G(potency=1, selectivity=2, solubility=-1, novelty=1),
    "solubility": G(solubility=2, selectivity=1, synthesis=0),
    "bioisostere": G(selectivity=1, metabolic=1, novelty=2, synthesis=-1),
}


def _merge_goals(category: str, **override) -> Dict:
    base = dict(CATEGORY_BASE_GOALS.get(category, G()))
    base.update(override)
    return base


@dataclass
class Frag:
    name: str
    smiles: str
    category: str
    goals: Dict
    tox: bool = False
    size_class: str = "auto"
    interaction_class: str = "auto"
    charge_class: str = "auto"
    source: str = "built_in"
    notes: str = ""

    @property
    def heavy(self) -> int:
        m = Chem.MolFromSmiles(self.smiles.replace("[*]", "[H]"))
        return m.GetNumHeavyAtoms() if m else 99


_FRAGMENT_ROWS = [
    ("methyl", "[*]C", "hydrophobic"),
    ("ethyl", "[*]CC", "hydrophobic"),
    ("n-propyl", "[*]CCC", "hydrophobic"),
    ("isopropyl", "[*]C(C)C", "hydrophobic"),
    ("n-butyl", "[*]CCCC", "hydrophobic"),
    ("tert-butyl", "[*]C(C)(C)C", "hydrophobic"),
    ("cyclopropyl", "[*]C1CC1", "hydrophobic"),
    ("cyclobutyl", "[*]C1CCC1", "hydrophobic"),
    ("cyclopentyl", "[*]C1CCCC1", "hydrophobic"),
    ("cyclohexyl", "[*]C1CCCCC1", "hydrophobic"),
    ("vinyl", "[*]C=C", "hydrophobic"),
    ("ethynyl", "[*]C#C", "hydrophobic"),
    ("hydroxyl", "[*]O", "polar"),
    ("methoxy", "[*]OC", "polar"),
    ("ethoxy", "[*]OCC", "polar"),
    ("isopropoxy", "[*]OC(C)C", "polar"),
    ("hydroxymethyl", "[*]CO", "polar"),
    ("2-hydroxyethyl", "[*]CCO", "polar"),
    ("acetyl", "[*]C(C)=O", "polar"),
    ("acetamido", "[*]NC(C)=O", "polar"),
    ("amide(C(=O)NH2)", "[*]C(N)=O", "polar"),
    ("N-methylamide", "[*]C(=O)NC", "polar"),
    ("urea", "[*]NC(=O)N", "polar"),
    ("methylsulfonyl", "[*]S(C)(=O)=O", "polar"),
    ("sulfonamide-NH2", "[*]S(N)(=O)=O", "polar"),
    ("amino", "[*]N", "basic"),
    ("methylamino", "[*]NC", "basic"),
    ("dimethylamino", "[*]N(C)C", "basic"),
    ("azetidine", "[*]N1CCC1", "basic"),
    ("pyrrolidine", "[*]N1CCCC1", "basic"),
    ("piperidine", "[*]N1CCCCC1", "basic"),
    ("morpholine", "[*]N1CCOCC1", "basic"),
    ("piperazine", "[*]N1CCNCC1", "basic"),
    ("N-methylpiperazine", "[*]N1CCN(C)CC1", "basic"),
    ("carboxyl", "[*]C(=O)O", "acidic"),
    ("carboxymethyl", "[*]CC(=O)O", "acidic"),
    ("sulfonic-acid", "[*]S(=O)(=O)O", "acidic"),
    ("sulfonamide", "[*]S(N)(=O)=O", "acidic"),
    ("tetrazole", "[*]c1nnn[nH]1", "acidic"),
    ("hydroxamic-acid", "[*]C(=O)NO", "acidic"),
    ("fluoro", "[*]F", "halogen"),
    ("chloro", "[*]Cl", "halogen"),
    ("bromo", "[*]Br", "halogen"),
    ("iodo", "[*]I", "halogen"),
    ("cyano", "[*]C#N", "halogen"),
    ("trifluoromethyl", "[*]C(F)(F)F", "halogen"),
    ("difluoromethyl", "[*]C(F)F", "halogen"),
    ("trifluoromethoxy", "[*]OC(F)(F)F", "halogen"),
    ("phenyl", "[*]c1ccccc1", "aromatic"),
    ("benzyl", "[*]Cc1ccccc1", "aromatic"),
    ("4-fluorophenyl", "[*]c1ccc(F)cc1", "aromatic"),
    ("4-chlorophenyl", "[*]c1ccc(Cl)cc1", "aromatic"),
    ("4-methylphenyl", "[*]c1ccc(C)cc1", "aromatic"),
    ("4-methoxyphenyl", "[*]c1ccc(OC)cc1", "aromatic"),
    ("pyridin-2-yl", "[*]c1ccccn1", "aromatic"),
    ("pyridin-3-yl", "[*]c1cccnc1", "aromatic"),
    ("pyridin-4-yl", "[*]c1ccncc1", "aromatic"),
    ("thiophen-2-yl", "[*]c1cccs1", "aromatic"),
    ("furan-2-yl", "[*]c1ccco1", "aromatic"),
    ("imidazol-1-yl", "[*]n1ccnc1", "aromatic"),
    ("pyrazol-1-yl", "[*]n1cccn1", "aromatic"),
    ("thiazol-2-yl", "[*]c1nccs1", "aromatic"),
    ("benzimidazolyl", "[*]c1nc2ccccc2[nH]1", "aromatic"),
    ("indol-3-yl", "[*]c1c[nH]c2ccccc12", "aromatic"),
    ("2-hydroxyethoxy", "[*]OCCO", "solubility"),
    ("PEG2", "[*]OCCOC", "solubility"),
    ("morpholinoethyl", "[*]CCN1CCOCC1", "solubility"),
    ("morpholine-carbonyl", "[*]C(=O)N1CCOCC1", "solubility"),
    ("N-methylpiperazine-carbonyl", "[*]C(=O)N1CCN(C)CC1", "solubility"),
    ("oxetan-3-yl", "[*]C1COC1", "bioisostere"),
    ("azetidin-3-yl", "[*]C1CNC1", "bioisostere"),
    ("tetrahydropyran-4-yl", "[*]C1CCOCC1", "bioisostere"),
    ("cyclopropyl-carbonyl", "[*]C(=O)C1CC1", "bioisostere"),
    ("difluorocyclopropyl", "[*]C1(F)CC1F", "bioisostere"),
    ("oxadiazole-methyl", "[*]Cc1nnco1", "bioisostere"),
    ("triazole-methyl", "[*]Cn1cncn1", "bioisostere"),
]


# ---------------------------------------------------------------------------
# Extended fragment library — baked in, no extra files needed for deployment
# 4,551 additional entries (MW ≤ 250 Da, deduplicated vs built-ins)
# To regenerate: python build_fragment_library.py  (dev tool only, not deployed)
# ---------------------------------------------------------------------------
_EXTENDED_ROWS: List[Tuple[str, str, str]] = [
    ('n-pentyl', '[*]CCCCC', 'hydrophobic'),
    ('n-hexyl', '[*]CCCCCC', 'hydrophobic'),
    ('n-heptyl', '[*]CCCCCCC', 'hydrophobic'),
    ('n-octyl', '[*]CCCCCCCC', 'hydrophobic'),
    ('sec-butyl', '[*]C(C)CC', 'hydrophobic'),
    ('neopentyl', '[*]CC(C)(C)C', 'hydrophobic'),
    ('2-methylbutyl', '[*]CC(C)CC', 'hydrophobic'),
    ('3-methylbutyl', '[*]CCC(C)C', 'hydrophobic'),
    ('2-ethylbutyl', '[*]CC(CC)CC', 'hydrophobic'),
    ('isobutyl', '[*]CC(C)C', 'hydrophobic'),
    ('cycloheptyl', '[*]C1CCCCCC1', 'hydrophobic'),
    ('2-methylcyclopropyl', '[*]C1CC1C', 'hydrophobic'),
    ('3-methylcyclobutyl', '[*]C1CCC1C', 'hydrophobic'),
    ('4-methylcyclohexyl', '[*]C1CCC(C)CC1', 'hydrophobic'),
    ('2-methylcyclohexyl', '[*]C1CCCCC1C', 'hydrophobic'),
    ('spiro[2.2]pentyl', '[*]C1(CC1)C1CC1', 'hydrophobic'),
    ('bicyclo[1.1.1]pentyl', '[*]C1(CC2)CC12', 'hydrophobic'),
    ('norbornyl', '[*]C1CC2CCC1C2', 'hydrophobic'),
    ('gem-dimethylcyclopropyl', '[*]C1(C)CC1C', 'hydrophobic'),
    ('spiro[3.3]heptyl', '[*]C1CCC12CCC2', 'hydrophobic'),
    ('n-propoxy', '[*]OCCC', 'polar'),
    ('n-butoxy', '[*]OCCCC', 'polar'),
    ('allyloxy', '[*]OCC=C', 'polar'),
    ('propargyloxy', '[*]OCC#C', 'polar'),
    ('cyclopropylmethoxy', '[*]OCC1CC1', 'polar'),
    ('3-methoxypropoxy', '[*]OCCCOC', 'solubility'),
    ('tetrahydrofurfuryloxy', '[*]OCC1CCCO1', 'polar'),
    ('methylthio', '[*]SC', 'polar'),
    ('ethylthio', '[*]SCC', 'polar'),
    ('phenylthio', '[*]Sc1ccccc1', 'aromatic'),
    ('methylsulfinyl', '[*]S(C)=O', 'polar'),
    ('N-ethylamino', '[*]NCC', 'basic'),
    ('N-propylamino', '[*]NCCC', 'basic'),
    ('N-isopropylamino', '[*]NC(C)C', 'basic'),
    ('N-butylamino', '[*]NCCCC', 'basic'),
    ('N-cyclopropylamino', '[*]NC1CC1', 'basic'),
    ('N,N-diethylamino', '[*]N(CC)CC', 'basic'),
    ('N,N-dipropylamino', '[*]N(CCC)CCC', 'basic'),
    ('methylfluoro', '[*]CF', 'halogen'),
    ('ethylfluoro', '[*]CCF', 'halogen'),
    ('methylchloro', '[*]CCl', 'halogen'),
    ('ethylchloro', '[*]CCCl', 'halogen'),
    ('methylbromo', '[*]CBr', 'halogen'),
    ('ethylbromo', '[*]CCBr', 'halogen'),
    ('gem-difluoroethyl', '[*]CC(F)F', 'halogen'),
    ('gem-difluoropropyl', '[*]CCC(F)F', 'halogen'),
    ('2-fluoroethoxy', '[*]OCCF', 'halogen'),
    ('2,2-difluoroethoxy', '[*]OCC(F)F', 'halogen'),
    ('3-fluoropropoxy', '[*]OCCCF', 'halogen'),
    ('3,3-difluoropropoxy', '[*]OCCC(F)F', 'halogen'),
    ('difluoromethoxy', '[*]OC(F)F', 'halogen'),
    ('3-fluoropropyl', '[*]CCCF', 'halogen'),
    ('cyanomethyl', '[*]CC#N', 'halogen'),
    ('cyanoethyl', '[*]CCC#N', 'halogen'),
    ('cyanopropyl', '[*]CCCC#N', 'halogen'),
    ('azepane', '[*]N1CCCCCC1', 'basic'),
    ('homopiperazine', '[*]N1CCNCCC1', 'basic'),
    ('thiomorpholine', '[*]N1CCSCC1', 'basic'),
    ('1,4-oxazepane', '[*]N1CCOCCC1', 'basic'),
    ('N-methylhomopiperazine', '[*]N1CCNC(C)CC1', 'basic'),
    ('N-acetylpiperazine', '[*]N1CCN(C(C)=O)CC1', 'basic'),
    ('N-Boc-piperazine', '[*]N1CCN(C(=O)OC(C)(C)C)CC1', 'basic'),
    ('piperidin-4-yl', '[*]C1CCNCC1', 'basic'),
    ('pyrrolidin-3-yl', '[*]C1CCNC1', 'basic'),
    ('piperidin-4-ylmethyl', '[*]CC1CCNCC1', 'basic'),
    ('morpholin-2-ylmethyl', '[*]CC1CNCCO1', 'basic'),
    ('1-methylpiperidin-4-yl', '[*]C1CCN(C)CC1', 'basic'),
    ('N-ethylamide', '[*]C(=O)NCC', 'polar'),
    ('N-propylamide', '[*]C(=O)NCCC', 'polar'),
    ('N-isopropylamide', '[*]C(=O)NC(C)C', 'polar'),
    ('piperidine-1-carbonyl', '[*]C(=O)N1CCCCC1', 'polar'),
    ('pyrrolidine-1-carbonyl', '[*]C(=O)N1CCCC1', 'polar'),
    ('azetidine-1-carbonyl', '[*]C(=O)N1CCC1', 'polar'),
    ('dimethylaminocarbonyl', '[*]C(=O)N(C)C', 'polar'),
    ('diethylaminocarbonyl', '[*]C(=O)N(CC)CC', 'polar'),
    ('N-methylsulfonamide', '[*]S(=O)(=O)NC', 'polar'),
    ('N-ethylsulfonamide', '[*]S(=O)(=O)NCC', 'polar'),
    ('N-dimethylsulfonamide', '[*]S(=O)(=O)N(C)C', 'polar'),
    ('morpholine-4-sulfonyl', '[*]S(=O)(=O)N1CCOCC1', 'polar'),
    ('piperidine-1-sulfonyl', '[*]S(=O)(=O)N1CCCCC1', 'polar'),
    ('methoxycarbonyl', '[*]C(=O)OC', 'polar'),
    ('ethoxycarbonyl', '[*]C(=O)OCC', 'polar'),
    ('tert-butoxycarbonyl', '[*]C(=O)OC(C)(C)C', 'polar'),
    ('4-OH-phenyl', '[*]c1ccc(O)cc1', 'aromatic'),
    ('4-OEt-phenyl', '[*]c1ccc(OCC)cc1', 'aromatic'),
    ('4-Et-phenyl', '[*]c1ccc(CC)cc1', 'aromatic'),
    ('4-iPr-phenyl', '[*]c1ccc(C(C)C)cc1', 'aromatic'),
    ('4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)cc1', 'aromatic'),
    ('4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('4-Br-phenyl', '[*]c1ccc(Br)cc1', 'aromatic'),
    ('4-I-phenyl', '[*]c1ccc(I)cc1', 'aromatic'),
    ('4-CN-phenyl', '[*]c1ccc(C#N)cc1', 'aromatic'),
    ('4-NH2-phenyl', '[*]c1ccc(N)cc1', 'aromatic'),
    ('4-NHMe-phenyl', '[*]c1ccc(NC)cc1', 'aromatic'),
    ('4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1', 'aromatic'),
    ('4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1', 'aromatic'),
    ('4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1', 'aromatic'),
    ('4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1', 'aromatic'),
    ('4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1', 'aromatic'),
    ('4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1', 'aromatic'),
    ('3-F-phenyl', '[*]c1cccc(F)c1', 'aromatic'),
    ('3-Cl-phenyl', '[*]c1cccc(Cl)c1', 'aromatic'),
    ('3-Me-phenyl', '[*]c1cccc(C)c1', 'aromatic'),
    ('3-OMe-phenyl', '[*]c1cccc(OC)c1', 'aromatic'),
    ('3-CF3-phenyl', '[*]c1cccc(C(F)(F)F)c1', 'aromatic'),
    ('3-CN-phenyl', '[*]c1cccc(C#N)c1', 'aromatic'),
    ('2-F-phenyl', '[*]c1ccccc1F', 'aromatic'),
    ('2-Cl-phenyl', '[*]c1ccccc1Cl', 'aromatic'),
    ('2-Me-phenyl', '[*]c1ccccc1C', 'aromatic'),
    ('2-OMe-phenyl', '[*]c1ccccc1OC', 'aromatic'),
    ('3,4-diF-phenyl', '[*]c1ccc(F)c(F)c1', 'aromatic'),
    ('3,4-diCl-phenyl', '[*]c1ccc(Cl)c(Cl)c1', 'aromatic'),
    ('3,5-diF-phenyl', '[*]c1cc(F)cc(F)c1', 'aromatic'),
    ('3,5-diCl-phenyl', '[*]c1cc(Cl)cc(Cl)c1', 'aromatic'),
    ('3,4-diMe-phenyl', '[*]c1ccc(C)c(C)c1', 'aromatic'),
    ('4-F-3-Me-phenyl', '[*]c1ccc(F)cc1C', 'aromatic'),
    ('4-Cl-3-CF3-phenyl', '[*]c1ccc(Cl)cc1C(F)(F)F', 'aromatic'),
    ('4-OMe-3-F-phenyl', '[*]c1ccc(OC)c(F)c1', 'aromatic'),
    ('3-OMe-4-F-phenyl', '[*]c1ccc(F)c(OC)c1', 'aromatic'),
    ('2,4-diF-phenyl', '[*]c1ccc(F)cc1F', 'aromatic'),
    ('2,4-diCl-phenyl', '[*]c1ccc(Cl)cc1Cl', 'aromatic'),
    ('2,6-diF-phenyl', '[*]c1c(F)cccc1F', 'aromatic'),
    ('2,6-diCl-phenyl', '[*]c1c(Cl)cccc1Cl', 'aromatic'),
    ('2,3-diF-phenyl', '[*]c1cccc(F)c1F', 'aromatic'),
    ('3,4,5-triF-phenyl', '[*]c1cc(F)c(F)c(F)c1', 'aromatic'),
    ('3,5-diMe-phenyl', '[*]c1c(C)cccc1C', 'aromatic'),
    ('pyrimidin-2-yl', '[*]c1ncccn1', 'aromatic'),
    ('pyrimidin-4-yl', '[*]c1ccncn1', 'aromatic'),
    ('pyrimidin-5-yl', '[*]c1cncnc1', 'aromatic'),
    ('pyrazin-2-yl', '[*]c1cnccn1', 'aromatic'),
    ('pyridazin-3-yl', '[*]c1ccnnc1', 'aromatic'),
    ('thiophen-3-yl', '[*]c1ccsc1', 'aromatic'),
    ('furan-3-yl', '[*]c1ccoc1', 'aromatic'),
    ('imidazol-2-yl', '[*]c1ncc[nH]1', 'aromatic'),
    ('imidazol-4-yl', '[*]c1c[nH]cn1', 'aromatic'),
    ('1,2,4-triazol-1-yl', '[*]n1cncn1', 'aromatic'),
    ('1,2,3-triazol-1-yl', '[*]n1ccnn1', 'aromatic'),
    ('1,2,3-triazol-4-yl', '[*]c1cnn[nH]1', 'aromatic'),
    ('oxazol-2-yl', '[*]c1ncco1', 'aromatic'),
    ('oxazol-4-yl', '[*]c1cnco1', 'aromatic'),
    ('isoxazol-3-yl', '[*]c1cc(no1)', 'aromatic'),
    ('isoxazol-5-yl', '[*]c1cc(on1)', 'aromatic'),
    ('isoxazol-4-yl', '[*]c1conc1', 'aromatic'),
    ('thiazol-4-yl', '[*]c1cncs1', 'aromatic'),
    ('isothiazol-3-yl', '[*]c1ccns1', 'aromatic'),
    ('1,3,4-oxadiazol-2-yl', '[*]c1nnco1', 'aromatic'),
    ('1,2,4-oxadiazol-3-yl', '[*]c1ncno1', 'aromatic'),
    ('1,3,4-thiadiazol-2-yl', '[*]c1nncs1', 'aromatic'),
    ('1,2,4-thiadiazol-3-yl', '[*]c1ncns1', 'aromatic'),
    ('5-Me-pyridin-2-yl', '[*]c1ccc(C)cn1', 'aromatic'),
    ('5-F-pyridin-2-yl', '[*]c1ccc(F)cn1', 'aromatic'),
    ('5-Cl-pyridin-2-yl', '[*]c1ccc(Cl)cn1', 'aromatic'),
    ('5-CF3-pyridin-2-yl', '[*]c1ccc(C(F)(F)F)cn1', 'aromatic'),
    ('6-Me-pyridin-2-yl', '[*]c1cccc(C)n1', 'aromatic'),
    ('4-Me-pyridin-2-yl', '[*]c1cc(C)ccn1', 'aromatic'),
    ('3-F-pyridin-2-yl', '[*]c1cccc(F)n1', 'aromatic'),
    ('5-Me-pyridin-3-yl', '[*]c1cncc(C)c1', 'aromatic'),
    ('6-Me-pyridin-3-yl', '[*]c1ccc(C)nc1', 'aromatic'),
    ('2-Me-pyridin-4-yl', '[*]c1cc(C)ncc1', 'aromatic'),
    ('5-Me-pyrimidin-2-yl', '[*]c1ncc(C)cn1', 'aromatic'),
    ('4-OMe-pyrimidin-2-yl', '[*]c1nc(OC)ccn1', 'aromatic'),
    ('4-NH2-pyrimidin-2-yl', '[*]c1nc(N)ccn1', 'aromatic'),
    ('5-Me-thiophen-2-yl', '[*]c1ccc(C)s1', 'aromatic'),
    ('4-Me-thiophen-2-yl', '[*]c1cc(C)cs1', 'aromatic'),
    ('5-F-thiophen-2-yl', '[*]c1ccc(F)s1', 'aromatic'),
    ('3-Me-thiophen-2-yl', '[*]c1ccsc1C', 'aromatic'),
    ('4-Me-furan-2-yl', '[*]c1cc(C)co1', 'aromatic'),
    ('5-Me-furan-2-yl', '[*]c1ccc(C)o1', 'aromatic'),
    ('3-Me-furan-2-yl', '[*]c1ccoc1C', 'aromatic'),
    ('1-Me-pyrazol-3-yl', '[*]c1ccn(C)n1', 'aromatic'),
    ('1-Me-pyrazol-4-yl', '[*]c1cn(C)nc1', 'aromatic'),
    ('4-Me-oxazol-2-yl', '[*]c1nc(C)co1', 'aromatic'),
    ('5-Me-oxazol-2-yl', '[*]c1ncc(C)o1', 'aromatic'),
    ('4-Me-thiazol-2-yl', '[*]c1nc(C)cs1', 'aromatic'),
    ('5-Me-thiazol-2-yl', '[*]c1sc(C)nc1', 'aromatic'),
    ('4-CF3-thiazol-2-yl', '[*]c1nc(C(F)(F)F)cs1', 'aromatic'),
    ('benzofuran-2-yl', '[*]c1cc2ccccc2o1', 'aromatic'),
    ('benzofuran-5-yl', '[*]c1ccc2occc2c1', 'aromatic'),
    ('benzothiophen-2-yl', '[*]c1cc2ccccc2s1', 'aromatic'),
    ('benzothiophen-5-yl', '[*]c1ccc2sccc2c1', 'aromatic'),
    ('benzoxazol-2-yl', '[*]c1nc2ccccc2o1', 'aromatic'),
    ('benzoxazol-5-yl', '[*]c1ccc2nc(oc2c1)', 'aromatic'),
    ('benzothiazol-2-yl', '[*]c1nc2ccccc2s1', 'aromatic'),
    ('benzothiazol-5-yl', '[*]c1ccc2nc(sc2c1)', 'aromatic'),
    ('1H-indol-2-yl', '[*]c1cc2ccccc2[nH]1', 'aromatic'),
    ('1H-indol-5-yl', '[*]c1ccc2[nH]ccc2c1', 'aromatic'),
    ('isoindol-5-yl', '[*]c1ccc2cc[nH]c2c1', 'aromatic'),
    ('1H-indazol-5-yl', '[*]c1ccc2[nH]ncc2c1', 'aromatic'),
    ('quinolin-2-yl', '[*]c1ccc2ncccc2c1', 'aromatic'),
    ('quinolin-3-yl', '[*]c1cnc2ccccc2c1', 'aromatic'),
    ('quinolin-6-yl', '[*]c1ccc2cccnc2c1', 'aromatic'),
    ('isoquinolin-1-yl', '[*]c1nccc2ccccc12', 'aromatic'),
    ('chroman-6-yl', '[*]c1ccc2CCCOc2c1', 'aromatic'),
    ('octahydroindol-1-yl', '[*]N1CCCCC1C1CCCC1', 'basic'),
    ('decahydroquinolin-1-yl', '[*]N1CCCCC2CCCCC12', 'basic'),
    ('2-azabicyclo[2.2.1]hept-5-en-2-yl', '[*]N1CC2CC1C=C2', 'basic'),
    ('3-azabicyclo[3.1.0]hexyl', '[*]N1CCC2(C1)CC2', 'basic'),
    ('2-oxa-5-azabicyclo[2.2.1]heptyl', '[*]N1CC2COC1C2', 'basic'),
    ('oxetan-2-yl', '[*]C1CCO1', 'bioisostere'),
    ('3-methyloxetan-3-yl', '[*]C1(C)COC1', 'bioisostere'),
    ('oxetane-3-methyl', '[*]CC1COC1', 'bioisostere'),
    ('tetrahydrofuran-2-yl', '[*]C1CCCO1', 'polar'),
    ('tetrahydrofuran-3-yl', '[*]C1COCC1', 'polar'),
    ('tetrahydrofurfuryl', '[*]CC1CCCO1', 'polar'),
    ('tetrahydropyran-2-yl', '[*]C1CCCCO1', 'bioisostere'),
    ('1,3-dioxolan-2-yl', '[*]C1OCCO1', 'polar'),
    ('1,3-dioxan-2-yl', '[*]C1OCCCO1', 'polar'),
    ('1-methylazetidine-3-yl', '[*]C1CN(C)C1', 'bioisostere'),
    ('thietane-3-yl', '[*]C1CSC1', 'bioisostere'),
    ('1,3-dithian-2-yl', '[*]C1SCCCS1', 'polar'),
    ('PEG2_2', '[*]OCCOCCO', 'solubility'),
    ('4-hydroxyalkyl-4', '[*]CCCO', 'solubility'),
    ('5-hydroxyalkyl-5', '[*]CCCCO', 'solubility'),
    ('morpholinomethyl', '[*]CN1CCOCC1', 'solubility'),
    ('3-morpholinopropyl', '[*]CCCN1CCOCC1', 'solubility'),
    ('piperidinomethyl', '[*]CN1CCCCC1', 'solubility'),
    ('piperazinomethyl', '[*]CN1CCNCC1', 'solubility'),
    ('N-methylpiperazinomethyl', '[*]CN1CCN(C)CC1', 'solubility'),
    ('2-piperidinoethyl', '[*]CCN1CCCCC1', 'solubility'),
    ('4-hydroxymethylpiperidyl', '[*]C1CCN(CC)CC1', 'solubility'),
    ('3-hydroxypyrrolidyl', '[*]N1CCC(O)C1', 'solubility'),
    ('3-hydroxymethylpiperidyl', '[*]N1CCCC(CO)C1', 'solubility'),
    ('dimethylaminoethyl', '[*]CCN(C)C', 'solubility'),
    ('dimethylaminopropyl', '[*]CCCN(C)C', 'solubility'),
    ('3-aminopropyl', '[*]CCCN', 'basic'),
    ('4-aminobutyl', '[*]CCCCN', 'basic'),
    ('2-aminoethoxy', '[*]OCCN', 'solubility'),
    ('tetrazole-CH2', '[*]Cc1nnn[nH]1', 'bioisostere'),
    ('acylsulfonamide', '[*]C(=O)NS(C)(=O)=O', 'bioisostere'),
    ('1-Me-tetrazol-5-yl', '[*]c1nnn(C)n1', 'bioisostere'),
    ('cyanamide', '[*]NC#N', 'bioisostere'),
    ('oxadiazol-2-one', '[*]c1nncn1O', 'bioisostere'),
    ('thiadiazolyl-methyl', '[*]Cc1nncs1', 'bioisostere'),
    ('bicyclo[1.1.1]pent-1-yl', '[*]C12CC(CC1)C2', 'bioisostere'),
    ('1-methylcyclopropyl', '[*]C1(C)CC1', 'bioisostere'),
    ('1-fluorocyclopropyl', '[*]C1(F)CC1', 'bioisostere'),
    ('1-trifluoromethylcyclopropyl', '[*]C1(C(F)(F)F)CC1', 'bioisostere'),
    ('1-oxa-6-azaspiro[3.3]heptyl', '[*]N1CC2(COC2)C1', 'bioisostere'),
    ('3-azaspiro[3.3]heptyl', '[*]N1CCC1(CC1)C1', 'bioisostere'),
    ('6-oxa-1-azaspiro[3.3]heptyl', '[*]N1CCC12OCC2', 'bioisostere'),
    ('1,2,3-triazol-4-ylmethyl', '[*]Cc1cnn[nH]1', 'bioisostere'),
    ('propargyl', '[*]CC#C', 'bioisostere'),
    ('prop-1-en-2-yl', '[*]C(=C)C', 'bioisostere'),
    ('4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1', 'halogen'),
    ('4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1', 'halogen'),
    ('4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1', 'polar'),
    ('4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1', 'polar'),
    ('4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1', 'aromatic'),
    ('4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1', 'basic'),
    ('4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1', 'basic'),
    ('4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1', 'basic'),
    ('3-Br-phenyl', '[*]c1cccc(Br)c1', 'aromatic'),
    ('3-Et-phenyl', '[*]c1cccc(CC)c1', 'aromatic'),
    ('3-iPr-phenyl', '[*]c1cccc(C(C)C)c1', 'aromatic'),
    ('3-tBu-phenyl', '[*]c1cccc(C(C)(C)C)c1', 'aromatic'),
    ('3-OEt-phenyl', '[*]c1cccc(OCC)c1', 'aromatic'),
    ('3-OH-phenyl', '[*]c1cccc(O)c1', 'aromatic'),
    ('3-CHF2-phenyl', '[*]c1cccc(C(F)F)c1', 'aromatic'),
    ('3-OCF3-phenyl', '[*]c1cccc(OC(F)(F)F)c1', 'aromatic'),
    ('3-NH2-phenyl', '[*]c1cccc(N)c1', 'aromatic'),
    ('3-NHMe-phenyl', '[*]c1cccc(NC)c1', 'aromatic'),
    ('3-NMe2-phenyl', '[*]c1cccc(N(C)C)c1', 'aromatic'),
    ('3-COOH-phenyl', '[*]c1cccc(C(=O)O)c1', 'aromatic'),
    ('3-COOMe-phenyl', '[*]c1cccc(C(=O)OC)c1', 'aromatic'),
    ('3-COMe-phenyl', '[*]c1cccc(C(C)=O)c1', 'aromatic'),
    ('3-SO2Me-phenyl', '[*]c1cccc(S(C)(=O)=O)c1', 'aromatic'),
    ('3-SO2NH2-phenyl', '[*]c1cccc(S(N)(=O)=O)c1', 'aromatic'),
    ('3-NHAc-phenyl', '[*]c1cccc(NC(C)=O)c1', 'aromatic'),
    ('3-NHSO2Me-phenyl', '[*]c1cccc(NS(C)(=O)=O)c1', 'aromatic'),
    ('3-cyclopropyl-phenyl', '[*]c1cccc(C1CC1)c1', 'aromatic'),
    ('3-morpholino-phenyl', '[*]c1cccc(N1CCOCC1)c1', 'aromatic'),
    ('3-piperidino-phenyl', '[*]c1cccc(N1CCCCC1)c1', 'aromatic'),
    ('3-pyrrolidino-phenyl', '[*]c1cccc(N1CCCC1)c1', 'aromatic'),
    ('2-Br-phenyl', '[*]c1ccccc1Br', 'aromatic'),
    ('2-Et-phenyl', '[*]c1ccccc1CC', 'aromatic'),
    ('2-iPr-phenyl', '[*]c1ccccc1C(C)C', 'aromatic'),
    ('2-tBu-phenyl', '[*]c1ccccc1C(C)(C)C', 'aromatic'),
    ('2-OEt-phenyl', '[*]c1ccccc1OCC', 'aromatic'),
    ('2-OH-phenyl', '[*]c1ccccc1O', 'aromatic'),
    ('2-CF3-phenyl', '[*]c1ccccc1C(F)(F)F', 'aromatic'),
    ('2-CHF2-phenyl', '[*]c1ccccc1C(F)F', 'aromatic'),
    ('2-OCF3-phenyl', '[*]c1ccccc1OC(F)(F)F', 'aromatic'),
    ('2-CN-phenyl', '[*]c1ccccc1C#N', 'aromatic'),
    ('2-NH2-phenyl', '[*]c1ccccc1N', 'aromatic'),
    ('2-NHMe-phenyl', '[*]c1ccccc1NC', 'aromatic'),
    ('2-NMe2-phenyl', '[*]c1ccccc1N(C)C', 'aromatic'),
    ('2-COOH-phenyl', '[*]c1ccccc1C(=O)O', 'aromatic'),
    ('2-COOMe-phenyl', '[*]c1ccccc1C(=O)OC', 'aromatic'),
    ('2-COMe-phenyl', '[*]c1ccccc1C(C)=O', 'aromatic'),
    ('2-SO2Me-phenyl', '[*]c1ccccc1S(C)(=O)=O', 'aromatic'),
    ('2-SO2NH2-phenyl', '[*]c1ccccc1S(N)(=O)=O', 'aromatic'),
    ('2-NHAc-phenyl', '[*]c1ccccc1NC(C)=O', 'aromatic'),
    ('2-NHSO2Me-phenyl', '[*]c1ccccc1NS(C)(=O)=O', 'aromatic'),
    ('2-cyclopropyl-phenyl', '[*]c1ccccc1C1CC1', 'aromatic'),
    ('2-morpholino-phenyl', '[*]c1ccccc1N1CCOCC1', 'aromatic'),
    ('2-piperidino-phenyl', '[*]c1ccccc1N1CCCCC1', 'aromatic'),
    ('2-pyrrolidino-phenyl', '[*]c1ccccc1N1CCCC1', 'aromatic'),
    ('3-F-4-Cl-phenyl', '[*]c1ccc(Cl)c(F)c1', 'aromatic'),
    ('3-F-4-Br-phenyl', '[*]c1ccc(Br)c(F)c1', 'aromatic'),
    ('3-F-4-Me-phenyl', '[*]c1ccc(C)c(F)c1', 'aromatic'),
    ('3-F-4-Et-phenyl', '[*]c1ccc(CC)c(F)c1', 'aromatic'),
    ('3-F-4-iPr-phenyl', '[*]c1ccc(C(C)C)c(F)c1', 'aromatic'),
    ('3-F-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)c(F)c1', 'aromatic'),
    ('3-F-4-OEt-phenyl', '[*]c1ccc(OCC)c(F)c1', 'aromatic'),
    ('3-F-4-OH-phenyl', '[*]c1ccc(O)c(F)c1', 'aromatic'),
    ('3-F-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(F)c1', 'aromatic'),
    ('3-F-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(F)c1', 'aromatic'),
    ('3-F-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(F)c1', 'aromatic'),
    ('3-F-4-CN-phenyl', '[*]c1ccc(C#N)c(F)c1', 'aromatic'),
    ('3-F-4-NH2-phenyl', '[*]c1ccc(N)c(F)c1', 'aromatic'),
    ('3-F-4-NHMe-phenyl', '[*]c1ccc(NC)c(F)c1', 'aromatic'),
    ('3-F-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(F)c1', 'aromatic'),
    ('3-F-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(F)c1', 'aromatic'),
    ('3-F-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(F)c1', 'aromatic'),
    ('3-F-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(F)c1', 'aromatic'),
    ('3-F-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(F)c1', 'aromatic'),
    ('3-F-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(F)c1', 'aromatic'),
    ('3-F-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(F)c1', 'aromatic'),
    ('3-F-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(F)c1', 'aromatic'),
    ('3-F-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(F)c1', 'aromatic'),
    ('3-F-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(F)c1', 'aromatic'),
    ('3-F-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(F)c1', 'aromatic'),
    ('3-F-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(F)c1', 'aromatic'),
    ('3-Cl-4-Br-phenyl', '[*]c1ccc(Br)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-Me-phenyl', '[*]c1ccc(C)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-Et-phenyl', '[*]c1ccc(CC)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-iPr-phenyl', '[*]c1ccc(C(C)C)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-OMe-phenyl', '[*]c1ccc(OC)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-OEt-phenyl', '[*]c1ccc(OCC)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-OH-phenyl', '[*]c1ccc(O)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-CN-phenyl', '[*]c1ccc(C#N)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-NH2-phenyl', '[*]c1ccc(N)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-NHMe-phenyl', '[*]c1ccc(NC)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(Cl)c1', 'aromatic'),
    ('3-Cl-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(Cl)c1', 'aromatic'),
    ('3-Br-4-Me-phenyl', '[*]c1ccc(C)c(Br)c1', 'aromatic'),
    ('3-Br-4-Et-phenyl', '[*]c1ccc(CC)c(Br)c1', 'aromatic'),
    ('3-Br-4-iPr-phenyl', '[*]c1ccc(C(C)C)c(Br)c1', 'aromatic'),
    ('3-Br-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)c(Br)c1', 'aromatic'),
    ('3-Br-4-OMe-phenyl', '[*]c1ccc(OC)c(Br)c1', 'aromatic'),
    ('3-Br-4-OEt-phenyl', '[*]c1ccc(OCC)c(Br)c1', 'aromatic'),
    ('3-Br-4-OH-phenyl', '[*]c1ccc(O)c(Br)c1', 'aromatic'),
    ('3-Br-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(Br)c1', 'aromatic'),
    ('3-Br-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(Br)c1', 'aromatic'),
    ('3-Br-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(Br)c1', 'aromatic'),
    ('3-Br-4-CN-phenyl', '[*]c1ccc(C#N)c(Br)c1', 'aromatic'),
    ('3-Br-4-NH2-phenyl', '[*]c1ccc(N)c(Br)c1', 'aromatic'),
    ('3-Br-4-NHMe-phenyl', '[*]c1ccc(NC)c(Br)c1', 'aromatic'),
    ('3-Br-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(Br)c1', 'aromatic'),
    ('3-Br-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(Br)c1', 'aromatic'),
    ('3-Br-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(Br)c1', 'aromatic'),
    ('3-Br-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(Br)c1', 'aromatic'),
    ('3-Br-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(Br)c1', 'aromatic'),
    ('3-Br-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(Br)c1', 'aromatic'),
    ('3-Br-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(Br)c1', 'aromatic'),
    ('3-Br-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(Br)c1', 'aromatic'),
    ('3-Br-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(Br)c1', 'aromatic'),
    ('3-Br-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(Br)c1', 'aromatic'),
    ('3-Br-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(Br)c1', 'aromatic'),
    ('3-Me-4-Et-phenyl', '[*]c1ccc(CC)c(C)c1', 'aromatic'),
    ('3-Me-4-iPr-phenyl', '[*]c1ccc(C(C)C)c(C)c1', 'aromatic'),
    ('3-Me-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)c(C)c1', 'aromatic'),
    ('3-Me-4-OMe-phenyl', '[*]c1ccc(OC)c(C)c1', 'aromatic'),
    ('3-Me-4-OEt-phenyl', '[*]c1ccc(OCC)c(C)c1', 'aromatic'),
    ('3-Me-4-OH-phenyl', '[*]c1ccc(O)c(C)c1', 'aromatic'),
    ('3-Me-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(C)c1', 'aromatic'),
    ('3-Me-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(C)c1', 'aromatic'),
    ('3-Me-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(C)c1', 'aromatic'),
    ('3-Me-4-CN-phenyl', '[*]c1ccc(C#N)c(C)c1', 'aromatic'),
    ('3-Me-4-NH2-phenyl', '[*]c1ccc(N)c(C)c1', 'aromatic'),
    ('3-Me-4-NHMe-phenyl', '[*]c1ccc(NC)c(C)c1', 'aromatic'),
    ('3-Me-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(C)c1', 'aromatic'),
    ('3-Me-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(C)c1', 'aromatic'),
    ('3-Me-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(C)c1', 'aromatic'),
    ('3-Me-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(C)c1', 'aromatic'),
    ('3-Me-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(C)c1', 'aromatic'),
    ('3-Me-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(C)c1', 'aromatic'),
    ('3-Me-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(C)c1', 'aromatic'),
    ('3-Me-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(C)c1', 'aromatic'),
    ('3-Me-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(C)c1', 'aromatic'),
    ('3-Me-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(C)c1', 'aromatic'),
    ('3-Me-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(C)c1', 'aromatic'),
    ('3-Me-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(C)c1', 'aromatic'),
    ('3-Et-4-iPr-phenyl', '[*]c1ccc(C(C)C)c(CC)c1', 'aromatic'),
    ('3-Et-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)c(CC)c1', 'aromatic'),
    ('3-Et-4-OMe-phenyl', '[*]c1ccc(OC)c(CC)c1', 'aromatic'),
    ('3-Et-4-OEt-phenyl', '[*]c1ccc(OCC)c(CC)c1', 'aromatic'),
    ('3-Et-4-OH-phenyl', '[*]c1ccc(O)c(CC)c1', 'aromatic'),
    ('3-Et-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(CC)c1', 'aromatic'),
    ('3-Et-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(CC)c1', 'aromatic'),
    ('3-Et-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(CC)c1', 'aromatic'),
    ('3-Et-4-CN-phenyl', '[*]c1ccc(C#N)c(CC)c1', 'aromatic'),
    ('3-Et-4-NH2-phenyl', '[*]c1ccc(N)c(CC)c1', 'aromatic'),
    ('3-Et-4-NHMe-phenyl', '[*]c1ccc(NC)c(CC)c1', 'aromatic'),
    ('3-Et-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(CC)c1', 'aromatic'),
    ('3-Et-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(CC)c1', 'aromatic'),
    ('3-Et-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(CC)c1', 'aromatic'),
    ('3-Et-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(CC)c1', 'aromatic'),
    ('3-Et-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(CC)c1', 'aromatic'),
    ('3-Et-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(CC)c1', 'aromatic'),
    ('3-Et-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(CC)c1', 'aromatic'),
    ('3-Et-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(CC)c1', 'aromatic'),
    ('3-Et-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(CC)c1', 'aromatic'),
    ('3-Et-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(CC)c1', 'aromatic'),
    ('3-Et-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(CC)c1', 'aromatic'),
    ('3-Et-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(CC)c1', 'aromatic'),
    ('3-iPr-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-OMe-phenyl', '[*]c1ccc(OC)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-OEt-phenyl', '[*]c1ccc(OCC)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-OH-phenyl', '[*]c1ccc(O)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-CN-phenyl', '[*]c1ccc(C#N)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-NH2-phenyl', '[*]c1ccc(N)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-NHMe-phenyl', '[*]c1ccc(NC)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(C(C)C)c1', 'aromatic'),
    ('3-iPr-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(C(C)C)c1', 'aromatic'),
    ('3-tBu-4-OMe-phenyl', '[*]c1ccc(OC)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-OEt-phenyl', '[*]c1ccc(OCC)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-OH-phenyl', '[*]c1ccc(O)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-CN-phenyl', '[*]c1ccc(C#N)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-NH2-phenyl', '[*]c1ccc(N)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-NHMe-phenyl', '[*]c1ccc(NC)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(C(C)(C)C)c1', 'aromatic'),
    ('3-tBu-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(C(C)(C)C)c1', 'aromatic'),
    ('3-OMe-4-OEt-phenyl', '[*]c1ccc(OCC)c(OC)c1', 'aromatic'),
    ('3-OMe-4-OH-phenyl', '[*]c1ccc(O)c(OC)c1', 'aromatic'),
    ('3-OMe-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(OC)c1', 'aromatic'),
    ('3-OMe-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(OC)c1', 'aromatic'),
    ('3-OMe-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(OC)c1', 'aromatic'),
    ('3-OMe-4-CN-phenyl', '[*]c1ccc(C#N)c(OC)c1', 'aromatic'),
    ('3-OMe-4-NH2-phenyl', '[*]c1ccc(N)c(OC)c1', 'aromatic'),
    ('3-OMe-4-NHMe-phenyl', '[*]c1ccc(NC)c(OC)c1', 'aromatic'),
    ('3-OMe-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(OC)c1', 'aromatic'),
    ('3-OMe-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(OC)c1', 'aromatic'),
    ('3-OMe-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(OC)c1', 'aromatic'),
    ('3-OMe-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(OC)c1', 'aromatic'),
    ('3-OMe-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(OC)c1', 'aromatic'),
    ('3-OMe-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(OC)c1', 'aromatic'),
    ('3-OMe-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(OC)c1', 'aromatic'),
    ('3-OMe-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(OC)c1', 'aromatic'),
    ('3-OMe-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(OC)c1', 'aromatic'),
    ('3-OMe-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(OC)c1', 'aromatic'),
    ('3-OMe-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(OC)c1', 'aromatic'),
    ('3-OMe-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(OC)c1', 'aromatic'),
    ('3-OEt-4-OH-phenyl', '[*]c1ccc(O)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-CN-phenyl', '[*]c1ccc(C#N)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-NH2-phenyl', '[*]c1ccc(N)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-NHMe-phenyl', '[*]c1ccc(NC)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(OCC)c1', 'aromatic'),
    ('3-OEt-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(OCC)c1', 'aromatic'),
    ('3-OH-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(O)c1', 'aromatic'),
    ('3-OH-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(O)c1', 'aromatic'),
    ('3-OH-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(O)c1', 'aromatic'),
    ('3-OH-4-CN-phenyl', '[*]c1ccc(C#N)c(O)c1', 'aromatic'),
    ('3-OH-4-NH2-phenyl', '[*]c1ccc(N)c(O)c1', 'aromatic'),
    ('3-OH-4-NHMe-phenyl', '[*]c1ccc(NC)c(O)c1', 'aromatic'),
    ('3-OH-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(O)c1', 'aromatic'),
    ('3-OH-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(O)c1', 'aromatic'),
    ('3-OH-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(O)c1', 'aromatic'),
    ('3-OH-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(O)c1', 'aromatic'),
    ('3-OH-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(O)c1', 'aromatic'),
    ('3-OH-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(O)c1', 'aromatic'),
    ('3-OH-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(O)c1', 'aromatic'),
    ('3-OH-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(O)c1', 'aromatic'),
    ('3-OH-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(O)c1', 'aromatic'),
    ('3-OH-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(O)c1', 'aromatic'),
    ('3-OH-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(O)c1', 'aromatic'),
    ('3-OH-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(O)c1', 'aromatic'),
    ('3-CF3-4-CHF2-phenyl', '[*]c1ccc(C(F)F)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-CN-phenyl', '[*]c1ccc(C#N)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-NH2-phenyl', '[*]c1ccc(N)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-NHMe-phenyl', '[*]c1ccc(NC)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CHF2-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-CN-phenyl', '[*]c1ccc(C#N)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-NH2-phenyl', '[*]c1ccc(N)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-NHMe-phenyl', '[*]c1ccc(NC)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(C(F)F)c1', 'aromatic'),
    ('3-CHF2-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(C(F)F)c1', 'aromatic'),
    ('3-OCF3-4-CN-phenyl', '[*]c1ccc(C#N)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-NH2-phenyl', '[*]c1ccc(N)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-NHMe-phenyl', '[*]c1ccc(NC)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-OCF3-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(OC(F)(F)F)c1', 'aromatic'),
    ('3-CN-4-NH2-phenyl', '[*]c1ccc(N)c(C#N)c1', 'aromatic'),
    ('3-CN-4-NHMe-phenyl', '[*]c1ccc(NC)c(C#N)c1', 'aromatic'),
    ('3-CN-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(C#N)c1', 'aromatic'),
    ('3-CN-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(C#N)c1', 'aromatic'),
    ('3-CN-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(C#N)c1', 'aromatic'),
    ('3-CN-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(C#N)c1', 'aromatic'),
    ('3-CN-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(C#N)c1', 'aromatic'),
    ('3-CN-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(C#N)c1', 'aromatic'),
    ('3-CN-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(C#N)c1', 'aromatic'),
    ('3-CN-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(C#N)c1', 'aromatic'),
    ('3-CN-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(C#N)c1', 'aromatic'),
    ('3-CN-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(C#N)c1', 'aromatic'),
    ('3-CN-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(C#N)c1', 'aromatic'),
    ('3-CN-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(C#N)c1', 'aromatic'),
    ('3-NH2-4-NHMe-phenyl', '[*]c1ccc(NC)c(N)c1', 'aromatic'),
    ('3-NH2-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(N)c1', 'aromatic'),
    ('3-NH2-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(N)c1', 'aromatic'),
    ('3-NH2-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(N)c1', 'aromatic'),
    ('3-NH2-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(N)c1', 'aromatic'),
    ('3-NH2-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(N)c1', 'aromatic'),
    ('3-NH2-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(N)c1', 'aromatic'),
    ('3-NH2-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(N)c1', 'aromatic'),
    ('3-NH2-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(N)c1', 'aromatic'),
    ('3-NH2-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(N)c1', 'aromatic'),
    ('3-NH2-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(N)c1', 'aromatic'),
    ('3-NH2-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(N)c1', 'aromatic'),
    ('3-NH2-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(N)c1', 'aromatic'),
    ('3-NHMe-4-NMe2-phenyl', '[*]c1ccc(N(C)C)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(NC)c1', 'aromatic'),
    ('3-NHMe-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(NC)c1', 'aromatic'),
    ('3-NMe2-4-COOH-phenyl', '[*]c1ccc(C(=O)O)c(N(C)C)c1', 'aromatic'),
    ('3-NMe2-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(N(C)C)c1', 'aromatic'),
    ('3-NMe2-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(N(C)C)c1', 'aromatic'),
    ('3-NMe2-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(N(C)C)c1', 'aromatic'),
    ('3-NMe2-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(N(C)C)c1', 'aromatic'),
    ('3-NMe2-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(N(C)C)c1', 'aromatic'),
    ('3-NMe2-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(N(C)C)c1', 'aromatic'),
    ('3-NMe2-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(N(C)C)c1', 'aromatic'),
    ('3-NMe2-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(N(C)C)c1', 'aromatic'),
    ('3-NMe2-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(N(C)C)c1', 'aromatic'),
    ('3-NMe2-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(N(C)C)c1', 'aromatic'),
    ('3-COOH-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)c(C(=O)O)c1', 'aromatic'),
    ('3-COOH-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(C(=O)O)c1', 'aromatic'),
    ('3-COOH-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(C(=O)O)c1', 'aromatic'),
    ('3-COOH-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(C(=O)O)c1', 'aromatic'),
    ('3-COOH-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(C(=O)O)c1', 'aromatic'),
    ('3-COOH-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(C(=O)O)c1', 'aromatic'),
    ('3-COOH-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(C(=O)O)c1', 'aromatic'),
    ('3-COOH-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(C(=O)O)c1', 'aromatic'),
    ('3-COOH-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(C(=O)O)c1', 'aromatic'),
    ('3-COOH-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(C(=O)O)c1', 'aromatic'),
    ('3-COOMe-4-COMe-phenyl', '[*]c1ccc(C(C)=O)c(C(=O)OC)c1', 'aromatic'),
    ('3-COOMe-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(C(=O)OC)c1', 'aromatic'),
    ('3-COOMe-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(C(=O)OC)c1', 'aromatic'),
    ('3-COOMe-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(C(=O)OC)c1', 'aromatic'),
    ('3-COOMe-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(C(=O)OC)c1', 'aromatic'),
    ('3-COOMe-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(C(=O)OC)c1', 'aromatic'),
    ('3-COOMe-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(C(=O)OC)c1', 'aromatic'),
    ('3-COOMe-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(C(=O)OC)c1', 'aromatic'),
    ('3-COOMe-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(C(=O)OC)c1', 'aromatic'),
    ('3-COMe-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)c(C(C)=O)c1', 'aromatic'),
    ('3-COMe-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(C(C)=O)c1', 'aromatic'),
    ('3-COMe-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(C(C)=O)c1', 'aromatic'),
    ('3-COMe-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(C(C)=O)c1', 'aromatic'),
    ('3-COMe-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(C(C)=O)c1', 'aromatic'),
    ('3-COMe-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(C(C)=O)c1', 'aromatic'),
    ('3-COMe-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(C(C)=O)c1', 'aromatic'),
    ('3-COMe-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(C(C)=O)c1', 'aromatic'),
    ('3-SO2Me-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)c(S(C)(=O)=O)c1', 'aromatic'),
    ('3-SO2Me-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(S(C)(=O)=O)c1', 'aromatic'),
    ('3-SO2Me-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(S(C)(=O)=O)c1', 'aromatic'),
    ('3-SO2Me-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(S(C)(=O)=O)c1', 'aromatic'),
    ('3-SO2Me-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(S(C)(=O)=O)c1', 'aromatic'),
    ('3-SO2Me-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(S(C)(=O)=O)c1', 'aromatic'),
    ('3-SO2Me-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(S(C)(=O)=O)c1', 'aromatic'),
    ('3-SO2NH2-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)c(S(N)(=O)=O)c1', 'aromatic'),
    ('3-SO2NH2-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(S(N)(=O)=O)c1', 'aromatic'),
    ('3-SO2NH2-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(S(N)(=O)=O)c1', 'aromatic'),
    ('3-SO2NH2-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(S(N)(=O)=O)c1', 'aromatic'),
    ('3-SO2NH2-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(S(N)(=O)=O)c1', 'aromatic'),
    ('3-NHAc-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)c(NC(C)=O)c1', 'aromatic'),
    ('3-NHAc-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(NC(C)=O)c1', 'aromatic'),
    ('3-NHAc-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(NC(C)=O)c1', 'aromatic'),
    ('3-NHAc-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(NC(C)=O)c1', 'aromatic'),
    ('3-NHAc-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(NC(C)=O)c1', 'aromatic'),
    ('3-NHSO2Me-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)c(NS(C)(=O)=O)c1', 'aromatic'),
    ('3-NHSO2Me-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(NS(C)(=O)=O)c1', 'aromatic'),
    ('3-cyclopropyl-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)c(C1CC1)c1', 'aromatic'),
    ('3-cyclopropyl-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(C1CC1)c1', 'aromatic'),
    ('3-cyclopropyl-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(C1CC1)c1', 'aromatic'),
    ('3-morpholino-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)c(N1CCOCC1)c1', 'aromatic'),
    ('3-morpholino-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(N1CCOCC1)c1', 'aromatic'),
    ('3-piperidino-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)c(N1CCCCC1)c1', 'aromatic'),
    ('2-F-4-Cl-phenyl', '[*]c1ccc(Cl)cc1F', 'aromatic'),
    ('2-F-4-Br-phenyl', '[*]c1ccc(Br)cc1F', 'aromatic'),
    ('2-F-4-Me-phenyl', '[*]c1ccc(C)cc1F', 'aromatic'),
    ('2-F-4-Et-phenyl', '[*]c1ccc(CC)cc1F', 'aromatic'),
    ('2-F-4-iPr-phenyl', '[*]c1ccc(C(C)C)cc1F', 'aromatic'),
    ('2-F-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)cc1F', 'aromatic'),
    ('2-F-4-OMe-phenyl', '[*]c1ccc(OC)cc1F', 'aromatic'),
    ('2-F-4-OEt-phenyl', '[*]c1ccc(OCC)cc1F', 'aromatic'),
    ('2-F-4-OH-phenyl', '[*]c1ccc(O)cc1F', 'aromatic'),
    ('2-F-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1F', 'aromatic'),
    ('2-F-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1F', 'aromatic'),
    ('2-F-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1F', 'aromatic'),
    ('2-F-4-CN-phenyl', '[*]c1ccc(C#N)cc1F', 'aromatic'),
    ('2-F-4-NH2-phenyl', '[*]c1ccc(N)cc1F', 'aromatic'),
    ('2-F-4-NHMe-phenyl', '[*]c1ccc(NC)cc1F', 'aromatic'),
    ('2-F-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1F', 'aromatic'),
    ('2-F-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1F', 'aromatic'),
    ('2-F-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1F', 'aromatic'),
    ('2-F-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1F', 'aromatic'),
    ('2-F-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1F', 'aromatic'),
    ('2-F-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1F', 'aromatic'),
    ('2-F-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1F', 'aromatic'),
    ('2-F-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1F', 'aromatic'),
    ('2-F-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1F', 'aromatic'),
    ('2-F-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1F', 'aromatic'),
    ('2-F-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1F', 'aromatic'),
    ('2-F-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1F', 'aromatic'),
    ('2-Cl-4-Br-phenyl', '[*]c1ccc(Br)cc1Cl', 'aromatic'),
    ('2-Cl-4-Me-phenyl', '[*]c1ccc(C)cc1Cl', 'aromatic'),
    ('2-Cl-4-Et-phenyl', '[*]c1ccc(CC)cc1Cl', 'aromatic'),
    ('2-Cl-4-iPr-phenyl', '[*]c1ccc(C(C)C)cc1Cl', 'aromatic'),
    ('2-Cl-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)cc1Cl', 'aromatic'),
    ('2-Cl-4-OMe-phenyl', '[*]c1ccc(OC)cc1Cl', 'aromatic'),
    ('2-Cl-4-OEt-phenyl', '[*]c1ccc(OCC)cc1Cl', 'aromatic'),
    ('2-Cl-4-OH-phenyl', '[*]c1ccc(O)cc1Cl', 'aromatic'),
    ('2-Cl-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1Cl', 'aromatic'),
    ('2-Cl-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1Cl', 'aromatic'),
    ('2-Cl-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1Cl', 'aromatic'),
    ('2-Cl-4-CN-phenyl', '[*]c1ccc(C#N)cc1Cl', 'aromatic'),
    ('2-Cl-4-NH2-phenyl', '[*]c1ccc(N)cc1Cl', 'aromatic'),
    ('2-Cl-4-NHMe-phenyl', '[*]c1ccc(NC)cc1Cl', 'aromatic'),
    ('2-Cl-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1Cl', 'aromatic'),
    ('2-Cl-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1Cl', 'aromatic'),
    ('2-Cl-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1Cl', 'aromatic'),
    ('2-Cl-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1Cl', 'aromatic'),
    ('2-Cl-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1Cl', 'aromatic'),
    ('2-Cl-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1Cl', 'aromatic'),
    ('2-Cl-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1Cl', 'aromatic'),
    ('2-Cl-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1Cl', 'aromatic'),
    ('2-Cl-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1Cl', 'aromatic'),
    ('2-Cl-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1Cl', 'aromatic'),
    ('2-Cl-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1Cl', 'aromatic'),
    ('2-Cl-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1Cl', 'aromatic'),
    ('2-Br-4-Me-phenyl', '[*]c1ccc(C)cc1Br', 'aromatic'),
    ('2-Br-4-Et-phenyl', '[*]c1ccc(CC)cc1Br', 'aromatic'),
    ('2-Br-4-iPr-phenyl', '[*]c1ccc(C(C)C)cc1Br', 'aromatic'),
    ('2-Br-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)cc1Br', 'aromatic'),
    ('2-Br-4-OMe-phenyl', '[*]c1ccc(OC)cc1Br', 'aromatic'),
    ('2-Br-4-OEt-phenyl', '[*]c1ccc(OCC)cc1Br', 'aromatic'),
    ('2-Br-4-OH-phenyl', '[*]c1ccc(O)cc1Br', 'aromatic'),
    ('2-Br-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1Br', 'aromatic'),
    ('2-Br-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1Br', 'aromatic'),
    ('2-Br-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1Br', 'aromatic'),
    ('2-Br-4-CN-phenyl', '[*]c1ccc(C#N)cc1Br', 'aromatic'),
    ('2-Br-4-NH2-phenyl', '[*]c1ccc(N)cc1Br', 'aromatic'),
    ('2-Br-4-NHMe-phenyl', '[*]c1ccc(NC)cc1Br', 'aromatic'),
    ('2-Br-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1Br', 'aromatic'),
    ('2-Br-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1Br', 'aromatic'),
    ('2-Br-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1Br', 'aromatic'),
    ('2-Br-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1Br', 'aromatic'),
    ('2-Br-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1Br', 'aromatic'),
    ('2-Br-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1Br', 'aromatic'),
    ('2-Br-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1Br', 'aromatic'),
    ('2-Br-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1Br', 'aromatic'),
    ('2-Br-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1Br', 'aromatic'),
    ('2-Br-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1Br', 'aromatic'),
    ('2-Br-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1Br', 'aromatic'),
    ('2-Me-4-Et-phenyl', '[*]c1ccc(CC)cc1C', 'aromatic'),
    ('2-Me-4-iPr-phenyl', '[*]c1ccc(C(C)C)cc1C', 'aromatic'),
    ('2-Me-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)cc1C', 'aromatic'),
    ('2-Me-4-OMe-phenyl', '[*]c1ccc(OC)cc1C', 'aromatic'),
    ('2-Me-4-OEt-phenyl', '[*]c1ccc(OCC)cc1C', 'aromatic'),
    ('2-Me-4-OH-phenyl', '[*]c1ccc(O)cc1C', 'aromatic'),
    ('2-Me-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1C', 'aromatic'),
    ('2-Me-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1C', 'aromatic'),
    ('2-Me-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1C', 'aromatic'),
    ('2-Me-4-CN-phenyl', '[*]c1ccc(C#N)cc1C', 'aromatic'),
    ('2-Me-4-NH2-phenyl', '[*]c1ccc(N)cc1C', 'aromatic'),
    ('2-Me-4-NHMe-phenyl', '[*]c1ccc(NC)cc1C', 'aromatic'),
    ('2-Me-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1C', 'aromatic'),
    ('2-Me-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1C', 'aromatic'),
    ('2-Me-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1C', 'aromatic'),
    ('2-Me-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1C', 'aromatic'),
    ('2-Me-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1C', 'aromatic'),
    ('2-Me-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1C', 'aromatic'),
    ('2-Me-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1C', 'aromatic'),
    ('2-Me-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1C', 'aromatic'),
    ('2-Me-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1C', 'aromatic'),
    ('2-Me-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1C', 'aromatic'),
    ('2-Me-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1C', 'aromatic'),
    ('2-Me-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1C', 'aromatic'),
    ('2-Et-4-iPr-phenyl', '[*]c1ccc(C(C)C)cc1CC', 'aromatic'),
    ('2-Et-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)cc1CC', 'aromatic'),
    ('2-Et-4-OMe-phenyl', '[*]c1ccc(OC)cc1CC', 'aromatic'),
    ('2-Et-4-OEt-phenyl', '[*]c1ccc(OCC)cc1CC', 'aromatic'),
    ('2-Et-4-OH-phenyl', '[*]c1ccc(O)cc1CC', 'aromatic'),
    ('2-Et-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1CC', 'aromatic'),
    ('2-Et-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1CC', 'aromatic'),
    ('2-Et-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1CC', 'aromatic'),
    ('2-Et-4-CN-phenyl', '[*]c1ccc(C#N)cc1CC', 'aromatic'),
    ('2-Et-4-NH2-phenyl', '[*]c1ccc(N)cc1CC', 'aromatic'),
    ('2-Et-4-NHMe-phenyl', '[*]c1ccc(NC)cc1CC', 'aromatic'),
    ('2-Et-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1CC', 'aromatic'),
    ('2-Et-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1CC', 'aromatic'),
    ('2-Et-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1CC', 'aromatic'),
    ('2-Et-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1CC', 'aromatic'),
    ('2-Et-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1CC', 'aromatic'),
    ('2-Et-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1CC', 'aromatic'),
    ('2-Et-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1CC', 'aromatic'),
    ('2-Et-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1CC', 'aromatic'),
    ('2-Et-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1CC', 'aromatic'),
    ('2-Et-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1CC', 'aromatic'),
    ('2-Et-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1CC', 'aromatic'),
    ('2-Et-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1CC', 'aromatic'),
    ('2-iPr-4-tBu-phenyl', '[*]c1ccc(C(C)(C)C)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-OMe-phenyl', '[*]c1ccc(OC)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-OEt-phenyl', '[*]c1ccc(OCC)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-OH-phenyl', '[*]c1ccc(O)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-CN-phenyl', '[*]c1ccc(C#N)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-NH2-phenyl', '[*]c1ccc(N)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-NHMe-phenyl', '[*]c1ccc(NC)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1C(C)C', 'aromatic'),
    ('2-iPr-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1C(C)C', 'aromatic'),
    ('2-tBu-4-OMe-phenyl', '[*]c1ccc(OC)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-OEt-phenyl', '[*]c1ccc(OCC)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-OH-phenyl', '[*]c1ccc(O)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-CN-phenyl', '[*]c1ccc(C#N)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-NH2-phenyl', '[*]c1ccc(N)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-NHMe-phenyl', '[*]c1ccc(NC)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1C(C)(C)C', 'aromatic'),
    ('2-tBu-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1C(C)(C)C', 'aromatic'),
    ('2-OMe-4-OEt-phenyl', '[*]c1ccc(OCC)cc1OC', 'aromatic'),
    ('2-OMe-4-OH-phenyl', '[*]c1ccc(O)cc1OC', 'aromatic'),
    ('2-OMe-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1OC', 'aromatic'),
    ('2-OMe-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1OC', 'aromatic'),
    ('2-OMe-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1OC', 'aromatic'),
    ('2-OMe-4-CN-phenyl', '[*]c1ccc(C#N)cc1OC', 'aromatic'),
    ('2-OMe-4-NH2-phenyl', '[*]c1ccc(N)cc1OC', 'aromatic'),
    ('2-OMe-4-NHMe-phenyl', '[*]c1ccc(NC)cc1OC', 'aromatic'),
    ('2-OMe-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1OC', 'aromatic'),
    ('2-OMe-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1OC', 'aromatic'),
    ('2-OMe-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1OC', 'aromatic'),
    ('2-OMe-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1OC', 'aromatic'),
    ('2-OMe-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1OC', 'aromatic'),
    ('2-OMe-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1OC', 'aromatic'),
    ('2-OMe-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1OC', 'aromatic'),
    ('2-OMe-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1OC', 'aromatic'),
    ('2-OMe-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1OC', 'aromatic'),
    ('2-OMe-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1OC', 'aromatic'),
    ('2-OMe-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1OC', 'aromatic'),
    ('2-OMe-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1OC', 'aromatic'),
    ('2-OEt-4-OH-phenyl', '[*]c1ccc(O)cc1OCC', 'aromatic'),
    ('2-OEt-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1OCC', 'aromatic'),
    ('2-OEt-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1OCC', 'aromatic'),
    ('2-OEt-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1OCC', 'aromatic'),
    ('2-OEt-4-CN-phenyl', '[*]c1ccc(C#N)cc1OCC', 'aromatic'),
    ('2-OEt-4-NH2-phenyl', '[*]c1ccc(N)cc1OCC', 'aromatic'),
    ('2-OEt-4-NHMe-phenyl', '[*]c1ccc(NC)cc1OCC', 'aromatic'),
    ('2-OEt-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1OCC', 'aromatic'),
    ('2-OEt-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1OCC', 'aromatic'),
    ('2-OEt-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1OCC', 'aromatic'),
    ('2-OEt-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1OCC', 'aromatic'),
    ('2-OEt-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1OCC', 'aromatic'),
    ('2-OEt-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1OCC', 'aromatic'),
    ('2-OEt-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1OCC', 'aromatic'),
    ('2-OEt-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1OCC', 'aromatic'),
    ('2-OEt-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1OCC', 'aromatic'),
    ('2-OEt-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1OCC', 'aromatic'),
    ('2-OEt-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1OCC', 'aromatic'),
    ('2-OEt-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1OCC', 'aromatic'),
    ('2-OH-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1O', 'aromatic'),
    ('2-OH-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1O', 'aromatic'),
    ('2-OH-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1O', 'aromatic'),
    ('2-OH-4-CN-phenyl', '[*]c1ccc(C#N)cc1O', 'aromatic'),
    ('2-OH-4-NH2-phenyl', '[*]c1ccc(N)cc1O', 'aromatic'),
    ('2-OH-4-NHMe-phenyl', '[*]c1ccc(NC)cc1O', 'aromatic'),
    ('2-OH-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1O', 'aromatic'),
    ('2-OH-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1O', 'aromatic'),
    ('2-OH-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1O', 'aromatic'),
    ('2-OH-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1O', 'aromatic'),
    ('2-OH-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1O', 'aromatic'),
    ('2-OH-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1O', 'aromatic'),
    ('2-OH-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1O', 'aromatic'),
    ('2-OH-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1O', 'aromatic'),
    ('2-OH-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1O', 'aromatic'),
    ('2-OH-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1O', 'aromatic'),
    ('2-OH-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1O', 'aromatic'),
    ('2-OH-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1O', 'aromatic'),
    ('2-CF3-4-CHF2-phenyl', '[*]c1ccc(C(F)F)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-CN-phenyl', '[*]c1ccc(C#N)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-NH2-phenyl', '[*]c1ccc(N)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-NHMe-phenyl', '[*]c1ccc(NC)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1C(F)(F)F', 'aromatic'),
    ('2-CHF2-4-OCF3-phenyl', '[*]c1ccc(OC(F)(F)F)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-CN-phenyl', '[*]c1ccc(C#N)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-NH2-phenyl', '[*]c1ccc(N)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-NHMe-phenyl', '[*]c1ccc(NC)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1C(F)F', 'aromatic'),
    ('2-CHF2-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1C(F)F', 'aromatic'),
    ('2-OCF3-4-CN-phenyl', '[*]c1ccc(C#N)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-NH2-phenyl', '[*]c1ccc(N)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-NHMe-phenyl', '[*]c1ccc(NC)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1OC(F)(F)F', 'aromatic'),
    ('2-OCF3-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1OC(F)(F)F', 'aromatic'),
    ('2-CN-4-NH2-phenyl', '[*]c1ccc(N)cc1C#N', 'aromatic'),
    ('2-CN-4-NHMe-phenyl', '[*]c1ccc(NC)cc1C#N', 'aromatic'),
    ('2-CN-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1C#N', 'aromatic'),
    ('2-CN-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1C#N', 'aromatic'),
    ('2-CN-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1C#N', 'aromatic'),
    ('2-CN-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1C#N', 'aromatic'),
    ('2-CN-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1C#N', 'aromatic'),
    ('2-CN-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1C#N', 'aromatic'),
    ('2-CN-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1C#N', 'aromatic'),
    ('2-CN-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1C#N', 'aromatic'),
    ('2-CN-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1C#N', 'aromatic'),
    ('2-CN-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1C#N', 'aromatic'),
    ('2-CN-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1C#N', 'aromatic'),
    ('2-CN-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1C#N', 'aromatic'),
    ('2-NH2-4-NHMe-phenyl', '[*]c1ccc(NC)cc1N', 'aromatic'),
    ('2-NH2-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1N', 'aromatic'),
    ('2-NH2-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1N', 'aromatic'),
    ('2-NH2-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1N', 'aromatic'),
    ('2-NH2-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1N', 'aromatic'),
    ('2-NH2-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1N', 'aromatic'),
    ('2-NH2-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1N', 'aromatic'),
    ('2-NH2-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1N', 'aromatic'),
    ('2-NH2-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1N', 'aromatic'),
    ('2-NH2-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1N', 'aromatic'),
    ('2-NH2-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1N', 'aromatic'),
    ('2-NH2-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1N', 'aromatic'),
    ('2-NH2-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1N', 'aromatic'),
    ('2-NHMe-4-NMe2-phenyl', '[*]c1ccc(N(C)C)cc1NC', 'aromatic'),
    ('2-NHMe-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1NC', 'aromatic'),
    ('2-NHMe-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1NC', 'aromatic'),
    ('2-NHMe-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1NC', 'aromatic'),
    ('2-NHMe-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1NC', 'aromatic'),
    ('2-NHMe-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1NC', 'aromatic'),
    ('2-NHMe-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1NC', 'aromatic'),
    ('2-NHMe-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1NC', 'aromatic'),
    ('2-NHMe-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1NC', 'aromatic'),
    ('2-NHMe-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1NC', 'aromatic'),
    ('2-NHMe-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1NC', 'aromatic'),
    ('2-NHMe-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1NC', 'aromatic'),
    ('2-NMe2-4-COOH-phenyl', '[*]c1ccc(C(=O)O)cc1N(C)C', 'aromatic'),
    ('2-NMe2-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1N(C)C', 'aromatic'),
    ('2-NMe2-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1N(C)C', 'aromatic'),
    ('2-NMe2-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1N(C)C', 'aromatic'),
    ('2-NMe2-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1N(C)C', 'aromatic'),
    ('2-NMe2-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1N(C)C', 'aromatic'),
    ('2-NMe2-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1N(C)C', 'aromatic'),
    ('2-NMe2-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1N(C)C', 'aromatic'),
    ('2-NMe2-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1N(C)C', 'aromatic'),
    ('2-NMe2-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1N(C)C', 'aromatic'),
    ('2-NMe2-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1N(C)C', 'aromatic'),
    ('2-COOH-4-COOMe-phenyl', '[*]c1ccc(C(=O)OC)cc1C(=O)O', 'aromatic'),
    ('2-COOH-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1C(=O)O', 'aromatic'),
    ('2-COOH-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1C(=O)O', 'aromatic'),
    ('2-COOH-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1C(=O)O', 'aromatic'),
    ('2-COOH-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1C(=O)O', 'aromatic'),
    ('2-COOH-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1C(=O)O', 'aromatic'),
    ('2-COOH-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1C(=O)O', 'aromatic'),
    ('2-COOH-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1C(=O)O', 'aromatic'),
    ('2-COOH-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1C(=O)O', 'aromatic'),
    ('2-COOH-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1C(=O)O', 'aromatic'),
    ('2-COOMe-4-COMe-phenyl', '[*]c1ccc(C(C)=O)cc1C(=O)OC', 'aromatic'),
    ('2-COOMe-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1C(=O)OC', 'aromatic'),
    ('2-COOMe-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1C(=O)OC', 'aromatic'),
    ('2-COOMe-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1C(=O)OC', 'aromatic'),
    ('2-COOMe-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1C(=O)OC', 'aromatic'),
    ('2-COOMe-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1C(=O)OC', 'aromatic'),
    ('2-COOMe-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1C(=O)OC', 'aromatic'),
    ('2-COOMe-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1C(=O)OC', 'aromatic'),
    ('2-COOMe-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1C(=O)OC', 'aromatic'),
    ('2-COMe-4-SO2Me-phenyl', '[*]c1ccc(S(C)(=O)=O)cc1C(C)=O', 'aromatic'),
    ('2-COMe-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1C(C)=O', 'aromatic'),
    ('2-COMe-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1C(C)=O', 'aromatic'),
    ('2-COMe-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1C(C)=O', 'aromatic'),
    ('2-COMe-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1C(C)=O', 'aromatic'),
    ('2-COMe-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1C(C)=O', 'aromatic'),
    ('2-COMe-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1C(C)=O', 'aromatic'),
    ('2-COMe-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1C(C)=O', 'aromatic'),
    ('2-SO2Me-4-SO2NH2-phenyl', '[*]c1ccc(S(N)(=O)=O)cc1S(C)(=O)=O', 'aromatic'),
    ('2-SO2Me-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1S(C)(=O)=O', 'aromatic'),
    ('2-SO2Me-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1S(C)(=O)=O', 'aromatic'),
    ('2-SO2Me-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1S(C)(=O)=O', 'aromatic'),
    ('2-SO2Me-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1S(C)(=O)=O', 'aromatic'),
    ('2-SO2Me-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1S(C)(=O)=O', 'aromatic'),
    ('2-SO2Me-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1S(C)(=O)=O', 'aromatic'),
    ('2-SO2NH2-4-NHAc-phenyl', '[*]c1ccc(NC(C)=O)cc1S(N)(=O)=O', 'aromatic'),
    ('2-SO2NH2-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1S(N)(=O)=O', 'aromatic'),
    ('2-SO2NH2-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1S(N)(=O)=O', 'aromatic'),
    ('2-SO2NH2-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1S(N)(=O)=O', 'aromatic'),
    ('2-SO2NH2-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1S(N)(=O)=O', 'aromatic'),
    ('2-NHAc-4-NHSO2Me-phenyl', '[*]c1ccc(NS(C)(=O)=O)cc1NC(C)=O', 'aromatic'),
    ('2-NHAc-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1NC(C)=O', 'aromatic'),
    ('2-NHAc-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1NC(C)=O', 'aromatic'),
    ('2-NHAc-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1NC(C)=O', 'aromatic'),
    ('2-NHAc-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1NC(C)=O', 'aromatic'),
    ('2-NHSO2Me-4-cyclopropyl-phenyl', '[*]c1ccc(C1CC1)cc1NS(C)(=O)=O', 'aromatic'),
    ('2-NHSO2Me-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1NS(C)(=O)=O', 'aromatic'),
    ('2-cyclopropyl-4-morpholino-phenyl', '[*]c1ccc(N1CCOCC1)cc1C1CC1', 'aromatic'),
    ('2-cyclopropyl-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1C1CC1', 'aromatic'),
    ('2-cyclopropyl-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1C1CC1', 'aromatic'),
    ('2-morpholino-4-piperidino-phenyl', '[*]c1ccc(N1CCCCC1)cc1N1CCOCC1', 'aromatic'),
    ('2-morpholino-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1N1CCOCC1', 'aromatic'),
    ('2-piperidino-4-pyrrolidino-phenyl', '[*]c1ccc(N1CCCC1)cc1N1CCCCC1', 'aromatic'),
    ('5-Br-pyridin-2-yl', '[*]c1ccc(Br)cn1', 'aromatic'),
    ('5-Et-pyridin-2-yl', '[*]c1ccc(CC)cn1', 'aromatic'),
    ('5-iPr-pyridin-2-yl', '[*]c1ccc(C(C)C)cn1', 'aromatic'),
    ('5-tBu-pyridin-2-yl', '[*]c1ccc(C(C)(C)C)cn1', 'aromatic'),
    ('5-OMe-pyridin-2-yl', '[*]c1ccc(OC)cn1', 'aromatic'),
    ('5-OEt-pyridin-2-yl', '[*]c1ccc(OCC)cn1', 'aromatic'),
    ('5-OH-pyridin-2-yl', '[*]c1ccc(O)cn1', 'aromatic'),
    ('5-CHF2-pyridin-2-yl', '[*]c1ccc(C(F)F)cn1', 'aromatic'),
    ('5-OCF3-pyridin-2-yl', '[*]c1ccc(OC(F)(F)F)cn1', 'aromatic'),
    ('5-CN-pyridin-2-yl', '[*]c1ccc(C#N)cn1', 'aromatic'),
    ('5-NH2-pyridin-2-yl', '[*]c1ccc(N)cn1', 'aromatic'),
    ('5-NHMe-pyridin-2-yl', '[*]c1ccc(NC)cn1', 'aromatic'),
    ('5-NMe2-pyridin-2-yl', '[*]c1ccc(N(C)C)cn1', 'aromatic'),
    ('5-COOH-pyridin-2-yl', '[*]c1ccc(C(=O)O)cn1', 'aromatic'),
    ('5-COOMe-pyridin-2-yl', '[*]c1ccc(C(=O)OC)cn1', 'aromatic'),
    ('5-COMe-pyridin-2-yl', '[*]c1ccc(C(C)=O)cn1', 'aromatic'),
    ('4-F-pyridin-2-yl', '[*]c1cc(F)ccn1', 'aromatic'),
    ('4-Cl-pyridin-2-yl', '[*]c1cc(Cl)ccn1', 'aromatic'),
    ('4-Br-pyridin-2-yl', '[*]c1cc(Br)ccn1', 'aromatic'),
    ('4-Et-pyridin-2-yl', '[*]c1cc(CC)ccn1', 'aromatic'),
    ('4-iPr-pyridin-2-yl', '[*]c1cc(C(C)C)ccn1', 'aromatic'),
    ('4-tBu-pyridin-2-yl', '[*]c1cc(C(C)(C)C)ccn1', 'aromatic'),
    ('4-OMe-pyridin-2-yl', '[*]c1cc(OC)ccn1', 'aromatic'),
    ('4-OEt-pyridin-2-yl', '[*]c1cc(OCC)ccn1', 'aromatic'),
    ('4-OH-pyridin-2-yl', '[*]c1cc(O)ccn1', 'aromatic'),
    ('4-CF3-pyridin-2-yl', '[*]c1cc(C(F)(F)F)ccn1', 'aromatic'),
    ('4-CHF2-pyridin-2-yl', '[*]c1cc(C(F)F)ccn1', 'aromatic'),
    ('4-OCF3-pyridin-2-yl', '[*]c1cc(OC(F)(F)F)ccn1', 'aromatic'),
    ('4-CN-pyridin-2-yl', '[*]c1cc(C#N)ccn1', 'aromatic'),
    ('4-NH2-pyridin-2-yl', '[*]c1cc(N)ccn1', 'aromatic'),
    ('4-NHMe-pyridin-2-yl', '[*]c1cc(NC)ccn1', 'aromatic'),
    ('4-NMe2-pyridin-2-yl', '[*]c1cc(N(C)C)ccn1', 'aromatic'),
    ('4-COOH-pyridin-2-yl', '[*]c1cc(C(=O)O)ccn1', 'aromatic'),
    ('4-COOMe-pyridin-2-yl', '[*]c1cc(C(=O)OC)ccn1', 'aromatic'),
    ('4-COMe-pyridin-2-yl', '[*]c1cc(C(C)=O)ccn1', 'aromatic'),
    ('3-Cl-pyridin-2-yl', '[*]c1cccc(Cl)n1', 'aromatic'),
    ('3-Br-pyridin-2-yl', '[*]c1cccc(Br)n1', 'aromatic'),
    ('3-Et-pyridin-2-yl', '[*]c1cccc(CC)n1', 'aromatic'),
    ('3-iPr-pyridin-2-yl', '[*]c1cccc(C(C)C)n1', 'aromatic'),
    ('3-tBu-pyridin-2-yl', '[*]c1cccc(C(C)(C)C)n1', 'aromatic'),
    ('3-OMe-pyridin-2-yl', '[*]c1cccc(OC)n1', 'aromatic'),
    ('3-OEt-pyridin-2-yl', '[*]c1cccc(OCC)n1', 'aromatic'),
    ('3-OH-pyridin-2-yl', '[*]c1cccc(O)n1', 'aromatic'),
    ('3-CF3-pyridin-2-yl', '[*]c1cccc(C(F)(F)F)n1', 'aromatic'),
    ('3-CHF2-pyridin-2-yl', '[*]c1cccc(C(F)F)n1', 'aromatic'),
    ('3-OCF3-pyridin-2-yl', '[*]c1cccc(OC(F)(F)F)n1', 'aromatic'),
    ('3-CN-pyridin-2-yl', '[*]c1cccc(C#N)n1', 'aromatic'),
    ('3-NH2-pyridin-2-yl', '[*]c1cccc(N)n1', 'aromatic'),
    ('3-NHMe-pyridin-2-yl', '[*]c1cccc(NC)n1', 'aromatic'),
    ('3-NMe2-pyridin-2-yl', '[*]c1cccc(N(C)C)n1', 'aromatic'),
    ('3-COOH-pyridin-2-yl', '[*]c1cccc(C(=O)O)n1', 'aromatic'),
    ('3-COOMe-pyridin-2-yl', '[*]c1cccc(C(=O)OC)n1', 'aromatic'),
    ('3-COMe-pyridin-2-yl', '[*]c1cccc(C(C)=O)n1', 'aromatic'),
    ('5-F-pyridin-3-yl', '[*]c1cncc(F)c1', 'aromatic'),
    ('5-Cl-pyridin-3-yl', '[*]c1cncc(Cl)c1', 'aromatic'),
    ('5-Br-pyridin-3-yl', '[*]c1cncc(Br)c1', 'aromatic'),
    ('5-Et-pyridin-3-yl', '[*]c1cncc(CC)c1', 'aromatic'),
    ('5-iPr-pyridin-3-yl', '[*]c1cncc(C(C)C)c1', 'aromatic'),
    ('5-tBu-pyridin-3-yl', '[*]c1cncc(C(C)(C)C)c1', 'aromatic'),
    ('5-OMe-pyridin-3-yl', '[*]c1cncc(OC)c1', 'aromatic'),
    ('5-OEt-pyridin-3-yl', '[*]c1cncc(OCC)c1', 'aromatic'),
    ('5-OH-pyridin-3-yl', '[*]c1cncc(O)c1', 'aromatic'),
    ('5-CF3-pyridin-3-yl', '[*]c1cncc(C(F)(F)F)c1', 'aromatic'),
    ('5-CHF2-pyridin-3-yl', '[*]c1cncc(C(F)F)c1', 'aromatic'),
    ('5-OCF3-pyridin-3-yl', '[*]c1cncc(OC(F)(F)F)c1', 'aromatic'),
    ('5-CN-pyridin-3-yl', '[*]c1cncc(C#N)c1', 'aromatic'),
    ('5-NH2-pyridin-3-yl', '[*]c1cncc(N)c1', 'aromatic'),
    ('5-NHMe-pyridin-3-yl', '[*]c1cncc(NC)c1', 'aromatic'),
    ('5-NMe2-pyridin-3-yl', '[*]c1cncc(N(C)C)c1', 'aromatic'),
    ('5-COOH-pyridin-3-yl', '[*]c1cncc(C(=O)O)c1', 'aromatic'),
    ('5-COOMe-pyridin-3-yl', '[*]c1cncc(C(=O)OC)c1', 'aromatic'),
    ('5-COMe-pyridin-3-yl', '[*]c1cncc(C(C)=O)c1', 'aromatic'),
    ('6-F-pyridin-3-yl', '[*]c1ccc(F)nc1', 'aromatic'),
    ('6-Cl-pyridin-3-yl', '[*]c1ccc(Cl)nc1', 'aromatic'),
    ('6-Br-pyridin-3-yl', '[*]c1ccc(Br)nc1', 'aromatic'),
    ('6-Et-pyridin-3-yl', '[*]c1ccc(CC)nc1', 'aromatic'),
    ('6-iPr-pyridin-3-yl', '[*]c1ccc(C(C)C)nc1', 'aromatic'),
    ('6-tBu-pyridin-3-yl', '[*]c1ccc(C(C)(C)C)nc1', 'aromatic'),
    ('6-OMe-pyridin-3-yl', '[*]c1ccc(OC)nc1', 'aromatic'),
    ('6-OEt-pyridin-3-yl', '[*]c1ccc(OCC)nc1', 'aromatic'),
    ('6-OH-pyridin-3-yl', '[*]c1ccc(O)nc1', 'aromatic'),
    ('6-CF3-pyridin-3-yl', '[*]c1ccc(C(F)(F)F)nc1', 'aromatic'),
    ('6-CHF2-pyridin-3-yl', '[*]c1ccc(C(F)F)nc1', 'aromatic'),
    ('6-OCF3-pyridin-3-yl', '[*]c1ccc(OC(F)(F)F)nc1', 'aromatic'),
    ('6-CN-pyridin-3-yl', '[*]c1ccc(C#N)nc1', 'aromatic'),
    ('6-NH2-pyridin-3-yl', '[*]c1ccc(N)nc1', 'aromatic'),
    ('6-NHMe-pyridin-3-yl', '[*]c1ccc(NC)nc1', 'aromatic'),
    ('6-NMe2-pyridin-3-yl', '[*]c1ccc(N(C)C)nc1', 'aromatic'),
    ('6-COOH-pyridin-3-yl', '[*]c1ccc(C(=O)O)nc1', 'aromatic'),
    ('6-COOMe-pyridin-3-yl', '[*]c1ccc(C(=O)OC)nc1', 'aromatic'),
    ('6-COMe-pyridin-3-yl', '[*]c1ccc(C(C)=O)nc1', 'aromatic'),
    ('2-F-pyridin-4-yl', '[*]c1cc(F)ncc1', 'aromatic'),
    ('2-Cl-pyridin-4-yl', '[*]c1cc(Cl)ncc1', 'aromatic'),
    ('2-Br-pyridin-4-yl', '[*]c1cc(Br)ncc1', 'aromatic'),
    ('2-Et-pyridin-4-yl', '[*]c1cc(CC)ncc1', 'aromatic'),
    ('2-iPr-pyridin-4-yl', '[*]c1cc(C(C)C)ncc1', 'aromatic'),
    ('2-tBu-pyridin-4-yl', '[*]c1cc(C(C)(C)C)ncc1', 'aromatic'),
    ('2-OMe-pyridin-4-yl', '[*]c1cc(OC)ncc1', 'aromatic'),
    ('2-OEt-pyridin-4-yl', '[*]c1cc(OCC)ncc1', 'aromatic'),
    ('2-OH-pyridin-4-yl', '[*]c1cc(O)ncc1', 'aromatic'),
    ('2-CF3-pyridin-4-yl', '[*]c1cc(C(F)(F)F)ncc1', 'aromatic'),
    ('2-CHF2-pyridin-4-yl', '[*]c1cc(C(F)F)ncc1', 'aromatic'),
    ('2-OCF3-pyridin-4-yl', '[*]c1cc(OC(F)(F)F)ncc1', 'aromatic'),
    ('2-CN-pyridin-4-yl', '[*]c1cc(C#N)ncc1', 'aromatic'),
    ('2-NH2-pyridin-4-yl', '[*]c1cc(N)ncc1', 'aromatic'),
    ('2-NHMe-pyridin-4-yl', '[*]c1cc(NC)ncc1', 'aromatic'),
    ('2-NMe2-pyridin-4-yl', '[*]c1cc(N(C)C)ncc1', 'aromatic'),
    ('2-COOH-pyridin-4-yl', '[*]c1cc(C(=O)O)ncc1', 'aromatic'),
    ('2-COOMe-pyridin-4-yl', '[*]c1cc(C(=O)OC)ncc1', 'aromatic'),
    ('2-COMe-pyridin-4-yl', '[*]c1cc(C(C)=O)ncc1', 'aromatic'),
    ('N-Me-piperidine', '[*]NC1CCCCC1', 'basic'),
    ('N-Et-piperidine', '[*]NCC1CCCCC1', 'basic'),
    ('N-iPr-piperidine', '[*]NC(C)C1CCCCC1', 'basic'),
    ('N-cPr-piperidine', '[*]NC1CC11CCCCC1', 'basic'),
    ('N-tBu-piperidine', '[*]NC(C)(C)C1CCCCC1', 'basic'),
    ('N-allyl-piperidine', '[*]NCC=C1CCCCC1', 'basic'),
    ('N-Me-pyrrolidine', '[*]NC1CCCC1', 'basic'),
    ('N-Et-pyrrolidine', '[*]NCC1CCCC1', 'basic'),
    ('N-iPr-pyrrolidine', '[*]NC(C)C1CCCC1', 'basic'),
    ('N-cPr-pyrrolidine', '[*]NC1CC11CCCC1', 'basic'),
    ('N-tBu-pyrrolidine', '[*]NC(C)(C)C1CCCC1', 'basic'),
    ('N-allyl-pyrrolidine', '[*]NCC=C1CCCC1', 'basic'),
    ('N-Me-azetidine', '[*]NC1CCC1', 'basic'),
    ('N-Et-azetidine', '[*]NCC1CCC1', 'basic'),
    ('N-iPr-azetidine', '[*]NC(C)C1CCC1', 'basic'),
    ('N-cPr-azetidine', '[*]NC1CC11CCC1', 'basic'),
    ('N-tBu-azetidine', '[*]NC(C)(C)C1CCC1', 'basic'),
    ('N-allyl-azetidine', '[*]NCC=C1CCC1', 'basic'),
    ('N-Me-morpholine', '[*]NC1CCOCC1', 'basic'),
    ('N-Et-morpholine', '[*]NCC1CCOCC1', 'basic'),
    ('N-iPr-morpholine', '[*]NC(C)C1CCOCC1', 'basic'),
    ('N-cPr-morpholine', '[*]NC1CC11CCOCC1', 'basic'),
    ('N-tBu-morpholine', '[*]NC(C)(C)C1CCOCC1', 'basic'),
    ('N-allyl-morpholine', '[*]NCC=C1CCOCC1', 'basic'),
    ('N-Me-piperazine', '[*]NC1CCNCC1', 'basic'),
    ('N-Et-piperazine', '[*]NCC1CCNCC1', 'basic'),
    ('N-iPr-piperazine', '[*]NC(C)C1CCNCC1', 'basic'),
    ('N-cPr-piperazine', '[*]NC1CC11CCNCC1', 'basic'),
    ('N-tBu-piperazine', '[*]NC(C)(C)C1CCNCC1', 'basic'),
    ('N-allyl-piperazine', '[*]NCC=C1CCNCC1', 'basic'),
    ('N-Me-azepane', '[*]NC1CCCCCC1', 'basic'),
    ('N-Et-azepane', '[*]NCC1CCCCCC1', 'basic'),
    ('N-iPr-azepane', '[*]NC(C)C1CCCCCC1', 'basic'),
    ('N-cPr-azepane', '[*]NC1CC11CCCCCC1', 'basic'),
    ('N-tBu-azepane', '[*]NC(C)(C)C1CCCCCC1', 'basic'),
    ('N-allyl-azepane', '[*]NCC=C1CCCCCC1', 'basic'),
    ('N-Me-homomorpholine', '[*]NC1CCCOCC1', 'basic'),
    ('N-Et-homomorpholine', '[*]NCC1CCCOCC1', 'basic'),
    ('N-iPr-homomorpholine', '[*]NC(C)C1CCCOCC1', 'basic'),
    ('N-cPr-homomorpholine', '[*]NC1CC11CCCOCC1', 'basic'),
    ('N-tBu-homomorpholine', '[*]NC(C)(C)C1CCCOCC1', 'basic'),
    ('N-allyl-homomorpholine', '[*]NCC=C1CCCOCC1', 'basic'),
    ('piperidin-3-yl-methyl', '[*]CC1CNCCC1', 'basic'),
    ('pyrrolidin-3-yl-methyl', '[*]CC1CNCC1', 'basic'),
    ('azetidin-3-yl-methyl', '[*]CC1CNC1', 'basic'),
    ('carbonyl-NH', '[*]C(=O)NN', 'polar'),
    ('carbonyl-NHMe', '[*]C(=O)NNC', 'polar'),
    ('carbonyl-NHEt', '[*]C(=O)NNCC', 'polar'),
    ('carbonyl-NHiPr', '[*]C(=O)NNC(C)C', 'polar'),
    ('carbonyl-NHtBu', '[*]C(=O)NNC(C)(C)C', 'polar'),
    ('carbonyl-NHcPr', '[*]C(=O)NNC1CC1', 'polar'),
    ('carbonyl-NHcPent', '[*]C(=O)NNC1CCCC1', 'polar'),
    ('carbonyl-NHcHex', '[*]C(=O)NNC1CCCCC1', 'polar'),
    ('carbonyl-NMe2', '[*]C(=O)NN(C)C', 'polar'),
    ('carbonyl-NEt2', '[*]C(=O)NN(CC)CC', 'polar'),
    ('carbonyl-NHBn', '[*]C(=O)NNCc1ccccc1', 'polar'),
    ('carbonyl-NH-4-FBn', '[*]C(=O)NNCc1ccc(F)cc1', 'polar'),
    ('carbonyl-piperidyl', '[*]C(=O)NN1CCCCC1', 'polar'),
    ('carbonyl-morpholyl', '[*]C(=O)NN1CCOCC1', 'polar'),
    ('carbonyl-pyrrolidyl', '[*]C(=O)NN1CCCC1', 'polar'),
    ('carbonyl-azetidinyl', '[*]C(=O)NN1CCC1', 'polar'),
    ('methylene-carbonyl-NH', '[*]CC(=O)NN', 'polar'),
    ('methylene-carbonyl-NHMe', '[*]CC(=O)NNC', 'polar'),
    ('methylene-carbonyl-NHEt', '[*]CC(=O)NNCC', 'polar'),
    ('methylene-carbonyl-NHiPr', '[*]CC(=O)NNC(C)C', 'polar'),
    ('methylene-carbonyl-NHtBu', '[*]CC(=O)NNC(C)(C)C', 'polar'),
    ('methylene-carbonyl-NHcPr', '[*]CC(=O)NNC1CC1', 'polar'),
    ('methylene-carbonyl-NHcPent', '[*]CC(=O)NNC1CCCC1', 'polar'),
    ('methylene-carbonyl-NHcHex', '[*]CC(=O)NNC1CCCCC1', 'polar'),
    ('methylene-carbonyl-NMe2', '[*]CC(=O)NN(C)C', 'polar'),
    ('methylene-carbonyl-NEt2', '[*]CC(=O)NN(CC)CC', 'polar'),
    ('methylene-carbonyl-NHBn', '[*]CC(=O)NNCc1ccccc1', 'polar'),
    ('methylene-carbonyl-NH-4-FBn', '[*]CC(=O)NNCc1ccc(F)cc1', 'polar'),
    ('methylene-carbonyl-piperidyl', '[*]CC(=O)NN1CCCCC1', 'polar'),
    ('methylene-carbonyl-morpholyl', '[*]CC(=O)NN1CCOCC1', 'polar'),
    ('methylene-carbonyl-pyrrolidyl', '[*]CC(=O)NN1CCCC1', 'polar'),
    ('methylene-carbonyl-azetidinyl', '[*]CC(=O)NN1CCC1', 'polar'),
    ('ethylene-carbonyl-NH', '[*]CCC(=O)NN', 'polar'),
    ('ethylene-carbonyl-NHMe', '[*]CCC(=O)NNC', 'polar'),
    ('ethylene-carbonyl-NHEt', '[*]CCC(=O)NNCC', 'polar'),
    ('ethylene-carbonyl-NHiPr', '[*]CCC(=O)NNC(C)C', 'polar'),
    ('ethylene-carbonyl-NHtBu', '[*]CCC(=O)NNC(C)(C)C', 'polar'),
    ('ethylene-carbonyl-NHcPr', '[*]CCC(=O)NNC1CC1', 'polar'),
    ('ethylene-carbonyl-NHcPent', '[*]CCC(=O)NNC1CCCC1', 'polar'),
    ('ethylene-carbonyl-NHcHex', '[*]CCC(=O)NNC1CCCCC1', 'polar'),
    ('ethylene-carbonyl-NMe2', '[*]CCC(=O)NN(C)C', 'polar'),
    ('ethylene-carbonyl-NEt2', '[*]CCC(=O)NN(CC)CC', 'polar'),
    ('ethylene-carbonyl-NHBn', '[*]CCC(=O)NNCc1ccccc1', 'polar'),
    ('ethylene-carbonyl-NH-4-FBn', '[*]CCC(=O)NNCc1ccc(F)cc1', 'polar'),
    ('ethylene-carbonyl-piperidyl', '[*]CCC(=O)NN1CCCCC1', 'polar'),
    ('ethylene-carbonyl-morpholyl', '[*]CCC(=O)NN1CCOCC1', 'polar'),
    ('ethylene-carbonyl-pyrrolidyl', '[*]CCC(=O)NN1CCCC1', 'polar'),
    ('ethylene-carbonyl-azetidinyl', '[*]CCC(=O)NN1CCC1', 'polar'),
    ('carbamate-NH', '[*]OC(=O)NN', 'polar'),
    ('carbamate-NHMe', '[*]OC(=O)NNC', 'polar'),
    ('carbamate-NHEt', '[*]OC(=O)NNCC', 'polar'),
    ('carbamate-NHiPr', '[*]OC(=O)NNC(C)C', 'polar'),
    ('carbamate-NHtBu', '[*]OC(=O)NNC(C)(C)C', 'polar'),
    ('carbamate-NHcPr', '[*]OC(=O)NNC1CC1', 'polar'),
    ('carbamate-NHcPent', '[*]OC(=O)NNC1CCCC1', 'polar'),
    ('carbamate-NHcHex', '[*]OC(=O)NNC1CCCCC1', 'polar'),
    ('carbamate-NMe2', '[*]OC(=O)NN(C)C', 'polar'),
    ('carbamate-NEt2', '[*]OC(=O)NN(CC)CC', 'polar'),
    ('carbamate-NHBn', '[*]OC(=O)NNCc1ccccc1', 'polar'),
    ('carbamate-NH-4-FBn', '[*]OC(=O)NNCc1ccc(F)cc1', 'polar'),
    ('carbamate-piperidyl', '[*]OC(=O)NN1CCCCC1', 'polar'),
    ('carbamate-morpholyl', '[*]OC(=O)NN1CCOCC1', 'polar'),
    ('carbamate-pyrrolidyl', '[*]OC(=O)NN1CCCC1', 'polar'),
    ('carbamate-azetidinyl', '[*]OC(=O)NN1CCC1', 'polar'),
    ('reverse-amide-NHMe', '[*]NC(=O)NC', 'polar'),
    ('reverse-amide-NHEt', '[*]NC(=O)NCC', 'polar'),
    ('reverse-amide-NHiPr', '[*]NC(=O)NC(C)C', 'polar'),
    ('reverse-amide-NHtBu', '[*]NC(=O)NC(C)(C)C', 'polar'),
    ('reverse-amide-NHcPr', '[*]NC(=O)NC1CC1', 'polar'),
    ('reverse-amide-NHcPent', '[*]NC(=O)NC1CCCC1', 'polar'),
    ('reverse-amide-NHcHex', '[*]NC(=O)NC1CCCCC1', 'polar'),
    ('reverse-amide-NMe2', '[*]NC(=O)N(C)C', 'polar'),
    ('reverse-amide-NEt2', '[*]NC(=O)N(CC)CC', 'polar'),
    ('reverse-amide-NHBn', '[*]NC(=O)NCc1ccccc1', 'polar'),
    ('reverse-amide-NH-4-FBn', '[*]NC(=O)NCc1ccc(F)cc1', 'polar'),
    ('reverse-amide-piperidyl', '[*]NC(=O)N1CCCCC1', 'polar'),
    ('reverse-amide-morpholyl', '[*]NC(=O)N1CCOCC1', 'polar'),
    ('reverse-amide-pyrrolidyl', '[*]NC(=O)N1CCCC1', 'polar'),
    ('reverse-amide-azetidinyl', '[*]NC(=O)N1CCC1', 'polar'),
    ('sulfonamide-NH', '[*]S(=O)(=O)NN', 'polar'),
    ('sulfonamide-NHMe', '[*]S(=O)(=O)NNC', 'polar'),
    ('sulfonamide-NHEt', '[*]S(=O)(=O)NNCC', 'polar'),
    ('sulfonamide-NHiPr', '[*]S(=O)(=O)NNC(C)C', 'polar'),
    ('sulfonamide-NHtBu', '[*]S(=O)(=O)NNC(C)(C)C', 'polar'),
    ('sulfonamide-NHcPr', '[*]S(=O)(=O)NNC1CC1', 'polar'),
    ('sulfonamide-NHcPent', '[*]S(=O)(=O)NNC1CCCC1', 'polar'),
    ('sulfonamide-NHcHex', '[*]S(=O)(=O)NNC1CCCCC1', 'polar'),
    ('sulfonamide-NMe2', '[*]S(=O)(=O)NN(C)C', 'polar'),
    ('sulfonamide-NEt2', '[*]S(=O)(=O)NN(CC)CC', 'polar'),
    ('sulfonamide-NHBn', '[*]S(=O)(=O)NNCc1ccccc1', 'polar'),
    ('sulfonamide-NH-4-FBn', '[*]S(=O)(=O)NNCc1ccc(F)cc1', 'polar'),
    ('sulfonamide-piperidyl', '[*]S(=O)(=O)NN1CCCCC1', 'polar'),
    ('sulfonamide-morpholyl', '[*]S(=O)(=O)NN1CCOCC1', 'polar'),
    ('sulfonamide-pyrrolidyl', '[*]S(=O)(=O)NN1CCCC1', 'polar'),
    ('sulfonamide-azetidinyl', '[*]S(=O)(=O)NN1CCC1', 'polar'),
    ('methyl-sulfonamide-NH', '[*]CS(=O)(=O)NN', 'polar'),
    ('methyl-sulfonamide-NHMe', '[*]CS(=O)(=O)NNC', 'polar'),
    ('methyl-sulfonamide-NHEt', '[*]CS(=O)(=O)NNCC', 'polar'),
    ('methyl-sulfonamide-NHiPr', '[*]CS(=O)(=O)NNC(C)C', 'polar'),
    ('methyl-sulfonamide-NHtBu', '[*]CS(=O)(=O)NNC(C)(C)C', 'polar'),
    ('methyl-sulfonamide-NHcPr', '[*]CS(=O)(=O)NNC1CC1', 'polar'),
    ('methyl-sulfonamide-NHcPent', '[*]CS(=O)(=O)NNC1CCCC1', 'polar'),
    ('methyl-sulfonamide-NHcHex', '[*]CS(=O)(=O)NNC1CCCCC1', 'polar'),
    ('methyl-sulfonamide-NMe2', '[*]CS(=O)(=O)NN(C)C', 'polar'),
    ('methyl-sulfonamide-NEt2', '[*]CS(=O)(=O)NN(CC)CC', 'polar'),
    ('methyl-sulfonamide-NHBn', '[*]CS(=O)(=O)NNCc1ccccc1', 'polar'),
    ('methyl-sulfonamide-NH-4-FBn', '[*]CS(=O)(=O)NNCc1ccc(F)cc1', 'polar'),
    ('methyl-sulfonamide-piperidyl', '[*]CS(=O)(=O)NN1CCCCC1', 'polar'),
    ('methyl-sulfonamide-morpholyl', '[*]CS(=O)(=O)NN1CCOCC1', 'polar'),
    ('methyl-sulfonamide-pyrrolidyl', '[*]CS(=O)(=O)NN1CCCC1', 'polar'),
    ('methyl-sulfonamide-azetidinyl', '[*]CS(=O)(=O)NN1CCC1', 'polar'),
    ('4-F-phenyl-methyl', '[*]Cc1ccc(F)cc1', 'aromatic'),
    ('4-Cl-phenyl-methyl', '[*]Cc1ccc(Cl)cc1', 'aromatic'),
    ('4-Me-phenyl-methyl', '[*]Cc1ccc(C)cc1', 'aromatic'),
    ('4-OMe-phenyl-methyl', '[*]Cc1ccc(OC)cc1', 'aromatic'),
    ('4-CF3-phenyl-methyl', '[*]Cc1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('3-F-phenyl-methyl', '[*]Cc1cccc(F)c1', 'aromatic'),
    ('3-Cl-phenyl-methyl', '[*]Cc1cccc(Cl)c1', 'aromatic'),
    ('3-CF3-phenyl-methyl', '[*]Cc1cccc(C(F)(F)F)c1', 'aromatic'),
    ('pyridin-2-yl-methyl', '[*]Cc1ccccn1', 'aromatic'),
    ('pyridin-3-yl-methyl', '[*]Cc1cccnc1', 'aromatic'),
    ('pyridin-4-yl-methyl', '[*]Cc1ccncc1', 'aromatic'),
    ('pyrimidin-2-yl-methyl', '[*]Cc1ncccn1', 'aromatic'),
    ('pyrimidin-5-yl-methyl', '[*]Cc1cncnc1', 'aromatic'),
    ('thiophen-2-yl-methyl', '[*]Cc1cccs1', 'aromatic'),
    ('thiophen-3-yl-methyl', '[*]Cc1ccsc1', 'aromatic'),
    ('furan-2-yl-methyl', '[*]Cc1ccco1', 'aromatic'),
    ('thiazol-2-yl-methyl', '[*]Cc1nccs1', 'aromatic'),
    ('oxazol-2-yl-methyl', '[*]Cc1ncco1', 'aromatic'),
    ('1-Me-pyrazol-3-yl-methyl', '[*]Cc1ccn(C)n1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-methyl', '[*]Cc1cn(C)nc1', 'aromatic'),
    ('isoxazol-3-yl-methyl', '[*]Cc1cc(no1)', 'aromatic'),
    ('isoxazol-5-yl-methyl', '[*]Cc1cc(on1)', 'aromatic'),
    ('benzimidazol-2-yl-methyl', '[*]Cc1nc2ccccc2[nH]1', 'aromatic'),
    ('quinolin-2-yl-methyl', '[*]Cc1ccc2ncccc2c1', 'aromatic'),
    ('indol-3-yl-methyl', '[*]Cc1c[nH]c2ccccc12', 'aromatic'),
    ('phenyl-ethyl', '[*]CCc1ccccc1', 'aromatic'),
    ('4-F-phenyl-ethyl', '[*]CCc1ccc(F)cc1', 'aromatic'),
    ('4-Cl-phenyl-ethyl', '[*]CCc1ccc(Cl)cc1', 'aromatic'),
    ('4-Me-phenyl-ethyl', '[*]CCc1ccc(C)cc1', 'aromatic'),
    ('4-OMe-phenyl-ethyl', '[*]CCc1ccc(OC)cc1', 'aromatic'),
    ('4-CF3-phenyl-ethyl', '[*]CCc1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('3-F-phenyl-ethyl', '[*]CCc1cccc(F)c1', 'aromatic'),
    ('3-Cl-phenyl-ethyl', '[*]CCc1cccc(Cl)c1', 'aromatic'),
    ('3-CF3-phenyl-ethyl', '[*]CCc1cccc(C(F)(F)F)c1', 'aromatic'),
    ('pyridin-2-yl-ethyl', '[*]CCc1ccccn1', 'aromatic'),
    ('pyridin-3-yl-ethyl', '[*]CCc1cccnc1', 'aromatic'),
    ('pyridin-4-yl-ethyl', '[*]CCc1ccncc1', 'aromatic'),
    ('pyrimidin-2-yl-ethyl', '[*]CCc1ncccn1', 'aromatic'),
    ('pyrimidin-5-yl-ethyl', '[*]CCc1cncnc1', 'aromatic'),
    ('thiophen-2-yl-ethyl', '[*]CCc1cccs1', 'aromatic'),
    ('thiophen-3-yl-ethyl', '[*]CCc1ccsc1', 'aromatic'),
    ('furan-2-yl-ethyl', '[*]CCc1ccco1', 'aromatic'),
    ('thiazol-2-yl-ethyl', '[*]CCc1nccs1', 'aromatic'),
    ('oxazol-2-yl-ethyl', '[*]CCc1ncco1', 'aromatic'),
    ('1-Me-pyrazol-3-yl-ethyl', '[*]CCc1ccn(C)n1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-ethyl', '[*]CCc1cn(C)nc1', 'aromatic'),
    ('isoxazol-3-yl-ethyl', '[*]CCc1cc(no1)', 'aromatic'),
    ('isoxazol-5-yl-ethyl', '[*]CCc1cc(on1)', 'aromatic'),
    ('1,2,4-triazol-1-yl-ethyl', '[*]CCn1cncn1', 'aromatic'),
    ('benzimidazol-2-yl-ethyl', '[*]CCc1nc2ccccc2[nH]1', 'aromatic'),
    ('quinolin-2-yl-ethyl', '[*]CCc1ccc2ncccc2c1', 'aromatic'),
    ('indol-3-yl-ethyl', '[*]CCc1c[nH]c2ccccc12', 'aromatic'),
    ('phenyl-propyl', '[*]CCCc1ccccc1', 'aromatic'),
    ('4-F-phenyl-propyl', '[*]CCCc1ccc(F)cc1', 'aromatic'),
    ('4-Cl-phenyl-propyl', '[*]CCCc1ccc(Cl)cc1', 'aromatic'),
    ('4-Me-phenyl-propyl', '[*]CCCc1ccc(C)cc1', 'aromatic'),
    ('4-OMe-phenyl-propyl', '[*]CCCc1ccc(OC)cc1', 'aromatic'),
    ('4-CF3-phenyl-propyl', '[*]CCCc1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('3-F-phenyl-propyl', '[*]CCCc1cccc(F)c1', 'aromatic'),
    ('3-Cl-phenyl-propyl', '[*]CCCc1cccc(Cl)c1', 'aromatic'),
    ('3-CF3-phenyl-propyl', '[*]CCCc1cccc(C(F)(F)F)c1', 'aromatic'),
    ('pyridin-2-yl-propyl', '[*]CCCc1ccccn1', 'aromatic'),
    ('pyridin-3-yl-propyl', '[*]CCCc1cccnc1', 'aromatic'),
    ('pyridin-4-yl-propyl', '[*]CCCc1ccncc1', 'aromatic'),
    ('pyrimidin-2-yl-propyl', '[*]CCCc1ncccn1', 'aromatic'),
    ('pyrimidin-5-yl-propyl', '[*]CCCc1cncnc1', 'aromatic'),
    ('thiophen-2-yl-propyl', '[*]CCCc1cccs1', 'aromatic'),
    ('thiophen-3-yl-propyl', '[*]CCCc1ccsc1', 'aromatic'),
    ('furan-2-yl-propyl', '[*]CCCc1ccco1', 'aromatic'),
    ('thiazol-2-yl-propyl', '[*]CCCc1nccs1', 'aromatic'),
    ('oxazol-2-yl-propyl', '[*]CCCc1ncco1', 'aromatic'),
    ('1-Me-pyrazol-3-yl-propyl', '[*]CCCc1ccn(C)n1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-propyl', '[*]CCCc1cn(C)nc1', 'aromatic'),
    ('isoxazol-3-yl-propyl', '[*]CCCc1cc(no1)', 'aromatic'),
    ('isoxazol-5-yl-propyl', '[*]CCCc1cc(on1)', 'aromatic'),
    ('1,2,4-triazol-1-yl-propyl', '[*]CCCn1cncn1', 'aromatic'),
    ('benzimidazol-2-yl-propyl', '[*]CCCc1nc2ccccc2[nH]1', 'aromatic'),
    ('quinolin-2-yl-propyl', '[*]CCCc1ccc2ncccc2c1', 'aromatic'),
    ('indol-3-yl-propyl', '[*]CCCc1c[nH]c2ccccc12', 'aromatic'),
    ('phenyl-oxy', '[*]Oc1ccccc1', 'aromatic'),
    ('4-F-phenyl-oxy', '[*]Oc1ccc(F)cc1', 'aromatic'),
    ('4-Cl-phenyl-oxy', '[*]Oc1ccc(Cl)cc1', 'aromatic'),
    ('4-Me-phenyl-oxy', '[*]Oc1ccc(C)cc1', 'aromatic'),
    ('4-OMe-phenyl-oxy', '[*]Oc1ccc(OC)cc1', 'aromatic'),
    ('4-CF3-phenyl-oxy', '[*]Oc1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('3-F-phenyl-oxy', '[*]Oc1cccc(F)c1', 'aromatic'),
    ('3-Cl-phenyl-oxy', '[*]Oc1cccc(Cl)c1', 'aromatic'),
    ('3-CF3-phenyl-oxy', '[*]Oc1cccc(C(F)(F)F)c1', 'aromatic'),
    ('pyridin-2-yl-oxy', '[*]Oc1ccccn1', 'aromatic'),
    ('pyridin-3-yl-oxy', '[*]Oc1cccnc1', 'aromatic'),
    ('pyridin-4-yl-oxy', '[*]Oc1ccncc1', 'aromatic'),
    ('pyrimidin-2-yl-oxy', '[*]Oc1ncccn1', 'aromatic'),
    ('pyrimidin-5-yl-oxy', '[*]Oc1cncnc1', 'aromatic'),
    ('thiophen-2-yl-oxy', '[*]Oc1cccs1', 'aromatic'),
    ('thiophen-3-yl-oxy', '[*]Oc1ccsc1', 'aromatic'),
    ('furan-2-yl-oxy', '[*]Oc1ccco1', 'aromatic'),
    ('thiazol-2-yl-oxy', '[*]Oc1nccs1', 'aromatic'),
    ('oxazol-2-yl-oxy', '[*]Oc1ncco1', 'aromatic'),
    ('1-Me-pyrazol-3-yl-oxy', '[*]Oc1ccn(C)n1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-oxy', '[*]Oc1cn(C)nc1', 'aromatic'),
    ('isoxazol-3-yl-oxy', '[*]Oc1cc(no1)', 'aromatic'),
    ('isoxazol-5-yl-oxy', '[*]Oc1cc(on1)', 'aromatic'),
    ('1,2,4-triazol-1-yl-oxy', '[*]On1cncn1', 'aromatic'),
    ('benzimidazol-2-yl-oxy', '[*]Oc1nc2ccccc2[nH]1', 'aromatic'),
    ('quinolin-2-yl-oxy', '[*]Oc1ccc2ncccc2c1', 'aromatic'),
    ('indol-3-yl-oxy', '[*]Oc1c[nH]c2ccccc12', 'aromatic'),
    ('phenyl-oxyethyl', '[*]OCCc1ccccc1', 'aromatic'),
    ('4-F-phenyl-oxyethyl', '[*]OCCc1ccc(F)cc1', 'aromatic'),
    ('4-Cl-phenyl-oxyethyl', '[*]OCCc1ccc(Cl)cc1', 'aromatic'),
    ('4-Me-phenyl-oxyethyl', '[*]OCCc1ccc(C)cc1', 'aromatic'),
    ('4-OMe-phenyl-oxyethyl', '[*]OCCc1ccc(OC)cc1', 'aromatic'),
    ('4-CF3-phenyl-oxyethyl', '[*]OCCc1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('3-F-phenyl-oxyethyl', '[*]OCCc1cccc(F)c1', 'aromatic'),
    ('3-Cl-phenyl-oxyethyl', '[*]OCCc1cccc(Cl)c1', 'aromatic'),
    ('3-CF3-phenyl-oxyethyl', '[*]OCCc1cccc(C(F)(F)F)c1', 'aromatic'),
    ('pyridin-2-yl-oxyethyl', '[*]OCCc1ccccn1', 'aromatic'),
    ('pyridin-3-yl-oxyethyl', '[*]OCCc1cccnc1', 'aromatic'),
    ('pyridin-4-yl-oxyethyl', '[*]OCCc1ccncc1', 'aromatic'),
    ('pyrimidin-2-yl-oxyethyl', '[*]OCCc1ncccn1', 'aromatic'),
    ('pyrimidin-5-yl-oxyethyl', '[*]OCCc1cncnc1', 'aromatic'),
    ('thiophen-2-yl-oxyethyl', '[*]OCCc1cccs1', 'aromatic'),
    ('thiophen-3-yl-oxyethyl', '[*]OCCc1ccsc1', 'aromatic'),
    ('furan-2-yl-oxyethyl', '[*]OCCc1ccco1', 'aromatic'),
    ('thiazol-2-yl-oxyethyl', '[*]OCCc1nccs1', 'aromatic'),
    ('oxazol-2-yl-oxyethyl', '[*]OCCc1ncco1', 'aromatic'),
    ('1-Me-pyrazol-3-yl-oxyethyl', '[*]OCCc1ccn(C)n1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-oxyethyl', '[*]OCCc1cn(C)nc1', 'aromatic'),
    ('isoxazol-3-yl-oxyethyl', '[*]OCCc1cc(no1)', 'aromatic'),
    ('isoxazol-5-yl-oxyethyl', '[*]OCCc1cc(on1)', 'aromatic'),
    ('1,2,4-triazol-1-yl-oxyethyl', '[*]OCCn1cncn1', 'aromatic'),
    ('benzimidazol-2-yl-oxyethyl', '[*]OCCc1nc2ccccc2[nH]1', 'aromatic'),
    ('quinolin-2-yl-oxyethyl', '[*]OCCc1ccc2ncccc2c1', 'aromatic'),
    ('indol-3-yl-oxyethyl', '[*]OCCc1c[nH]c2ccccc12', 'aromatic'),
    ('3-OMe-4-OMe-phenyl', '[*]c1ccc(OC)c(OC)c1', 'aromatic'),
    ('3-OMe-4-Et-phenyl', '[*]c1ccc(CC)c(OC)c1', 'aromatic'),
    ('3-CF3-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-OH-phenyl', '[*]c1ccc(O)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-Et-phenyl', '[*]c1ccc(CC)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-4-OEt-phenyl', '[*]c1ccc(OCC)c(C(F)(F)F)c1', 'aromatic'),
    ('3-CN-4-CN-phenyl', '[*]c1ccc(C#N)c(C#N)c1', 'aromatic'),
    ('3-CN-4-OH-phenyl', '[*]c1ccc(O)c(C#N)c1', 'aromatic'),
    ('3-CN-4-Et-phenyl', '[*]c1ccc(CC)c(C#N)c1', 'aromatic'),
    ('3-CN-4-OEt-phenyl', '[*]c1ccc(OCC)c(C#N)c1', 'aromatic'),
    ('3-OH-4-OH-phenyl', '[*]c1ccc(O)c(O)c1', 'aromatic'),
    ('3-OH-4-Et-phenyl', '[*]c1ccc(CC)c(O)c1', 'aromatic'),
    ('3-OH-4-OEt-phenyl', '[*]c1ccc(OCC)c(O)c1', 'aromatic'),
    ('3-NH2-4-NH2-phenyl', '[*]c1ccc(N)c(N)c1', 'aromatic'),
    ('3-NH2-4-Et-phenyl', '[*]c1ccc(CC)c(N)c1', 'aromatic'),
    ('3-NH2-4-OEt-phenyl', '[*]c1ccc(OCC)c(N)c1', 'aromatic'),
    ('3-Et-4-Et-phenyl', '[*]c1ccc(CC)c(CC)c1', 'aromatic'),
    ('3-OEt-4-OEt-phenyl', '[*]c1ccc(OCC)c(OCC)c1', 'aromatic'),
    ('3-F-5-Cl-phenyl', '[*]c1cc(F)cc(Cl)c1', 'aromatic'),
    ('3-F-5-Me-phenyl', '[*]c1cc(F)cc(C)c1', 'aromatic'),
    ('3-F-5-OMe-phenyl', '[*]c1cc(F)cc(OC)c1', 'aromatic'),
    ('3-F-5-CF3-phenyl', '[*]c1cc(F)cc(C(F)(F)F)c1', 'aromatic'),
    ('3-F-5-CN-phenyl', '[*]c1cc(F)cc(C#N)c1', 'aromatic'),
    ('3-F-5-OH-phenyl', '[*]c1cc(F)cc(O)c1', 'aromatic'),
    ('3-F-5-NH2-phenyl', '[*]c1cc(F)cc(N)c1', 'aromatic'),
    ('3-F-5-Et-phenyl', '[*]c1cc(F)cc(CC)c1', 'aromatic'),
    ('3-F-5-OEt-phenyl', '[*]c1cc(F)cc(OCC)c1', 'aromatic'),
    ('3-Cl-5-Me-phenyl', '[*]c1cc(Cl)cc(C)c1', 'aromatic'),
    ('3-Cl-5-OMe-phenyl', '[*]c1cc(Cl)cc(OC)c1', 'aromatic'),
    ('3-Cl-5-CF3-phenyl', '[*]c1cc(Cl)cc(C(F)(F)F)c1', 'aromatic'),
    ('3-Cl-5-CN-phenyl', '[*]c1cc(Cl)cc(C#N)c1', 'aromatic'),
    ('3-Cl-5-OH-phenyl', '[*]c1cc(Cl)cc(O)c1', 'aromatic'),
    ('3-Cl-5-NH2-phenyl', '[*]c1cc(Cl)cc(N)c1', 'aromatic'),
    ('3-Cl-5-Et-phenyl', '[*]c1cc(Cl)cc(CC)c1', 'aromatic'),
    ('3-Cl-5-OEt-phenyl', '[*]c1cc(Cl)cc(OCC)c1', 'aromatic'),
    ('3-Me-5-Me-phenyl', '[*]c1cc(C)cc(C)c1', 'aromatic'),
    ('3-Me-5-OMe-phenyl', '[*]c1cc(C)cc(OC)c1', 'aromatic'),
    ('3-Me-5-CF3-phenyl', '[*]c1cc(C)cc(C(F)(F)F)c1', 'aromatic'),
    ('3-Me-5-CN-phenyl', '[*]c1cc(C)cc(C#N)c1', 'aromatic'),
    ('3-Me-5-OH-phenyl', '[*]c1cc(C)cc(O)c1', 'aromatic'),
    ('3-Me-5-NH2-phenyl', '[*]c1cc(C)cc(N)c1', 'aromatic'),
    ('3-Me-5-Et-phenyl', '[*]c1cc(C)cc(CC)c1', 'aromatic'),
    ('3-Me-5-OEt-phenyl', '[*]c1cc(C)cc(OCC)c1', 'aromatic'),
    ('3-OMe-5-OMe-phenyl', '[*]c1cc(OC)cc(OC)c1', 'aromatic'),
    ('3-OMe-5-CF3-phenyl', '[*]c1cc(OC)cc(C(F)(F)F)c1', 'aromatic'),
    ('3-OMe-5-CN-phenyl', '[*]c1cc(OC)cc(C#N)c1', 'aromatic'),
    ('3-OMe-5-OH-phenyl', '[*]c1cc(OC)cc(O)c1', 'aromatic'),
    ('3-OMe-5-NH2-phenyl', '[*]c1cc(OC)cc(N)c1', 'aromatic'),
    ('3-OMe-5-Et-phenyl', '[*]c1cc(OC)cc(CC)c1', 'aromatic'),
    ('3-OMe-5-OEt-phenyl', '[*]c1cc(OC)cc(OCC)c1', 'aromatic'),
    ('3-CF3-5-CF3-phenyl', '[*]c1cc(C(F)(F)F)cc(C(F)(F)F)c1', 'aromatic'),
    ('3-CF3-5-CN-phenyl', '[*]c1cc(C(F)(F)F)cc(C#N)c1', 'aromatic'),
    ('3-CF3-5-OH-phenyl', '[*]c1cc(C(F)(F)F)cc(O)c1', 'aromatic'),
    ('3-CF3-5-NH2-phenyl', '[*]c1cc(C(F)(F)F)cc(N)c1', 'aromatic'),
    ('3-CF3-5-Et-phenyl', '[*]c1cc(C(F)(F)F)cc(CC)c1', 'aromatic'),
    ('3-CF3-5-OEt-phenyl', '[*]c1cc(C(F)(F)F)cc(OCC)c1', 'aromatic'),
    ('3-CN-5-CN-phenyl', '[*]c1cc(C#N)cc(C#N)c1', 'aromatic'),
    ('3-CN-5-OH-phenyl', '[*]c1cc(C#N)cc(O)c1', 'aromatic'),
    ('3-CN-5-NH2-phenyl', '[*]c1cc(C#N)cc(N)c1', 'aromatic'),
    ('3-CN-5-Et-phenyl', '[*]c1cc(C#N)cc(CC)c1', 'aromatic'),
    ('3-CN-5-OEt-phenyl', '[*]c1cc(C#N)cc(OCC)c1', 'aromatic'),
    ('3-OH-5-OH-phenyl', '[*]c1cc(O)cc(O)c1', 'aromatic'),
    ('3-OH-5-NH2-phenyl', '[*]c1cc(O)cc(N)c1', 'aromatic'),
    ('3-OH-5-Et-phenyl', '[*]c1cc(O)cc(CC)c1', 'aromatic'),
    ('3-OH-5-OEt-phenyl', '[*]c1cc(O)cc(OCC)c1', 'aromatic'),
    ('3-NH2-5-NH2-phenyl', '[*]c1cc(N)cc(N)c1', 'aromatic'),
    ('3-NH2-5-Et-phenyl', '[*]c1cc(N)cc(CC)c1', 'aromatic'),
    ('3-NH2-5-OEt-phenyl', '[*]c1cc(N)cc(OCC)c1', 'aromatic'),
    ('3-Et-5-Et-phenyl', '[*]c1cc(CC)cc(CC)c1', 'aromatic'),
    ('3-Et-5-OEt-phenyl', '[*]c1cc(CC)cc(OCC)c1', 'aromatic'),
    ('3-OEt-5-OEt-phenyl', '[*]c1cc(OCC)cc(OCC)c1', 'aromatic'),
    ('2-Me-4-Me-phenyl', '[*]c1ccc(C)cc1C', 'aromatic'),
    ('2-OMe-4-OMe-phenyl', '[*]c1ccc(OC)cc1OC', 'aromatic'),
    ('2-OMe-4-Et-phenyl', '[*]c1ccc(CC)cc1OC', 'aromatic'),
    ('2-CF3-4-CF3-phenyl', '[*]c1ccc(C(F)(F)F)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-OH-phenyl', '[*]c1ccc(O)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-Et-phenyl', '[*]c1ccc(CC)cc1C(F)(F)F', 'aromatic'),
    ('2-CF3-4-OEt-phenyl', '[*]c1ccc(OCC)cc1C(F)(F)F', 'aromatic'),
    ('2-CN-4-CN-phenyl', '[*]c1ccc(C#N)cc1C#N', 'aromatic'),
    ('2-CN-4-OH-phenyl', '[*]c1ccc(O)cc1C#N', 'aromatic'),
    ('2-CN-4-Et-phenyl', '[*]c1ccc(CC)cc1C#N', 'aromatic'),
    ('2-CN-4-OEt-phenyl', '[*]c1ccc(OCC)cc1C#N', 'aromatic'),
    ('2-OH-4-OH-phenyl', '[*]c1ccc(O)cc1O', 'aromatic'),
    ('2-OH-4-Et-phenyl', '[*]c1ccc(CC)cc1O', 'aromatic'),
    ('2-OH-4-OEt-phenyl', '[*]c1ccc(OCC)cc1O', 'aromatic'),
    ('2-NH2-4-NH2-phenyl', '[*]c1ccc(N)cc1N', 'aromatic'),
    ('2-NH2-4-Et-phenyl', '[*]c1ccc(CC)cc1N', 'aromatic'),
    ('2-NH2-4-OEt-phenyl', '[*]c1ccc(OCC)cc1N', 'aromatic'),
    ('2-Et-4-Et-phenyl', '[*]c1ccc(CC)cc1CC', 'aromatic'),
    ('2-OEt-4-OEt-phenyl', '[*]c1ccc(OCC)cc1OCC', 'aromatic'),
    ('2-F-3-Cl-phenyl', '[*]c1cccc(Cl)c1F', 'aromatic'),
    ('2-F-3-Me-phenyl', '[*]c1cccc(C)c1F', 'aromatic'),
    ('2-F-3-OMe-phenyl', '[*]c1cccc(OC)c1F', 'aromatic'),
    ('2-F-3-CF3-phenyl', '[*]c1cccc(C(F)(F)F)c1F', 'aromatic'),
    ('2-F-3-CN-phenyl', '[*]c1cccc(C#N)c1F', 'aromatic'),
    ('2-F-3-OH-phenyl', '[*]c1cccc(O)c1F', 'aromatic'),
    ('2-F-3-NH2-phenyl', '[*]c1cccc(N)c1F', 'aromatic'),
    ('2-F-3-Et-phenyl', '[*]c1cccc(CC)c1F', 'aromatic'),
    ('2-F-3-OEt-phenyl', '[*]c1cccc(OCC)c1F', 'aromatic'),
    ('2-Cl-3-Cl-phenyl', '[*]c1cccc(Cl)c1Cl', 'aromatic'),
    ('2-Cl-3-Me-phenyl', '[*]c1cccc(C)c1Cl', 'aromatic'),
    ('2-Cl-3-OMe-phenyl', '[*]c1cccc(OC)c1Cl', 'aromatic'),
    ('2-Cl-3-CF3-phenyl', '[*]c1cccc(C(F)(F)F)c1Cl', 'aromatic'),
    ('2-Cl-3-CN-phenyl', '[*]c1cccc(C#N)c1Cl', 'aromatic'),
    ('2-Cl-3-OH-phenyl', '[*]c1cccc(O)c1Cl', 'aromatic'),
    ('2-Cl-3-NH2-phenyl', '[*]c1cccc(N)c1Cl', 'aromatic'),
    ('2-Cl-3-Et-phenyl', '[*]c1cccc(CC)c1Cl', 'aromatic'),
    ('2-Cl-3-OEt-phenyl', '[*]c1cccc(OCC)c1Cl', 'aromatic'),
    ('2-Me-3-Me-phenyl', '[*]c1cccc(C)c1C', 'aromatic'),
    ('2-Me-3-OMe-phenyl', '[*]c1cccc(OC)c1C', 'aromatic'),
    ('2-Me-3-CF3-phenyl', '[*]c1cccc(C(F)(F)F)c1C', 'aromatic'),
    ('2-Me-3-CN-phenyl', '[*]c1cccc(C#N)c1C', 'aromatic'),
    ('2-Me-3-OH-phenyl', '[*]c1cccc(O)c1C', 'aromatic'),
    ('2-Me-3-NH2-phenyl', '[*]c1cccc(N)c1C', 'aromatic'),
    ('2-Me-3-Et-phenyl', '[*]c1cccc(CC)c1C', 'aromatic'),
    ('2-Me-3-OEt-phenyl', '[*]c1cccc(OCC)c1C', 'aromatic'),
    ('2-OMe-3-OMe-phenyl', '[*]c1cccc(OC)c1OC', 'aromatic'),
    ('2-OMe-3-CF3-phenyl', '[*]c1cccc(C(F)(F)F)c1OC', 'aromatic'),
    ('2-OMe-3-CN-phenyl', '[*]c1cccc(C#N)c1OC', 'aromatic'),
    ('2-OMe-3-OH-phenyl', '[*]c1cccc(O)c1OC', 'aromatic'),
    ('2-OMe-3-NH2-phenyl', '[*]c1cccc(N)c1OC', 'aromatic'),
    ('2-OMe-3-Et-phenyl', '[*]c1cccc(CC)c1OC', 'aromatic'),
    ('2-OMe-3-OEt-phenyl', '[*]c1cccc(OCC)c1OC', 'aromatic'),
    ('2-CF3-3-CF3-phenyl', '[*]c1cccc(C(F)(F)F)c1C(F)(F)F', 'aromatic'),
    ('2-CF3-3-CN-phenyl', '[*]c1cccc(C#N)c1C(F)(F)F', 'aromatic'),
    ('2-CF3-3-OH-phenyl', '[*]c1cccc(O)c1C(F)(F)F', 'aromatic'),
    ('2-CF3-3-NH2-phenyl', '[*]c1cccc(N)c1C(F)(F)F', 'aromatic'),
    ('2-CF3-3-Et-phenyl', '[*]c1cccc(CC)c1C(F)(F)F', 'aromatic'),
    ('2-CF3-3-OEt-phenyl', '[*]c1cccc(OCC)c1C(F)(F)F', 'aromatic'),
    ('2-CN-3-CN-phenyl', '[*]c1cccc(C#N)c1C#N', 'aromatic'),
    ('2-CN-3-OH-phenyl', '[*]c1cccc(O)c1C#N', 'aromatic'),
    ('2-CN-3-NH2-phenyl', '[*]c1cccc(N)c1C#N', 'aromatic'),
    ('2-CN-3-Et-phenyl', '[*]c1cccc(CC)c1C#N', 'aromatic'),
    ('2-CN-3-OEt-phenyl', '[*]c1cccc(OCC)c1C#N', 'aromatic'),
    ('2-OH-3-OH-phenyl', '[*]c1cccc(O)c1O', 'aromatic'),
    ('2-OH-3-NH2-phenyl', '[*]c1cccc(N)c1O', 'aromatic'),
    ('2-OH-3-Et-phenyl', '[*]c1cccc(CC)c1O', 'aromatic'),
    ('2-OH-3-OEt-phenyl', '[*]c1cccc(OCC)c1O', 'aromatic'),
    ('2-NH2-3-NH2-phenyl', '[*]c1cccc(N)c1N', 'aromatic'),
    ('2-NH2-3-Et-phenyl', '[*]c1cccc(CC)c1N', 'aromatic'),
    ('2-NH2-3-OEt-phenyl', '[*]c1cccc(OCC)c1N', 'aromatic'),
    ('2-Et-3-Et-phenyl', '[*]c1cccc(CC)c1CC', 'aromatic'),
    ('2-Et-3-OEt-phenyl', '[*]c1cccc(OCC)c1CC', 'aromatic'),
    ('2-OEt-3-OEt-phenyl', '[*]c1cccc(OCC)c1OCC', 'aromatic'),
    ('2-F-6-Cl-phenyl', '[*]c1c(F)cccc1Cl', 'aromatic'),
    ('2-F-6-Me-phenyl', '[*]c1c(F)cccc1C', 'aromatic'),
    ('2-F-6-OMe-phenyl', '[*]c1c(F)cccc1OC', 'aromatic'),
    ('2-F-6-CF3-phenyl', '[*]c1c(F)cccc1C(F)(F)F', 'aromatic'),
    ('2-F-6-CN-phenyl', '[*]c1c(F)cccc1C#N', 'aromatic'),
    ('2-F-6-OH-phenyl', '[*]c1c(F)cccc1O', 'aromatic'),
    ('2-F-6-NH2-phenyl', '[*]c1c(F)cccc1N', 'aromatic'),
    ('2-F-6-Et-phenyl', '[*]c1c(F)cccc1CC', 'aromatic'),
    ('2-F-6-OEt-phenyl', '[*]c1c(F)cccc1OCC', 'aromatic'),
    ('2-Cl-6-Me-phenyl', '[*]c1c(Cl)cccc1C', 'aromatic'),
    ('2-Cl-6-OMe-phenyl', '[*]c1c(Cl)cccc1OC', 'aromatic'),
    ('2-Cl-6-CF3-phenyl', '[*]c1c(Cl)cccc1C(F)(F)F', 'aromatic'),
    ('2-Cl-6-CN-phenyl', '[*]c1c(Cl)cccc1C#N', 'aromatic'),
    ('2-Cl-6-OH-phenyl', '[*]c1c(Cl)cccc1O', 'aromatic'),
    ('2-Cl-6-NH2-phenyl', '[*]c1c(Cl)cccc1N', 'aromatic'),
    ('2-Cl-6-Et-phenyl', '[*]c1c(Cl)cccc1CC', 'aromatic'),
    ('2-Cl-6-OEt-phenyl', '[*]c1c(Cl)cccc1OCC', 'aromatic'),
    ('2-Me-6-OMe-phenyl', '[*]c1c(C)cccc1OC', 'aromatic'),
    ('2-Me-6-CF3-phenyl', '[*]c1c(C)cccc1C(F)(F)F', 'aromatic'),
    ('2-Me-6-CN-phenyl', '[*]c1c(C)cccc1C#N', 'aromatic'),
    ('2-Me-6-OH-phenyl', '[*]c1c(C)cccc1O', 'aromatic'),
    ('2-Me-6-NH2-phenyl', '[*]c1c(C)cccc1N', 'aromatic'),
    ('2-Me-6-Et-phenyl', '[*]c1c(C)cccc1CC', 'aromatic'),
    ('2-Me-6-OEt-phenyl', '[*]c1c(C)cccc1OCC', 'aromatic'),
    ('2-OMe-6-OMe-phenyl', '[*]c1c(OC)cccc1OC', 'aromatic'),
    ('2-OMe-6-CF3-phenyl', '[*]c1c(OC)cccc1C(F)(F)F', 'aromatic'),
    ('2-OMe-6-CN-phenyl', '[*]c1c(OC)cccc1C#N', 'aromatic'),
    ('2-OMe-6-OH-phenyl', '[*]c1c(OC)cccc1O', 'aromatic'),
    ('2-OMe-6-NH2-phenyl', '[*]c1c(OC)cccc1N', 'aromatic'),
    ('2-OMe-6-Et-phenyl', '[*]c1c(OC)cccc1CC', 'aromatic'),
    ('2-OMe-6-OEt-phenyl', '[*]c1c(OC)cccc1OCC', 'aromatic'),
    ('2-CF3-6-CF3-phenyl', '[*]c1c(C(F)(F)F)cccc1C(F)(F)F', 'aromatic'),
    ('2-CF3-6-CN-phenyl', '[*]c1c(C(F)(F)F)cccc1C#N', 'aromatic'),
    ('2-CF3-6-OH-phenyl', '[*]c1c(C(F)(F)F)cccc1O', 'aromatic'),
    ('2-CF3-6-NH2-phenyl', '[*]c1c(C(F)(F)F)cccc1N', 'aromatic'),
    ('2-CF3-6-Et-phenyl', '[*]c1c(C(F)(F)F)cccc1CC', 'aromatic'),
    ('2-CF3-6-OEt-phenyl', '[*]c1c(C(F)(F)F)cccc1OCC', 'aromatic'),
    ('2-CN-6-CN-phenyl', '[*]c1c(C#N)cccc1C#N', 'aromatic'),
    ('2-CN-6-OH-phenyl', '[*]c1c(C#N)cccc1O', 'aromatic'),
    ('2-CN-6-NH2-phenyl', '[*]c1c(C#N)cccc1N', 'aromatic'),
    ('2-CN-6-Et-phenyl', '[*]c1c(C#N)cccc1CC', 'aromatic'),
    ('2-CN-6-OEt-phenyl', '[*]c1c(C#N)cccc1OCC', 'aromatic'),
    ('2-OH-6-OH-phenyl', '[*]c1c(O)cccc1O', 'aromatic'),
    ('2-OH-6-NH2-phenyl', '[*]c1c(O)cccc1N', 'aromatic'),
    ('2-OH-6-Et-phenyl', '[*]c1c(O)cccc1CC', 'aromatic'),
    ('2-OH-6-OEt-phenyl', '[*]c1c(O)cccc1OCC', 'aromatic'),
    ('2-NH2-6-NH2-phenyl', '[*]c1c(N)cccc1N', 'aromatic'),
    ('2-NH2-6-Et-phenyl', '[*]c1c(N)cccc1CC', 'aromatic'),
    ('2-NH2-6-OEt-phenyl', '[*]c1c(N)cccc1OCC', 'aromatic'),
    ('2-Et-6-Et-phenyl', '[*]c1c(CC)cccc1CC', 'aromatic'),
    ('2-Et-6-OEt-phenyl', '[*]c1c(CC)cccc1OCC', 'aromatic'),
    ('2-OEt-6-OEt-phenyl', '[*]c1c(OCC)cccc1OCC', 'aromatic'),
    ('5-SO2Me-pyridin-2-yl', '[*]c1ccc(S(C)(=O)=O)cn1', 'aromatic'),
    ('5-cyclopropyl-pyridin-2-yl', '[*]c1ccc(C1CC1)cn1', 'aromatic'),
    ('5-NHAc-pyridin-2-yl', '[*]c1ccc(NC(C)=O)cn1', 'aromatic'),
    ('5-morpholino-pyridin-2-yl', '[*]c1ccc(N1CCOCC1)cn1', 'aromatic'),
    ('5-piperidino-pyridin-2-yl', '[*]c1ccc(N1CCCCC1)cn1', 'aromatic'),
    ('5-pyrrolidino-pyridin-2-yl', '[*]c1ccc(N1CCCC1)cn1', 'aromatic'),
    ('5-SMe-pyridin-2-yl', '[*]c1ccc(SC)cn1', 'aromatic'),
    ('5-NHSO2Me-pyridin-2-yl', '[*]c1ccc(NS(C)(=O)=O)cn1', 'aromatic'),
    ('4-SO2Me-pyridin-2-yl', '[*]c1cc(S(C)(=O)=O)ccn1', 'aromatic'),
    ('4-cyclopropyl-pyridin-2-yl', '[*]c1cc(C1CC1)ccn1', 'aromatic'),
    ('4-NHAc-pyridin-2-yl', '[*]c1cc(NC(C)=O)ccn1', 'aromatic'),
    ('4-morpholino-pyridin-2-yl', '[*]c1cc(N1CCOCC1)ccn1', 'aromatic'),
    ('4-piperidino-pyridin-2-yl', '[*]c1cc(N1CCCCC1)ccn1', 'aromatic'),
    ('4-pyrrolidino-pyridin-2-yl', '[*]c1cc(N1CCCC1)ccn1', 'aromatic'),
    ('4-SMe-pyridin-2-yl', '[*]c1cc(SC)ccn1', 'aromatic'),
    ('4-NHSO2Me-pyridin-2-yl', '[*]c1cc(NS(C)(=O)=O)ccn1', 'aromatic'),
    ('3-SO2Me-pyridin-2-yl', '[*]c1cccc(S(C)(=O)=O)n1', 'aromatic'),
    ('3-cyclopropyl-pyridin-2-yl', '[*]c1cccc(C1CC1)n1', 'aromatic'),
    ('3-NHAc-pyridin-2-yl', '[*]c1cccc(NC(C)=O)n1', 'aromatic'),
    ('3-morpholino-pyridin-2-yl', '[*]c1cccc(N1CCOCC1)n1', 'aromatic'),
    ('3-piperidino-pyridin-2-yl', '[*]c1cccc(N1CCCCC1)n1', 'aromatic'),
    ('3-pyrrolidino-pyridin-2-yl', '[*]c1cccc(N1CCCC1)n1', 'aromatic'),
    ('3-SMe-pyridin-2-yl', '[*]c1cccc(SC)n1', 'aromatic'),
    ('3-NHSO2Me-pyridin-2-yl', '[*]c1cccc(NS(C)(=O)=O)n1', 'aromatic'),
    ('6-SO2Me-pyridin-3-yl', '[*]c1ccc(S(C)(=O)=O)nc1', 'aromatic'),
    ('6-cyclopropyl-pyridin-3-yl', '[*]c1ccc(C1CC1)nc1', 'aromatic'),
    ('6-NHAc-pyridin-3-yl', '[*]c1ccc(NC(C)=O)nc1', 'aromatic'),
    ('6-morpholino-pyridin-3-yl', '[*]c1ccc(N1CCOCC1)nc1', 'aromatic'),
    ('6-piperidino-pyridin-3-yl', '[*]c1ccc(N1CCCCC1)nc1', 'aromatic'),
    ('6-pyrrolidino-pyridin-3-yl', '[*]c1ccc(N1CCCC1)nc1', 'aromatic'),
    ('6-SMe-pyridin-3-yl', '[*]c1ccc(SC)nc1', 'aromatic'),
    ('6-NHSO2Me-pyridin-3-yl', '[*]c1ccc(NS(C)(=O)=O)nc1', 'aromatic'),
    ('5-SO2Me-pyridin-3-yl', '[*]c1cncc(S(C)(=O)=O)c1', 'aromatic'),
    ('5-cyclopropyl-pyridin-3-yl', '[*]c1cncc(C1CC1)c1', 'aromatic'),
    ('5-NHAc-pyridin-3-yl', '[*]c1cncc(NC(C)=O)c1', 'aromatic'),
    ('5-morpholino-pyridin-3-yl', '[*]c1cncc(N1CCOCC1)c1', 'aromatic'),
    ('5-piperidino-pyridin-3-yl', '[*]c1cncc(N1CCCCC1)c1', 'aromatic'),
    ('5-pyrrolidino-pyridin-3-yl', '[*]c1cncc(N1CCCC1)c1', 'aromatic'),
    ('5-SMe-pyridin-3-yl', '[*]c1cncc(SC)c1', 'aromatic'),
    ('5-NHSO2Me-pyridin-3-yl', '[*]c1cncc(NS(C)(=O)=O)c1', 'aromatic'),
    ('2-SO2Me-pyridin-4-yl', '[*]c1cc(S(C)(=O)=O)ncc1', 'aromatic'),
    ('2-cyclopropyl-pyridin-4-yl', '[*]c1cc(C1CC1)ncc1', 'aromatic'),
    ('2-NHAc-pyridin-4-yl', '[*]c1cc(NC(C)=O)ncc1', 'aromatic'),
    ('2-morpholino-pyridin-4-yl', '[*]c1cc(N1CCOCC1)ncc1', 'aromatic'),
    ('2-piperidino-pyridin-4-yl', '[*]c1cc(N1CCCCC1)ncc1', 'aromatic'),
    ('2-pyrrolidino-pyridin-4-yl', '[*]c1cc(N1CCCC1)ncc1', 'aromatic'),
    ('2-SMe-pyridin-4-yl', '[*]c1cc(SC)ncc1', 'aromatic'),
    ('2-NHSO2Me-pyridin-4-yl', '[*]c1cc(NS(C)(=O)=O)ncc1', 'aromatic'),
    ('3-cyclopropyl-pyridin-4-yl', '[*]c1ccnc(C1CC1)c1', 'aromatic'),
    ('3-morpholino-pyridin-4-yl', '[*]c1ccnc(N1CCOCC1)c1', 'aromatic'),
    ('3-piperidino-pyridin-4-yl', '[*]c1ccnc(N1CCCCC1)c1', 'aromatic'),
    ('3-pyrrolidino-pyridin-4-yl', '[*]c1ccnc(N1CCCC1)c1', 'aromatic'),
    ('4-F-pyrimidin-2-yl', '[*]c1nc(F)ccn1', 'aromatic'),
    ('4-Cl-pyrimidin-2-yl', '[*]c1nc(Cl)ccn1', 'aromatic'),
    ('4-Br-pyrimidin-2-yl', '[*]c1nc(Br)ccn1', 'aromatic'),
    ('4-Me-pyrimidin-2-yl', '[*]c1nc(C)ccn1', 'aromatic'),
    ('4-Et-pyrimidin-2-yl', '[*]c1nc(CC)ccn1', 'aromatic'),
    ('4-iPr-pyrimidin-2-yl', '[*]c1nc(C(C)C)ccn1', 'aromatic'),
    ('4-OEt-pyrimidin-2-yl', '[*]c1nc(OCC)ccn1', 'aromatic'),
    ('4-OH-pyrimidin-2-yl', '[*]c1nc(O)ccn1', 'aromatic'),
    ('4-CF3-pyrimidin-2-yl', '[*]c1nc(C(F)(F)F)ccn1', 'aromatic'),
    ('4-CHF2-pyrimidin-2-yl', '[*]c1nc(C(F)F)ccn1', 'aromatic'),
    ('4-CN-pyrimidin-2-yl', '[*]c1nc(C#N)ccn1', 'aromatic'),
    ('4-NHMe-pyrimidin-2-yl', '[*]c1nc(NC)ccn1', 'aromatic'),
    ('4-NMe2-pyrimidin-2-yl', '[*]c1nc(N(C)C)ccn1', 'aromatic'),
    ('4-COOH-pyrimidin-2-yl', '[*]c1nc(C(=O)O)ccn1', 'aromatic'),
    ('4-COOMe-pyrimidin-2-yl', '[*]c1nc(C(=O)OC)ccn1', 'aromatic'),
    ('4-COMe-pyrimidin-2-yl', '[*]c1nc(C(C)=O)ccn1', 'aromatic'),
    ('4-SO2Me-pyrimidin-2-yl', '[*]c1nc(S(C)(=O)=O)ccn1', 'aromatic'),
    ('4-cyclopropyl-pyrimidin-2-yl', '[*]c1nc(C1CC1)ccn1', 'aromatic'),
    ('4-NHAc-pyrimidin-2-yl', '[*]c1nc(NC(C)=O)ccn1', 'aromatic'),
    ('4-morpholino-pyrimidin-2-yl', '[*]c1nc(N1CCOCC1)ccn1', 'aromatic'),
    ('4-piperidino-pyrimidin-2-yl', '[*]c1nc(N1CCCCC1)ccn1', 'aromatic'),
    ('4-pyrrolidino-pyrimidin-2-yl', '[*]c1nc(N1CCCC1)ccn1', 'aromatic'),
    ('4-OCF3-pyrimidin-2-yl', '[*]c1nc(OC(F)(F)F)ccn1', 'aromatic'),
    ('4-SMe-pyrimidin-2-yl', '[*]c1nc(SC)ccn1', 'aromatic'),
    ('4-NHSO2Me-pyrimidin-2-yl', '[*]c1nc(NS(C)(=O)=O)ccn1', 'aromatic'),
    ('5-F-pyrimidin-2-yl', '[*]c1ncc(F)cn1', 'aromatic'),
    ('5-Cl-pyrimidin-2-yl', '[*]c1ncc(Cl)cn1', 'aromatic'),
    ('5-Br-pyrimidin-2-yl', '[*]c1ncc(Br)cn1', 'aromatic'),
    ('5-Et-pyrimidin-2-yl', '[*]c1ncc(CC)cn1', 'aromatic'),
    ('5-iPr-pyrimidin-2-yl', '[*]c1ncc(C(C)C)cn1', 'aromatic'),
    ('5-OMe-pyrimidin-2-yl', '[*]c1ncc(OC)cn1', 'aromatic'),
    ('5-OEt-pyrimidin-2-yl', '[*]c1ncc(OCC)cn1', 'aromatic'),
    ('5-OH-pyrimidin-2-yl', '[*]c1ncc(O)cn1', 'aromatic'),
    ('5-CF3-pyrimidin-2-yl', '[*]c1ncc(C(F)(F)F)cn1', 'aromatic'),
    ('5-CHF2-pyrimidin-2-yl', '[*]c1ncc(C(F)F)cn1', 'aromatic'),
    ('5-CN-pyrimidin-2-yl', '[*]c1ncc(C#N)cn1', 'aromatic'),
    ('5-NH2-pyrimidin-2-yl', '[*]c1ncc(N)cn1', 'aromatic'),
    ('5-NHMe-pyrimidin-2-yl', '[*]c1ncc(NC)cn1', 'aromatic'),
    ('5-NMe2-pyrimidin-2-yl', '[*]c1ncc(N(C)C)cn1', 'aromatic'),
    ('5-COOH-pyrimidin-2-yl', '[*]c1ncc(C(=O)O)cn1', 'aromatic'),
    ('5-COOMe-pyrimidin-2-yl', '[*]c1ncc(C(=O)OC)cn1', 'aromatic'),
    ('5-COMe-pyrimidin-2-yl', '[*]c1ncc(C(C)=O)cn1', 'aromatic'),
    ('5-SO2Me-pyrimidin-2-yl', '[*]c1ncc(S(C)(=O)=O)cn1', 'aromatic'),
    ('5-cyclopropyl-pyrimidin-2-yl', '[*]c1ncc(C1CC1)cn1', 'aromatic'),
    ('5-NHAc-pyrimidin-2-yl', '[*]c1ncc(NC(C)=O)cn1', 'aromatic'),
    ('5-morpholino-pyrimidin-2-yl', '[*]c1ncc(N1CCOCC1)cn1', 'aromatic'),
    ('5-piperidino-pyrimidin-2-yl', '[*]c1ncc(N1CCCCC1)cn1', 'aromatic'),
    ('5-pyrrolidino-pyrimidin-2-yl', '[*]c1ncc(N1CCCC1)cn1', 'aromatic'),
    ('5-OCF3-pyrimidin-2-yl', '[*]c1ncc(OC(F)(F)F)cn1', 'aromatic'),
    ('5-SMe-pyrimidin-2-yl', '[*]c1ncc(SC)cn1', 'aromatic'),
    ('5-NHSO2Me-pyrimidin-2-yl', '[*]c1ncc(NS(C)(=O)=O)cn1', 'aromatic'),
    ('5-F-pyrimidin-4-yl', '[*]c1ncnc(F)c1', 'aromatic'),
    ('5-Cl-pyrimidin-4-yl', '[*]c1ncnc(Cl)c1', 'aromatic'),
    ('5-Br-pyrimidin-4-yl', '[*]c1ncnc(Br)c1', 'aromatic'),
    ('5-Me-pyrimidin-4-yl', '[*]c1ncnc(C)c1', 'aromatic'),
    ('5-Et-pyrimidin-4-yl', '[*]c1ncnc(CC)c1', 'aromatic'),
    ('5-iPr-pyrimidin-4-yl', '[*]c1ncnc(C(C)C)c1', 'aromatic'),
    ('5-OMe-pyrimidin-4-yl', '[*]c1ncnc(OC)c1', 'aromatic'),
    ('5-OEt-pyrimidin-4-yl', '[*]c1ncnc(OCC)c1', 'aromatic'),
    ('5-OH-pyrimidin-4-yl', '[*]c1ncnc(O)c1', 'aromatic'),
    ('5-CF3-pyrimidin-4-yl', '[*]c1ncnc(C(F)(F)F)c1', 'aromatic'),
    ('5-CHF2-pyrimidin-4-yl', '[*]c1ncnc(C(F)F)c1', 'aromatic'),
    ('5-CN-pyrimidin-4-yl', '[*]c1ncnc(C#N)c1', 'aromatic'),
    ('5-NH2-pyrimidin-4-yl', '[*]c1ncnc(N)c1', 'aromatic'),
    ('5-NHMe-pyrimidin-4-yl', '[*]c1ncnc(NC)c1', 'aromatic'),
    ('5-NMe2-pyrimidin-4-yl', '[*]c1ncnc(N(C)C)c1', 'aromatic'),
    ('5-COOH-pyrimidin-4-yl', '[*]c1ncnc(C(=O)O)c1', 'aromatic'),
    ('5-COOMe-pyrimidin-4-yl', '[*]c1ncnc(C(=O)OC)c1', 'aromatic'),
    ('5-COMe-pyrimidin-4-yl', '[*]c1ncnc(C(C)=O)c1', 'aromatic'),
    ('5-SO2Me-pyrimidin-4-yl', '[*]c1ncnc(S(C)(=O)=O)c1', 'aromatic'),
    ('5-cyclopropyl-pyrimidin-4-yl', '[*]c1ncnc(C1CC1)c1', 'aromatic'),
    ('5-NHAc-pyrimidin-4-yl', '[*]c1ncnc(NC(C)=O)c1', 'aromatic'),
    ('5-morpholino-pyrimidin-4-yl', '[*]c1ncnc(N1CCOCC1)c1', 'aromatic'),
    ('5-piperidino-pyrimidin-4-yl', '[*]c1ncnc(N1CCCCC1)c1', 'aromatic'),
    ('5-pyrrolidino-pyrimidin-4-yl', '[*]c1ncnc(N1CCCC1)c1', 'aromatic'),
    ('5-OCF3-pyrimidin-4-yl', '[*]c1ncnc(OC(F)(F)F)c1', 'aromatic'),
    ('5-SMe-pyrimidin-4-yl', '[*]c1ncnc(SC)c1', 'aromatic'),
    ('5-NHSO2Me-pyrimidin-4-yl', '[*]c1ncnc(NS(C)(=O)=O)c1', 'aromatic'),
    ('3-F-pyrazin-2-yl', '[*]c1cncc(F)n1', 'aromatic'),
    ('3-Cl-pyrazin-2-yl', '[*]c1cncc(Cl)n1', 'aromatic'),
    ('3-Br-pyrazin-2-yl', '[*]c1cncc(Br)n1', 'aromatic'),
    ('3-Me-pyrazin-2-yl', '[*]c1cncc(C)n1', 'aromatic'),
    ('3-Et-pyrazin-2-yl', '[*]c1cncc(CC)n1', 'aromatic'),
    ('3-iPr-pyrazin-2-yl', '[*]c1cncc(C(C)C)n1', 'aromatic'),
    ('3-OMe-pyrazin-2-yl', '[*]c1cncc(OC)n1', 'aromatic'),
    ('3-OEt-pyrazin-2-yl', '[*]c1cncc(OCC)n1', 'aromatic'),
    ('3-OH-pyrazin-2-yl', '[*]c1cncc(O)n1', 'aromatic'),
    ('3-CF3-pyrazin-2-yl', '[*]c1cncc(C(F)(F)F)n1', 'aromatic'),
    ('3-CHF2-pyrazin-2-yl', '[*]c1cncc(C(F)F)n1', 'aromatic'),
    ('3-CN-pyrazin-2-yl', '[*]c1cncc(C#N)n1', 'aromatic'),
    ('3-NH2-pyrazin-2-yl', '[*]c1cncc(N)n1', 'aromatic'),
    ('3-NHMe-pyrazin-2-yl', '[*]c1cncc(NC)n1', 'aromatic'),
    ('3-NMe2-pyrazin-2-yl', '[*]c1cncc(N(C)C)n1', 'aromatic'),
    ('3-COOH-pyrazin-2-yl', '[*]c1cncc(C(=O)O)n1', 'aromatic'),
    ('3-COOMe-pyrazin-2-yl', '[*]c1cncc(C(=O)OC)n1', 'aromatic'),
    ('3-COMe-pyrazin-2-yl', '[*]c1cncc(C(C)=O)n1', 'aromatic'),
    ('3-SO2Me-pyrazin-2-yl', '[*]c1cncc(S(C)(=O)=O)n1', 'aromatic'),
    ('3-cyclopropyl-pyrazin-2-yl', '[*]c1cncc(C1CC1)n1', 'aromatic'),
    ('3-NHAc-pyrazin-2-yl', '[*]c1cncc(NC(C)=O)n1', 'aromatic'),
    ('3-morpholino-pyrazin-2-yl', '[*]c1cncc(N1CCOCC1)n1', 'aromatic'),
    ('3-piperidino-pyrazin-2-yl', '[*]c1cncc(N1CCCCC1)n1', 'aromatic'),
    ('3-pyrrolidino-pyrazin-2-yl', '[*]c1cncc(N1CCCC1)n1', 'aromatic'),
    ('3-OCF3-pyrazin-2-yl', '[*]c1cncc(OC(F)(F)F)n1', 'aromatic'),
    ('3-SMe-pyrazin-2-yl', '[*]c1cncc(SC)n1', 'aromatic'),
    ('3-NHSO2Me-pyrazin-2-yl', '[*]c1cncc(NS(C)(=O)=O)n1', 'aromatic'),
    ('3-F-thiophen-2-yl', '[*]c1ccsc1F', 'aromatic'),
    ('3-Cl-thiophen-2-yl', '[*]c1ccsc1Cl', 'aromatic'),
    ('3-Br-thiophen-2-yl', '[*]c1ccsc1Br', 'aromatic'),
    ('3-Et-thiophen-2-yl', '[*]c1ccsc1CC', 'aromatic'),
    ('3-iPr-thiophen-2-yl', '[*]c1ccsc1C(C)C', 'aromatic'),
    ('3-OMe-thiophen-2-yl', '[*]c1ccsc1OC', 'aromatic'),
    ('3-OEt-thiophen-2-yl', '[*]c1ccsc1OCC', 'aromatic'),
    ('3-OH-thiophen-2-yl', '[*]c1ccsc1O', 'aromatic'),
    ('3-CF3-thiophen-2-yl', '[*]c1ccsc1C(F)(F)F', 'aromatic'),
    ('3-CHF2-thiophen-2-yl', '[*]c1ccsc1C(F)F', 'aromatic'),
    ('3-CN-thiophen-2-yl', '[*]c1ccsc1C#N', 'aromatic'),
    ('3-NH2-thiophen-2-yl', '[*]c1ccsc1N', 'aromatic'),
    ('3-NHMe-thiophen-2-yl', '[*]c1ccsc1NC', 'aromatic'),
    ('3-NMe2-thiophen-2-yl', '[*]c1ccsc1N(C)C', 'aromatic'),
    ('3-COOH-thiophen-2-yl', '[*]c1ccsc1C(=O)O', 'aromatic'),
    ('3-COOMe-thiophen-2-yl', '[*]c1ccsc1C(=O)OC', 'aromatic'),
    ('3-COMe-thiophen-2-yl', '[*]c1ccsc1C(C)=O', 'aromatic'),
    ('3-SO2Me-thiophen-2-yl', '[*]c1ccsc1S(C)(=O)=O', 'aromatic'),
    ('3-cyclopropyl-thiophen-2-yl', '[*]c1ccsc1C1CC1', 'aromatic'),
    ('3-NHAc-thiophen-2-yl', '[*]c1ccsc1NC(C)=O', 'aromatic'),
    ('3-morpholino-thiophen-2-yl', '[*]c1ccsc1N1CCOCC1', 'aromatic'),
    ('3-piperidino-thiophen-2-yl', '[*]c1ccsc1N1CCCCC1', 'aromatic'),
    ('3-pyrrolidino-thiophen-2-yl', '[*]c1ccsc1N1CCCC1', 'aromatic'),
    ('3-OCF3-thiophen-2-yl', '[*]c1ccsc1OC(F)(F)F', 'aromatic'),
    ('3-SMe-thiophen-2-yl', '[*]c1ccsc1SC', 'aromatic'),
    ('3-NHSO2Me-thiophen-2-yl', '[*]c1ccsc1NS(C)(=O)=O', 'aromatic'),
    ('4-F-thiophen-2-yl', '[*]c1cc(F)cs1', 'aromatic'),
    ('4-Cl-thiophen-2-yl', '[*]c1cc(Cl)cs1', 'aromatic'),
    ('4-Br-thiophen-2-yl', '[*]c1cc(Br)cs1', 'aromatic'),
    ('4-Et-thiophen-2-yl', '[*]c1cc(CC)cs1', 'aromatic'),
    ('4-iPr-thiophen-2-yl', '[*]c1cc(C(C)C)cs1', 'aromatic'),
    ('4-OMe-thiophen-2-yl', '[*]c1cc(OC)cs1', 'aromatic'),
    ('4-OEt-thiophen-2-yl', '[*]c1cc(OCC)cs1', 'aromatic'),
    ('4-OH-thiophen-2-yl', '[*]c1cc(O)cs1', 'aromatic'),
    ('4-CF3-thiophen-2-yl', '[*]c1cc(C(F)(F)F)cs1', 'aromatic'),
    ('4-CHF2-thiophen-2-yl', '[*]c1cc(C(F)F)cs1', 'aromatic'),
    ('4-CN-thiophen-2-yl', '[*]c1cc(C#N)cs1', 'aromatic'),
    ('4-NH2-thiophen-2-yl', '[*]c1cc(N)cs1', 'aromatic'),
    ('4-NHMe-thiophen-2-yl', '[*]c1cc(NC)cs1', 'aromatic'),
    ('4-NMe2-thiophen-2-yl', '[*]c1cc(N(C)C)cs1', 'aromatic'),
    ('4-COOH-thiophen-2-yl', '[*]c1cc(C(=O)O)cs1', 'aromatic'),
    ('4-COOMe-thiophen-2-yl', '[*]c1cc(C(=O)OC)cs1', 'aromatic'),
    ('4-COMe-thiophen-2-yl', '[*]c1cc(C(C)=O)cs1', 'aromatic'),
    ('4-SO2Me-thiophen-2-yl', '[*]c1cc(S(C)(=O)=O)cs1', 'aromatic'),
    ('4-cyclopropyl-thiophen-2-yl', '[*]c1cc(C1CC1)cs1', 'aromatic'),
    ('4-NHAc-thiophen-2-yl', '[*]c1cc(NC(C)=O)cs1', 'aromatic'),
    ('4-morpholino-thiophen-2-yl', '[*]c1cc(N1CCOCC1)cs1', 'aromatic'),
    ('4-piperidino-thiophen-2-yl', '[*]c1cc(N1CCCCC1)cs1', 'aromatic'),
    ('4-pyrrolidino-thiophen-2-yl', '[*]c1cc(N1CCCC1)cs1', 'aromatic'),
    ('4-OCF3-thiophen-2-yl', '[*]c1cc(OC(F)(F)F)cs1', 'aromatic'),
    ('4-SMe-thiophen-2-yl', '[*]c1cc(SC)cs1', 'aromatic'),
    ('4-NHSO2Me-thiophen-2-yl', '[*]c1cc(NS(C)(=O)=O)cs1', 'aromatic'),
    ('5-Cl-thiophen-2-yl', '[*]c1ccc(Cl)s1', 'aromatic'),
    ('5-Br-thiophen-2-yl', '[*]c1ccc(Br)s1', 'aromatic'),
    ('5-Et-thiophen-2-yl', '[*]c1ccc(CC)s1', 'aromatic'),
    ('5-iPr-thiophen-2-yl', '[*]c1ccc(C(C)C)s1', 'aromatic'),
    ('5-OMe-thiophen-2-yl', '[*]c1ccc(OC)s1', 'aromatic'),
    ('5-OEt-thiophen-2-yl', '[*]c1ccc(OCC)s1', 'aromatic'),
    ('5-OH-thiophen-2-yl', '[*]c1ccc(O)s1', 'aromatic'),
    ('5-CF3-thiophen-2-yl', '[*]c1ccc(C(F)(F)F)s1', 'aromatic'),
    ('5-CHF2-thiophen-2-yl', '[*]c1ccc(C(F)F)s1', 'aromatic'),
    ('5-CN-thiophen-2-yl', '[*]c1ccc(C#N)s1', 'aromatic'),
    ('5-NH2-thiophen-2-yl', '[*]c1ccc(N)s1', 'aromatic'),
    ('5-NHMe-thiophen-2-yl', '[*]c1ccc(NC)s1', 'aromatic'),
    ('5-NMe2-thiophen-2-yl', '[*]c1ccc(N(C)C)s1', 'aromatic'),
    ('5-COOH-thiophen-2-yl', '[*]c1ccc(C(=O)O)s1', 'aromatic'),
    ('5-COOMe-thiophen-2-yl', '[*]c1ccc(C(=O)OC)s1', 'aromatic'),
    ('5-COMe-thiophen-2-yl', '[*]c1ccc(C(C)=O)s1', 'aromatic'),
    ('5-SO2Me-thiophen-2-yl', '[*]c1ccc(S(C)(=O)=O)s1', 'aromatic'),
    ('5-cyclopropyl-thiophen-2-yl', '[*]c1ccc(C1CC1)s1', 'aromatic'),
    ('5-NHAc-thiophen-2-yl', '[*]c1ccc(NC(C)=O)s1', 'aromatic'),
    ('5-morpholino-thiophen-2-yl', '[*]c1ccc(N1CCOCC1)s1', 'aromatic'),
    ('5-piperidino-thiophen-2-yl', '[*]c1ccc(N1CCCCC1)s1', 'aromatic'),
    ('5-pyrrolidino-thiophen-2-yl', '[*]c1ccc(N1CCCC1)s1', 'aromatic'),
    ('5-OCF3-thiophen-2-yl', '[*]c1ccc(OC(F)(F)F)s1', 'aromatic'),
    ('5-SMe-thiophen-2-yl', '[*]c1ccc(SC)s1', 'aromatic'),
    ('5-NHSO2Me-thiophen-2-yl', '[*]c1ccc(NS(C)(=O)=O)s1', 'aromatic'),
    ('2-F-thiophen-3-yl', '[*]c1cc(F)sc1', 'aromatic'),
    ('2-Cl-thiophen-3-yl', '[*]c1cc(Cl)sc1', 'aromatic'),
    ('2-Br-thiophen-3-yl', '[*]c1cc(Br)sc1', 'aromatic'),
    ('2-Me-thiophen-3-yl', '[*]c1cc(C)sc1', 'aromatic'),
    ('2-Et-thiophen-3-yl', '[*]c1cc(CC)sc1', 'aromatic'),
    ('2-iPr-thiophen-3-yl', '[*]c1cc(C(C)C)sc1', 'aromatic'),
    ('2-OMe-thiophen-3-yl', '[*]c1cc(OC)sc1', 'aromatic'),
    ('2-OEt-thiophen-3-yl', '[*]c1cc(OCC)sc1', 'aromatic'),
    ('2-OH-thiophen-3-yl', '[*]c1cc(O)sc1', 'aromatic'),
    ('2-CF3-thiophen-3-yl', '[*]c1cc(C(F)(F)F)sc1', 'aromatic'),
    ('2-CHF2-thiophen-3-yl', '[*]c1cc(C(F)F)sc1', 'aromatic'),
    ('2-CN-thiophen-3-yl', '[*]c1cc(C#N)sc1', 'aromatic'),
    ('2-NH2-thiophen-3-yl', '[*]c1cc(N)sc1', 'aromatic'),
    ('2-NHMe-thiophen-3-yl', '[*]c1cc(NC)sc1', 'aromatic'),
    ('2-NMe2-thiophen-3-yl', '[*]c1cc(N(C)C)sc1', 'aromatic'),
    ('2-COOH-thiophen-3-yl', '[*]c1cc(C(=O)O)sc1', 'aromatic'),
    ('2-COOMe-thiophen-3-yl', '[*]c1cc(C(=O)OC)sc1', 'aromatic'),
    ('2-COMe-thiophen-3-yl', '[*]c1cc(C(C)=O)sc1', 'aromatic'),
    ('2-SO2Me-thiophen-3-yl', '[*]c1cc(S(C)(=O)=O)sc1', 'aromatic'),
    ('2-NHAc-thiophen-3-yl', '[*]c1cc(NC(C)=O)sc1', 'aromatic'),
    ('2-OCF3-thiophen-3-yl', '[*]c1cc(OC(F)(F)F)sc1', 'aromatic'),
    ('2-SMe-thiophen-3-yl', '[*]c1cc(SC)sc1', 'aromatic'),
    ('2-NHSO2Me-thiophen-3-yl', '[*]c1cc(NS(C)(=O)=O)sc1', 'aromatic'),
    ('3-F-furan-2-yl', '[*]c1ccoc1F', 'aromatic'),
    ('3-Cl-furan-2-yl', '[*]c1ccoc1Cl', 'aromatic'),
    ('3-Br-furan-2-yl', '[*]c1ccoc1Br', 'aromatic'),
    ('3-Et-furan-2-yl', '[*]c1ccoc1CC', 'aromatic'),
    ('3-iPr-furan-2-yl', '[*]c1ccoc1C(C)C', 'aromatic'),
    ('3-OMe-furan-2-yl', '[*]c1ccoc1OC', 'aromatic'),
    ('3-OEt-furan-2-yl', '[*]c1ccoc1OCC', 'aromatic'),
    ('3-OH-furan-2-yl', '[*]c1ccoc1O', 'aromatic'),
    ('3-CF3-furan-2-yl', '[*]c1ccoc1C(F)(F)F', 'aromatic'),
    ('3-CHF2-furan-2-yl', '[*]c1ccoc1C(F)F', 'aromatic'),
    ('3-CN-furan-2-yl', '[*]c1ccoc1C#N', 'aromatic'),
    ('3-NH2-furan-2-yl', '[*]c1ccoc1N', 'aromatic'),
    ('3-NHMe-furan-2-yl', '[*]c1ccoc1NC', 'aromatic'),
    ('3-NMe2-furan-2-yl', '[*]c1ccoc1N(C)C', 'aromatic'),
    ('3-COOH-furan-2-yl', '[*]c1ccoc1C(=O)O', 'aromatic'),
    ('3-COOMe-furan-2-yl', '[*]c1ccoc1C(=O)OC', 'aromatic'),
    ('3-COMe-furan-2-yl', '[*]c1ccoc1C(C)=O', 'aromatic'),
    ('3-SO2Me-furan-2-yl', '[*]c1ccoc1S(C)(=O)=O', 'aromatic'),
    ('3-cyclopropyl-furan-2-yl', '[*]c1ccoc1C1CC1', 'aromatic'),
    ('3-NHAc-furan-2-yl', '[*]c1ccoc1NC(C)=O', 'aromatic'),
    ('3-morpholino-furan-2-yl', '[*]c1ccoc1N1CCOCC1', 'aromatic'),
    ('3-piperidino-furan-2-yl', '[*]c1ccoc1N1CCCCC1', 'aromatic'),
    ('3-pyrrolidino-furan-2-yl', '[*]c1ccoc1N1CCCC1', 'aromatic'),
    ('3-OCF3-furan-2-yl', '[*]c1ccoc1OC(F)(F)F', 'aromatic'),
    ('3-SMe-furan-2-yl', '[*]c1ccoc1SC', 'aromatic'),
    ('3-NHSO2Me-furan-2-yl', '[*]c1ccoc1NS(C)(=O)=O', 'aromatic'),
    ('5-F-furan-2-yl', '[*]c1ccc(F)o1', 'aromatic'),
    ('5-Cl-furan-2-yl', '[*]c1ccc(Cl)o1', 'aromatic'),
    ('5-Br-furan-2-yl', '[*]c1ccc(Br)o1', 'aromatic'),
    ('5-Et-furan-2-yl', '[*]c1ccc(CC)o1', 'aromatic'),
    ('5-iPr-furan-2-yl', '[*]c1ccc(C(C)C)o1', 'aromatic'),
    ('5-OMe-furan-2-yl', '[*]c1ccc(OC)o1', 'aromatic'),
    ('5-OEt-furan-2-yl', '[*]c1ccc(OCC)o1', 'aromatic'),
    ('5-OH-furan-2-yl', '[*]c1ccc(O)o1', 'aromatic'),
    ('5-CF3-furan-2-yl', '[*]c1ccc(C(F)(F)F)o1', 'aromatic'),
    ('5-CHF2-furan-2-yl', '[*]c1ccc(C(F)F)o1', 'aromatic'),
    ('5-CN-furan-2-yl', '[*]c1ccc(C#N)o1', 'aromatic'),
    ('5-NH2-furan-2-yl', '[*]c1ccc(N)o1', 'aromatic'),
    ('5-NHMe-furan-2-yl', '[*]c1ccc(NC)o1', 'aromatic'),
    ('5-NMe2-furan-2-yl', '[*]c1ccc(N(C)C)o1', 'aromatic'),
    ('5-COOH-furan-2-yl', '[*]c1ccc(C(=O)O)o1', 'aromatic'),
    ('5-COOMe-furan-2-yl', '[*]c1ccc(C(=O)OC)o1', 'aromatic'),
    ('5-COMe-furan-2-yl', '[*]c1ccc(C(C)=O)o1', 'aromatic'),
    ('5-SO2Me-furan-2-yl', '[*]c1ccc(S(C)(=O)=O)o1', 'aromatic'),
    ('5-cyclopropyl-furan-2-yl', '[*]c1ccc(C1CC1)o1', 'aromatic'),
    ('5-NHAc-furan-2-yl', '[*]c1ccc(NC(C)=O)o1', 'aromatic'),
    ('5-morpholino-furan-2-yl', '[*]c1ccc(N1CCOCC1)o1', 'aromatic'),
    ('5-piperidino-furan-2-yl', '[*]c1ccc(N1CCCCC1)o1', 'aromatic'),
    ('5-pyrrolidino-furan-2-yl', '[*]c1ccc(N1CCCC1)o1', 'aromatic'),
    ('5-OCF3-furan-2-yl', '[*]c1ccc(OC(F)(F)F)o1', 'aromatic'),
    ('5-SMe-furan-2-yl', '[*]c1ccc(SC)o1', 'aromatic'),
    ('5-NHSO2Me-furan-2-yl', '[*]c1ccc(NS(C)(=O)=O)o1', 'aromatic'),
    ('3-F-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1F', 'aromatic'),
    ('3-Cl-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1Cl', 'aromatic'),
    ('3-Br-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1Br', 'aromatic'),
    ('3-Me-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1C', 'aromatic'),
    ('3-Et-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1CC', 'aromatic'),
    ('3-iPr-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1C(C)C', 'aromatic'),
    ('3-OMe-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1OC', 'aromatic'),
    ('3-OEt-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1OCC', 'aromatic'),
    ('3-OH-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1O', 'aromatic'),
    ('3-CF3-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1C(F)(F)F', 'aromatic'),
    ('3-CHF2-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1C(F)F', 'aromatic'),
    ('3-CN-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1C#N', 'aromatic'),
    ('3-NH2-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1N', 'aromatic'),
    ('3-NHMe-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1NC', 'aromatic'),
    ('3-NMe2-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1N(C)C', 'aromatic'),
    ('3-COOH-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1C(=O)O', 'aromatic'),
    ('3-COOMe-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1C(=O)OC', 'aromatic'),
    ('3-COMe-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1C(C)=O', 'aromatic'),
    ('3-SO2Me-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1S(C)(=O)=O', 'aromatic'),
    ('3-cyclopropyl-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1C1CC1', 'aromatic'),
    ('3-NHAc-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1NC(C)=O', 'aromatic'),
    ('3-morpholino-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1N1CCOCC1', 'aromatic'),
    ('3-piperidino-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1N1CCCCC1', 'aromatic'),
    ('3-pyrrolidino-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1N1CCCC1', 'aromatic'),
    ('3-OCF3-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1OC(F)(F)F', 'aromatic'),
    ('3-SMe-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1SC', 'aromatic'),
    ('3-NHSO2Me-1-Me-pyrazol-5-yl', '[*]c1cnn(C)c1NS(C)(=O)=O', 'aromatic'),
    ('4-F-1-Me-pyrazol-3-yl', '[*]c1c(F)cn(C)n1', 'aromatic'),
    ('4-Cl-1-Me-pyrazol-3-yl', '[*]c1c(Cl)cn(C)n1', 'aromatic'),
    ('4-Br-1-Me-pyrazol-3-yl', '[*]c1c(Br)cn(C)n1', 'aromatic'),
    ('4-Me-1-Me-pyrazol-3-yl', '[*]c1c(C)cn(C)n1', 'aromatic'),
    ('4-Et-1-Me-pyrazol-3-yl', '[*]c1c(CC)cn(C)n1', 'aromatic'),
    ('4-iPr-1-Me-pyrazol-3-yl', '[*]c1c(C(C)C)cn(C)n1', 'aromatic'),
    ('4-OMe-1-Me-pyrazol-3-yl', '[*]c1c(OC)cn(C)n1', 'aromatic'),
    ('4-OEt-1-Me-pyrazol-3-yl', '[*]c1c(OCC)cn(C)n1', 'aromatic'),
    ('4-OH-1-Me-pyrazol-3-yl', '[*]c1c(O)cn(C)n1', 'aromatic'),
    ('4-CF3-1-Me-pyrazol-3-yl', '[*]c1c(C(F)(F)F)cn(C)n1', 'aromatic'),
    ('4-CHF2-1-Me-pyrazol-3-yl', '[*]c1c(C(F)F)cn(C)n1', 'aromatic'),
    ('4-CN-1-Me-pyrazol-3-yl', '[*]c1c(C#N)cn(C)n1', 'aromatic'),
    ('4-NH2-1-Me-pyrazol-3-yl', '[*]c1c(N)cn(C)n1', 'aromatic'),
    ('4-NHMe-1-Me-pyrazol-3-yl', '[*]c1c(NC)cn(C)n1', 'aromatic'),
    ('4-NMe2-1-Me-pyrazol-3-yl', '[*]c1c(N(C)C)cn(C)n1', 'aromatic'),
    ('4-COOH-1-Me-pyrazol-3-yl', '[*]c1c(C(=O)O)cn(C)n1', 'aromatic'),
    ('4-COOMe-1-Me-pyrazol-3-yl', '[*]c1c(C(=O)OC)cn(C)n1', 'aromatic'),
    ('4-COMe-1-Me-pyrazol-3-yl', '[*]c1c(C(C)=O)cn(C)n1', 'aromatic'),
    ('4-SO2Me-1-Me-pyrazol-3-yl', '[*]c1c(S(C)(=O)=O)cn(C)n1', 'aromatic'),
    ('4-NHAc-1-Me-pyrazol-3-yl', '[*]c1c(NC(C)=O)cn(C)n1', 'aromatic'),
    ('4-OCF3-1-Me-pyrazol-3-yl', '[*]c1c(OC(F)(F)F)cn(C)n1', 'aromatic'),
    ('4-SMe-1-Me-pyrazol-3-yl', '[*]c1c(SC)cn(C)n1', 'aromatic'),
    ('4-NHSO2Me-1-Me-pyrazol-3-yl', '[*]c1c(NS(C)(=O)=O)cn(C)n1', 'aromatic'),
    ('4-F-1-Me-imidazol-2-yl', '[*]c1nc(F)cn1C', 'aromatic'),
    ('4-Cl-1-Me-imidazol-2-yl', '[*]c1nc(Cl)cn1C', 'aromatic'),
    ('4-Br-1-Me-imidazol-2-yl', '[*]c1nc(Br)cn1C', 'aromatic'),
    ('4-Me-1-Me-imidazol-2-yl', '[*]c1nc(C)cn1C', 'aromatic'),
    ('4-Et-1-Me-imidazol-2-yl', '[*]c1nc(CC)cn1C', 'aromatic'),
    ('4-iPr-1-Me-imidazol-2-yl', '[*]c1nc(C(C)C)cn1C', 'aromatic'),
    ('4-OMe-1-Me-imidazol-2-yl', '[*]c1nc(OC)cn1C', 'aromatic'),
    ('4-OEt-1-Me-imidazol-2-yl', '[*]c1nc(OCC)cn1C', 'aromatic'),
    ('4-OH-1-Me-imidazol-2-yl', '[*]c1nc(O)cn1C', 'aromatic'),
    ('4-CF3-1-Me-imidazol-2-yl', '[*]c1nc(C(F)(F)F)cn1C', 'aromatic'),
    ('4-CHF2-1-Me-imidazol-2-yl', '[*]c1nc(C(F)F)cn1C', 'aromatic'),
    ('4-CN-1-Me-imidazol-2-yl', '[*]c1nc(C#N)cn1C', 'aromatic'),
    ('4-NH2-1-Me-imidazol-2-yl', '[*]c1nc(N)cn1C', 'aromatic'),
    ('4-NHMe-1-Me-imidazol-2-yl', '[*]c1nc(NC)cn1C', 'aromatic'),
    ('4-NMe2-1-Me-imidazol-2-yl', '[*]c1nc(N(C)C)cn1C', 'aromatic'),
    ('4-COOH-1-Me-imidazol-2-yl', '[*]c1nc(C(=O)O)cn1C', 'aromatic'),
    ('4-COOMe-1-Me-imidazol-2-yl', '[*]c1nc(C(=O)OC)cn1C', 'aromatic'),
    ('4-COMe-1-Me-imidazol-2-yl', '[*]c1nc(C(C)=O)cn1C', 'aromatic'),
    ('4-SO2Me-1-Me-imidazol-2-yl', '[*]c1nc(S(C)(=O)=O)cn1C', 'aromatic'),
    ('4-cyclopropyl-1-Me-imidazol-2-yl', '[*]c1nc(C1CC1)cn1C', 'aromatic'),
    ('4-NHAc-1-Me-imidazol-2-yl', '[*]c1nc(NC(C)=O)cn1C', 'aromatic'),
    ('4-morpholino-1-Me-imidazol-2-yl', '[*]c1nc(N1CCOCC1)cn1C', 'aromatic'),
    ('4-piperidino-1-Me-imidazol-2-yl', '[*]c1nc(N1CCCCC1)cn1C', 'aromatic'),
    ('4-pyrrolidino-1-Me-imidazol-2-yl', '[*]c1nc(N1CCCC1)cn1C', 'aromatic'),
    ('4-OCF3-1-Me-imidazol-2-yl', '[*]c1nc(OC(F)(F)F)cn1C', 'aromatic'),
    ('4-SMe-1-Me-imidazol-2-yl', '[*]c1nc(SC)cn1C', 'aromatic'),
    ('4-NHSO2Me-1-Me-imidazol-2-yl', '[*]c1nc(NS(C)(=O)=O)cn1C', 'aromatic'),
    ('4-F-oxazol-2-yl', '[*]c1nc(F)co1', 'aromatic'),
    ('4-Cl-oxazol-2-yl', '[*]c1nc(Cl)co1', 'aromatic'),
    ('4-Br-oxazol-2-yl', '[*]c1nc(Br)co1', 'aromatic'),
    ('4-Et-oxazol-2-yl', '[*]c1nc(CC)co1', 'aromatic'),
    ('4-iPr-oxazol-2-yl', '[*]c1nc(C(C)C)co1', 'aromatic'),
    ('4-OMe-oxazol-2-yl', '[*]c1nc(OC)co1', 'aromatic'),
    ('4-OEt-oxazol-2-yl', '[*]c1nc(OCC)co1', 'aromatic'),
    ('4-OH-oxazol-2-yl', '[*]c1nc(O)co1', 'aromatic'),
    ('4-CF3-oxazol-2-yl', '[*]c1nc(C(F)(F)F)co1', 'aromatic'),
    ('4-CHF2-oxazol-2-yl', '[*]c1nc(C(F)F)co1', 'aromatic'),
    ('4-CN-oxazol-2-yl', '[*]c1nc(C#N)co1', 'aromatic'),
    ('4-NH2-oxazol-2-yl', '[*]c1nc(N)co1', 'aromatic'),
    ('4-NHMe-oxazol-2-yl', '[*]c1nc(NC)co1', 'aromatic'),
    ('4-NMe2-oxazol-2-yl', '[*]c1nc(N(C)C)co1', 'aromatic'),
    ('4-COOH-oxazol-2-yl', '[*]c1nc(C(=O)O)co1', 'aromatic'),
    ('4-COOMe-oxazol-2-yl', '[*]c1nc(C(=O)OC)co1', 'aromatic'),
    ('4-COMe-oxazol-2-yl', '[*]c1nc(C(C)=O)co1', 'aromatic'),
    ('4-SO2Me-oxazol-2-yl', '[*]c1nc(S(C)(=O)=O)co1', 'aromatic'),
    ('4-cyclopropyl-oxazol-2-yl', '[*]c1nc(C1CC1)co1', 'aromatic'),
    ('4-NHAc-oxazol-2-yl', '[*]c1nc(NC(C)=O)co1', 'aromatic'),
    ('4-morpholino-oxazol-2-yl', '[*]c1nc(N1CCOCC1)co1', 'aromatic'),
    ('4-piperidino-oxazol-2-yl', '[*]c1nc(N1CCCCC1)co1', 'aromatic'),
    ('4-pyrrolidino-oxazol-2-yl', '[*]c1nc(N1CCCC1)co1', 'aromatic'),
    ('4-OCF3-oxazol-2-yl', '[*]c1nc(OC(F)(F)F)co1', 'aromatic'),
    ('4-SMe-oxazol-2-yl', '[*]c1nc(SC)co1', 'aromatic'),
    ('4-NHSO2Me-oxazol-2-yl', '[*]c1nc(NS(C)(=O)=O)co1', 'aromatic'),
    ('5-F-oxazol-2-yl', '[*]c1ncc(F)o1', 'aromatic'),
    ('5-Cl-oxazol-2-yl', '[*]c1ncc(Cl)o1', 'aromatic'),
    ('5-Br-oxazol-2-yl', '[*]c1ncc(Br)o1', 'aromatic'),
    ('5-Et-oxazol-2-yl', '[*]c1ncc(CC)o1', 'aromatic'),
    ('5-iPr-oxazol-2-yl', '[*]c1ncc(C(C)C)o1', 'aromatic'),
    ('5-OMe-oxazol-2-yl', '[*]c1ncc(OC)o1', 'aromatic'),
    ('5-OEt-oxazol-2-yl', '[*]c1ncc(OCC)o1', 'aromatic'),
    ('5-OH-oxazol-2-yl', '[*]c1ncc(O)o1', 'aromatic'),
    ('5-CF3-oxazol-2-yl', '[*]c1ncc(C(F)(F)F)o1', 'aromatic'),
    ('5-CHF2-oxazol-2-yl', '[*]c1ncc(C(F)F)o1', 'aromatic'),
    ('5-CN-oxazol-2-yl', '[*]c1ncc(C#N)o1', 'aromatic'),
    ('5-NH2-oxazol-2-yl', '[*]c1ncc(N)o1', 'aromatic'),
    ('5-NHMe-oxazol-2-yl', '[*]c1ncc(NC)o1', 'aromatic'),
    ('5-NMe2-oxazol-2-yl', '[*]c1ncc(N(C)C)o1', 'aromatic'),
    ('5-COOH-oxazol-2-yl', '[*]c1ncc(C(=O)O)o1', 'aromatic'),
    ('5-COOMe-oxazol-2-yl', '[*]c1ncc(C(=O)OC)o1', 'aromatic'),
    ('5-COMe-oxazol-2-yl', '[*]c1ncc(C(C)=O)o1', 'aromatic'),
    ('5-SO2Me-oxazol-2-yl', '[*]c1ncc(S(C)(=O)=O)o1', 'aromatic'),
    ('5-cyclopropyl-oxazol-2-yl', '[*]c1ncc(C1CC1)o1', 'aromatic'),
    ('5-NHAc-oxazol-2-yl', '[*]c1ncc(NC(C)=O)o1', 'aromatic'),
    ('5-morpholino-oxazol-2-yl', '[*]c1ncc(N1CCOCC1)o1', 'aromatic'),
    ('5-piperidino-oxazol-2-yl', '[*]c1ncc(N1CCCCC1)o1', 'aromatic'),
    ('5-pyrrolidino-oxazol-2-yl', '[*]c1ncc(N1CCCC1)o1', 'aromatic'),
    ('5-OCF3-oxazol-2-yl', '[*]c1ncc(OC(F)(F)F)o1', 'aromatic'),
    ('5-SMe-oxazol-2-yl', '[*]c1ncc(SC)o1', 'aromatic'),
    ('5-NHSO2Me-oxazol-2-yl', '[*]c1ncc(NS(C)(=O)=O)o1', 'aromatic'),
    ('4-F-thiazol-2-yl', '[*]c1nc(F)cs1', 'aromatic'),
    ('4-Cl-thiazol-2-yl', '[*]c1nc(Cl)cs1', 'aromatic'),
    ('4-Br-thiazol-2-yl', '[*]c1nc(Br)cs1', 'aromatic'),
    ('4-Et-thiazol-2-yl', '[*]c1nc(CC)cs1', 'aromatic'),
    ('4-iPr-thiazol-2-yl', '[*]c1nc(C(C)C)cs1', 'aromatic'),
    ('4-OMe-thiazol-2-yl', '[*]c1nc(OC)cs1', 'aromatic'),
    ('4-OEt-thiazol-2-yl', '[*]c1nc(OCC)cs1', 'aromatic'),
    ('4-OH-thiazol-2-yl', '[*]c1nc(O)cs1', 'aromatic'),
    ('4-CHF2-thiazol-2-yl', '[*]c1nc(C(F)F)cs1', 'aromatic'),
    ('4-CN-thiazol-2-yl', '[*]c1nc(C#N)cs1', 'aromatic'),
    ('4-NH2-thiazol-2-yl', '[*]c1nc(N)cs1', 'aromatic'),
    ('4-NHMe-thiazol-2-yl', '[*]c1nc(NC)cs1', 'aromatic'),
    ('4-NMe2-thiazol-2-yl', '[*]c1nc(N(C)C)cs1', 'aromatic'),
    ('4-COOH-thiazol-2-yl', '[*]c1nc(C(=O)O)cs1', 'aromatic'),
    ('4-COOMe-thiazol-2-yl', '[*]c1nc(C(=O)OC)cs1', 'aromatic'),
    ('4-COMe-thiazol-2-yl', '[*]c1nc(C(C)=O)cs1', 'aromatic'),
    ('4-SO2Me-thiazol-2-yl', '[*]c1nc(S(C)(=O)=O)cs1', 'aromatic'),
    ('4-cyclopropyl-thiazol-2-yl', '[*]c1nc(C1CC1)cs1', 'aromatic'),
    ('4-NHAc-thiazol-2-yl', '[*]c1nc(NC(C)=O)cs1', 'aromatic'),
    ('4-morpholino-thiazol-2-yl', '[*]c1nc(N1CCOCC1)cs1', 'aromatic'),
    ('4-piperidino-thiazol-2-yl', '[*]c1nc(N1CCCCC1)cs1', 'aromatic'),
    ('4-pyrrolidino-thiazol-2-yl', '[*]c1nc(N1CCCC1)cs1', 'aromatic'),
    ('4-OCF3-thiazol-2-yl', '[*]c1nc(OC(F)(F)F)cs1', 'aromatic'),
    ('4-SMe-thiazol-2-yl', '[*]c1nc(SC)cs1', 'aromatic'),
    ('4-NHSO2Me-thiazol-2-yl', '[*]c1nc(NS(C)(=O)=O)cs1', 'aromatic'),
    ('5-F-thiazol-2-yl', '[*]c1sc(F)nc1', 'aromatic'),
    ('5-Cl-thiazol-2-yl', '[*]c1sc(Cl)nc1', 'aromatic'),
    ('5-Br-thiazol-2-yl', '[*]c1sc(Br)nc1', 'aromatic'),
    ('5-Et-thiazol-2-yl', '[*]c1sc(CC)nc1', 'aromatic'),
    ('5-iPr-thiazol-2-yl', '[*]c1sc(C(C)C)nc1', 'aromatic'),
    ('5-OMe-thiazol-2-yl', '[*]c1sc(OC)nc1', 'aromatic'),
    ('5-OEt-thiazol-2-yl', '[*]c1sc(OCC)nc1', 'aromatic'),
    ('5-OH-thiazol-2-yl', '[*]c1sc(O)nc1', 'aromatic'),
    ('5-CF3-thiazol-2-yl', '[*]c1sc(C(F)(F)F)nc1', 'aromatic'),
    ('5-CHF2-thiazol-2-yl', '[*]c1sc(C(F)F)nc1', 'aromatic'),
    ('5-CN-thiazol-2-yl', '[*]c1sc(C#N)nc1', 'aromatic'),
    ('5-NH2-thiazol-2-yl', '[*]c1sc(N)nc1', 'aromatic'),
    ('5-NHMe-thiazol-2-yl', '[*]c1sc(NC)nc1', 'aromatic'),
    ('5-NMe2-thiazol-2-yl', '[*]c1sc(N(C)C)nc1', 'aromatic'),
    ('5-COOH-thiazol-2-yl', '[*]c1sc(C(=O)O)nc1', 'aromatic'),
    ('5-COOMe-thiazol-2-yl', '[*]c1sc(C(=O)OC)nc1', 'aromatic'),
    ('5-COMe-thiazol-2-yl', '[*]c1sc(C(C)=O)nc1', 'aromatic'),
    ('5-SO2Me-thiazol-2-yl', '[*]c1sc(S(C)(=O)=O)nc1', 'aromatic'),
    ('5-NHAc-thiazol-2-yl', '[*]c1sc(NC(C)=O)nc1', 'aromatic'),
    ('5-OCF3-thiazol-2-yl', '[*]c1sc(OC(F)(F)F)nc1', 'aromatic'),
    ('5-SMe-thiazol-2-yl', '[*]c1sc(SC)nc1', 'aromatic'),
    ('5-NHSO2Me-thiazol-2-yl', '[*]c1sc(NS(C)(=O)=O)nc1', 'aromatic'),
    ('4-OH-piperidinyl', '[*]C1CCNCC1O', 'basic'),
    ('4-OMe-piperidinyl', '[*]C1CCNCC1OC', 'basic'),
    ('4-F-piperidinyl', '[*]C1CCNCC1F', 'basic'),
    ('4-Me-piperidinyl', '[*]C1CCNCC1C', 'basic'),
    ('4-Et-piperidinyl', '[*]C1CCNCC1CC', 'basic'),
    ('4-NH2-piperidinyl', '[*]C1CCNCC1N', 'basic'),
    ('4-NHMe-piperidinyl', '[*]C1CCNCC1NC', 'basic'),
    ('4-NMe2-piperidinyl', '[*]C1CCNCC1N(C)C', 'basic'),
    ('4-COOH-piperidinyl', '[*]C1CCNCC1C(=O)O', 'basic'),
    ('4-COMe-piperidinyl', '[*]C1CCNCC1C(C)=O', 'basic'),
    ('4-CN-piperidinyl', '[*]C1CCNCC1C#N', 'basic'),
    ('4-CF3-piperidinyl', '[*]C1CCNCC1C(F)(F)F', 'basic'),
    ('4-Cl-piperidinyl', '[*]C1CCNCC1Cl', 'basic'),
    ('4-OH-1-Me-piperidinyl', '[*]C1CCN(C)CC1O', 'basic'),
    ('4-OMe-1-Me-piperidinyl', '[*]C1CCN(C)CC1OC', 'basic'),
    ('4-F-1-Me-piperidinyl', '[*]C1CCN(C)CC1F', 'basic'),
    ('4-Me-1-Me-piperidinyl', '[*]C1CCN(C)CC1C', 'basic'),
    ('4-Et-1-Me-piperidinyl', '[*]C1CCN(C)CC1CC', 'basic'),
    ('4-NH2-1-Me-piperidinyl', '[*]C1CCN(C)CC1N', 'basic'),
    ('4-NHMe-1-Me-piperidinyl', '[*]C1CCN(C)CC1NC', 'basic'),
    ('4-NMe2-1-Me-piperidinyl', '[*]C1CCN(C)CC1N(C)C', 'basic'),
    ('4-COOH-1-Me-piperidinyl', '[*]C1CCN(C)CC1C(=O)O', 'basic'),
    ('4-COMe-1-Me-piperidinyl', '[*]C1CCN(C)CC1C(C)=O', 'basic'),
    ('4-CN-1-Me-piperidinyl', '[*]C1CCN(C)CC1C#N', 'basic'),
    ('4-CF3-1-Me-piperidinyl', '[*]C1CCN(C)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-Me-piperidinyl', '[*]C1CCN(C)CC1Cl', 'basic'),
    ('4-OH-1-Et-piperidinyl', '[*]C1CCN(CC)CC1O', 'basic'),
    ('4-OMe-1-Et-piperidinyl', '[*]C1CCN(CC)CC1OC', 'basic'),
    ('4-F-1-Et-piperidinyl', '[*]C1CCN(CC)CC1F', 'basic'),
    ('4-Me-1-Et-piperidinyl', '[*]C1CCN(CC)CC1C', 'basic'),
    ('4-Et-1-Et-piperidinyl', '[*]C1CCN(CC)CC1CC', 'basic'),
    ('4-NH2-1-Et-piperidinyl', '[*]C1CCN(CC)CC1N', 'basic'),
    ('4-NHMe-1-Et-piperidinyl', '[*]C1CCN(CC)CC1NC', 'basic'),
    ('4-NMe2-1-Et-piperidinyl', '[*]C1CCN(CC)CC1N(C)C', 'basic'),
    ('4-COOH-1-Et-piperidinyl', '[*]C1CCN(CC)CC1C(=O)O', 'basic'),
    ('4-COMe-1-Et-piperidinyl', '[*]C1CCN(CC)CC1C(C)=O', 'basic'),
    ('4-CN-1-Et-piperidinyl', '[*]C1CCN(CC)CC1C#N', 'basic'),
    ('4-CF3-1-Et-piperidinyl', '[*]C1CCN(CC)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-Et-piperidinyl', '[*]C1CCN(CC)CC1Cl', 'basic'),
    ('4-OH-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1O', 'basic'),
    ('4-OMe-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1OC', 'basic'),
    ('4-F-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1F', 'basic'),
    ('4-Me-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1C', 'basic'),
    ('4-Et-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1CC', 'basic'),
    ('4-NH2-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1N', 'basic'),
    ('4-NHMe-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1NC', 'basic'),
    ('4-NMe2-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1N(C)C', 'basic'),
    ('4-COOH-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1C(=O)O', 'basic'),
    ('4-COMe-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1C(C)=O', 'basic'),
    ('4-CN-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1C#N', 'basic'),
    ('4-CF3-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-iPr-piperidinyl', '[*]C1CCN(C(C)C)CC1Cl', 'basic'),
    ('4-OH-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1O', 'basic'),
    ('4-OMe-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1OC', 'basic'),
    ('4-F-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1F', 'basic'),
    ('4-Me-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1C', 'basic'),
    ('4-Et-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1CC', 'basic'),
    ('4-NH2-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1N', 'basic'),
    ('4-NHMe-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1NC', 'basic'),
    ('4-NMe2-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1N(C)C', 'basic'),
    ('4-COOH-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1C(=O)O', 'basic'),
    ('4-COMe-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1C(C)=O', 'basic'),
    ('4-CN-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1C#N', 'basic'),
    ('4-CF3-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-cPr-piperidinyl', '[*]C1CCN(C1CC1)CC1Cl', 'basic'),
    ('4-OH-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1O', 'basic'),
    ('4-OMe-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1OC', 'basic'),
    ('4-F-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1F', 'basic'),
    ('4-Me-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1C', 'basic'),
    ('4-Et-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1CC', 'basic'),
    ('4-NH2-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1N', 'basic'),
    ('4-NHMe-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1NC', 'basic'),
    ('4-NMe2-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1N(C)C', 'basic'),
    ('4-COOH-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1C(=O)O', 'basic'),
    ('4-COMe-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1C(C)=O', 'basic'),
    ('4-CN-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1C#N', 'basic'),
    ('4-CF3-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-Bn-piperidinyl', '[*]C1CCN(Cc1ccccc1)CC1Cl', 'basic'),
    ('4-OH-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1O', 'basic'),
    ('4-OMe-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1OC', 'basic'),
    ('4-F-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1F', 'basic'),
    ('4-Me-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1C', 'basic'),
    ('4-Et-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1CC', 'basic'),
    ('4-NH2-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1N', 'basic'),
    ('4-NHMe-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1NC', 'basic'),
    ('4-NMe2-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1N(C)C', 'basic'),
    ('4-COOH-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1C(=O)O', 'basic'),
    ('4-COMe-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1C(C)=O', 'basic'),
    ('4-CN-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1C#N', 'basic'),
    ('4-Cl-1-4-FBn-piperidinyl', '[*]C1CCN(Cc1ccc(F)cc1)CC1Cl', 'basic'),
    ('4-OH-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1O', 'basic'),
    ('4-OMe-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1OC', 'basic'),
    ('4-F-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1F', 'basic'),
    ('4-Me-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1C', 'basic'),
    ('4-Et-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1CC', 'basic'),
    ('4-NH2-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1N', 'basic'),
    ('4-NHMe-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1NC', 'basic'),
    ('4-NMe2-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1N(C)C', 'basic'),
    ('4-COOH-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1C(=O)O', 'basic'),
    ('4-COMe-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1C(C)=O', 'basic'),
    ('4-CN-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1C#N', 'basic'),
    ('4-CF3-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-Ac-piperidinyl', '[*]C1CCN(C(C)=O)CC1Cl', 'basic'),
    ('4-OH-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1O', 'basic'),
    ('4-OMe-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1OC', 'basic'),
    ('4-F-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1F', 'basic'),
    ('4-Me-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1C', 'basic'),
    ('4-Et-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1CC', 'basic'),
    ('4-NH2-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1N', 'basic'),
    ('4-NHMe-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1NC', 'basic'),
    ('4-NMe2-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1N(C)C', 'basic'),
    ('4-COOH-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1C(=O)O', 'basic'),
    ('4-COMe-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1C(C)=O', 'basic'),
    ('4-CN-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1C#N', 'basic'),
    ('4-CF3-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-MeSO2-piperidinyl', '[*]C1CCN(S(C)(=O)=O)CC1Cl', 'basic'),
    ('4-OH-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1O', 'basic'),
    ('4-OMe-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1OC', 'basic'),
    ('4-F-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1F', 'basic'),
    ('4-Me-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1C', 'basic'),
    ('4-Et-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1CC', 'basic'),
    ('4-NH2-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1N', 'basic'),
    ('4-NHMe-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1NC', 'basic'),
    ('4-NMe2-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1N(C)C', 'basic'),
    ('4-COOH-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1C(=O)O', 'basic'),
    ('4-COMe-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1C(C)=O', 'basic'),
    ('4-CN-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1C#N', 'basic'),
    ('4-CF3-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-CH2CN-piperidinyl', '[*]C1CCN(CC#N)CC1Cl', 'basic'),
    ('4-OH-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1O', 'basic'),
    ('4-OMe-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1OC', 'basic'),
    ('4-F-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1F', 'basic'),
    ('4-Me-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1C', 'basic'),
    ('4-Et-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1CC', 'basic'),
    ('4-NH2-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1N', 'basic'),
    ('4-NHMe-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1NC', 'basic'),
    ('4-NMe2-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1N(C)C', 'basic'),
    ('4-COOH-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1C(=O)O', 'basic'),
    ('4-COMe-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1C(C)=O', 'basic'),
    ('4-CN-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1C#N', 'basic'),
    ('4-CF3-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-CH2OH-piperidinyl', '[*]C1CCN(CO)CC1Cl', 'basic'),
    ('4-OH-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1O', 'basic'),
    ('4-OMe-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1OC', 'basic'),
    ('4-F-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1F', 'basic'),
    ('4-Me-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1C', 'basic'),
    ('4-Et-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1CC', 'basic'),
    ('4-NH2-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1N', 'basic'),
    ('4-NHMe-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1NC', 'basic'),
    ('4-NMe2-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1N(C)C', 'basic'),
    ('4-COOH-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1C(=O)O', 'basic'),
    ('4-COMe-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1C(C)=O', 'basic'),
    ('4-CN-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1C#N', 'basic'),
    ('4-CF3-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-CH2OMe-piperidinyl', '[*]C1CCN(COC)CC1Cl', 'basic'),
    ('4-OH-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1O', 'basic'),
    ('4-OMe-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1OC', 'basic'),
    ('4-F-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1F', 'basic'),
    ('4-Me-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1C', 'basic'),
    ('4-Et-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1CC', 'basic'),
    ('4-NH2-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1N', 'basic'),
    ('4-NHMe-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1NC', 'basic'),
    ('4-NMe2-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1N(C)C', 'basic'),
    ('4-COOH-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1C(=O)O', 'basic'),
    ('4-COMe-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1C(C)=O', 'basic'),
    ('4-CN-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1C#N', 'basic'),
    ('4-CF3-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-allyl-piperidinyl', '[*]C1CCN(CC=C)CC1Cl', 'basic'),
    ('4-OH-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1O', 'basic'),
    ('4-OMe-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1OC', 'basic'),
    ('4-F-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1F', 'basic'),
    ('4-Me-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1C', 'basic'),
    ('4-Et-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1CC', 'basic'),
    ('4-NH2-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1N', 'basic'),
    ('4-NHMe-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1NC', 'basic'),
    ('4-NMe2-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1N(C)C', 'basic'),
    ('4-COOH-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1C(=O)O', 'basic'),
    ('4-COMe-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1C(C)=O', 'basic'),
    ('4-CN-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1C#N', 'basic'),
    ('4-CF3-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-prop-2-yn-1-yl-piperidinyl', '[*]C1CCN(CC#C)CC1Cl', 'basic'),
    ('4-OH-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1O', 'basic'),
    ('4-OMe-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1OC', 'basic'),
    ('4-F-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1F', 'basic'),
    ('4-Me-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1C', 'basic'),
    ('4-Et-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1CC', 'basic'),
    ('4-NH2-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1N', 'basic'),
    ('4-NHMe-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1NC', 'basic'),
    ('4-NMe2-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1N(C)C', 'basic'),
    ('4-COOH-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1C(=O)O', 'basic'),
    ('4-COMe-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1C(C)=O', 'basic'),
    ('4-CN-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1C#N', 'basic'),
    ('4-CF3-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-2-hydroxyethyl-piperidinyl', '[*]C1CCN(CCO)CC1Cl', 'basic'),
    ('4-OH-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1O', 'basic'),
    ('4-OMe-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1OC', 'basic'),
    ('4-F-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1F', 'basic'),
    ('4-Me-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1C', 'basic'),
    ('4-Et-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1CC', 'basic'),
    ('4-NH2-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1N', 'basic'),
    ('4-NHMe-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1NC', 'basic'),
    ('4-NMe2-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1N(C)C', 'basic'),
    ('4-COOH-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1C(=O)O', 'basic'),
    ('4-COMe-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1C(C)=O', 'basic'),
    ('4-CN-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1C#N', 'basic'),
    ('4-CF3-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-3-hydroxypropyl-piperidinyl', '[*]C1CCN(CCCO)CC1Cl', 'basic'),
    ('4-OH-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1O', 'basic'),
    ('4-OMe-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1OC', 'basic'),
    ('4-F-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1F', 'basic'),
    ('4-Me-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1C', 'basic'),
    ('4-Et-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1CC', 'basic'),
    ('4-NH2-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1N', 'basic'),
    ('4-NHMe-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1NC', 'basic'),
    ('4-NMe2-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1N(C)C', 'basic'),
    ('4-COOH-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1C(=O)O', 'basic'),
    ('4-COMe-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1C(C)=O', 'basic'),
    ('4-CN-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1C#N', 'basic'),
    ('4-CF3-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-cyclobutyl-piperidinyl', '[*]C1CCN(C1CCC1)CC1Cl', 'basic'),
    ('4-OH-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1O', 'basic'),
    ('4-OMe-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1OC', 'basic'),
    ('4-F-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1F', 'basic'),
    ('4-Me-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1C', 'basic'),
    ('4-Et-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1CC', 'basic'),
    ('4-NH2-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1N', 'basic'),
    ('4-NHMe-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1NC', 'basic'),
    ('4-NMe2-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1N(C)C', 'basic'),
    ('4-COOH-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1C(=O)O', 'basic'),
    ('4-COMe-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1C(C)=O', 'basic'),
    ('4-CN-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1C#N', 'basic'),
    ('4-CF3-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-cyclopentyl-piperidinyl', '[*]C1CCN(C1CCCC1)CC1Cl', 'basic'),
    ('4-OH-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1O', 'basic'),
    ('4-OMe-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1OC', 'basic'),
    ('4-F-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1F', 'basic'),
    ('4-Me-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1C', 'basic'),
    ('4-Et-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1CC', 'basic'),
    ('4-NH2-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1N', 'basic'),
    ('4-NHMe-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1NC', 'basic'),
    ('4-NMe2-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1N(C)C', 'basic'),
    ('4-COOH-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1C(=O)O', 'basic'),
    ('4-COMe-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1C(C)=O', 'basic'),
    ('4-CN-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1C#N', 'basic'),
    ('4-CF3-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1C(F)(F)F', 'basic'),
    ('4-Cl-1-tBu-piperidinyl', '[*]C1CCN(C(C)(C)C)CC1Cl', 'basic'),
    ('3-OH-pyrrolidinyl', '[*]C1CNCC1O', 'basic'),
    ('3-OMe-pyrrolidinyl', '[*]C1CNCC1OC', 'basic'),
    ('3-F-pyrrolidinyl', '[*]C1CNCC1F', 'basic'),
    ('3-Me-pyrrolidinyl', '[*]C1CNCC1C', 'basic'),
    ('3-NH2-pyrrolidinyl', '[*]C1CNCC1N', 'basic'),
    ('3-CN-pyrrolidinyl', '[*]C1CNCC1C#N', 'basic'),
    ('3-OH-1-Me-pyrrolidinyl', '[*]C1CN(C)CC1O', 'basic'),
    ('3-OMe-1-Me-pyrrolidinyl', '[*]C1CN(C)CC1OC', 'basic'),
    ('3-F-1-Me-pyrrolidinyl', '[*]C1CN(C)CC1F', 'basic'),
    ('3-Me-1-Me-pyrrolidinyl', '[*]C1CN(C)CC1C', 'basic'),
    ('3-NH2-1-Me-pyrrolidinyl', '[*]C1CN(C)CC1N', 'basic'),
    ('3-CN-1-Me-pyrrolidinyl', '[*]C1CN(C)CC1C#N', 'basic'),
    ('3-OH-1-Et-pyrrolidinyl', '[*]C1CN(CC)CC1O', 'basic'),
    ('3-OMe-1-Et-pyrrolidinyl', '[*]C1CN(CC)CC1OC', 'basic'),
    ('3-F-1-Et-pyrrolidinyl', '[*]C1CN(CC)CC1F', 'basic'),
    ('3-Me-1-Et-pyrrolidinyl', '[*]C1CN(CC)CC1C', 'basic'),
    ('3-NH2-1-Et-pyrrolidinyl', '[*]C1CN(CC)CC1N', 'basic'),
    ('3-CN-1-Et-pyrrolidinyl', '[*]C1CN(CC)CC1C#N', 'basic'),
    ('3-OH-1-Ac-pyrrolidinyl', '[*]C1CN(C(C)=O)CC1O', 'basic'),
    ('3-OMe-1-Ac-pyrrolidinyl', '[*]C1CN(C(C)=O)CC1OC', 'basic'),
    ('3-F-1-Ac-pyrrolidinyl', '[*]C1CN(C(C)=O)CC1F', 'basic'),
    ('3-Me-1-Ac-pyrrolidinyl', '[*]C1CN(C(C)=O)CC1C', 'basic'),
    ('3-NH2-1-Ac-pyrrolidinyl', '[*]C1CN(C(C)=O)CC1N', 'basic'),
    ('3-CN-1-Ac-pyrrolidinyl', '[*]C1CN(C(C)=O)CC1C#N', 'basic'),
    ('3-OH-1-Bn-pyrrolidinyl', '[*]C1CN(Cc1ccccc1)CC1O', 'basic'),
    ('3-OMe-1-Bn-pyrrolidinyl', '[*]C1CN(Cc1ccccc1)CC1OC', 'basic'),
    ('3-F-1-Bn-pyrrolidinyl', '[*]C1CN(Cc1ccccc1)CC1F', 'basic'),
    ('3-Me-1-Bn-pyrrolidinyl', '[*]C1CN(Cc1ccccc1)CC1C', 'basic'),
    ('3-NH2-1-Bn-pyrrolidinyl', '[*]C1CN(Cc1ccccc1)CC1N', 'basic'),
    ('3-CN-1-Bn-pyrrolidinyl', '[*]C1CN(Cc1ccccc1)CC1C#N', 'basic'),
    ('3-OH-1-cPr-pyrrolidinyl', '[*]C1CN(C1CC1)CC1O', 'basic'),
    ('3-OMe-1-cPr-pyrrolidinyl', '[*]C1CN(C1CC1)CC1OC', 'basic'),
    ('3-F-1-cPr-pyrrolidinyl', '[*]C1CN(C1CC1)CC1F', 'basic'),
    ('3-Me-1-cPr-pyrrolidinyl', '[*]C1CN(C1CC1)CC1C', 'basic'),
    ('3-NH2-1-cPr-pyrrolidinyl', '[*]C1CN(C1CC1)CC1N', 'basic'),
    ('3-CN-1-cPr-pyrrolidinyl', '[*]C1CN(C1CC1)CC1C#N', 'basic'),
    ('C-OMe', '[*]COC', 'polar'),
    ('C-OEt', '[*]COCC', 'polar'),
    ('C-NH2', '[*]CN', 'basic'),
    ('C-NHMe', '[*]CNC', 'basic'),
    ('C-NMe2', '[*]CN(C)C', 'basic'),
    ('C-CF3', '[*]CC(F)(F)F', 'halogen'),
    ('C-COOMe', '[*]CC(=O)OC', 'polar'),
    ('C-SO2Me', '[*]CS(=O)(=O)C', 'polar'),
    ('C-SO2NH2', '[*]CS(=O)(=O)N', 'polar'),
    ('C-pyrrolidino', '[*]CN1CCCC1', 'basic'),
    ('C-azetidino', '[*]CN1CCC1', 'basic'),
    ('C-imidazolyl', '[*]Cc1ccn[nH]1', 'basic'),
    ('CC-OMe', '[*]CCOC', 'polar'),
    ('CC-OEt', '[*]CCOCC', 'polar'),
    ('CC-NH2', '[*]CCN', 'basic'),
    ('CC-NHMe', '[*]CCNC', 'basic'),
    ('CC-CF3', '[*]CCC(F)(F)F', 'halogen'),
    ('CC-COOH', '[*]CCC(=O)O', 'acidic'),
    ('CC-COOMe', '[*]CCC(=O)OC', 'polar'),
    ('CC-SO2Me', '[*]CCS(=O)(=O)C', 'polar'),
    ('CC-SO2NH2', '[*]CCS(=O)(=O)N', 'polar'),
    ('CC-pyrrolidino', '[*]CCN1CCCC1', 'basic'),
    ('CC-N-methylpiperazino', '[*]CCN1CCN(C)CC1', 'basic'),
    ('CC-azetidino', '[*]CCN1CCC1', 'basic'),
    ('CC-tetrazolyl', '[*]CCc1nnn[nH]1', 'acidic'),
    ('CC-oxadiazolyl', '[*]CCc1nnco1', 'bioisostere'),
    ('CC-imidazolyl', '[*]CCc1ccn[nH]1', 'basic'),
    ('CC-oxetanyl', '[*]CCC1COC1', 'bioisostere'),
    ('CCC-OMe', '[*]CCCOC', 'polar'),
    ('CCC-OEt', '[*]CCCOCC', 'polar'),
    ('CCC-Cl', '[*]CCCCl', 'halogen'),
    ('CCC-Br', '[*]CCCBr', 'halogen'),
    ('CCC-NHMe', '[*]CCCNC', 'basic'),
    ('CCC-CF3', '[*]CCCC(F)(F)F', 'halogen'),
    ('CCC-COOH', '[*]CCCC(=O)O', 'acidic'),
    ('CCC-COOMe', '[*]CCCC(=O)OC', 'polar'),
    ('CCC-SO2Me', '[*]CCCS(=O)(=O)C', 'polar'),
    ('CCC-SO2NH2', '[*]CCCS(=O)(=O)N', 'polar'),
    ('CCC-piperidino', '[*]CCCN1CCCCC1', 'basic'),
    ('CCC-pyrrolidino', '[*]CCCN1CCCC1', 'basic'),
    ('CCC-N-methylpiperazino', '[*]CCCN1CCN(C)CC1', 'basic'),
    ('CCC-azetidino', '[*]CCCN1CCC1', 'basic'),
    ('CCC-tetrazolyl', '[*]CCCc1nnn[nH]1', 'acidic'),
    ('CCC-oxadiazolyl', '[*]CCCc1nnco1', 'bioisostere'),
    ('CCC-imidazolyl', '[*]CCCc1ccn[nH]1', 'basic'),
    ('CCC-oxetanyl', '[*]CCCC1COC1', 'bioisostere'),
    ('CCCC-OMe', '[*]CCCCOC', 'polar'),
    ('CCCC-OEt', '[*]CCCCOCC', 'polar'),
    ('CCCC-F', '[*]CCCCF', 'halogen'),
    ('CCCC-Cl', '[*]CCCCCl', 'halogen'),
    ('CCCC-Br', '[*]CCCCBr', 'halogen'),
    ('CCCC-NHMe', '[*]CCCCNC', 'basic'),
    ('CCCC-NMe2', '[*]CCCCN(C)C', 'basic'),
    ('CCCC-CN', '[*]CCCCC#N', 'halogen'),
    ('CCCC-CF3', '[*]CCCCC(F)(F)F', 'halogen'),
    ('CCCC-COOH', '[*]CCCCC(=O)O', 'acidic'),
    ('CCCC-COOMe', '[*]CCCCC(=O)OC', 'polar'),
    ('CCCC-SO2Me', '[*]CCCCS(=O)(=O)C', 'polar'),
    ('CCCC-SO2NH2', '[*]CCCCS(=O)(=O)N', 'polar'),
    ('CCCC-morpholino', '[*]CCCCN1CCOCC1', 'basic'),
    ('CCCC-piperidino', '[*]CCCCN1CCCCC1', 'basic'),
    ('CCCC-pyrrolidino', '[*]CCCCN1CCCC1', 'basic'),
    ('CCCC-N-methylpiperazino', '[*]CCCCN1CCN(C)CC1', 'basic'),
    ('CCCC-azetidino', '[*]CCCCN1CCC1', 'basic'),
    ('CCCC-tetrazolyl', '[*]CCCCc1nnn[nH]1', 'acidic'),
    ('CCCC-oxadiazolyl', '[*]CCCCc1nnco1', 'bioisostere'),
    ('CCCC-imidazolyl', '[*]CCCCc1ccn[nH]1', 'basic'),
    ('CCCC-oxetanyl', '[*]CCCCC1COC1', 'bioisostere'),
    ('CO-OH', '[*]COO', 'polar'),
    ('CO-OMe', '[*]COOC', 'polar'),
    ('CO-OEt', '[*]COOCC', 'polar'),
    ('CO-F', '[*]COF', 'halogen'),
    ('CO-Cl', '[*]COCl', 'halogen'),
    ('CO-Br', '[*]COBr', 'halogen'),
    ('CO-NH2', '[*]CON', 'basic'),
    ('CO-NHMe', '[*]CONC', 'basic'),
    ('CO-NMe2', '[*]CON(C)C', 'basic'),
    ('CO-CN', '[*]COC#N', 'halogen'),
    ('CO-CF3', '[*]COC(F)(F)F', 'halogen'),
    ('CO-COOH', '[*]COC(=O)O', 'acidic'),
    ('CO-COOMe', '[*]COC(=O)OC', 'polar'),
    ('CO-SO2Me', '[*]COS(=O)(=O)C', 'polar'),
    ('CO-SO2NH2', '[*]COS(=O)(=O)N', 'polar'),
    ('CO-morpholino', '[*]CON1CCOCC1', 'basic'),
    ('CO-piperidino', '[*]CON1CCCCC1', 'basic'),
    ('CO-pyrrolidino', '[*]CON1CCCC1', 'basic'),
    ('CO-N-methylpiperazino', '[*]CON1CCN(C)CC1', 'basic'),
    ('CO-azetidino', '[*]CON1CCC1', 'basic'),
    ('CO-tetrazolyl', '[*]COc1nnn[nH]1', 'acidic'),
    ('CO-oxadiazolyl', '[*]COc1nnco1', 'bioisostere'),
    ('CO-imidazolyl', '[*]COc1ccn[nH]1', 'basic'),
    ('CO-oxetanyl', '[*]COC1COC1', 'bioisostere'),
    ('CCO-OH', '[*]CCOO', 'polar'),
    ('CCO-OMe', '[*]CCOOC', 'polar'),
    ('CCO-OEt', '[*]CCOOCC', 'polar'),
    ('CCO-F', '[*]CCOF', 'halogen'),
    ('CCO-Cl', '[*]CCOCl', 'halogen'),
    ('CCO-Br', '[*]CCOBr', 'halogen'),
    ('CCO-NH2', '[*]CCON', 'basic'),
    ('CCO-NHMe', '[*]CCONC', 'basic'),
    ('CCO-NMe2', '[*]CCON(C)C', 'basic'),
    ('CCO-CN', '[*]CCOC#N', 'halogen'),
    ('CCO-CF3', '[*]CCOC(F)(F)F', 'halogen'),
    ('CCO-COOH', '[*]CCOC(=O)O', 'acidic'),
    ('CCO-COOMe', '[*]CCOC(=O)OC', 'polar'),
    ('CCO-SO2Me', '[*]CCOS(=O)(=O)C', 'polar'),
    ('CCO-SO2NH2', '[*]CCOS(=O)(=O)N', 'polar'),
    ('CCO-morpholino', '[*]CCON1CCOCC1', 'basic'),
    ('CCO-piperidino', '[*]CCON1CCCCC1', 'basic'),
    ('CCO-pyrrolidino', '[*]CCON1CCCC1', 'basic'),
    ('CCO-N-methylpiperazino', '[*]CCON1CCN(C)CC1', 'basic'),
    ('CCO-azetidino', '[*]CCON1CCC1', 'basic'),
    ('CCO-tetrazolyl', '[*]CCOc1nnn[nH]1', 'acidic'),
    ('CCO-oxadiazolyl', '[*]CCOc1nnco1', 'bioisostere'),
    ('CCO-imidazolyl', '[*]CCOc1ccn[nH]1', 'basic'),
    ('CCO-oxetanyl', '[*]CCOC1COC1', 'bioisostere'),
    ('CCCO-OH', '[*]CCCOO', 'polar'),
    ('CCCO-OMe', '[*]CCCOOC', 'polar'),
    ('CCCO-OEt', '[*]CCCOOCC', 'polar'),
    ('CCCO-F', '[*]CCCOF', 'halogen'),
    ('CCCO-Cl', '[*]CCCOCl', 'halogen'),
    ('CCCO-Br', '[*]CCCOBr', 'halogen'),
    ('CCCO-NH2', '[*]CCCON', 'basic'),
    ('CCCO-NHMe', '[*]CCCONC', 'basic'),
    ('CCCO-NMe2', '[*]CCCON(C)C', 'basic'),
    ('CCCO-CN', '[*]CCCOC#N', 'halogen'),
    ('CCCO-CF3', '[*]CCCOC(F)(F)F', 'halogen'),
    ('CCCO-COOH', '[*]CCCOC(=O)O', 'acidic'),
    ('CCCO-COOMe', '[*]CCCOC(=O)OC', 'polar'),
    ('CCCO-SO2Me', '[*]CCCOS(=O)(=O)C', 'polar'),
    ('CCCO-SO2NH2', '[*]CCCOS(=O)(=O)N', 'polar'),
    ('CCCO-morpholino', '[*]CCCON1CCOCC1', 'basic'),
    ('CCCO-piperidino', '[*]CCCON1CCCCC1', 'basic'),
    ('CCCO-pyrrolidino', '[*]CCCON1CCCC1', 'basic'),
    ('CCCO-N-methylpiperazino', '[*]CCCON1CCN(C)CC1', 'basic'),
    ('CCCO-azetidino', '[*]CCCON1CCC1', 'basic'),
    ('CCCO-tetrazolyl', '[*]CCCOc1nnn[nH]1', 'acidic'),
    ('CCCO-oxadiazolyl', '[*]CCCOc1nnco1', 'bioisostere'),
    ('CCCO-imidazolyl', '[*]CCCOc1ccn[nH]1', 'basic'),
    ('CCCO-oxetanyl', '[*]CCCOC1COC1', 'bioisostere'),
    ('CN-OH', '[*]CNO', 'polar'),
    ('CN-OMe', '[*]CNOC', 'polar'),
    ('CN-OEt', '[*]CNOCC', 'polar'),
    ('CN-F', '[*]CNF', 'halogen'),
    ('CN-Cl', '[*]CNCl', 'halogen'),
    ('CN-Br', '[*]CNBr', 'halogen'),
    ('CN-NH2', '[*]CNN', 'basic'),
    ('CN-NHMe', '[*]CNNC', 'basic'),
    ('CN-NMe2', '[*]CNN(C)C', 'basic'),
    ('CN-CN', '[*]CNC#N', 'halogen'),
    ('CN-CF3', '[*]CNC(F)(F)F', 'halogen'),
    ('CN-COOH', '[*]CNC(=O)O', 'acidic'),
    ('CN-COOMe', '[*]CNC(=O)OC', 'polar'),
    ('CN-SO2Me', '[*]CNS(=O)(=O)C', 'polar'),
    ('CN-SO2NH2', '[*]CNS(=O)(=O)N', 'polar'),
    ('CN-morpholino', '[*]CNN1CCOCC1', 'basic'),
    ('CN-piperidino', '[*]CNN1CCCCC1', 'basic'),
    ('CN-pyrrolidino', '[*]CNN1CCCC1', 'basic'),
    ('CN-N-methylpiperazino', '[*]CNN1CCN(C)CC1', 'basic'),
    ('CN-azetidino', '[*]CNN1CCC1', 'basic'),
    ('CN-tetrazolyl', '[*]CNc1nnn[nH]1', 'acidic'),
    ('CN-oxadiazolyl', '[*]CNc1nnco1', 'bioisostere'),
    ('CN-imidazolyl', '[*]CNc1ccn[nH]1', 'basic'),
    ('CN-oxetanyl', '[*]CNC1COC1', 'bioisostere'),
    ('CCN-OH', '[*]CCNO', 'polar'),
    ('CCN-OMe', '[*]CCNOC', 'polar'),
    ('CCN-OEt', '[*]CCNOCC', 'polar'),
    ('CCN-F', '[*]CCNF', 'halogen'),
    ('CCN-Cl', '[*]CCNCl', 'halogen'),
    ('CCN-Br', '[*]CCNBr', 'halogen'),
    ('CCN-NH2', '[*]CCNN', 'basic'),
    ('CCN-NHMe', '[*]CCNNC', 'basic'),
    ('CCN-NMe2', '[*]CCNN(C)C', 'basic'),
    ('CCN-CN', '[*]CCNC#N', 'halogen'),
    ('CCN-CF3', '[*]CCNC(F)(F)F', 'halogen'),
    ('CCN-COOH', '[*]CCNC(=O)O', 'acidic'),
    ('CCN-COOMe', '[*]CCNC(=O)OC', 'polar'),
    ('CCN-SO2Me', '[*]CCNS(=O)(=O)C', 'polar'),
    ('CCN-SO2NH2', '[*]CCNS(=O)(=O)N', 'polar'),
    ('CCN-morpholino', '[*]CCNN1CCOCC1', 'basic'),
    ('CCN-piperidino', '[*]CCNN1CCCCC1', 'basic'),
    ('CCN-pyrrolidino', '[*]CCNN1CCCC1', 'basic'),
    ('CCN-N-methylpiperazino', '[*]CCNN1CCN(C)CC1', 'basic'),
    ('CCN-azetidino', '[*]CCNN1CCC1', 'basic'),
    ('CCN-tetrazolyl', '[*]CCNc1nnn[nH]1', 'acidic'),
    ('CCN-oxadiazolyl', '[*]CCNc1nnco1', 'bioisostere'),
    ('CCN-imidazolyl', '[*]CCNc1ccn[nH]1', 'basic'),
    ('CCN-oxetanyl', '[*]CCNC1COC1', 'bioisostere'),
    ('CCCN-OH', '[*]CCCNO', 'polar'),
    ('CCCN-OMe', '[*]CCCNOC', 'polar'),
    ('CCCN-OEt', '[*]CCCNOCC', 'polar'),
    ('CCCN-F', '[*]CCCNF', 'halogen'),
    ('CCCN-Cl', '[*]CCCNCl', 'halogen'),
    ('CCCN-Br', '[*]CCCNBr', 'halogen'),
    ('CCCN-NH2', '[*]CCCNN', 'basic'),
    ('CCCN-NHMe', '[*]CCCNNC', 'basic'),
    ('CCCN-NMe2', '[*]CCCNN(C)C', 'basic'),
    ('CCCN-CN', '[*]CCCNC#N', 'halogen'),
    ('CCCN-CF3', '[*]CCCNC(F)(F)F', 'halogen'),
    ('CCCN-COOH', '[*]CCCNC(=O)O', 'acidic'),
    ('CCCN-COOMe', '[*]CCCNC(=O)OC', 'polar'),
    ('CCCN-SO2Me', '[*]CCCNS(=O)(=O)C', 'polar'),
    ('CCCN-SO2NH2', '[*]CCCNS(=O)(=O)N', 'polar'),
    ('CCCN-morpholino', '[*]CCCNN1CCOCC1', 'basic'),
    ('CCCN-piperidino', '[*]CCCNN1CCCCC1', 'basic'),
    ('CCCN-pyrrolidino', '[*]CCCNN1CCCC1', 'basic'),
    ('CCCN-N-methylpiperazino', '[*]CCCNN1CCN(C)CC1', 'basic'),
    ('CCCN-azetidino', '[*]CCCNN1CCC1', 'basic'),
    ('CCCN-tetrazolyl', '[*]CCCNc1nnn[nH]1', 'acidic'),
    ('CCCN-oxadiazolyl', '[*]CCCNc1nnco1', 'bioisostere'),
    ('CCCN-imidazolyl', '[*]CCCNc1ccn[nH]1', 'basic'),
    ('CCCN-oxetanyl', '[*]CCCNC1COC1', 'bioisostere'),
    ('C=C-OH', '[*]C=CO', 'polar'),
    ('C=C-OMe', '[*]C=COC', 'polar'),
    ('C=C-OEt', '[*]C=COCC', 'polar'),
    ('C=C-F', '[*]C=CF', 'halogen'),
    ('C=C-Cl', '[*]C=CCl', 'halogen'),
    ('C=C-Br', '[*]C=CBr', 'halogen'),
    ('C=C-NH2', '[*]C=CN', 'basic'),
    ('C=C-NHMe', '[*]C=CNC', 'basic'),
    ('C=C-NMe2', '[*]C=CN(C)C', 'basic'),
    ('C=C-CN', '[*]C=CC#N', 'halogen'),
    ('C=C-CF3', '[*]C=CC(F)(F)F', 'halogen'),
    ('C=C-COOH', '[*]C=CC(=O)O', 'acidic'),
    ('C=C-COOMe', '[*]C=CC(=O)OC', 'polar'),
    ('C=C-SO2Me', '[*]C=CS(=O)(=O)C', 'polar'),
    ('C=C-SO2NH2', '[*]C=CS(=O)(=O)N', 'polar'),
    ('C=C-morpholino', '[*]C=CN1CCOCC1', 'basic'),
    ('C=C-piperidino', '[*]C=CN1CCCCC1', 'basic'),
    ('C=C-pyrrolidino', '[*]C=CN1CCCC1', 'basic'),
    ('C=C-N-methylpiperazino', '[*]C=CN1CCN(C)CC1', 'basic'),
    ('C=C-azetidino', '[*]C=CN1CCC1', 'basic'),
    ('C=C-tetrazolyl', '[*]C=Cc1nnn[nH]1', 'acidic'),
    ('C=C-oxadiazolyl', '[*]C=Cc1nnco1', 'bioisostere'),
    ('C=C-imidazolyl', '[*]C=Cc1ccn[nH]1', 'basic'),
    ('C=C-oxetanyl', '[*]C=CC1COC1', 'bioisostere'),
    ('C#C-OH', '[*]C#CO', 'polar'),
    ('C#C-OMe', '[*]C#COC', 'polar'),
    ('C#C-OEt', '[*]C#COCC', 'polar'),
    ('C#C-F', '[*]C#CF', 'halogen'),
    ('C#C-Cl', '[*]C#CCl', 'halogen'),
    ('C#C-Br', '[*]C#CBr', 'halogen'),
    ('C#C-NH2', '[*]C#CN', 'basic'),
    ('C#C-NHMe', '[*]C#CNC', 'basic'),
    ('C#C-NMe2', '[*]C#CN(C)C', 'basic'),
    ('C#C-CN', '[*]C#CC#N', 'halogen'),
    ('C#C-CF3', '[*]C#CC(F)(F)F', 'halogen'),
    ('C#C-COOH', '[*]C#CC(=O)O', 'acidic'),
    ('C#C-COOMe', '[*]C#CC(=O)OC', 'polar'),
    ('C#C-SO2Me', '[*]C#CS(=O)(=O)C', 'polar'),
    ('C#C-SO2NH2', '[*]C#CS(=O)(=O)N', 'polar'),
    ('C#C-morpholino', '[*]C#CN1CCOCC1', 'basic'),
    ('C#C-piperidino', '[*]C#CN1CCCCC1', 'basic'),
    ('C#C-pyrrolidino', '[*]C#CN1CCCC1', 'basic'),
    ('C#C-N-methylpiperazino', '[*]C#CN1CCN(C)CC1', 'basic'),
    ('C#C-azetidino', '[*]C#CN1CCC1', 'basic'),
    ('C#C-tetrazolyl', '[*]C#Cc1nnn[nH]1', 'acidic'),
    ('C#C-oxadiazolyl', '[*]C#Cc1nnco1', 'bioisostere'),
    ('C#C-imidazolyl', '[*]C#Cc1ccn[nH]1', 'basic'),
    ('C#C-oxetanyl', '[*]C#CC1COC1', 'bioisostere'),
    ('C(F)(F)-OH', '[*]C(F)(F)O', 'polar'),
    ('C(F)(F)-OMe', '[*]C(F)(F)OC', 'polar'),
    ('C(F)(F)-OEt', '[*]C(F)(F)OCC', 'polar'),
    ('C(F)(F)-Cl', '[*]C(F)(F)Cl', 'halogen'),
    ('C(F)(F)-Br', '[*]C(F)(F)Br', 'halogen'),
    ('C(F)(F)-NH2', '[*]C(F)(F)N', 'basic'),
    ('C(F)(F)-NHMe', '[*]C(F)(F)NC', 'basic'),
    ('C(F)(F)-NMe2', '[*]C(F)(F)N(C)C', 'basic'),
    ('C(F)(F)-CN', '[*]C(F)(F)C#N', 'halogen'),
    ('C(F)(F)-CF3', '[*]C(F)(F)C(F)(F)F', 'halogen'),
    ('C(F)(F)-COOH', '[*]C(F)(F)C(=O)O', 'acidic'),
    ('C(F)(F)-COOMe', '[*]C(F)(F)C(=O)OC', 'polar'),
    ('C(F)(F)-SO2Me', '[*]C(F)(F)S(=O)(=O)C', 'polar'),
    ('C(F)(F)-SO2NH2', '[*]C(F)(F)S(=O)(=O)N', 'polar'),
    ('C(F)(F)-morpholino', '[*]C(F)(F)N1CCOCC1', 'basic'),
    ('C(F)(F)-piperidino', '[*]C(F)(F)N1CCCCC1', 'basic'),
    ('C(F)(F)-pyrrolidino', '[*]C(F)(F)N1CCCC1', 'basic'),
    ('C(F)(F)-N-methylpiperazino', '[*]C(F)(F)N1CCN(C)CC1', 'basic'),
    ('C(F)(F)-azetidino', '[*]C(F)(F)N1CCC1', 'basic'),
    ('C(F)(F)-tetrazolyl', '[*]C(F)(F)c1nnn[nH]1', 'acidic'),
    ('C(F)(F)-oxadiazolyl', '[*]C(F)(F)c1nnco1', 'bioisostere'),
    ('C(F)(F)-imidazolyl', '[*]C(F)(F)c1ccn[nH]1', 'basic'),
    ('C(F)(F)-oxetanyl', '[*]C(F)(F)C1COC1', 'bioisostere'),
    ('C(C)-OH', '[*]C(C)O', 'polar'),
    ('C(C)-OMe', '[*]C(C)OC', 'polar'),
    ('C(C)-OEt', '[*]C(C)OCC', 'polar'),
    ('C(C)-F', '[*]C(C)F', 'halogen'),
    ('C(C)-Cl', '[*]C(C)Cl', 'halogen'),
    ('C(C)-Br', '[*]C(C)Br', 'halogen'),
    ('C(C)-NH2', '[*]C(C)N', 'basic'),
    ('C(C)-NHMe', '[*]C(C)NC', 'basic'),
    ('C(C)-NMe2', '[*]C(C)N(C)C', 'basic'),
    ('C(C)-CN', '[*]C(C)C#N', 'halogen'),
    ('C(C)-CF3', '[*]C(C)C(F)(F)F', 'halogen'),
    ('C(C)-COOH', '[*]C(C)C(=O)O', 'acidic'),
    ('C(C)-COOMe', '[*]C(C)C(=O)OC', 'polar'),
    ('C(C)-SO2Me', '[*]C(C)S(=O)(=O)C', 'polar'),
    ('C(C)-SO2NH2', '[*]C(C)S(=O)(=O)N', 'polar'),
    ('C(C)-morpholino', '[*]C(C)N1CCOCC1', 'basic'),
    ('C(C)-piperidino', '[*]C(C)N1CCCCC1', 'basic'),
    ('C(C)-pyrrolidino', '[*]C(C)N1CCCC1', 'basic'),
    ('C(C)-N-methylpiperazino', '[*]C(C)N1CCN(C)CC1', 'basic'),
    ('C(C)-azetidino', '[*]C(C)N1CCC1', 'basic'),
    ('C(C)-tetrazolyl', '[*]C(C)c1nnn[nH]1', 'acidic'),
    ('C(C)-oxadiazolyl', '[*]C(C)c1nnco1', 'bioisostere'),
    ('C(C)-imidazolyl', '[*]C(C)c1ccn[nH]1', 'basic'),
    ('C(C)-oxetanyl', '[*]C(C)C1COC1', 'bioisostere'),
    ('C(C)(C)-OH', '[*]C(C)(C)O', 'polar'),
    ('C(C)(C)-OMe', '[*]C(C)(C)OC', 'polar'),
    ('C(C)(C)-OEt', '[*]C(C)(C)OCC', 'polar'),
    ('C(C)(C)-F', '[*]C(C)(C)F', 'halogen'),
    ('C(C)(C)-Cl', '[*]C(C)(C)Cl', 'halogen'),
    ('C(C)(C)-Br', '[*]C(C)(C)Br', 'halogen'),
    ('C(C)(C)-NH2', '[*]C(C)(C)N', 'basic'),
    ('C(C)(C)-NHMe', '[*]C(C)(C)NC', 'basic'),
    ('C(C)(C)-NMe2', '[*]C(C)(C)N(C)C', 'basic'),
    ('C(C)(C)-CN', '[*]C(C)(C)C#N', 'halogen'),
    ('C(C)(C)-CF3', '[*]C(C)(C)C(F)(F)F', 'halogen'),
    ('C(C)(C)-COOH', '[*]C(C)(C)C(=O)O', 'acidic'),
    ('C(C)(C)-COOMe', '[*]C(C)(C)C(=O)OC', 'polar'),
    ('C(C)(C)-SO2Me', '[*]C(C)(C)S(=O)(=O)C', 'polar'),
    ('C(C)(C)-SO2NH2', '[*]C(C)(C)S(=O)(=O)N', 'polar'),
    ('C(C)(C)-morpholino', '[*]C(C)(C)N1CCOCC1', 'basic'),
    ('C(C)(C)-piperidino', '[*]C(C)(C)N1CCCCC1', 'basic'),
    ('C(C)(C)-pyrrolidino', '[*]C(C)(C)N1CCCC1', 'basic'),
    ('C(C)(C)-N-methylpiperazino', '[*]C(C)(C)N1CCN(C)CC1', 'basic'),
    ('C(C)(C)-azetidino', '[*]C(C)(C)N1CCC1', 'basic'),
    ('C(C)(C)-tetrazolyl', '[*]C(C)(C)c1nnn[nH]1', 'acidic'),
    ('C(C)(C)-oxadiazolyl', '[*]C(C)(C)c1nnco1', 'bioisostere'),
    ('C(C)(C)-imidazolyl', '[*]C(C)(C)c1ccn[nH]1', 'basic'),
    ('C(C)(C)-oxetanyl', '[*]C(C)(C)C1COC1', 'bioisostere'),
    ('C1CC1-OH', '[*]C1CC1O', 'polar'),
    ('C1CC1-OMe', '[*]C1CC1OC', 'polar'),
    ('C1CC1-OEt', '[*]C1CC1OCC', 'polar'),
    ('C1CC1-F', '[*]C1CC1F', 'halogen'),
    ('C1CC1-Cl', '[*]C1CC1Cl', 'halogen'),
    ('C1CC1-Br', '[*]C1CC1Br', 'halogen'),
    ('C1CC1-NH2', '[*]C1CC1N', 'basic'),
    ('C1CC1-NHMe', '[*]C1CC1NC', 'basic'),
    ('C1CC1-NMe2', '[*]C1CC1N(C)C', 'basic'),
    ('C1CC1-CN', '[*]C1CC1C#N', 'halogen'),
    ('C1CC1-CF3', '[*]C1CC1C(F)(F)F', 'halogen'),
    ('C1CC1-COOH', '[*]C1CC1C(=O)O', 'acidic'),
    ('C1CC1-COOMe', '[*]C1CC1C(=O)OC', 'polar'),
    ('C1CC1-SO2Me', '[*]C1CC1S(=O)(=O)C', 'polar'),
    ('C1CC1-SO2NH2', '[*]C1CC1S(=O)(=O)N', 'polar'),
    ('C1CC1-morpholino', '[*]C1CC1N1CCOCC1', 'basic'),
    ('C1CC1-piperidino', '[*]C1CC1N1CCCCC1', 'basic'),
    ('C1CC1-pyrrolidino', '[*]C1CC1N1CCCC1', 'basic'),
    ('C1CC1-N-methylpiperazino', '[*]C1CC1N1CCN(C)CC1', 'basic'),
    ('C1CC1-azetidino', '[*]C1CC1N1CCC1', 'basic'),
    ('C1CC1-tetrazolyl', '[*]C1CC1c1nnn[nH]1', 'acidic'),
    ('C1CC1-oxadiazolyl', '[*]C1CC1c1nnco1', 'bioisostere'),
    ('C1CC1-imidazolyl', '[*]C1CC1c1ccn[nH]1', 'basic'),
    ('C1CC1-oxetanyl', '[*]C1CC1C1COC1', 'bioisostere'),
    ('C1CCC1-OH', '[*]C1CCC1O', 'polar'),
    ('C1CCC1-OMe', '[*]C1CCC1OC', 'polar'),
    ('C1CCC1-OEt', '[*]C1CCC1OCC', 'polar'),
    ('C1CCC1-F', '[*]C1CCC1F', 'halogen'),
    ('C1CCC1-Cl', '[*]C1CCC1Cl', 'halogen'),
    ('C1CCC1-Br', '[*]C1CCC1Br', 'halogen'),
    ('C1CCC1-NH2', '[*]C1CCC1N', 'basic'),
    ('C1CCC1-NHMe', '[*]C1CCC1NC', 'basic'),
    ('C1CCC1-NMe2', '[*]C1CCC1N(C)C', 'basic'),
    ('C1CCC1-CN', '[*]C1CCC1C#N', 'halogen'),
    ('C1CCC1-CF3', '[*]C1CCC1C(F)(F)F', 'halogen'),
    ('C1CCC1-COOH', '[*]C1CCC1C(=O)O', 'acidic'),
    ('C1CCC1-COOMe', '[*]C1CCC1C(=O)OC', 'polar'),
    ('C1CCC1-SO2Me', '[*]C1CCC1S(=O)(=O)C', 'polar'),
    ('C1CCC1-SO2NH2', '[*]C1CCC1S(=O)(=O)N', 'polar'),
    ('C1CCC1-morpholino', '[*]C1CCC1N1CCOCC1', 'basic'),
    ('C1CCC1-piperidino', '[*]C1CCC1N1CCCCC1', 'basic'),
    ('C1CCC1-pyrrolidino', '[*]C1CCC1N1CCCC1', 'basic'),
    ('C1CCC1-N-methylpiperazino', '[*]C1CCC1N1CCN(C)CC1', 'basic'),
    ('C1CCC1-azetidino', '[*]C1CCC1N1CCC1', 'basic'),
    ('C1CCC1-tetrazolyl', '[*]C1CCC1c1nnn[nH]1', 'acidic'),
    ('C1CCC1-oxadiazolyl', '[*]C1CCC1c1nnco1', 'bioisostere'),
    ('C1CCC1-imidazolyl', '[*]C1CCC1c1ccn[nH]1', 'basic'),
    ('C1CCC1-oxetanyl', '[*]C1CCC1C1COC1', 'bioisostere'),
    ('C(=O)-SO2Me', '[*]C(=O)S(=O)(=O)C', 'polar'),
    ('C(=O)-SO2NH2', '[*]C(=O)S(=O)(=O)N', 'polar'),
    ('S(=O)(=O)-OMe', '[*]S(=O)(=O)OC', 'polar'),
    ('S(=O)(=O)-OEt', '[*]S(=O)(=O)OCC', 'polar'),
    ('S(=O)(=O)-F', '[*]S(=O)(=O)F', 'halogen'),
    ('S(=O)(=O)-Cl', '[*]S(=O)(=O)Cl', 'halogen'),
    ('S(=O)(=O)-Br', '[*]S(=O)(=O)Br', 'halogen'),
    ('S(=O)(=O)-CN', '[*]S(=O)(=O)C#N', 'halogen'),
    ('S(=O)(=O)-CF3', '[*]S(=O)(=O)C(F)(F)F', 'halogen'),
    ('S(=O)(=O)-COOH', '[*]S(=O)(=O)C(=O)O', 'acidic'),
    ('S(=O)(=O)-COOMe', '[*]S(=O)(=O)C(=O)OC', 'polar'),
    ('S(=O)(=O)-SO2Me', '[*]S(=O)(=O)S(=O)(=O)C', 'polar'),
    ('S(=O)(=O)-SO2NH2', '[*]S(=O)(=O)S(=O)(=O)N', 'polar'),
    ('S(=O)(=O)-pyrrolidino', '[*]S(=O)(=O)N1CCCC1', 'basic'),
    ('S(=O)(=O)-N-methylpiperazino', '[*]S(=O)(=O)N1CCN(C)CC1', 'basic'),
    ('S(=O)(=O)-azetidino', '[*]S(=O)(=O)N1CCC1', 'basic'),
    ('S(=O)(=O)-tetrazolyl', '[*]S(=O)(=O)c1nnn[nH]1', 'acidic'),
    ('S(=O)(=O)-oxadiazolyl', '[*]S(=O)(=O)c1nnco1', 'bioisostere'),
    ('S(=O)(=O)-imidazolyl', '[*]S(=O)(=O)c1ccn[nH]1', 'basic'),
    ('S(=O)(=O)-oxetanyl', '[*]S(=O)(=O)C1COC1', 'bioisostere'),
    ('OC-OH', '[*]OCO', 'polar'),
    ('OC-OMe', '[*]OCOC', 'polar'),
    ('OC-OEt', '[*]OCOCC', 'polar'),
    ('OC-F', '[*]OCF', 'halogen'),
    ('OC-Cl', '[*]OCCl', 'halogen'),
    ('OC-Br', '[*]OCBr', 'halogen'),
    ('OC-NH2', '[*]OCN', 'basic'),
    ('OC-NHMe', '[*]OCNC', 'basic'),
    ('OC-NMe2', '[*]OCN(C)C', 'basic'),
    ('OC-CN', '[*]OCC#N', 'halogen'),
    ('OC-CF3', '[*]OCC(F)(F)F', 'halogen'),
    ('OC-COOH', '[*]OCC(=O)O', 'acidic'),
    ('OC-COOMe', '[*]OCC(=O)OC', 'polar'),
    ('OC-SO2Me', '[*]OCS(=O)(=O)C', 'polar'),
    ('OC-SO2NH2', '[*]OCS(=O)(=O)N', 'polar'),
    ('OC-morpholino', '[*]OCN1CCOCC1', 'basic'),
    ('OC-piperidino', '[*]OCN1CCCCC1', 'basic'),
    ('OC-pyrrolidino', '[*]OCN1CCCC1', 'basic'),
    ('OC-N-methylpiperazino', '[*]OCN1CCN(C)CC1', 'basic'),
    ('OC-azetidino', '[*]OCN1CCC1', 'basic'),
    ('OC-tetrazolyl', '[*]OCc1nnn[nH]1', 'acidic'),
    ('OC-oxadiazolyl', '[*]OCc1nnco1', 'bioisostere'),
    ('OC-imidazolyl', '[*]OCc1ccn[nH]1', 'basic'),
    ('OC-oxetanyl', '[*]OCC1COC1', 'bioisostere'),
    ('OCC-OEt', '[*]OCCOCC', 'polar'),
    ('OCC-Cl', '[*]OCCCl', 'halogen'),
    ('OCC-Br', '[*]OCCBr', 'halogen'),
    ('OCC-NHMe', '[*]OCCNC', 'basic'),
    ('OCC-NMe2', '[*]OCCN(C)C', 'basic'),
    ('OCC-CN', '[*]OCCC#N', 'halogen'),
    ('OCC-CF3', '[*]OCCC(F)(F)F', 'halogen'),
    ('OCC-COOH', '[*]OCCC(=O)O', 'acidic'),
    ('OCC-COOMe', '[*]OCCC(=O)OC', 'polar'),
    ('OCC-SO2Me', '[*]OCCS(=O)(=O)C', 'polar'),
    ('OCC-SO2NH2', '[*]OCCS(=O)(=O)N', 'polar'),
    ('OCC-morpholino', '[*]OCCN1CCOCC1', 'basic'),
    ('OCC-piperidino', '[*]OCCN1CCCCC1', 'basic'),
    ('OCC-pyrrolidino', '[*]OCCN1CCCC1', 'basic'),
    ('OCC-N-methylpiperazino', '[*]OCCN1CCN(C)CC1', 'basic'),
    ('OCC-azetidino', '[*]OCCN1CCC1', 'basic'),
    ('OCC-tetrazolyl', '[*]OCCc1nnn[nH]1', 'acidic'),
    ('OCC-oxadiazolyl', '[*]OCCc1nnco1', 'bioisostere'),
    ('OCC-imidazolyl', '[*]OCCc1ccn[nH]1', 'basic'),
    ('OCC-oxetanyl', '[*]OCCC1COC1', 'bioisostere'),
    ('OCCC-OH', '[*]OCCCO', 'polar'),
    ('OCCC-OEt', '[*]OCCCOCC', 'polar'),
    ('OCCC-Cl', '[*]OCCCCl', 'halogen'),
    ('OCCC-Br', '[*]OCCCBr', 'halogen'),
    ('OCCC-NH2', '[*]OCCCN', 'basic'),
    ('OCCC-NHMe', '[*]OCCCNC', 'basic'),
    ('OCCC-NMe2', '[*]OCCCN(C)C', 'basic'),
    ('OCCC-CN', '[*]OCCCC#N', 'halogen'),
    ('OCCC-CF3', '[*]OCCCC(F)(F)F', 'halogen'),
    ('OCCC-COOH', '[*]OCCCC(=O)O', 'acidic'),
    ('OCCC-COOMe', '[*]OCCCC(=O)OC', 'polar'),
    ('OCCC-SO2Me', '[*]OCCCS(=O)(=O)C', 'polar'),
    ('OCCC-SO2NH2', '[*]OCCCS(=O)(=O)N', 'polar'),
    ('OCCC-morpholino', '[*]OCCCN1CCOCC1', 'basic'),
    ('OCCC-piperidino', '[*]OCCCN1CCCCC1', 'basic'),
    ('OCCC-pyrrolidino', '[*]OCCCN1CCCC1', 'basic'),
    ('OCCC-N-methylpiperazino', '[*]OCCCN1CCN(C)CC1', 'basic'),
    ('OCCC-azetidino', '[*]OCCCN1CCC1', 'basic'),
    ('OCCC-tetrazolyl', '[*]OCCCc1nnn[nH]1', 'acidic'),
    ('OCCC-oxadiazolyl', '[*]OCCCc1nnco1', 'bioisostere'),
    ('OCCC-imidazolyl', '[*]OCCCc1ccn[nH]1', 'basic'),
    ('OCCC-oxetanyl', '[*]OCCCC1COC1', 'bioisostere'),
    ('NC-OH', '[*]NCO', 'polar'),
    ('NC-OMe', '[*]NCOC', 'polar'),
    ('NC-OEt', '[*]NCOCC', 'polar'),
    ('NC-F', '[*]NCF', 'halogen'),
    ('NC-Cl', '[*]NCCl', 'halogen'),
    ('NC-Br', '[*]NCBr', 'halogen'),
    ('NC-NH2', '[*]NCN', 'basic'),
    ('NC-NHMe', '[*]NCNC', 'basic'),
    ('NC-NMe2', '[*]NCN(C)C', 'basic'),
    ('NC-CN', '[*]NCC#N', 'halogen'),
    ('NC-CF3', '[*]NCC(F)(F)F', 'halogen'),
    ('NC-COOH', '[*]NCC(=O)O', 'acidic'),
    ('NC-COOMe', '[*]NCC(=O)OC', 'polar'),
    ('NC-SO2Me', '[*]NCS(=O)(=O)C', 'polar'),
    ('NC-SO2NH2', '[*]NCS(=O)(=O)N', 'polar'),
    ('NC-morpholino', '[*]NCN1CCOCC1', 'basic'),
    ('NC-piperidino', '[*]NCN1CCCCC1', 'basic'),
    ('NC-pyrrolidino', '[*]NCN1CCCC1', 'basic'),
    ('NC-N-methylpiperazino', '[*]NCN1CCN(C)CC1', 'basic'),
    ('NC-azetidino', '[*]NCN1CCC1', 'basic'),
    ('NC-tetrazolyl', '[*]NCc1nnn[nH]1', 'acidic'),
    ('NC-oxadiazolyl', '[*]NCc1nnco1', 'bioisostere'),
    ('NC-imidazolyl', '[*]NCc1ccn[nH]1', 'basic'),
    ('NC-oxetanyl', '[*]NCC1COC1', 'bioisostere'),
    ('NCC-OH', '[*]NCCO', 'polar'),
    ('NCC-OMe', '[*]NCCOC', 'polar'),
    ('NCC-OEt', '[*]NCCOCC', 'polar'),
    ('NCC-F', '[*]NCCF', 'halogen'),
    ('NCC-Cl', '[*]NCCCl', 'halogen'),
    ('NCC-Br', '[*]NCCBr', 'halogen'),
    ('NCC-NH2', '[*]NCCN', 'basic'),
    ('NCC-NHMe', '[*]NCCNC', 'basic'),
    ('NCC-NMe2', '[*]NCCN(C)C', 'basic'),
    ('NCC-CN', '[*]NCCC#N', 'halogen'),
    ('NCC-CF3', '[*]NCCC(F)(F)F', 'halogen'),
    ('NCC-COOH', '[*]NCCC(=O)O', 'acidic'),
    ('NCC-COOMe', '[*]NCCC(=O)OC', 'polar'),
    ('NCC-SO2Me', '[*]NCCS(=O)(=O)C', 'polar'),
    ('NCC-SO2NH2', '[*]NCCS(=O)(=O)N', 'polar'),
    ('NCC-morpholino', '[*]NCCN1CCOCC1', 'basic'),
    ('NCC-piperidino', '[*]NCCN1CCCCC1', 'basic'),
    ('NCC-pyrrolidino', '[*]NCCN1CCCC1', 'basic'),
    ('NCC-N-methylpiperazino', '[*]NCCN1CCN(C)CC1', 'basic'),
    ('NCC-azetidino', '[*]NCCN1CCC1', 'basic'),
    ('NCC-tetrazolyl', '[*]NCCc1nnn[nH]1', 'acidic'),
    ('NCC-oxadiazolyl', '[*]NCCc1nnco1', 'bioisostere'),
    ('NCC-imidazolyl', '[*]NCCc1ccn[nH]1', 'basic'),
    ('NCC-oxetanyl', '[*]NCCC1COC1', 'bioisostere'),
    ('N(C)C-OH', '[*]N(C)CO', 'polar'),
    ('N(C)C-OMe', '[*]N(C)COC', 'polar'),
    ('N(C)C-OEt', '[*]N(C)COCC', 'polar'),
    ('N(C)C-F', '[*]N(C)CF', 'halogen'),
    ('N(C)C-Cl', '[*]N(C)CCl', 'halogen'),
    ('N(C)C-Br', '[*]N(C)CBr', 'halogen'),
    ('N(C)C-NH2', '[*]N(C)CN', 'basic'),
    ('N(C)C-NHMe', '[*]N(C)CNC', 'basic'),
    ('N(C)C-NMe2', '[*]N(C)CN(C)C', 'basic'),
    ('N(C)C-CN', '[*]N(C)CC#N', 'halogen'),
    ('N(C)C-CF3', '[*]N(C)CC(F)(F)F', 'halogen'),
    ('N(C)C-COOH', '[*]N(C)CC(=O)O', 'acidic'),
    ('N(C)C-COOMe', '[*]N(C)CC(=O)OC', 'polar'),
    ('N(C)C-SO2Me', '[*]N(C)CS(=O)(=O)C', 'polar'),
    ('N(C)C-SO2NH2', '[*]N(C)CS(=O)(=O)N', 'polar'),
    ('N(C)C-morpholino', '[*]N(C)CN1CCOCC1', 'basic'),
    ('N(C)C-piperidino', '[*]N(C)CN1CCCCC1', 'basic'),
    ('N(C)C-pyrrolidino', '[*]N(C)CN1CCCC1', 'basic'),
    ('N(C)C-N-methylpiperazino', '[*]N(C)CN1CCN(C)CC1', 'basic'),
    ('N(C)C-azetidino', '[*]N(C)CN1CCC1', 'basic'),
    ('N(C)C-tetrazolyl', '[*]N(C)Cc1nnn[nH]1', 'acidic'),
    ('N(C)C-oxadiazolyl', '[*]N(C)Cc1nnco1', 'bioisostere'),
    ('N(C)C-imidazolyl', '[*]N(C)Cc1ccn[nH]1', 'basic'),
    ('N(C)C-oxetanyl', '[*]N(C)CC1COC1', 'bioisostere'),
    ('c1ccccc1-N-methylpiperazino', '[*]c1ccccc1N1CCN(C)CC1', 'basic'),
    ('c1ccccc1-azetidino', '[*]c1ccccc1N1CCC1', 'basic'),
    ('c1ccccc1-tetrazolyl', '[*]c1ccccc1c1nnn[nH]1', 'acidic'),
    ('c1ccccc1-oxadiazolyl', '[*]c1ccccc1c1nnco1', 'bioisostere'),
    ('c1ccccc1-imidazolyl', '[*]c1ccccc1c1ccn[nH]1', 'basic'),
    ('c1ccccc1-oxetanyl', '[*]c1ccccc1C1COC1', 'bioisostere'),
    ('c1cccnc1-OH', '[*]c1cccnc1O', 'polar'),
    ('c1cccnc1-OMe', '[*]c1cccnc1OC', 'polar'),
    ('c1cccnc1-OEt', '[*]c1cccnc1OCC', 'polar'),
    ('c1cccnc1-F', '[*]c1cccnc1F', 'halogen'),
    ('c1cccnc1-Cl', '[*]c1cccnc1Cl', 'halogen'),
    ('c1cccnc1-Br', '[*]c1cccnc1Br', 'halogen'),
    ('c1cccnc1-NH2', '[*]c1cccnc1N', 'basic'),
    ('c1cccnc1-NHMe', '[*]c1cccnc1NC', 'basic'),
    ('c1cccnc1-NMe2', '[*]c1cccnc1N(C)C', 'basic'),
    ('c1cccnc1-CN', '[*]c1cccnc1C#N', 'halogen'),
    ('c1cccnc1-CF3', '[*]c1cccnc1C(F)(F)F', 'halogen'),
    ('c1cccnc1-COOH', '[*]c1cccnc1C(=O)O', 'acidic'),
    ('c1cccnc1-COOMe', '[*]c1cccnc1C(=O)OC', 'polar'),
    ('c1cccnc1-SO2Me', '[*]c1cccnc1S(=O)(=O)C', 'polar'),
    ('c1cccnc1-SO2NH2', '[*]c1cccnc1S(=O)(=O)N', 'polar'),
    ('c1cccnc1-morpholino', '[*]c1cccnc1N1CCOCC1', 'basic'),
    ('c1cccnc1-piperidino', '[*]c1cccnc1N1CCCCC1', 'basic'),
    ('c1cccnc1-pyrrolidino', '[*]c1cccnc1N1CCCC1', 'basic'),
    ('c1cccnc1-N-methylpiperazino', '[*]c1cccnc1N1CCN(C)CC1', 'basic'),
    ('c1cccnc1-azetidino', '[*]c1cccnc1N1CCC1', 'basic'),
    ('c1cccnc1-tetrazolyl', '[*]c1cccnc1c1nnn[nH]1', 'acidic'),
    ('c1cccnc1-oxadiazolyl', '[*]c1cccnc1c1nnco1', 'bioisostere'),
    ('c1cccnc1-imidazolyl', '[*]c1cccnc1c1ccn[nH]1', 'basic'),
    ('c1cccnc1-oxetanyl', '[*]c1cccnc1C1COC1', 'bioisostere'),
    ('piperazine-ethyl', '[*]CCN1CCNCC1', 'basic'),
    ('azepane-methyl', '[*]CN1CCCCCC1', 'basic'),
    ('azepane-ethyl', '[*]CCN1CCCCCC1', 'basic'),
    ('1,4-oxazepane-methyl', '[*]CN1CCOCCC1', 'basic'),
    ('1,4-oxazepane-ethyl', '[*]CCN1CCOCCC1', 'basic'),
    ('carbonyl-NHtBu_2', '[*]C(=O)NC(C)(C)C', 'polar'),
    ('carbonyl-NHcPr_2', '[*]C(=O)NC1CC1', 'polar'),
    ('carbonyl-NHcBu', '[*]C(=O)NC1CCC1', 'polar'),
    ('carbonyl-NHcPent_2', '[*]C(=O)NC1CCCC1', 'polar'),
    ('carbonyl-NHcHex_2', '[*]C(=O)NC1CCCCC1', 'polar'),
    ('carbonyl-NMeEt', '[*]C(=O)N(C)CC', 'polar'),
    ('carbonyl-NHBn_2', '[*]C(=O)NCc1ccccc1', 'polar'),
    ('carbonyl-NH(4-FBn)', '[*]C(=O)NCc1ccc(F)cc1', 'polar'),
    ('carbonyl-NH(4-ClBn)', '[*]C(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('carbonyl-NH(4-MeBn)', '[*]C(=O)NCc1ccc(C)cc1', 'polar'),
    ('carbonyl-azepanyl', '[*]C(=O)N1CCCCCC1', 'polar'),
    ('carbonyl-NHOMe', '[*]C(=O)NOC', 'polar'),
    ('carbonyl-NHOEt', '[*]C(=O)NOCC', 'polar'),
    ('carbonyl-NH(2-pyridyl)', '[*]C(=O)Nc1ccccn1', 'polar'),
    ('carbonyl-NH(3-pyridyl)', '[*]C(=O)Nc1cccnc1', 'polar'),
    ('carbonyl-NH(4-pyridyl)', '[*]C(=O)Nc1ccncc1', 'polar'),
    ('carbonyl-NH(pyrimidinyl)', '[*]C(=O)Nc1ncccn1', 'polar'),
    ('carbonyl-NH(thiazolyl)', '[*]C(=O)Nc1nccs1', 'polar'),
    ('carbonyl-NHPh', '[*]C(=O)Nc1ccccc1', 'polar'),
    ('carbonyl-NHMe-OH', '[*]C(=O)N(C)CCO', 'polar'),
    ('carbonyl-N(Me)(Bn)', '[*]C(=O)N(C)Cc1ccccc1', 'polar'),
    ('carbonyl-NH(2-HOEt)', '[*]C(=O)NCCO', 'polar'),
    ('carbonyl-N(2-HOEt)2', '[*]C(=O)N(CCO)CCO', 'polar'),
    ('carbonyl-3-hydroxypyrrolidinyl', '[*]C(=O)N1CCC(O)C1', 'polar'),
    ('carbonyl-4-hydroxypiperidyl', '[*]C(=O)N1CCC(O)CC1', 'polar'),
    ('carbonyl-3-fluoropyrrolidinyl', '[*]C(=O)N1CCC(F)C1', 'polar'),
    ('carbonyl-3-methylpiperidyl', '[*]C(=O)N1CCCC(C)C1', 'polar'),
    ('carbonyl-4-methylpiperidyl', '[*]C(=O)N1CCC(C)CC1', 'polar'),
    ('C-carbonyl-NH2', '[*]CC(=O)N', 'polar'),
    ('C-carbonyl-NHMe', '[*]CC(=O)NC', 'polar'),
    ('C-carbonyl-NHEt', '[*]CC(=O)NCC', 'polar'),
    ('C-carbonyl-NHiPr', '[*]CC(=O)NC(C)C', 'polar'),
    ('C-carbonyl-NHtBu', '[*]CC(=O)NC(C)(C)C', 'polar'),
    ('C-carbonyl-NHcPr', '[*]CC(=O)NC1CC1', 'polar'),
    ('C-carbonyl-NHcBu', '[*]CC(=O)NC1CCC1', 'polar'),
    ('C-carbonyl-NHcPent', '[*]CC(=O)NC1CCCC1', 'polar'),
    ('C-carbonyl-NHcHex', '[*]CC(=O)NC1CCCCC1', 'polar'),
    ('C-carbonyl-NMe2', '[*]CC(=O)N(C)C', 'polar'),
    ('C-carbonyl-NEt2', '[*]CC(=O)N(CC)CC', 'polar'),
    ('C-carbonyl-NMeEt', '[*]CC(=O)N(C)CC', 'polar'),
    ('C-carbonyl-NHBn', '[*]CC(=O)NCc1ccccc1', 'polar'),
    ('C-carbonyl-NH(4-FBn)', '[*]CC(=O)NCc1ccc(F)cc1', 'polar'),
    ('C-carbonyl-NH(4-ClBn)', '[*]CC(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('C-carbonyl-NH(4-MeBn)', '[*]CC(=O)NCc1ccc(C)cc1', 'polar'),
    ('C-carbonyl-piperidyl', '[*]CC(=O)N1CCCCC1', 'polar'),
    ('C-carbonyl-pyrrolidyl', '[*]CC(=O)N1CCCC1', 'polar'),
    ('C-carbonyl-morpholyl', '[*]CC(=O)N1CCOCC1', 'polar'),
    ('C-carbonyl-azetidinyl', '[*]CC(=O)N1CCC1', 'polar'),
    ('C-carbonyl-N-Me-piperazinyl', '[*]CC(=O)N1CCN(C)CC1', 'polar'),
    ('C-carbonyl-azepanyl', '[*]CC(=O)N1CCCCCC1', 'polar'),
    ('C-carbonyl-NHOMe', '[*]CC(=O)NOC', 'polar'),
    ('C-carbonyl-NHOEt', '[*]CC(=O)NOCC', 'polar'),
    ('C-carbonyl-NH(2-pyridyl)', '[*]CC(=O)Nc1ccccn1', 'polar'),
    ('C-carbonyl-NH(3-pyridyl)', '[*]CC(=O)Nc1cccnc1', 'polar'),
    ('C-carbonyl-NH(4-pyridyl)', '[*]CC(=O)Nc1ccncc1', 'polar'),
    ('C-carbonyl-NH(pyrimidinyl)', '[*]CC(=O)Nc1ncccn1', 'polar'),
    ('C-carbonyl-NH(thiazolyl)', '[*]CC(=O)Nc1nccs1', 'polar'),
    ('C-carbonyl-NHPh', '[*]CC(=O)Nc1ccccc1', 'polar'),
    ('C-carbonyl-NHMe-OH', '[*]CC(=O)N(C)CCO', 'polar'),
    ('C-carbonyl-N(Me)(Bn)', '[*]CC(=O)N(C)Cc1ccccc1', 'polar'),
    ('C-carbonyl-NH(2-HOEt)', '[*]CC(=O)NCCO', 'polar'),
    ('C-carbonyl-N(2-HOEt)2', '[*]CC(=O)N(CCO)CCO', 'polar'),
    ('C-carbonyl-3-hydroxypyrrolidinyl', '[*]CC(=O)N1CCC(O)C1', 'polar'),
    ('C-carbonyl-4-hydroxypiperidyl', '[*]CC(=O)N1CCC(O)CC1', 'polar'),
    ('C-carbonyl-3-fluoropyrrolidinyl', '[*]CC(=O)N1CCC(F)C1', 'polar'),
    ('C-carbonyl-3-methylpiperidyl', '[*]CC(=O)N1CCCC(C)C1', 'polar'),
    ('C-carbonyl-4-methylpiperidyl', '[*]CC(=O)N1CCC(C)CC1', 'polar'),
    ('CC-carbonyl-NH2', '[*]CCC(=O)N', 'polar'),
    ('CC-carbonyl-NHMe', '[*]CCC(=O)NC', 'polar'),
    ('CC-carbonyl-NHEt', '[*]CCC(=O)NCC', 'polar'),
    ('CC-carbonyl-NHiPr', '[*]CCC(=O)NC(C)C', 'polar'),
    ('CC-carbonyl-NHtBu', '[*]CCC(=O)NC(C)(C)C', 'polar'),
    ('CC-carbonyl-NHcPr', '[*]CCC(=O)NC1CC1', 'polar'),
    ('CC-carbonyl-NHcBu', '[*]CCC(=O)NC1CCC1', 'polar'),
    ('CC-carbonyl-NHcPent', '[*]CCC(=O)NC1CCCC1', 'polar'),
    ('CC-carbonyl-NHcHex', '[*]CCC(=O)NC1CCCCC1', 'polar'),
    ('CC-carbonyl-NMe2', '[*]CCC(=O)N(C)C', 'polar'),
    ('CC-carbonyl-NEt2', '[*]CCC(=O)N(CC)CC', 'polar'),
    ('CC-carbonyl-NMeEt', '[*]CCC(=O)N(C)CC', 'polar'),
    ('CC-carbonyl-NHBn', '[*]CCC(=O)NCc1ccccc1', 'polar'),
    ('CC-carbonyl-NH(4-FBn)', '[*]CCC(=O)NCc1ccc(F)cc1', 'polar'),
    ('CC-carbonyl-NH(4-ClBn)', '[*]CCC(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('CC-carbonyl-NH(4-MeBn)', '[*]CCC(=O)NCc1ccc(C)cc1', 'polar'),
    ('CC-carbonyl-piperidyl', '[*]CCC(=O)N1CCCCC1', 'polar'),
    ('CC-carbonyl-pyrrolidyl', '[*]CCC(=O)N1CCCC1', 'polar'),
    ('CC-carbonyl-morpholyl', '[*]CCC(=O)N1CCOCC1', 'polar'),
    ('CC-carbonyl-azetidinyl', '[*]CCC(=O)N1CCC1', 'polar'),
    ('CC-carbonyl-N-Me-piperazinyl', '[*]CCC(=O)N1CCN(C)CC1', 'polar'),
    ('CC-carbonyl-azepanyl', '[*]CCC(=O)N1CCCCCC1', 'polar'),
    ('CC-carbonyl-NHOMe', '[*]CCC(=O)NOC', 'polar'),
    ('CC-carbonyl-NHOEt', '[*]CCC(=O)NOCC', 'polar'),
    ('CC-carbonyl-NH(2-pyridyl)', '[*]CCC(=O)Nc1ccccn1', 'polar'),
    ('CC-carbonyl-NH(3-pyridyl)', '[*]CCC(=O)Nc1cccnc1', 'polar'),
    ('CC-carbonyl-NH(4-pyridyl)', '[*]CCC(=O)Nc1ccncc1', 'polar'),
    ('CC-carbonyl-NH(pyrimidinyl)', '[*]CCC(=O)Nc1ncccn1', 'polar'),
    ('CC-carbonyl-NH(thiazolyl)', '[*]CCC(=O)Nc1nccs1', 'polar'),
    ('CC-carbonyl-NHPh', '[*]CCC(=O)Nc1ccccc1', 'polar'),
    ('CC-carbonyl-NHMe-OH', '[*]CCC(=O)N(C)CCO', 'polar'),
    ('CC-carbonyl-N(Me)(Bn)', '[*]CCC(=O)N(C)Cc1ccccc1', 'polar'),
    ('CC-carbonyl-NH(2-HOEt)', '[*]CCC(=O)NCCO', 'polar'),
    ('CC-carbonyl-N(2-HOEt)2', '[*]CCC(=O)N(CCO)CCO', 'polar'),
    ('CC-carbonyl-3-hydroxypyrrolidinyl', '[*]CCC(=O)N1CCC(O)C1', 'polar'),
    ('CC-carbonyl-4-hydroxypiperidyl', '[*]CCC(=O)N1CCC(O)CC1', 'polar'),
    ('CC-carbonyl-3-fluoropyrrolidinyl', '[*]CCC(=O)N1CCC(F)C1', 'polar'),
    ('CC-carbonyl-3-methylpiperidyl', '[*]CCC(=O)N1CCCC(C)C1', 'polar'),
    ('CC-carbonyl-4-methylpiperidyl', '[*]CCC(=O)N1CCC(C)CC1', 'polar'),
    ('CCC-carbonyl-NH2', '[*]CCCC(=O)N', 'polar'),
    ('CCC-carbonyl-NHMe', '[*]CCCC(=O)NC', 'polar'),
    ('CCC-carbonyl-NHEt', '[*]CCCC(=O)NCC', 'polar'),
    ('CCC-carbonyl-NHiPr', '[*]CCCC(=O)NC(C)C', 'polar'),
    ('CCC-carbonyl-NHtBu', '[*]CCCC(=O)NC(C)(C)C', 'polar'),
    ('CCC-carbonyl-NHcPr', '[*]CCCC(=O)NC1CC1', 'polar'),
    ('CCC-carbonyl-NHcBu', '[*]CCCC(=O)NC1CCC1', 'polar'),
    ('CCC-carbonyl-NHcPent', '[*]CCCC(=O)NC1CCCC1', 'polar'),
    ('CCC-carbonyl-NHcHex', '[*]CCCC(=O)NC1CCCCC1', 'polar'),
    ('CCC-carbonyl-NMe2', '[*]CCCC(=O)N(C)C', 'polar'),
    ('CCC-carbonyl-NEt2', '[*]CCCC(=O)N(CC)CC', 'polar'),
    ('CCC-carbonyl-NMeEt', '[*]CCCC(=O)N(C)CC', 'polar'),
    ('CCC-carbonyl-NHBn', '[*]CCCC(=O)NCc1ccccc1', 'polar'),
    ('CCC-carbonyl-NH(4-FBn)', '[*]CCCC(=O)NCc1ccc(F)cc1', 'polar'),
    ('CCC-carbonyl-NH(4-ClBn)', '[*]CCCC(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('CCC-carbonyl-NH(4-MeBn)', '[*]CCCC(=O)NCc1ccc(C)cc1', 'polar'),
    ('CCC-carbonyl-piperidyl', '[*]CCCC(=O)N1CCCCC1', 'polar'),
    ('CCC-carbonyl-pyrrolidyl', '[*]CCCC(=O)N1CCCC1', 'polar'),
    ('CCC-carbonyl-morpholyl', '[*]CCCC(=O)N1CCOCC1', 'polar'),
    ('CCC-carbonyl-azetidinyl', '[*]CCCC(=O)N1CCC1', 'polar'),
    ('CCC-carbonyl-N-Me-piperazinyl', '[*]CCCC(=O)N1CCN(C)CC1', 'polar'),
    ('CCC-carbonyl-azepanyl', '[*]CCCC(=O)N1CCCCCC1', 'polar'),
    ('CCC-carbonyl-NHOMe', '[*]CCCC(=O)NOC', 'polar'),
    ('CCC-carbonyl-NHOEt', '[*]CCCC(=O)NOCC', 'polar'),
    ('CCC-carbonyl-NH(2-pyridyl)', '[*]CCCC(=O)Nc1ccccn1', 'polar'),
    ('CCC-carbonyl-NH(3-pyridyl)', '[*]CCCC(=O)Nc1cccnc1', 'polar'),
    ('CCC-carbonyl-NH(4-pyridyl)', '[*]CCCC(=O)Nc1ccncc1', 'polar'),
    ('CCC-carbonyl-NH(pyrimidinyl)', '[*]CCCC(=O)Nc1ncccn1', 'polar'),
    ('CCC-carbonyl-NH(thiazolyl)', '[*]CCCC(=O)Nc1nccs1', 'polar'),
    ('CCC-carbonyl-NHPh', '[*]CCCC(=O)Nc1ccccc1', 'polar'),
    ('CCC-carbonyl-NHMe-OH', '[*]CCCC(=O)N(C)CCO', 'polar'),
    ('CCC-carbonyl-N(Me)(Bn)', '[*]CCCC(=O)N(C)Cc1ccccc1', 'polar'),
    ('CCC-carbonyl-NH(2-HOEt)', '[*]CCCC(=O)NCCO', 'polar'),
    ('CCC-carbonyl-N(2-HOEt)2', '[*]CCCC(=O)N(CCO)CCO', 'polar'),
    ('CCC-carbonyl-3-hydroxypyrrolidinyl', '[*]CCCC(=O)N1CCC(O)C1', 'polar'),
    ('CCC-carbonyl-4-hydroxypiperidyl', '[*]CCCC(=O)N1CCC(O)CC1', 'polar'),
    ('CCC-carbonyl-3-fluoropyrrolidinyl', '[*]CCCC(=O)N1CCC(F)C1', 'polar'),
    ('CCC-carbonyl-3-methylpiperidyl', '[*]CCCC(=O)N1CCCC(C)C1', 'polar'),
    ('CCC-carbonyl-4-methylpiperidyl', '[*]CCCC(=O)N1CCC(C)CC1', 'polar'),
    ('C1CC1-carbonyl-NH2', '[*]C1CC1C(=O)N', 'polar'),
    ('C1CC1-carbonyl-NHMe', '[*]C1CC1C(=O)NC', 'polar'),
    ('C1CC1-carbonyl-NHEt', '[*]C1CC1C(=O)NCC', 'polar'),
    ('C1CC1-carbonyl-NHiPr', '[*]C1CC1C(=O)NC(C)C', 'polar'),
    ('C1CC1-carbonyl-NHtBu', '[*]C1CC1C(=O)NC(C)(C)C', 'polar'),
    ('C1CC1-carbonyl-NHcPr', '[*]C1CC1C(=O)NC1CC1', 'polar'),
    ('C1CC1-carbonyl-NHcBu', '[*]C1CC1C(=O)NC1CCC1', 'polar'),
    ('C1CC1-carbonyl-NHcPent', '[*]C1CC1C(=O)NC1CCCC1', 'polar'),
    ('C1CC1-carbonyl-NHcHex', '[*]C1CC1C(=O)NC1CCCCC1', 'polar'),
    ('C1CC1-carbonyl-NMe2', '[*]C1CC1C(=O)N(C)C', 'polar'),
    ('C1CC1-carbonyl-NEt2', '[*]C1CC1C(=O)N(CC)CC', 'polar'),
    ('C1CC1-carbonyl-NMeEt', '[*]C1CC1C(=O)N(C)CC', 'polar'),
    ('C1CC1-carbonyl-NHBn', '[*]C1CC1C(=O)NCc1ccccc1', 'polar'),
    ('C1CC1-carbonyl-NH(4-FBn)', '[*]C1CC1C(=O)NCc1ccc(F)cc1', 'polar'),
    ('C1CC1-carbonyl-NH(4-ClBn)', '[*]C1CC1C(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('C1CC1-carbonyl-NH(4-MeBn)', '[*]C1CC1C(=O)NCc1ccc(C)cc1', 'polar'),
    ('C1CC1-carbonyl-piperidyl', '[*]C1CC1C(=O)N1CCCCC1', 'polar'),
    ('C1CC1-carbonyl-pyrrolidyl', '[*]C1CC1C(=O)N1CCCC1', 'polar'),
    ('C1CC1-carbonyl-morpholyl', '[*]C1CC1C(=O)N1CCOCC1', 'polar'),
    ('C1CC1-carbonyl-azetidinyl', '[*]C1CC1C(=O)N1CCC1', 'polar'),
    ('C1CC1-carbonyl-N-Me-piperazinyl', '[*]C1CC1C(=O)N1CCN(C)CC1', 'polar'),
    ('C1CC1-carbonyl-azepanyl', '[*]C1CC1C(=O)N1CCCCCC1', 'polar'),
    ('C1CC1-carbonyl-NHOMe', '[*]C1CC1C(=O)NOC', 'polar'),
    ('C1CC1-carbonyl-NHOEt', '[*]C1CC1C(=O)NOCC', 'polar'),
    ('C1CC1-carbonyl-NH(2-pyridyl)', '[*]C1CC1C(=O)Nc1ccccn1', 'polar'),
    ('C1CC1-carbonyl-NH(3-pyridyl)', '[*]C1CC1C(=O)Nc1cccnc1', 'polar'),
    ('C1CC1-carbonyl-NH(4-pyridyl)', '[*]C1CC1C(=O)Nc1ccncc1', 'polar'),
    ('C1CC1-carbonyl-NH(pyrimidinyl)', '[*]C1CC1C(=O)Nc1ncccn1', 'polar'),
    ('C1CC1-carbonyl-NH(thiazolyl)', '[*]C1CC1C(=O)Nc1nccs1', 'polar'),
    ('C1CC1-carbonyl-NHPh', '[*]C1CC1C(=O)Nc1ccccc1', 'polar'),
    ('C1CC1-carbonyl-NHMe-OH', '[*]C1CC1C(=O)N(C)CCO', 'polar'),
    ('C1CC1-carbonyl-N(Me)(Bn)', '[*]C1CC1C(=O)N(C)Cc1ccccc1', 'polar'),
    ('C1CC1-carbonyl-NH(2-HOEt)', '[*]C1CC1C(=O)NCCO', 'polar'),
    ('C1CC1-carbonyl-N(2-HOEt)2', '[*]C1CC1C(=O)N(CCO)CCO', 'polar'),
    ('C1CC1-carbonyl-3-hydroxypyrrolidinyl', '[*]C1CC1C(=O)N1CCC(O)C1', 'polar'),
    ('C1CC1-carbonyl-4-hydroxypiperidyl', '[*]C1CC1C(=O)N1CCC(O)CC1', 'polar'),
    ('C1CC1-carbonyl-3-fluoropyrrolidinyl', '[*]C1CC1C(=O)N1CCC(F)C1', 'polar'),
    ('C1CC1-carbonyl-3-methylpiperidyl', '[*]C1CC1C(=O)N1CCCC(C)C1', 'polar'),
    ('C1CC1-carbonyl-4-methylpiperidyl', '[*]C1CC1C(=O)N1CCC(C)CC1', 'polar'),
    ('C1CCC1-carbonyl-NH2', '[*]C1CCC1C(=O)N', 'polar'),
    ('C1CCC1-carbonyl-NHMe', '[*]C1CCC1C(=O)NC', 'polar'),
    ('C1CCC1-carbonyl-NHEt', '[*]C1CCC1C(=O)NCC', 'polar'),
    ('C1CCC1-carbonyl-NHiPr', '[*]C1CCC1C(=O)NC(C)C', 'polar'),
    ('C1CCC1-carbonyl-NHtBu', '[*]C1CCC1C(=O)NC(C)(C)C', 'polar'),
    ('C1CCC1-carbonyl-NHcPr', '[*]C1CCC1C(=O)NC1CC1', 'polar'),
    ('C1CCC1-carbonyl-NHcBu', '[*]C1CCC1C(=O)NC1CCC1', 'polar'),
    ('C1CCC1-carbonyl-NHcPent', '[*]C1CCC1C(=O)NC1CCCC1', 'polar'),
    ('C1CCC1-carbonyl-NHcHex', '[*]C1CCC1C(=O)NC1CCCCC1', 'polar'),
    ('C1CCC1-carbonyl-NMe2', '[*]C1CCC1C(=O)N(C)C', 'polar'),
    ('C1CCC1-carbonyl-NEt2', '[*]C1CCC1C(=O)N(CC)CC', 'polar'),
    ('C1CCC1-carbonyl-NMeEt', '[*]C1CCC1C(=O)N(C)CC', 'polar'),
    ('C1CCC1-carbonyl-NHBn', '[*]C1CCC1C(=O)NCc1ccccc1', 'polar'),
    ('C1CCC1-carbonyl-NH(4-FBn)', '[*]C1CCC1C(=O)NCc1ccc(F)cc1', 'polar'),
    ('C1CCC1-carbonyl-NH(4-ClBn)', '[*]C1CCC1C(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('C1CCC1-carbonyl-NH(4-MeBn)', '[*]C1CCC1C(=O)NCc1ccc(C)cc1', 'polar'),
    ('C1CCC1-carbonyl-piperidyl', '[*]C1CCC1C(=O)N1CCCCC1', 'polar'),
    ('C1CCC1-carbonyl-pyrrolidyl', '[*]C1CCC1C(=O)N1CCCC1', 'polar'),
    ('C1CCC1-carbonyl-morpholyl', '[*]C1CCC1C(=O)N1CCOCC1', 'polar'),
    ('C1CCC1-carbonyl-azetidinyl', '[*]C1CCC1C(=O)N1CCC1', 'polar'),
    ('C1CCC1-carbonyl-N-Me-piperazinyl', '[*]C1CCC1C(=O)N1CCN(C)CC1', 'polar'),
    ('C1CCC1-carbonyl-azepanyl', '[*]C1CCC1C(=O)N1CCCCCC1', 'polar'),
    ('C1CCC1-carbonyl-NHOMe', '[*]C1CCC1C(=O)NOC', 'polar'),
    ('C1CCC1-carbonyl-NHOEt', '[*]C1CCC1C(=O)NOCC', 'polar'),
    ('C1CCC1-carbonyl-NH(2-pyridyl)', '[*]C1CCC1C(=O)Nc1ccccn1', 'polar'),
    ('C1CCC1-carbonyl-NH(3-pyridyl)', '[*]C1CCC1C(=O)Nc1cccnc1', 'polar'),
    ('C1CCC1-carbonyl-NH(4-pyridyl)', '[*]C1CCC1C(=O)Nc1ccncc1', 'polar'),
    ('C1CCC1-carbonyl-NH(pyrimidinyl)', '[*]C1CCC1C(=O)Nc1ncccn1', 'polar'),
    ('C1CCC1-carbonyl-NH(thiazolyl)', '[*]C1CCC1C(=O)Nc1nccs1', 'polar'),
    ('C1CCC1-carbonyl-NHPh', '[*]C1CCC1C(=O)Nc1ccccc1', 'polar'),
    ('C1CCC1-carbonyl-NHMe-OH', '[*]C1CCC1C(=O)N(C)CCO', 'polar'),
    ('C1CCC1-carbonyl-N(Me)(Bn)', '[*]C1CCC1C(=O)N(C)Cc1ccccc1', 'polar'),
    ('C1CCC1-carbonyl-NH(2-HOEt)', '[*]C1CCC1C(=O)NCCO', 'polar'),
    ('C1CCC1-carbonyl-N(2-HOEt)2', '[*]C1CCC1C(=O)N(CCO)CCO', 'polar'),
    ('C1CCC1-carbonyl-3-hydroxypyrrolidinyl', '[*]C1CCC1C(=O)N1CCC(O)C1', 'polar'),
    ('C1CCC1-carbonyl-4-hydroxypiperidyl', '[*]C1CCC1C(=O)N1CCC(O)CC1', 'polar'),
    ('C1CCC1-carbonyl-3-fluoropyrrolidinyl', '[*]C1CCC1C(=O)N1CCC(F)C1', 'polar'),
    ('C1CCC1-carbonyl-3-methylpiperidyl', '[*]C1CCC1C(=O)N1CCCC(C)C1', 'polar'),
    ('C1CCC1-carbonyl-4-methylpiperidyl', '[*]C1CCC1C(=O)N1CCC(C)CC1', 'polar'),
    ('CO-carbonyl-NH2', '[*]COC(=O)N', 'polar'),
    ('CO-carbonyl-NHMe', '[*]COC(=O)NC', 'polar'),
    ('CO-carbonyl-NHEt', '[*]COC(=O)NCC', 'polar'),
    ('CO-carbonyl-NHiPr', '[*]COC(=O)NC(C)C', 'polar'),
    ('CO-carbonyl-NHtBu', '[*]COC(=O)NC(C)(C)C', 'polar'),
    ('CO-carbonyl-NHcPr', '[*]COC(=O)NC1CC1', 'polar'),
    ('CO-carbonyl-NHcBu', '[*]COC(=O)NC1CCC1', 'polar'),
    ('CO-carbonyl-NHcPent', '[*]COC(=O)NC1CCCC1', 'polar'),
    ('CO-carbonyl-NHcHex', '[*]COC(=O)NC1CCCCC1', 'polar'),
    ('CO-carbonyl-NMe2', '[*]COC(=O)N(C)C', 'polar'),
    ('CO-carbonyl-NEt2', '[*]COC(=O)N(CC)CC', 'polar'),
    ('CO-carbonyl-NMeEt', '[*]COC(=O)N(C)CC', 'polar'),
    ('CO-carbonyl-NHBn', '[*]COC(=O)NCc1ccccc1', 'polar'),
    ('CO-carbonyl-NH(4-FBn)', '[*]COC(=O)NCc1ccc(F)cc1', 'polar'),
    ('CO-carbonyl-NH(4-ClBn)', '[*]COC(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('CO-carbonyl-NH(4-MeBn)', '[*]COC(=O)NCc1ccc(C)cc1', 'polar'),
    ('CO-carbonyl-piperidyl', '[*]COC(=O)N1CCCCC1', 'polar'),
    ('CO-carbonyl-pyrrolidyl', '[*]COC(=O)N1CCCC1', 'polar'),
    ('CO-carbonyl-morpholyl', '[*]COC(=O)N1CCOCC1', 'polar'),
    ('CO-carbonyl-azetidinyl', '[*]COC(=O)N1CCC1', 'polar'),
    ('CO-carbonyl-N-Me-piperazinyl', '[*]COC(=O)N1CCN(C)CC1', 'polar'),
    ('CO-carbonyl-azepanyl', '[*]COC(=O)N1CCCCCC1', 'polar'),
    ('CO-carbonyl-NHOMe', '[*]COC(=O)NOC', 'polar'),
    ('CO-carbonyl-NHOEt', '[*]COC(=O)NOCC', 'polar'),
    ('CO-carbonyl-NH(2-pyridyl)', '[*]COC(=O)Nc1ccccn1', 'polar'),
    ('CO-carbonyl-NH(3-pyridyl)', '[*]COC(=O)Nc1cccnc1', 'polar'),
    ('CO-carbonyl-NH(4-pyridyl)', '[*]COC(=O)Nc1ccncc1', 'polar'),
    ('CO-carbonyl-NH(pyrimidinyl)', '[*]COC(=O)Nc1ncccn1', 'polar'),
    ('CO-carbonyl-NH(thiazolyl)', '[*]COC(=O)Nc1nccs1', 'polar'),
    ('CO-carbonyl-NHPh', '[*]COC(=O)Nc1ccccc1', 'polar'),
    ('CO-carbonyl-NHMe-OH', '[*]COC(=O)N(C)CCO', 'polar'),
    ('CO-carbonyl-N(Me)(Bn)', '[*]COC(=O)N(C)Cc1ccccc1', 'polar'),
    ('CO-carbonyl-NH(2-HOEt)', '[*]COC(=O)NCCO', 'polar'),
    ('CO-carbonyl-N(2-HOEt)2', '[*]COC(=O)N(CCO)CCO', 'polar'),
    ('CO-carbonyl-3-hydroxypyrrolidinyl', '[*]COC(=O)N1CCC(O)C1', 'polar'),
    ('CO-carbonyl-4-hydroxypiperidyl', '[*]COC(=O)N1CCC(O)CC1', 'polar'),
    ('CO-carbonyl-3-fluoropyrrolidinyl', '[*]COC(=O)N1CCC(F)C1', 'polar'),
    ('CO-carbonyl-3-methylpiperidyl', '[*]COC(=O)N1CCCC(C)C1', 'polar'),
    ('CO-carbonyl-4-methylpiperidyl', '[*]COC(=O)N1CCC(C)CC1', 'polar'),
    ('CCO-carbonyl-NH2', '[*]CCOC(=O)N', 'polar'),
    ('CCO-carbonyl-NHMe', '[*]CCOC(=O)NC', 'polar'),
    ('CCO-carbonyl-NHEt', '[*]CCOC(=O)NCC', 'polar'),
    ('CCO-carbonyl-NHiPr', '[*]CCOC(=O)NC(C)C', 'polar'),
    ('CCO-carbonyl-NHtBu', '[*]CCOC(=O)NC(C)(C)C', 'polar'),
    ('CCO-carbonyl-NHcPr', '[*]CCOC(=O)NC1CC1', 'polar'),
    ('CCO-carbonyl-NHcBu', '[*]CCOC(=O)NC1CCC1', 'polar'),
    ('CCO-carbonyl-NHcPent', '[*]CCOC(=O)NC1CCCC1', 'polar'),
    ('CCO-carbonyl-NHcHex', '[*]CCOC(=O)NC1CCCCC1', 'polar'),
    ('CCO-carbonyl-NMe2', '[*]CCOC(=O)N(C)C', 'polar'),
    ('CCO-carbonyl-NEt2', '[*]CCOC(=O)N(CC)CC', 'polar'),
    ('CCO-carbonyl-NMeEt', '[*]CCOC(=O)N(C)CC', 'polar'),
    ('CCO-carbonyl-NHBn', '[*]CCOC(=O)NCc1ccccc1', 'polar'),
    ('CCO-carbonyl-NH(4-FBn)', '[*]CCOC(=O)NCc1ccc(F)cc1', 'polar'),
    ('CCO-carbonyl-NH(4-ClBn)', '[*]CCOC(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('CCO-carbonyl-NH(4-MeBn)', '[*]CCOC(=O)NCc1ccc(C)cc1', 'polar'),
    ('CCO-carbonyl-piperidyl', '[*]CCOC(=O)N1CCCCC1', 'polar'),
    ('CCO-carbonyl-pyrrolidyl', '[*]CCOC(=O)N1CCCC1', 'polar'),
    ('CCO-carbonyl-morpholyl', '[*]CCOC(=O)N1CCOCC1', 'polar'),
    ('CCO-carbonyl-azetidinyl', '[*]CCOC(=O)N1CCC1', 'polar'),
    ('CCO-carbonyl-N-Me-piperazinyl', '[*]CCOC(=O)N1CCN(C)CC1', 'polar'),
    ('CCO-carbonyl-azepanyl', '[*]CCOC(=O)N1CCCCCC1', 'polar'),
    ('CCO-carbonyl-NHOMe', '[*]CCOC(=O)NOC', 'polar'),
    ('CCO-carbonyl-NHOEt', '[*]CCOC(=O)NOCC', 'polar'),
    ('CCO-carbonyl-NH(2-pyridyl)', '[*]CCOC(=O)Nc1ccccn1', 'polar'),
    ('CCO-carbonyl-NH(3-pyridyl)', '[*]CCOC(=O)Nc1cccnc1', 'polar'),
    ('CCO-carbonyl-NH(4-pyridyl)', '[*]CCOC(=O)Nc1ccncc1', 'polar'),
    ('CCO-carbonyl-NH(pyrimidinyl)', '[*]CCOC(=O)Nc1ncccn1', 'polar'),
    ('CCO-carbonyl-NH(thiazolyl)', '[*]CCOC(=O)Nc1nccs1', 'polar'),
    ('CCO-carbonyl-NHPh', '[*]CCOC(=O)Nc1ccccc1', 'polar'),
    ('CCO-carbonyl-NHMe-OH', '[*]CCOC(=O)N(C)CCO', 'polar'),
    ('CCO-carbonyl-N(Me)(Bn)', '[*]CCOC(=O)N(C)Cc1ccccc1', 'polar'),
    ('CCO-carbonyl-NH(2-HOEt)', '[*]CCOC(=O)NCCO', 'polar'),
    ('CCO-carbonyl-N(2-HOEt)2', '[*]CCOC(=O)N(CCO)CCO', 'polar'),
    ('CCO-carbonyl-3-hydroxypyrrolidinyl', '[*]CCOC(=O)N1CCC(O)C1', 'polar'),
    ('CCO-carbonyl-4-hydroxypiperidyl', '[*]CCOC(=O)N1CCC(O)CC1', 'polar'),
    ('CCO-carbonyl-3-fluoropyrrolidinyl', '[*]CCOC(=O)N1CCC(F)C1', 'polar'),
    ('CCO-carbonyl-3-methylpiperidyl', '[*]CCOC(=O)N1CCCC(C)C1', 'polar'),
    ('CCO-carbonyl-4-methylpiperidyl', '[*]CCOC(=O)N1CCC(C)CC1', 'polar'),
    ('C(C)-carbonyl-NH2', '[*]C(C)C(=O)N', 'polar'),
    ('C(C)-carbonyl-NHMe', '[*]C(C)C(=O)NC', 'polar'),
    ('C(C)-carbonyl-NHEt', '[*]C(C)C(=O)NCC', 'polar'),
    ('C(C)-carbonyl-NHiPr', '[*]C(C)C(=O)NC(C)C', 'polar'),
    ('C(C)-carbonyl-NHtBu', '[*]C(C)C(=O)NC(C)(C)C', 'polar'),
    ('C(C)-carbonyl-NHcPr', '[*]C(C)C(=O)NC1CC1', 'polar'),
    ('C(C)-carbonyl-NHcBu', '[*]C(C)C(=O)NC1CCC1', 'polar'),
    ('C(C)-carbonyl-NHcPent', '[*]C(C)C(=O)NC1CCCC1', 'polar'),
    ('C(C)-carbonyl-NHcHex', '[*]C(C)C(=O)NC1CCCCC1', 'polar'),
    ('C(C)-carbonyl-NMe2', '[*]C(C)C(=O)N(C)C', 'polar'),
    ('C(C)-carbonyl-NEt2', '[*]C(C)C(=O)N(CC)CC', 'polar'),
    ('C(C)-carbonyl-NMeEt', '[*]C(C)C(=O)N(C)CC', 'polar'),
    ('C(C)-carbonyl-NHBn', '[*]C(C)C(=O)NCc1ccccc1', 'polar'),
    ('C(C)-carbonyl-NH(4-FBn)', '[*]C(C)C(=O)NCc1ccc(F)cc1', 'polar'),
    ('C(C)-carbonyl-NH(4-ClBn)', '[*]C(C)C(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('C(C)-carbonyl-NH(4-MeBn)', '[*]C(C)C(=O)NCc1ccc(C)cc1', 'polar'),
    ('C(C)-carbonyl-piperidyl', '[*]C(C)C(=O)N1CCCCC1', 'polar'),
    ('C(C)-carbonyl-pyrrolidyl', '[*]C(C)C(=O)N1CCCC1', 'polar'),
    ('C(C)-carbonyl-morpholyl', '[*]C(C)C(=O)N1CCOCC1', 'polar'),
    ('C(C)-carbonyl-azetidinyl', '[*]C(C)C(=O)N1CCC1', 'polar'),
    ('C(C)-carbonyl-N-Me-piperazinyl', '[*]C(C)C(=O)N1CCN(C)CC1', 'polar'),
    ('C(C)-carbonyl-azepanyl', '[*]C(C)C(=O)N1CCCCCC1', 'polar'),
    ('C(C)-carbonyl-NHOMe', '[*]C(C)C(=O)NOC', 'polar'),
    ('C(C)-carbonyl-NHOEt', '[*]C(C)C(=O)NOCC', 'polar'),
    ('C(C)-carbonyl-NH(2-pyridyl)', '[*]C(C)C(=O)Nc1ccccn1', 'polar'),
    ('C(C)-carbonyl-NH(3-pyridyl)', '[*]C(C)C(=O)Nc1cccnc1', 'polar'),
    ('C(C)-carbonyl-NH(4-pyridyl)', '[*]C(C)C(=O)Nc1ccncc1', 'polar'),
    ('C(C)-carbonyl-NH(pyrimidinyl)', '[*]C(C)C(=O)Nc1ncccn1', 'polar'),
    ('C(C)-carbonyl-NH(thiazolyl)', '[*]C(C)C(=O)Nc1nccs1', 'polar'),
    ('C(C)-carbonyl-NHPh', '[*]C(C)C(=O)Nc1ccccc1', 'polar'),
    ('C(C)-carbonyl-NHMe-OH', '[*]C(C)C(=O)N(C)CCO', 'polar'),
    ('C(C)-carbonyl-N(Me)(Bn)', '[*]C(C)C(=O)N(C)Cc1ccccc1', 'polar'),
    ('C(C)-carbonyl-NH(2-HOEt)', '[*]C(C)C(=O)NCCO', 'polar'),
    ('C(C)-carbonyl-N(2-HOEt)2', '[*]C(C)C(=O)N(CCO)CCO', 'polar'),
    ('C(C)-carbonyl-3-hydroxypyrrolidinyl', '[*]C(C)C(=O)N1CCC(O)C1', 'polar'),
    ('C(C)-carbonyl-4-hydroxypiperidyl', '[*]C(C)C(=O)N1CCC(O)CC1', 'polar'),
    ('C(C)-carbonyl-3-fluoropyrrolidinyl', '[*]C(C)C(=O)N1CCC(F)C1', 'polar'),
    ('C(C)-carbonyl-3-methylpiperidyl', '[*]C(C)C(=O)N1CCCC(C)C1', 'polar'),
    ('C(C)-carbonyl-4-methylpiperidyl', '[*]C(C)C(=O)N1CCC(C)CC1', 'polar'),
    ('c1ccccc1-carbonyl-NH2', '[*]c1ccccc1C(=O)N', 'polar'),
    ('c1ccccc1-carbonyl-NHMe', '[*]c1ccccc1C(=O)NC', 'polar'),
    ('c1ccccc1-carbonyl-NHEt', '[*]c1ccccc1C(=O)NCC', 'polar'),
    ('c1ccccc1-carbonyl-NHiPr', '[*]c1ccccc1C(=O)NC(C)C', 'polar'),
    ('c1ccccc1-carbonyl-NHtBu', '[*]c1ccccc1C(=O)NC(C)(C)C', 'polar'),
    ('c1ccccc1-carbonyl-NHcPr', '[*]c1ccccc1C(=O)NC1CC1', 'polar'),
    ('c1ccccc1-carbonyl-NHcBu', '[*]c1ccccc1C(=O)NC1CCC1', 'polar'),
    ('c1ccccc1-carbonyl-NHcPent', '[*]c1ccccc1C(=O)NC1CCCC1', 'polar'),
    ('c1ccccc1-carbonyl-NHcHex', '[*]c1ccccc1C(=O)NC1CCCCC1', 'polar'),
    ('c1ccccc1-carbonyl-NMe2', '[*]c1ccccc1C(=O)N(C)C', 'polar'),
    ('c1ccccc1-carbonyl-NEt2', '[*]c1ccccc1C(=O)N(CC)CC', 'polar'),
    ('c1ccccc1-carbonyl-NMeEt', '[*]c1ccccc1C(=O)N(C)CC', 'polar'),
    ('c1ccccc1-carbonyl-NHBn', '[*]c1ccccc1C(=O)NCc1ccccc1', 'polar'),
    ('c1ccccc1-carbonyl-NH(4-FBn)', '[*]c1ccccc1C(=O)NCc1ccc(F)cc1', 'polar'),
    ('c1ccccc1-carbonyl-NH(4-ClBn)', '[*]c1ccccc1C(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('c1ccccc1-carbonyl-NH(4-MeBn)', '[*]c1ccccc1C(=O)NCc1ccc(C)cc1', 'polar'),
    ('c1ccccc1-carbonyl-piperidyl', '[*]c1ccccc1C(=O)N1CCCCC1', 'polar'),
    ('c1ccccc1-carbonyl-pyrrolidyl', '[*]c1ccccc1C(=O)N1CCCC1', 'polar'),
    ('c1ccccc1-carbonyl-morpholyl', '[*]c1ccccc1C(=O)N1CCOCC1', 'polar'),
    ('c1ccccc1-carbonyl-azetidinyl', '[*]c1ccccc1C(=O)N1CCC1', 'polar'),
    ('c1ccccc1-carbonyl-N-Me-piperazinyl', '[*]c1ccccc1C(=O)N1CCN(C)CC1', 'polar'),
    ('c1ccccc1-carbonyl-azepanyl', '[*]c1ccccc1C(=O)N1CCCCCC1', 'polar'),
    ('c1ccccc1-carbonyl-NHOMe', '[*]c1ccccc1C(=O)NOC', 'polar'),
    ('c1ccccc1-carbonyl-NHOEt', '[*]c1ccccc1C(=O)NOCC', 'polar'),
    ('c1ccccc1-carbonyl-NH(2-pyridyl)', '[*]c1ccccc1C(=O)Nc1ccccn1', 'polar'),
    ('c1ccccc1-carbonyl-NH(3-pyridyl)', '[*]c1ccccc1C(=O)Nc1cccnc1', 'polar'),
    ('c1ccccc1-carbonyl-NH(4-pyridyl)', '[*]c1ccccc1C(=O)Nc1ccncc1', 'polar'),
    ('c1ccccc1-carbonyl-NH(pyrimidinyl)', '[*]c1ccccc1C(=O)Nc1ncccn1', 'polar'),
    ('c1ccccc1-carbonyl-NH(thiazolyl)', '[*]c1ccccc1C(=O)Nc1nccs1', 'polar'),
    ('c1ccccc1-carbonyl-NHPh', '[*]c1ccccc1C(=O)Nc1ccccc1', 'polar'),
    ('c1ccccc1-carbonyl-NHMe-OH', '[*]c1ccccc1C(=O)N(C)CCO', 'polar'),
    ('c1ccccc1-carbonyl-N(Me)(Bn)', '[*]c1ccccc1C(=O)N(C)Cc1ccccc1', 'polar'),
    ('c1ccccc1-carbonyl-NH(2-HOEt)', '[*]c1ccccc1C(=O)NCCO', 'polar'),
    ('c1ccccc1-carbonyl-N(2-HOEt)2', '[*]c1ccccc1C(=O)N(CCO)CCO', 'polar'),
    ('c1ccccc1-carbonyl-3-hydroxypyrrolidinyl', '[*]c1ccccc1C(=O)N1CCC(O)C1', 'polar'),
    ('c1ccccc1-carbonyl-4-hydroxypiperidyl', '[*]c1ccccc1C(=O)N1CCC(O)CC1', 'polar'),
    ('c1ccccc1-carbonyl-3-fluoropyrrolidinyl', '[*]c1ccccc1C(=O)N1CCC(F)C1', 'polar'),
    ('c1ccccc1-carbonyl-3-methylpiperidyl', '[*]c1ccccc1C(=O)N1CCCC(C)C1', 'polar'),
    ('c1ccccc1-carbonyl-4-methylpiperidyl', '[*]c1ccccc1C(=O)N1CCC(C)CC1', 'polar'),
    ('C=C-carbonyl-NH2', '[*]C=CC(=O)N', 'polar'),
    ('C=C-carbonyl-NHMe', '[*]C=CC(=O)NC', 'polar'),
    ('C=C-carbonyl-NHEt', '[*]C=CC(=O)NCC', 'polar'),
    ('C=C-carbonyl-NHiPr', '[*]C=CC(=O)NC(C)C', 'polar'),
    ('C=C-carbonyl-NHtBu', '[*]C=CC(=O)NC(C)(C)C', 'polar'),
    ('C=C-carbonyl-NHcPr', '[*]C=CC(=O)NC1CC1', 'polar'),
    ('C=C-carbonyl-NHcBu', '[*]C=CC(=O)NC1CCC1', 'polar'),
    ('C=C-carbonyl-NHcPent', '[*]C=CC(=O)NC1CCCC1', 'polar'),
    ('C=C-carbonyl-NHcHex', '[*]C=CC(=O)NC1CCCCC1', 'polar'),
    ('C=C-carbonyl-NMe2', '[*]C=CC(=O)N(C)C', 'polar'),
    ('C=C-carbonyl-NEt2', '[*]C=CC(=O)N(CC)CC', 'polar'),
    ('C=C-carbonyl-NMeEt', '[*]C=CC(=O)N(C)CC', 'polar'),
    ('C=C-carbonyl-NHBn', '[*]C=CC(=O)NCc1ccccc1', 'polar'),
    ('C=C-carbonyl-NH(4-FBn)', '[*]C=CC(=O)NCc1ccc(F)cc1', 'polar'),
    ('C=C-carbonyl-NH(4-ClBn)', '[*]C=CC(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('C=C-carbonyl-NH(4-MeBn)', '[*]C=CC(=O)NCc1ccc(C)cc1', 'polar'),
    ('C=C-carbonyl-piperidyl', '[*]C=CC(=O)N1CCCCC1', 'polar'),
    ('C=C-carbonyl-pyrrolidyl', '[*]C=CC(=O)N1CCCC1', 'polar'),
    ('C=C-carbonyl-morpholyl', '[*]C=CC(=O)N1CCOCC1', 'polar'),
    ('C=C-carbonyl-azetidinyl', '[*]C=CC(=O)N1CCC1', 'polar'),
    ('C=C-carbonyl-N-Me-piperazinyl', '[*]C=CC(=O)N1CCN(C)CC1', 'polar'),
    ('C=C-carbonyl-azepanyl', '[*]C=CC(=O)N1CCCCCC1', 'polar'),
    ('C=C-carbonyl-NHOMe', '[*]C=CC(=O)NOC', 'polar'),
    ('C=C-carbonyl-NHOEt', '[*]C=CC(=O)NOCC', 'polar'),
    ('C=C-carbonyl-NH(2-pyridyl)', '[*]C=CC(=O)Nc1ccccn1', 'polar'),
    ('C=C-carbonyl-NH(3-pyridyl)', '[*]C=CC(=O)Nc1cccnc1', 'polar'),
    ('C=C-carbonyl-NH(4-pyridyl)', '[*]C=CC(=O)Nc1ccncc1', 'polar'),
    ('C=C-carbonyl-NH(pyrimidinyl)', '[*]C=CC(=O)Nc1ncccn1', 'polar'),
    ('C=C-carbonyl-NH(thiazolyl)', '[*]C=CC(=O)Nc1nccs1', 'polar'),
    ('C=C-carbonyl-NHPh', '[*]C=CC(=O)Nc1ccccc1', 'polar'),
    ('C=C-carbonyl-NHMe-OH', '[*]C=CC(=O)N(C)CCO', 'polar'),
    ('C=C-carbonyl-N(Me)(Bn)', '[*]C=CC(=O)N(C)Cc1ccccc1', 'polar'),
    ('C=C-carbonyl-NH(2-HOEt)', '[*]C=CC(=O)NCCO', 'polar'),
    ('C=C-carbonyl-N(2-HOEt)2', '[*]C=CC(=O)N(CCO)CCO', 'polar'),
    ('C=C-carbonyl-3-hydroxypyrrolidinyl', '[*]C=CC(=O)N1CCC(O)C1', 'polar'),
    ('C=C-carbonyl-4-hydroxypiperidyl', '[*]C=CC(=O)N1CCC(O)CC1', 'polar'),
    ('C=C-carbonyl-3-fluoropyrrolidinyl', '[*]C=CC(=O)N1CCC(F)C1', 'polar'),
    ('C=C-carbonyl-3-methylpiperidyl', '[*]C=CC(=O)N1CCCC(C)C1', 'polar'),
    ('C=C-carbonyl-4-methylpiperidyl', '[*]C=CC(=O)N1CCC(C)CC1', 'polar'),
    ('C(F)(F)-carbonyl-NH2', '[*]C(F)(F)C(=O)N', 'polar'),
    ('C(F)(F)-carbonyl-NHMe', '[*]C(F)(F)C(=O)NC', 'polar'),
    ('C(F)(F)-carbonyl-NHEt', '[*]C(F)(F)C(=O)NCC', 'polar'),
    ('C(F)(F)-carbonyl-NHiPr', '[*]C(F)(F)C(=O)NC(C)C', 'polar'),
    ('C(F)(F)-carbonyl-NHtBu', '[*]C(F)(F)C(=O)NC(C)(C)C', 'polar'),
    ('C(F)(F)-carbonyl-NHcPr', '[*]C(F)(F)C(=O)NC1CC1', 'polar'),
    ('C(F)(F)-carbonyl-NHcBu', '[*]C(F)(F)C(=O)NC1CCC1', 'polar'),
    ('C(F)(F)-carbonyl-NHcPent', '[*]C(F)(F)C(=O)NC1CCCC1', 'polar'),
    ('C(F)(F)-carbonyl-NHcHex', '[*]C(F)(F)C(=O)NC1CCCCC1', 'polar'),
    ('C(F)(F)-carbonyl-NMe2', '[*]C(F)(F)C(=O)N(C)C', 'polar'),
    ('C(F)(F)-carbonyl-NEt2', '[*]C(F)(F)C(=O)N(CC)CC', 'polar'),
    ('C(F)(F)-carbonyl-NMeEt', '[*]C(F)(F)C(=O)N(C)CC', 'polar'),
    ('C(F)(F)-carbonyl-NHBn', '[*]C(F)(F)C(=O)NCc1ccccc1', 'polar'),
    ('C(F)(F)-carbonyl-NH(4-FBn)', '[*]C(F)(F)C(=O)NCc1ccc(F)cc1', 'polar'),
    ('C(F)(F)-carbonyl-NH(4-ClBn)', '[*]C(F)(F)C(=O)NCc1ccc(Cl)cc1', 'polar'),
    ('C(F)(F)-carbonyl-NH(4-MeBn)', '[*]C(F)(F)C(=O)NCc1ccc(C)cc1', 'polar'),
    ('C(F)(F)-carbonyl-piperidyl', '[*]C(F)(F)C(=O)N1CCCCC1', 'polar'),
    ('C(F)(F)-carbonyl-pyrrolidyl', '[*]C(F)(F)C(=O)N1CCCC1', 'polar'),
    ('C(F)(F)-carbonyl-morpholyl', '[*]C(F)(F)C(=O)N1CCOCC1', 'polar'),
    ('C(F)(F)-carbonyl-azetidinyl', '[*]C(F)(F)C(=O)N1CCC1', 'polar'),
    ('C(F)(F)-carbonyl-N-Me-piperazinyl', '[*]C(F)(F)C(=O)N1CCN(C)CC1', 'polar'),
    ('C(F)(F)-carbonyl-azepanyl', '[*]C(F)(F)C(=O)N1CCCCCC1', 'polar'),
    ('C(F)(F)-carbonyl-NHOMe', '[*]C(F)(F)C(=O)NOC', 'polar'),
    ('C(F)(F)-carbonyl-NHOEt', '[*]C(F)(F)C(=O)NOCC', 'polar'),
    ('C(F)(F)-carbonyl-NH(2-pyridyl)', '[*]C(F)(F)C(=O)Nc1ccccn1', 'polar'),
    ('C(F)(F)-carbonyl-NH(3-pyridyl)', '[*]C(F)(F)C(=O)Nc1cccnc1', 'polar'),
    ('C(F)(F)-carbonyl-NH(4-pyridyl)', '[*]C(F)(F)C(=O)Nc1ccncc1', 'polar'),
    ('C(F)(F)-carbonyl-NH(pyrimidinyl)', '[*]C(F)(F)C(=O)Nc1ncccn1', 'polar'),
    ('C(F)(F)-carbonyl-NH(thiazolyl)', '[*]C(F)(F)C(=O)Nc1nccs1', 'polar'),
    ('C(F)(F)-carbonyl-NHPh', '[*]C(F)(F)C(=O)Nc1ccccc1', 'polar'),
    ('C(F)(F)-carbonyl-NHMe-OH', '[*]C(F)(F)C(=O)N(C)CCO', 'polar'),
    ('C(F)(F)-carbonyl-N(Me)(Bn)', '[*]C(F)(F)C(=O)N(C)Cc1ccccc1', 'polar'),
    ('C(F)(F)-carbonyl-NH(2-HOEt)', '[*]C(F)(F)C(=O)NCCO', 'polar'),
    ('C(F)(F)-carbonyl-N(2-HOEt)2', '[*]C(F)(F)C(=O)N(CCO)CCO', 'polar'),
    ('C(F)(F)-carbonyl-3-hydroxypyrrolidinyl', '[*]C(F)(F)C(=O)N1CCC(O)C1', 'polar'),
    ('C(F)(F)-carbonyl-4-hydroxypiperidyl', '[*]C(F)(F)C(=O)N1CCC(O)CC1', 'polar'),
    ('C(F)(F)-carbonyl-3-fluoropyrrolidinyl', '[*]C(F)(F)C(=O)N1CCC(F)C1', 'polar'),
    ('C(F)(F)-carbonyl-3-methylpiperidyl', '[*]C(F)(F)C(=O)N1CCCC(C)C1', 'polar'),
    ('C(F)(F)-carbonyl-4-methylpiperidyl', '[*]C(F)(F)C(=O)N1CCC(C)CC1', 'polar'),
    ('sulfonyl-Et', '[*]S(=O)(=O)CC', 'polar'),
    ('sulfonyl-iPr', '[*]S(=O)(=O)C(C)C', 'polar'),
    ('sulfonyl-cPr', '[*]S(=O)(=O)C1CC1', 'polar'),
    ('sulfonyl-tBu', '[*]S(=O)(=O)C(C)(C)C', 'polar'),
    ('sulfonyl-Ph', '[*]S(=O)(=O)c1ccccc1', 'polar'),
    ('sulfonyl-4-FPh', '[*]S(=O)(=O)c1ccc(F)cc1', 'polar'),
    ('sulfonyl-Bn', '[*]S(=O)(=O)Cc1ccccc1', 'polar'),
    ('sulfonyl-CH2CF3', '[*]S(=O)(=O)CC(F)(F)F', 'polar'),
    ('sulfonyl-NHcPr', '[*]S(=O)(=O)NC1CC1', 'polar'),
    ('sulfonyl-NHiPr', '[*]S(=O)(=O)NC(C)C', 'polar'),
    ('C-sulfonyl-Et', '[*]CS(=O)(=O)CC', 'polar'),
    ('C-sulfonyl-iPr', '[*]CS(=O)(=O)C(C)C', 'polar'),
    ('C-sulfonyl-cPr', '[*]CS(=O)(=O)C1CC1', 'polar'),
    ('C-sulfonyl-tBu', '[*]CS(=O)(=O)C(C)(C)C', 'polar'),
    ('C-sulfonyl-Ph', '[*]CS(=O)(=O)c1ccccc1', 'polar'),
    ('C-sulfonyl-4-FPh', '[*]CS(=O)(=O)c1ccc(F)cc1', 'polar'),
    ('C-sulfonyl-Bn', '[*]CS(=O)(=O)Cc1ccccc1', 'polar'),
    ('C-sulfonyl-CF3', '[*]CS(=O)(=O)C(F)(F)F', 'polar'),
    ('C-sulfonyl-CH2CF3', '[*]CS(=O)(=O)CC(F)(F)F', 'polar'),
    ('C-sulfonyl-NHMe', '[*]CS(=O)(=O)NC', 'polar'),
    ('C-sulfonyl-NMe2', '[*]CS(=O)(=O)N(C)C', 'polar'),
    ('C-sulfonyl-NHEt', '[*]CS(=O)(=O)NCC', 'polar'),
    ('C-sulfonyl-NHcPr', '[*]CS(=O)(=O)NC1CC1', 'polar'),
    ('C-sulfonyl-NHiPr', '[*]CS(=O)(=O)NC(C)C', 'polar'),
    ('C-sulfonyl-piperidyl', '[*]CS(=O)(=O)N1CCCCC1', 'polar'),
    ('C-sulfonyl-morpholyl', '[*]CS(=O)(=O)N1CCOCC1', 'polar'),
    ('C-sulfonyl-pyrrolidyl', '[*]CS(=O)(=O)N1CCCC1', 'polar'),
    ('C-sulfonyl-azetidinyl', '[*]CS(=O)(=O)N1CCC1', 'polar'),
    ('CC-sulfonyl-Et', '[*]CCS(=O)(=O)CC', 'polar'),
    ('CC-sulfonyl-iPr', '[*]CCS(=O)(=O)C(C)C', 'polar'),
    ('CC-sulfonyl-cPr', '[*]CCS(=O)(=O)C1CC1', 'polar'),
    ('CC-sulfonyl-tBu', '[*]CCS(=O)(=O)C(C)(C)C', 'polar'),
    ('CC-sulfonyl-Ph', '[*]CCS(=O)(=O)c1ccccc1', 'polar'),
    ('CC-sulfonyl-4-FPh', '[*]CCS(=O)(=O)c1ccc(F)cc1', 'polar'),
    ('CC-sulfonyl-Bn', '[*]CCS(=O)(=O)Cc1ccccc1', 'polar'),
    ('CC-sulfonyl-CF3', '[*]CCS(=O)(=O)C(F)(F)F', 'polar'),
    ('CC-sulfonyl-CH2CF3', '[*]CCS(=O)(=O)CC(F)(F)F', 'polar'),
    ('CC-sulfonyl-NHMe', '[*]CCS(=O)(=O)NC', 'polar'),
    ('CC-sulfonyl-NMe2', '[*]CCS(=O)(=O)N(C)C', 'polar'),
    ('CC-sulfonyl-NHEt', '[*]CCS(=O)(=O)NCC', 'polar'),
    ('CC-sulfonyl-NHcPr', '[*]CCS(=O)(=O)NC1CC1', 'polar'),
    ('CC-sulfonyl-NHiPr', '[*]CCS(=O)(=O)NC(C)C', 'polar'),
    ('CC-sulfonyl-piperidyl', '[*]CCS(=O)(=O)N1CCCCC1', 'polar'),
    ('CC-sulfonyl-morpholyl', '[*]CCS(=O)(=O)N1CCOCC1', 'polar'),
    ('CC-sulfonyl-pyrrolidyl', '[*]CCS(=O)(=O)N1CCCC1', 'polar'),
    ('CC-sulfonyl-azetidinyl', '[*]CCS(=O)(=O)N1CCC1', 'polar'),
    ('c1ccccc1-sulfonyl-Et', '[*]c1ccccc1S(=O)(=O)CC', 'polar'),
    ('c1ccccc1-sulfonyl-iPr', '[*]c1ccccc1S(=O)(=O)C(C)C', 'polar'),
    ('c1ccccc1-sulfonyl-cPr', '[*]c1ccccc1S(=O)(=O)C1CC1', 'polar'),
    ('c1ccccc1-sulfonyl-tBu', '[*]c1ccccc1S(=O)(=O)C(C)(C)C', 'polar'),
    ('c1ccccc1-sulfonyl-Ph', '[*]c1ccccc1S(=O)(=O)c1ccccc1', 'polar'),
    ('c1ccccc1-sulfonyl-4-FPh', '[*]c1ccccc1S(=O)(=O)c1ccc(F)cc1', 'polar'),
    ('c1ccccc1-sulfonyl-Bn', '[*]c1ccccc1S(=O)(=O)Cc1ccccc1', 'polar'),
    ('c1ccccc1-sulfonyl-CF3', '[*]c1ccccc1S(=O)(=O)C(F)(F)F', 'polar'),
    ('c1ccccc1-sulfonyl-CH2CF3', '[*]c1ccccc1S(=O)(=O)CC(F)(F)F', 'polar'),
    ('c1ccccc1-sulfonyl-NHMe', '[*]c1ccccc1S(=O)(=O)NC', 'polar'),
    ('c1ccccc1-sulfonyl-NMe2', '[*]c1ccccc1S(=O)(=O)N(C)C', 'polar'),
    ('c1ccccc1-sulfonyl-NHEt', '[*]c1ccccc1S(=O)(=O)NCC', 'polar'),
    ('c1ccccc1-sulfonyl-NHcPr', '[*]c1ccccc1S(=O)(=O)NC1CC1', 'polar'),
    ('c1ccccc1-sulfonyl-NHiPr', '[*]c1ccccc1S(=O)(=O)NC(C)C', 'polar'),
    ('c1ccccc1-sulfonyl-piperidyl', '[*]c1ccccc1S(=O)(=O)N1CCCCC1', 'polar'),
    ('c1ccccc1-sulfonyl-morpholyl', '[*]c1ccccc1S(=O)(=O)N1CCOCC1', 'polar'),
    ('c1ccccc1-sulfonyl-pyrrolidyl', '[*]c1ccccc1S(=O)(=O)N1CCCC1', 'polar'),
    ('c1ccccc1-sulfonyl-azetidinyl', '[*]c1ccccc1S(=O)(=O)N1CCC1', 'polar'),
    ('C1CC1-sulfonyl-Et', '[*]C1CC1S(=O)(=O)CC', 'polar'),
    ('C1CC1-sulfonyl-iPr', '[*]C1CC1S(=O)(=O)C(C)C', 'polar'),
    ('C1CC1-sulfonyl-cPr', '[*]C1CC1S(=O)(=O)C1CC1', 'polar'),
    ('C1CC1-sulfonyl-tBu', '[*]C1CC1S(=O)(=O)C(C)(C)C', 'polar'),
    ('C1CC1-sulfonyl-Ph', '[*]C1CC1S(=O)(=O)c1ccccc1', 'polar'),
    ('C1CC1-sulfonyl-4-FPh', '[*]C1CC1S(=O)(=O)c1ccc(F)cc1', 'polar'),
    ('C1CC1-sulfonyl-Bn', '[*]C1CC1S(=O)(=O)Cc1ccccc1', 'polar'),
    ('C1CC1-sulfonyl-CF3', '[*]C1CC1S(=O)(=O)C(F)(F)F', 'polar'),
    ('C1CC1-sulfonyl-CH2CF3', '[*]C1CC1S(=O)(=O)CC(F)(F)F', 'polar'),
    ('C1CC1-sulfonyl-NHMe', '[*]C1CC1S(=O)(=O)NC', 'polar'),
    ('C1CC1-sulfonyl-NMe2', '[*]C1CC1S(=O)(=O)N(C)C', 'polar'),
    ('C1CC1-sulfonyl-NHEt', '[*]C1CC1S(=O)(=O)NCC', 'polar'),
    ('C1CC1-sulfonyl-NHcPr', '[*]C1CC1S(=O)(=O)NC1CC1', 'polar'),
    ('C1CC1-sulfonyl-NHiPr', '[*]C1CC1S(=O)(=O)NC(C)C', 'polar'),
    ('C1CC1-sulfonyl-piperidyl', '[*]C1CC1S(=O)(=O)N1CCCCC1', 'polar'),
    ('C1CC1-sulfonyl-morpholyl', '[*]C1CC1S(=O)(=O)N1CCOCC1', 'polar'),
    ('C1CC1-sulfonyl-pyrrolidyl', '[*]C1CC1S(=O)(=O)N1CCCC1', 'polar'),
    ('C1CC1-sulfonyl-azetidinyl', '[*]C1CC1S(=O)(=O)N1CCC1', 'polar'),
    ('phenyl-phenyl', '[*]c1ccccc1c1ccccc1', 'aromatic'),
    ('phenyl-4-F-phenyl', '[*]c1ccccc1c1ccc(F)cc1', 'aromatic'),
    ('phenyl-4-Cl-phenyl', '[*]c1ccccc1c1ccc(Cl)cc1', 'aromatic'),
    ('phenyl-4-Me-phenyl', '[*]c1ccccc1c1ccc(C)cc1', 'aromatic'),
    ('phenyl-4-OMe-phenyl', '[*]c1ccccc1c1ccc(OC)cc1', 'aromatic'),
    ('phenyl-4-CF3-phenyl', '[*]c1ccccc1c1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('phenyl-pyridin-2-yl', '[*]c1ccccc1c1ccccn1', 'aromatic'),
    ('phenyl-pyridin-3-yl', '[*]c1ccccc1c1cccnc1', 'aromatic'),
    ('phenyl-pyridin-4-yl', '[*]c1ccccc1c1ccncc1', 'aromatic'),
    ('phenyl-pyrimidin-2-yl', '[*]c1ccccc1c1ncccn1', 'aromatic'),
    ('phenyl-thiophen-2-yl', '[*]c1ccccc1c1cccs1', 'aromatic'),
    ('phenyl-furan-2-yl', '[*]c1ccccc1c1ccco1', 'aromatic'),
    ('phenyl-thiazol-2-yl', '[*]c1ccccc1c1nccs1', 'aromatic'),
    ('phenyl-1-Me-pyrazol-4-yl', '[*]c1ccccc1c1cn(C)nc1', 'aromatic'),
    ('pyridin-3-yl-phenyl', '[*]c1cccnc1c1ccccc1', 'aromatic'),
    ('pyridin-3-yl-4-F-phenyl', '[*]c1cccnc1c1ccc(F)cc1', 'aromatic'),
    ('pyridin-3-yl-4-Cl-phenyl', '[*]c1cccnc1c1ccc(Cl)cc1', 'aromatic'),
    ('pyridin-3-yl-4-Me-phenyl', '[*]c1cccnc1c1ccc(C)cc1', 'aromatic'),
    ('pyridin-3-yl-4-OMe-phenyl', '[*]c1cccnc1c1ccc(OC)cc1', 'aromatic'),
    ('pyridin-3-yl-4-CF3-phenyl', '[*]c1cccnc1c1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('pyridin-3-yl-pyridin-2-yl', '[*]c1cccnc1c1ccccn1', 'aromatic'),
    ('pyridin-3-yl-pyridin-3-yl', '[*]c1cccnc1c1cccnc1', 'aromatic'),
    ('pyridin-3-yl-pyridin-4-yl', '[*]c1cccnc1c1ccncc1', 'aromatic'),
    ('pyridin-3-yl-pyrimidin-2-yl', '[*]c1cccnc1c1ncccn1', 'aromatic'),
    ('pyridin-3-yl-thiophen-2-yl', '[*]c1cccnc1c1cccs1', 'aromatic'),
    ('pyridin-3-yl-furan-2-yl', '[*]c1cccnc1c1ccco1', 'aromatic'),
    ('pyridin-3-yl-thiazol-2-yl', '[*]c1cccnc1c1nccs1', 'aromatic'),
    ('pyridin-3-yl-1-Me-pyrazol-4-yl', '[*]c1cccnc1c1cn(C)nc1', 'aromatic'),
    ('pyridin-4-yl-phenyl', '[*]c1ccncc1c1ccccc1', 'aromatic'),
    ('pyridin-4-yl-4-F-phenyl', '[*]c1ccncc1c1ccc(F)cc1', 'aromatic'),
    ('pyridin-4-yl-4-Cl-phenyl', '[*]c1ccncc1c1ccc(Cl)cc1', 'aromatic'),
    ('pyridin-4-yl-4-Me-phenyl', '[*]c1ccncc1c1ccc(C)cc1', 'aromatic'),
    ('pyridin-4-yl-4-OMe-phenyl', '[*]c1ccncc1c1ccc(OC)cc1', 'aromatic'),
    ('pyridin-4-yl-4-CF3-phenyl', '[*]c1ccncc1c1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('pyridin-4-yl-pyridin-2-yl', '[*]c1ccncc1c1ccccn1', 'aromatic'),
    ('pyridin-4-yl-pyridin-3-yl', '[*]c1ccncc1c1cccnc1', 'aromatic'),
    ('pyridin-4-yl-pyridin-4-yl', '[*]c1ccncc1c1ccncc1', 'aromatic'),
    ('pyridin-4-yl-pyrimidin-2-yl', '[*]c1ccncc1c1ncccn1', 'aromatic'),
    ('pyridin-4-yl-thiophen-2-yl', '[*]c1ccncc1c1cccs1', 'aromatic'),
    ('pyridin-4-yl-furan-2-yl', '[*]c1ccncc1c1ccco1', 'aromatic'),
    ('pyridin-4-yl-thiazol-2-yl', '[*]c1ccncc1c1nccs1', 'aromatic'),
    ('pyridin-4-yl-1-Me-pyrazol-4-yl', '[*]c1ccncc1c1cn(C)nc1', 'aromatic'),
    ('pyrimidin-5-yl-phenyl', '[*]c1cncnc1c1ccccc1', 'aromatic'),
    ('pyrimidin-5-yl-4-F-phenyl', '[*]c1cncnc1c1ccc(F)cc1', 'aromatic'),
    ('pyrimidin-5-yl-4-Cl-phenyl', '[*]c1cncnc1c1ccc(Cl)cc1', 'aromatic'),
    ('pyrimidin-5-yl-4-Me-phenyl', '[*]c1cncnc1c1ccc(C)cc1', 'aromatic'),
    ('pyrimidin-5-yl-4-OMe-phenyl', '[*]c1cncnc1c1ccc(OC)cc1', 'aromatic'),
    ('pyrimidin-5-yl-4-CF3-phenyl', '[*]c1cncnc1c1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('pyrimidin-5-yl-pyridin-2-yl', '[*]c1cncnc1c1ccccn1', 'aromatic'),
    ('pyrimidin-5-yl-pyridin-3-yl', '[*]c1cncnc1c1cccnc1', 'aromatic'),
    ('pyrimidin-5-yl-pyridin-4-yl', '[*]c1cncnc1c1ccncc1', 'aromatic'),
    ('pyrimidin-5-yl-pyrimidin-2-yl', '[*]c1cncnc1c1ncccn1', 'aromatic'),
    ('pyrimidin-5-yl-thiophen-2-yl', '[*]c1cncnc1c1cccs1', 'aromatic'),
    ('pyrimidin-5-yl-furan-2-yl', '[*]c1cncnc1c1ccco1', 'aromatic'),
    ('pyrimidin-5-yl-thiazol-2-yl', '[*]c1cncnc1c1nccs1', 'aromatic'),
    ('pyrimidin-5-yl-1-Me-pyrazol-4-yl', '[*]c1cncnc1c1cn(C)nc1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-phenyl', '[*]c1cn(C)nc1c1ccccc1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-4-F-phenyl', '[*]c1cn(C)nc1c1ccc(F)cc1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-4-Cl-phenyl', '[*]c1cn(C)nc1c1ccc(Cl)cc1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-4-Me-phenyl', '[*]c1cn(C)nc1c1ccc(C)cc1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-4-OMe-phenyl', '[*]c1cn(C)nc1c1ccc(OC)cc1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-4-CF3-phenyl', '[*]c1cn(C)nc1c1ccc(C(F)(F)F)cc1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-pyridin-2-yl', '[*]c1cn(C)nc1c1ccccn1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-pyridin-3-yl', '[*]c1cn(C)nc1c1cccnc1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-pyridin-4-yl', '[*]c1cn(C)nc1c1ccncc1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-pyrimidin-2-yl', '[*]c1cn(C)nc1c1ncccn1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-thiophen-2-yl', '[*]c1cn(C)nc1c1cccs1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-furan-2-yl', '[*]c1cn(C)nc1c1ccco1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-thiazol-2-yl', '[*]c1cn(C)nc1c1nccs1', 'aromatic'),
    ('1-Me-pyrazol-4-yl-1-Me-pyrazol-4-yl', '[*]c1cn(C)nc1c1cn(C)nc1', 'aromatic'),
    ('oxy-CC-cyclopropyl', '[*]OCCC1CC1', 'polar'),
    ('oxy-CCC-cyclopropyl', '[*]OCCCC1CC1', 'polar'),
    ('oxy-C(C)-F', '[*]OC(C)F', 'polar'),
    ('oxy-C(C)-Cl', '[*]OC(C)Cl', 'polar'),
    ('oxy-C(C)-Br', '[*]OC(C)Br', 'polar'),
    ('oxy-C(C)-OH', '[*]OC(C)O', 'polar'),
    ('oxy-C(C)-OMe', '[*]OC(C)OC', 'polar'),
    ('oxy-C(C)-NH2', '[*]OC(C)N', 'polar'),
    ('oxy-C(C)-CN', '[*]OC(C)C#N', 'polar'),
    ('oxy-C(C)-CF3', '[*]OC(C)C(F)(F)F', 'polar'),
    ('oxy-C(C)-cyclopropyl', '[*]OC(C)C1CC1', 'polar'),
    ('oxy-C(C)-morpholino', '[*]OC(C)N1CCOCC1', 'polar'),
    ('oxy-C(C)-piperidino', '[*]OC(C)N1CCCCC1', 'polar'),
    ('oxy-C(C)-F2', '[*]OC(C)(F)F', 'polar'),
    ('oxy-C(C)C-F', '[*]OC(C)CF', 'polar'),
    ('oxy-C(C)C-Cl', '[*]OC(C)CCl', 'polar'),
    ('oxy-C(C)C-Br', '[*]OC(C)CBr', 'polar'),
    ('oxy-C(C)C-OH', '[*]OC(C)CO', 'polar'),
    ('oxy-C(C)C-OMe', '[*]OC(C)COC', 'polar'),
    ('oxy-C(C)C-NH2', '[*]OC(C)CN', 'polar'),
    ('oxy-C(C)C-CN', '[*]OC(C)CC#N', 'polar'),
    ('oxy-C(C)C-CF3', '[*]OC(C)CC(F)(F)F', 'polar'),
    ('oxy-C(C)C-Me', '[*]OC(C)CC', 'polar'),
    ('oxy-C(C)C-cyclopropyl', '[*]OC(C)CC1CC1', 'polar'),
    ('oxy-C(C)C-morpholino', '[*]OC(C)CN1CCOCC1', 'polar'),
    ('oxy-C(C)C-piperidino', '[*]OC(C)CN1CCCCC1', 'polar'),
    ('oxy-C(C)C-F2', '[*]OC(C)C(F)F', 'polar'),
    ('oxy-C1CC1-H', '[*]OC1CC1', 'polar'),
    ('oxy-C1CC1-F', '[*]OC1CC1F', 'polar'),
    ('oxy-C1CC1-Cl', '[*]OC1CC1Cl', 'polar'),
    ('oxy-C1CC1-Br', '[*]OC1CC1Br', 'polar'),
    ('oxy-C1CC1-OH', '[*]OC1CC1O', 'polar'),
    ('oxy-C1CC1-OMe', '[*]OC1CC1OC', 'polar'),
    ('oxy-C1CC1-NH2', '[*]OC1CC1N', 'polar'),
    ('oxy-C1CC1-CN', '[*]OC1CC1C#N', 'polar'),
    ('oxy-C1CC1-CF3', '[*]OC1CC1C(F)(F)F', 'polar'),
    ('oxy-C1CC1-Me', '[*]OC1CC1C', 'polar'),
    ('oxy-C1CC1-cyclopropyl', '[*]OC1CC1C1CC1', 'polar'),
    ('oxy-C1CC1-morpholino', '[*]OC1CC1N1CCOCC1', 'polar'),
    ('oxy-C1CC1-piperidino', '[*]OC1CC1N1CCCCC1', 'polar'),
    ('oxy-C1CC1-F2', '[*]OC1CC1(F)F', 'polar'),
    ('oxy-C1CCC1-H', '[*]OC1CCC1', 'polar'),
    ('oxy-C1CCC1-F', '[*]OC1CCC1F', 'polar'),
    ('oxy-C1CCC1-Cl', '[*]OC1CCC1Cl', 'polar'),
    ('oxy-C1CCC1-Br', '[*]OC1CCC1Br', 'polar'),
    ('oxy-C1CCC1-OH', '[*]OC1CCC1O', 'polar'),
    ('oxy-C1CCC1-OMe', '[*]OC1CCC1OC', 'polar'),
    ('oxy-C1CCC1-NH2', '[*]OC1CCC1N', 'polar'),
    ('oxy-C1CCC1-CN', '[*]OC1CCC1C#N', 'polar'),
    ('oxy-C1CCC1-CF3', '[*]OC1CCC1C(F)(F)F', 'polar'),
    ('oxy-C1CCC1-Me', '[*]OC1CCC1C', 'polar'),
    ('oxy-C1CCC1-cyclopropyl', '[*]OC1CCC1C1CC1', 'polar'),
    ('oxy-C1CCC1-morpholino', '[*]OC1CCC1N1CCOCC1', 'polar'),
    ('oxy-C1CCC1-piperidino', '[*]OC1CCC1N1CCCCC1', 'polar'),
    ('oxy-C1CCC1-F2', '[*]OC1CCC1(F)F', 'polar'),
    ('oxy-C(F)(F)-Cl', '[*]OC(F)(F)Cl', 'polar'),
    ('oxy-C(F)(F)-Br', '[*]OC(F)(F)Br', 'polar'),
    ('oxy-C(F)(F)-OH', '[*]OC(F)(F)O', 'polar'),
    ('oxy-C(F)(F)-OMe', '[*]OC(F)(F)OC', 'polar'),
    ('oxy-C(F)(F)-NH2', '[*]OC(F)(F)N', 'polar'),
    ('oxy-C(F)(F)-CN', '[*]OC(F)(F)C#N', 'polar'),
    ('oxy-C(F)(F)-CF3', '[*]OC(F)(F)C(F)(F)F', 'polar'),
    ('oxy-C(F)(F)-cyclopropyl', '[*]OC(F)(F)C1CC1', 'polar'),
    ('oxy-C(F)(F)-morpholino', '[*]OC(F)(F)N1CCOCC1', 'polar'),
    ('oxy-C(F)(F)-piperidino', '[*]OC(F)(F)N1CCCCC1', 'polar'),
    ('oxy-C=C-H', '[*]OC=C', 'polar'),
    ('oxy-C=C-F', '[*]OC=CF', 'polar'),
    ('oxy-C=C-Cl', '[*]OC=CCl', 'polar'),
    ('oxy-C=C-Br', '[*]OC=CBr', 'polar'),
    ('oxy-C=C-OH', '[*]OC=CO', 'polar'),
    ('oxy-C=C-OMe', '[*]OC=COC', 'polar'),
    ('oxy-C=C-NH2', '[*]OC=CN', 'polar'),
    ('oxy-C=C-CN', '[*]OC=CC#N', 'polar'),
    ('oxy-C=C-CF3', '[*]OC=CC(F)(F)F', 'polar'),
    ('oxy-C=C-Me', '[*]OC=CC', 'polar'),
    ('oxy-C=C-cyclopropyl', '[*]OC=CC1CC1', 'polar'),
    ('oxy-C=C-morpholino', '[*]OC=CN1CCOCC1', 'polar'),
    ('oxy-C=C-piperidino', '[*]OC=CN1CCCCC1', 'polar'),
    ('oxy-C=C-F2', '[*]OC=C(F)F', 'polar'),
    ('oxy-C#C-H', '[*]OC#C', 'polar'),
    ('oxy-C#C-F', '[*]OC#CF', 'polar'),
    ('oxy-C#C-Cl', '[*]OC#CCl', 'polar'),
    ('oxy-C#C-Br', '[*]OC#CBr', 'polar'),
    ('oxy-C#C-OH', '[*]OC#CO', 'polar'),
    ('oxy-C#C-OMe', '[*]OC#COC', 'polar'),
    ('oxy-C#C-NH2', '[*]OC#CN', 'polar'),
    ('oxy-C#C-CN', '[*]OC#CC#N', 'polar'),
    ('oxy-C#C-CF3', '[*]OC#CC(F)(F)F', 'polar'),
    ('oxy-C#C-Me', '[*]OC#CC', 'polar'),
    ('oxy-C#C-cyclopropyl', '[*]OC#CC1CC1', 'polar'),
    ('oxy-C#C-morpholino', '[*]OC#CN1CCOCC1', 'polar'),
    ('oxy-C#C-piperidino', '[*]OC#CN1CCCCC1', 'polar'),
    ('amino-C-cyclopropyl', '[*]NCC1CC1', 'polar'),
    ('amino-C-F2', '[*]NC(F)F', 'polar'),
    ('amino-CC-cyclopropyl', '[*]NCCC1CC1', 'polar'),
    ('amino-CC-F2', '[*]NCC(F)F', 'polar'),
    ('amino-CCC-F', '[*]NCCCF', 'polar'),
    ('amino-CCC-Cl', '[*]NCCCCl', 'polar'),
    ('amino-CCC-Br', '[*]NCCCBr', 'polar'),
    ('amino-CCC-OH', '[*]NCCCO', 'polar'),
    ('amino-CCC-OMe', '[*]NCCCOC', 'polar'),
    ('amino-CCC-NH2', '[*]NCCCN', 'polar'),
    ('amino-CCC-CN', '[*]NCCCC#N', 'polar'),
    ('amino-CCC-CF3', '[*]NCCCC(F)(F)F', 'polar'),
    ('amino-CCC-cyclopropyl', '[*]NCCCC1CC1', 'polar'),
    ('amino-CCC-morpholino', '[*]NCCCN1CCOCC1', 'polar'),
    ('amino-CCC-piperidino', '[*]NCCCN1CCCCC1', 'polar'),
    ('amino-CCC-F2', '[*]NCCC(F)F', 'polar'),
    ('amino-C(C)-F', '[*]NC(C)F', 'polar'),
    ('amino-C(C)-Cl', '[*]NC(C)Cl', 'polar'),
    ('amino-C(C)-Br', '[*]NC(C)Br', 'polar'),
    ('amino-C(C)-OH', '[*]NC(C)O', 'polar'),
    ('amino-C(C)-OMe', '[*]NC(C)OC', 'polar'),
    ('amino-C(C)-NH2', '[*]NC(C)N', 'polar'),
    ('amino-C(C)-CN', '[*]NC(C)C#N', 'polar'),
    ('amino-C(C)-CF3', '[*]NC(C)C(F)(F)F', 'polar'),
    ('amino-C(C)-cyclopropyl', '[*]NC(C)C1CC1', 'polar'),
    ('amino-C(C)-morpholino', '[*]NC(C)N1CCOCC1', 'polar'),
    ('amino-C(C)-piperidino', '[*]NC(C)N1CCCCC1', 'polar'),
    ('amino-C(C)-F2', '[*]NC(C)(F)F', 'polar'),
    ('amino-C(C)C-F', '[*]NC(C)CF', 'polar'),
    ('amino-C(C)C-Cl', '[*]NC(C)CCl', 'polar'),
    ('amino-C(C)C-Br', '[*]NC(C)CBr', 'polar'),
    ('amino-C(C)C-OH', '[*]NC(C)CO', 'polar'),
    ('amino-C(C)C-OMe', '[*]NC(C)COC', 'polar'),
    ('amino-C(C)C-NH2', '[*]NC(C)CN', 'polar'),
    ('amino-C(C)C-CN', '[*]NC(C)CC#N', 'polar'),
    ('amino-C(C)C-CF3', '[*]NC(C)CC(F)(F)F', 'polar'),
    ('amino-C(C)C-Me', '[*]NC(C)CC', 'polar'),
    ('amino-C(C)C-cyclopropyl', '[*]NC(C)CC1CC1', 'polar'),
    ('amino-C(C)C-morpholino', '[*]NC(C)CN1CCOCC1', 'polar'),
    ('amino-C(C)C-piperidino', '[*]NC(C)CN1CCCCC1', 'polar'),
    ('amino-C(C)C-F2', '[*]NC(C)C(F)F', 'polar'),
    ('amino-C1CC1-F', '[*]NC1CC1F', 'polar'),
    ('amino-C1CC1-Cl', '[*]NC1CC1Cl', 'polar'),
    ('amino-C1CC1-Br', '[*]NC1CC1Br', 'polar'),
    ('amino-C1CC1-OH', '[*]NC1CC1O', 'polar'),
    ('amino-C1CC1-OMe', '[*]NC1CC1OC', 'polar'),
    ('amino-C1CC1-NH2', '[*]NC1CC1N', 'polar'),
    ('amino-C1CC1-CN', '[*]NC1CC1C#N', 'polar'),
    ('amino-C1CC1-CF3', '[*]NC1CC1C(F)(F)F', 'polar'),
    ('amino-C1CC1-Me', '[*]NC1CC1C', 'polar'),
    ('amino-C1CC1-cyclopropyl', '[*]NC1CC1C1CC1', 'polar'),
    ('amino-C1CC1-morpholino', '[*]NC1CC1N1CCOCC1', 'polar'),
    ('amino-C1CC1-piperidino', '[*]NC1CC1N1CCCCC1', 'polar'),
    ('amino-C1CC1-F2', '[*]NC1CC1(F)F', 'polar'),
    ('amino-C1CCC1-F', '[*]NC1CCC1F', 'polar'),
    ('amino-C1CCC1-Cl', '[*]NC1CCC1Cl', 'polar'),
    ('amino-C1CCC1-Br', '[*]NC1CCC1Br', 'polar'),
    ('amino-C1CCC1-OH', '[*]NC1CCC1O', 'polar'),
    ('amino-C1CCC1-OMe', '[*]NC1CCC1OC', 'polar'),
    ('amino-C1CCC1-NH2', '[*]NC1CCC1N', 'polar'),
    ('amino-C1CCC1-CN', '[*]NC1CCC1C#N', 'polar'),
    ('amino-C1CCC1-CF3', '[*]NC1CCC1C(F)(F)F', 'polar'),
    ('amino-C1CCC1-Me', '[*]NC1CCC1C', 'polar'),
    ('amino-C1CCC1-cyclopropyl', '[*]NC1CCC1C1CC1', 'polar'),
    ('amino-C1CCC1-morpholino', '[*]NC1CCC1N1CCOCC1', 'polar'),
    ('amino-C1CCC1-piperidino', '[*]NC1CCC1N1CCCCC1', 'polar'),
    ('amino-C1CCC1-F2', '[*]NC1CCC1(F)F', 'polar'),
    ('amino-C(F)(F)-F', '[*]NC(F)(F)F', 'polar'),
    ('amino-C(F)(F)-Cl', '[*]NC(F)(F)Cl', 'polar'),
    ('amino-C(F)(F)-Br', '[*]NC(F)(F)Br', 'polar'),
    ('amino-C(F)(F)-OH', '[*]NC(F)(F)O', 'polar'),
    ('amino-C(F)(F)-OMe', '[*]NC(F)(F)OC', 'polar'),
    ('amino-C(F)(F)-NH2', '[*]NC(F)(F)N', 'polar'),
    ('amino-C(F)(F)-CN', '[*]NC(F)(F)C#N', 'polar'),
    ('amino-C(F)(F)-CF3', '[*]NC(F)(F)C(F)(F)F', 'polar'),
    ('amino-C(F)(F)-cyclopropyl', '[*]NC(F)(F)C1CC1', 'polar'),
    ('amino-C(F)(F)-morpholino', '[*]NC(F)(F)N1CCOCC1', 'polar'),
    ('amino-C(F)(F)-piperidino', '[*]NC(F)(F)N1CCCCC1', 'polar'),
    ('amino-C=C-H', '[*]NC=C', 'polar'),
    ('amino-C=C-F', '[*]NC=CF', 'polar'),
    ('amino-C=C-Cl', '[*]NC=CCl', 'polar'),
    ('amino-C=C-Br', '[*]NC=CBr', 'polar'),
    ('amino-C=C-OH', '[*]NC=CO', 'polar'),
    ('amino-C=C-OMe', '[*]NC=COC', 'polar'),
    ('amino-C=C-NH2', '[*]NC=CN', 'polar'),
    ('amino-C=C-CN', '[*]NC=CC#N', 'polar'),
    ('amino-C=C-CF3', '[*]NC=CC(F)(F)F', 'polar'),
    ('amino-C=C-Me', '[*]NC=CC', 'polar'),
    ('amino-C=C-cyclopropyl', '[*]NC=CC1CC1', 'polar'),
    ('amino-C=C-morpholino', '[*]NC=CN1CCOCC1', 'polar'),
    ('amino-C=C-piperidino', '[*]NC=CN1CCCCC1', 'polar'),
    ('amino-C=C-F2', '[*]NC=C(F)F', 'polar'),
    ('amino-C#C-H', '[*]NC#C', 'polar'),
    ('amino-C#C-F', '[*]NC#CF', 'polar'),
    ('amino-C#C-Cl', '[*]NC#CCl', 'polar'),
    ('amino-C#C-Br', '[*]NC#CBr', 'polar'),
    ('amino-C#C-OH', '[*]NC#CO', 'polar'),
    ('amino-C#C-OMe', '[*]NC#COC', 'polar'),
    ('amino-C#C-NH2', '[*]NC#CN', 'polar'),
    ('amino-C#C-CN', '[*]NC#CC#N', 'polar'),
    ('amino-C#C-CF3', '[*]NC#CC(F)(F)F', 'polar'),
    ('amino-C#C-Me', '[*]NC#CC', 'polar'),
    ('amino-C#C-cyclopropyl', '[*]NC#CC1CC1', 'polar'),
    ('amino-C#C-morpholino', '[*]NC#CN1CCOCC1', 'polar'),
    ('amino-C#C-piperidino', '[*]NC#CN1CCCCC1', 'polar'),
    ('thio-C-F', '[*]SCF', 'polar'),
    ('thio-C-Cl', '[*]SCCl', 'polar'),
    ('thio-C-Br', '[*]SCBr', 'polar'),
    ('thio-C-OH', '[*]SCO', 'polar'),
    ('thio-C-OMe', '[*]SCOC', 'polar'),
    ('thio-C-NH2', '[*]SCN', 'polar'),
    ('thio-C-CN', '[*]SCC#N', 'polar'),
    ('thio-C-CF3', '[*]SCC(F)(F)F', 'polar'),
    ('thio-C-cyclopropyl', '[*]SCC1CC1', 'polar'),
    ('thio-C-morpholino', '[*]SCN1CCOCC1', 'polar'),
    ('thio-C-piperidino', '[*]SCN1CCCCC1', 'polar'),
    ('thio-C-F2', '[*]SC(F)F', 'polar'),
    ('thio-CC-F', '[*]SCCF', 'polar'),
    ('thio-CC-Cl', '[*]SCCCl', 'polar'),
    ('thio-CC-Br', '[*]SCCBr', 'polar'),
    ('thio-CC-OH', '[*]SCCO', 'polar'),
    ('thio-CC-OMe', '[*]SCCOC', 'polar'),
    ('thio-CC-NH2', '[*]SCCN', 'polar'),
    ('thio-CC-CN', '[*]SCCC#N', 'polar'),
    ('thio-CC-CF3', '[*]SCCC(F)(F)F', 'polar'),
    ('thio-CC-Me', '[*]SCCC', 'polar'),
    ('thio-CC-cyclopropyl', '[*]SCCC1CC1', 'polar'),
    ('thio-CC-morpholino', '[*]SCCN1CCOCC1', 'polar'),
    ('thio-CC-piperidino', '[*]SCCN1CCCCC1', 'polar'),
    ('thio-CC-F2', '[*]SCC(F)F', 'polar'),
    ('thio-CCC-F', '[*]SCCCF', 'polar'),
    ('thio-CCC-Cl', '[*]SCCCCl', 'polar'),
    ('thio-CCC-Br', '[*]SCCCBr', 'polar'),
    ('thio-CCC-OH', '[*]SCCCO', 'polar'),
    ('thio-CCC-OMe', '[*]SCCCOC', 'polar'),
    ('thio-CCC-NH2', '[*]SCCCN', 'polar'),
    ('thio-CCC-CN', '[*]SCCCC#N', 'polar'),
    ('thio-CCC-CF3', '[*]SCCCC(F)(F)F', 'polar'),
    ('thio-CCC-Me', '[*]SCCCC', 'polar'),
    ('thio-CCC-cyclopropyl', '[*]SCCCC1CC1', 'polar'),
    ('thio-CCC-morpholino', '[*]SCCCN1CCOCC1', 'polar'),
    ('thio-CCC-piperidino', '[*]SCCCN1CCCCC1', 'polar'),
    ('thio-CCC-F2', '[*]SCCC(F)F', 'polar'),
    ('thio-C(C)-F', '[*]SC(C)F', 'polar'),
    ('thio-C(C)-Cl', '[*]SC(C)Cl', 'polar'),
    ('thio-C(C)-Br', '[*]SC(C)Br', 'polar'),
    ('thio-C(C)-OH', '[*]SC(C)O', 'polar'),
    ('thio-C(C)-OMe', '[*]SC(C)OC', 'polar'),
    ('thio-C(C)-NH2', '[*]SC(C)N', 'polar'),
    ('thio-C(C)-CN', '[*]SC(C)C#N', 'polar'),
    ('thio-C(C)-CF3', '[*]SC(C)C(F)(F)F', 'polar'),
    ('thio-C(C)-Me', '[*]SC(C)C', 'polar'),
    ('thio-C(C)-cyclopropyl', '[*]SC(C)C1CC1', 'polar'),
    ('thio-C(C)-morpholino', '[*]SC(C)N1CCOCC1', 'polar'),
    ('thio-C(C)-piperidino', '[*]SC(C)N1CCCCC1', 'polar'),
    ('thio-C(C)-F2', '[*]SC(C)(F)F', 'polar'),
    ('thio-C(C)C-F', '[*]SC(C)CF', 'polar'),
    ('thio-C(C)C-Cl', '[*]SC(C)CCl', 'polar'),
    ('thio-C(C)C-Br', '[*]SC(C)CBr', 'polar'),
    ('thio-C(C)C-OH', '[*]SC(C)CO', 'polar'),
    ('thio-C(C)C-OMe', '[*]SC(C)COC', 'polar'),
    ('thio-C(C)C-NH2', '[*]SC(C)CN', 'polar'),
    ('thio-C(C)C-CN', '[*]SC(C)CC#N', 'polar'),
    ('thio-C(C)C-CF3', '[*]SC(C)CC(F)(F)F', 'polar'),
    ('thio-C(C)C-Me', '[*]SC(C)CC', 'polar'),
    ('thio-C(C)C-cyclopropyl', '[*]SC(C)CC1CC1', 'polar'),
    ('thio-C(C)C-morpholino', '[*]SC(C)CN1CCOCC1', 'polar'),
    ('thio-C(C)C-piperidino', '[*]SC(C)CN1CCCCC1', 'polar'),
    ('thio-C(C)C-F2', '[*]SC(C)C(F)F', 'polar'),
    ('thio-C1CC1-H', '[*]SC1CC1', 'polar'),
    ('thio-C1CC1-F', '[*]SC1CC1F', 'polar'),
    ('thio-C1CC1-Cl', '[*]SC1CC1Cl', 'polar'),
    ('thio-C1CC1-Br', '[*]SC1CC1Br', 'polar'),
    ('thio-C1CC1-OH', '[*]SC1CC1O', 'polar'),
    ('thio-C1CC1-OMe', '[*]SC1CC1OC', 'polar'),
    ('thio-C1CC1-NH2', '[*]SC1CC1N', 'polar'),
    ('thio-C1CC1-CN', '[*]SC1CC1C#N', 'polar'),
    ('thio-C1CC1-CF3', '[*]SC1CC1C(F)(F)F', 'polar'),
    ('thio-C1CC1-Me', '[*]SC1CC1C', 'polar'),
    ('thio-C1CC1-cyclopropyl', '[*]SC1CC1C1CC1', 'polar'),
    ('thio-C1CC1-morpholino', '[*]SC1CC1N1CCOCC1', 'polar'),
    ('thio-C1CC1-piperidino', '[*]SC1CC1N1CCCCC1', 'polar'),
    ('thio-C1CC1-F2', '[*]SC1CC1(F)F', 'polar'),
    ('thio-C1CCC1-H', '[*]SC1CCC1', 'polar'),
    ('thio-C1CCC1-F', '[*]SC1CCC1F', 'polar'),
    ('thio-C1CCC1-Cl', '[*]SC1CCC1Cl', 'polar'),
    ('thio-C1CCC1-Br', '[*]SC1CCC1Br', 'polar'),
    ('thio-C1CCC1-OH', '[*]SC1CCC1O', 'polar'),
    ('thio-C1CCC1-OMe', '[*]SC1CCC1OC', 'polar'),
    ('thio-C1CCC1-NH2', '[*]SC1CCC1N', 'polar'),
    ('thio-C1CCC1-CN', '[*]SC1CCC1C#N', 'polar'),
    ('thio-C1CCC1-CF3', '[*]SC1CCC1C(F)(F)F', 'polar'),
    ('thio-C1CCC1-Me', '[*]SC1CCC1C', 'polar'),
    ('thio-C1CCC1-cyclopropyl', '[*]SC1CCC1C1CC1', 'polar'),
    ('thio-C1CCC1-morpholino', '[*]SC1CCC1N1CCOCC1', 'polar'),
    ('thio-C1CCC1-piperidino', '[*]SC1CCC1N1CCCCC1', 'polar'),
    ('thio-C1CCC1-F2', '[*]SC1CCC1(F)F', 'polar'),
    ('thio-C(F)(F)-F', '[*]SC(F)(F)F', 'polar'),
    ('thio-C(F)(F)-Cl', '[*]SC(F)(F)Cl', 'polar'),
    ('thio-C(F)(F)-Br', '[*]SC(F)(F)Br', 'polar'),
    ('thio-C(F)(F)-OH', '[*]SC(F)(F)O', 'polar'),
    ('thio-C(F)(F)-OMe', '[*]SC(F)(F)OC', 'polar'),
    ('thio-C(F)(F)-NH2', '[*]SC(F)(F)N', 'polar'),
    ('thio-C(F)(F)-CN', '[*]SC(F)(F)C#N', 'polar'),
    ('thio-C(F)(F)-CF3', '[*]SC(F)(F)C(F)(F)F', 'polar'),
    ('thio-C(F)(F)-cyclopropyl', '[*]SC(F)(F)C1CC1', 'polar'),
    ('thio-C(F)(F)-morpholino', '[*]SC(F)(F)N1CCOCC1', 'polar'),
    ('thio-C(F)(F)-piperidino', '[*]SC(F)(F)N1CCCCC1', 'polar'),
    ('thio-C=C-H', '[*]SC=C', 'polar'),
    ('thio-C=C-F', '[*]SC=CF', 'polar'),
    ('thio-C=C-Cl', '[*]SC=CCl', 'polar'),
    ('thio-C=C-Br', '[*]SC=CBr', 'polar'),
    ('thio-C=C-OH', '[*]SC=CO', 'polar'),
    ('thio-C=C-OMe', '[*]SC=COC', 'polar'),
    ('thio-C=C-NH2', '[*]SC=CN', 'polar'),
    ('thio-C=C-CN', '[*]SC=CC#N', 'polar'),
    ('thio-C=C-CF3', '[*]SC=CC(F)(F)F', 'polar'),
    ('thio-C=C-Me', '[*]SC=CC', 'polar'),
    ('thio-C=C-cyclopropyl', '[*]SC=CC1CC1', 'polar'),
    ('thio-C=C-morpholino', '[*]SC=CN1CCOCC1', 'polar'),
    ('thio-C=C-piperidino', '[*]SC=CN1CCCCC1', 'polar'),
    ('thio-C=C-F2', '[*]SC=C(F)F', 'polar'),
    ('thio-C#C-H', '[*]SC#C', 'polar'),
    ('thio-C#C-F', '[*]SC#CF', 'polar'),
    ('thio-C#C-Cl', '[*]SC#CCl', 'polar'),
    ('thio-C#C-Br', '[*]SC#CBr', 'polar'),
    ('thio-C#C-OH', '[*]SC#CO', 'polar'),
    ('thio-C#C-OMe', '[*]SC#COC', 'polar'),
    ('thio-C#C-NH2', '[*]SC#CN', 'polar'),
    ('thio-C#C-CN', '[*]SC#CC#N', 'polar'),
    ('thio-C#C-CF3', '[*]SC#CC(F)(F)F', 'polar'),
    ('thio-C#C-Me', '[*]SC#CC', 'polar'),
    ('thio-C#C-cyclopropyl', '[*]SC#CC1CC1', 'polar'),
    ('thio-C#C-morpholino', '[*]SC#CN1CCOCC1', 'polar'),
    ('thio-C#C-piperidino', '[*]SC#CN1CCCCC1', 'polar'),
    ('acyloxy-C-F', '[*]C(=O)OCF', 'polar'),
    ('acyloxy-C-Cl', '[*]C(=O)OCCl', 'polar'),
    ('acyloxy-C-Br', '[*]C(=O)OCBr', 'polar'),
    ('acyloxy-C-OH', '[*]C(=O)OCO', 'polar'),
    ('acyloxy-C-OMe', '[*]C(=O)OCOC', 'polar'),
    ('acyloxy-C-NH2', '[*]C(=O)OCN', 'polar'),
    ('acyloxy-C-CN', '[*]C(=O)OCC#N', 'polar'),
    ('acyloxy-C-CF3', '[*]C(=O)OCC(F)(F)F', 'polar'),
    ('acyloxy-C-cyclopropyl', '[*]C(=O)OCC1CC1', 'polar'),
    ('acyloxy-C-morpholino', '[*]C(=O)OCN1CCOCC1', 'polar'),
    ('acyloxy-C-piperidino', '[*]C(=O)OCN1CCCCC1', 'polar'),
    ('acyloxy-C-F2', '[*]C(=O)OC(F)F', 'polar'),
    ('acyloxy-CC-F', '[*]C(=O)OCCF', 'polar'),
    ('acyloxy-CC-Cl', '[*]C(=O)OCCCl', 'polar'),
    ('acyloxy-CC-Br', '[*]C(=O)OCCBr', 'polar'),
    ('acyloxy-CC-OH', '[*]C(=O)OCCO', 'polar'),
    ('acyloxy-CC-OMe', '[*]C(=O)OCCOC', 'polar'),
    ('acyloxy-CC-NH2', '[*]C(=O)OCCN', 'polar'),
    ('acyloxy-CC-CN', '[*]C(=O)OCCC#N', 'polar'),
    ('acyloxy-CC-CF3', '[*]C(=O)OCCC(F)(F)F', 'polar'),
    ('acyloxy-CC-Me', '[*]C(=O)OCCC', 'polar'),
    ('acyloxy-CC-cyclopropyl', '[*]C(=O)OCCC1CC1', 'polar'),
    ('acyloxy-CC-morpholino', '[*]C(=O)OCCN1CCOCC1', 'polar'),
    ('acyloxy-CC-piperidino', '[*]C(=O)OCCN1CCCCC1', 'polar'),
    ('acyloxy-CC-F2', '[*]C(=O)OCC(F)F', 'polar'),
    ('acyloxy-CCC-F', '[*]C(=O)OCCCF', 'polar'),
    ('acyloxy-CCC-Cl', '[*]C(=O)OCCCCl', 'polar'),
    ('acyloxy-CCC-Br', '[*]C(=O)OCCCBr', 'polar'),
    ('acyloxy-CCC-OH', '[*]C(=O)OCCCO', 'polar'),
    ('acyloxy-CCC-OMe', '[*]C(=O)OCCCOC', 'polar'),
    ('acyloxy-CCC-NH2', '[*]C(=O)OCCCN', 'polar'),
    ('acyloxy-CCC-CN', '[*]C(=O)OCCCC#N', 'polar'),
    ('acyloxy-CCC-CF3', '[*]C(=O)OCCCC(F)(F)F', 'polar'),
    ('acyloxy-CCC-Me', '[*]C(=O)OCCCC', 'polar'),
    ('acyloxy-CCC-cyclopropyl', '[*]C(=O)OCCCC1CC1', 'polar'),
    ('acyloxy-CCC-morpholino', '[*]C(=O)OCCCN1CCOCC1', 'polar'),
    ('acyloxy-CCC-piperidino', '[*]C(=O)OCCCN1CCCCC1', 'polar'),
    ('acyloxy-CCC-F2', '[*]C(=O)OCCC(F)F', 'polar'),
    ('acyloxy-C(C)-F', '[*]C(=O)OC(C)F', 'polar'),
    ('acyloxy-C(C)-Cl', '[*]C(=O)OC(C)Cl', 'polar'),
    ('acyloxy-C(C)-Br', '[*]C(=O)OC(C)Br', 'polar'),
    ('acyloxy-C(C)-OH', '[*]C(=O)OC(C)O', 'polar'),
    ('acyloxy-C(C)-OMe', '[*]C(=O)OC(C)OC', 'polar'),
    ('acyloxy-C(C)-NH2', '[*]C(=O)OC(C)N', 'polar'),
    ('acyloxy-C(C)-CN', '[*]C(=O)OC(C)C#N', 'polar'),
    ('acyloxy-C(C)-CF3', '[*]C(=O)OC(C)C(F)(F)F', 'polar'),
    ('acyloxy-C(C)-Me', '[*]C(=O)OC(C)C', 'polar'),
    ('acyloxy-C(C)-cyclopropyl', '[*]C(=O)OC(C)C1CC1', 'polar'),
    ('acyloxy-C(C)-morpholino', '[*]C(=O)OC(C)N1CCOCC1', 'polar'),
    ('acyloxy-C(C)-piperidino', '[*]C(=O)OC(C)N1CCCCC1', 'polar'),
    ('acyloxy-C(C)-F2', '[*]C(=O)OC(C)(F)F', 'polar'),
    ('acyloxy-C(C)C-F', '[*]C(=O)OC(C)CF', 'polar'),
    ('acyloxy-C(C)C-Cl', '[*]C(=O)OC(C)CCl', 'polar'),
    ('acyloxy-C(C)C-Br', '[*]C(=O)OC(C)CBr', 'polar'),
    ('acyloxy-C(C)C-OH', '[*]C(=O)OC(C)CO', 'polar'),
    ('acyloxy-C(C)C-OMe', '[*]C(=O)OC(C)COC', 'polar'),
    ('acyloxy-C(C)C-NH2', '[*]C(=O)OC(C)CN', 'polar'),
    ('acyloxy-C(C)C-CN', '[*]C(=O)OC(C)CC#N', 'polar'),
    ('acyloxy-C(C)C-CF3', '[*]C(=O)OC(C)CC(F)(F)F', 'polar'),
    ('acyloxy-C(C)C-Me', '[*]C(=O)OC(C)CC', 'polar'),
    ('acyloxy-C(C)C-cyclopropyl', '[*]C(=O)OC(C)CC1CC1', 'polar'),
    ('acyloxy-C(C)C-morpholino', '[*]C(=O)OC(C)CN1CCOCC1', 'polar'),
    ('acyloxy-C(C)C-piperidino', '[*]C(=O)OC(C)CN1CCCCC1', 'polar'),
    ('acyloxy-C(C)C-F2', '[*]C(=O)OC(C)C(F)F', 'polar'),
    ('acyloxy-C1CC1-H', '[*]C(=O)OC1CC1', 'polar'),
    ('acyloxy-C1CC1-F', '[*]C(=O)OC1CC1F', 'polar'),
    ('acyloxy-C1CC1-Cl', '[*]C(=O)OC1CC1Cl', 'polar'),
    ('acyloxy-C1CC1-Br', '[*]C(=O)OC1CC1Br', 'polar'),
    ('acyloxy-C1CC1-OH', '[*]C(=O)OC1CC1O', 'polar'),
    ('acyloxy-C1CC1-OMe', '[*]C(=O)OC1CC1OC', 'polar'),
    ('acyloxy-C1CC1-NH2', '[*]C(=O)OC1CC1N', 'polar'),
    ('acyloxy-C1CC1-CN', '[*]C(=O)OC1CC1C#N', 'polar'),
    ('acyloxy-C1CC1-CF3', '[*]C(=O)OC1CC1C(F)(F)F', 'polar'),
    ('acyloxy-C1CC1-Me', '[*]C(=O)OC1CC1C', 'polar'),
    ('acyloxy-C1CC1-cyclopropyl', '[*]C(=O)OC1CC1C1CC1', 'polar'),
    ('acyloxy-C1CC1-morpholino', '[*]C(=O)OC1CC1N1CCOCC1', 'polar'),
    ('acyloxy-C1CC1-piperidino', '[*]C(=O)OC1CC1N1CCCCC1', 'polar'),
    ('acyloxy-C1CC1-F2', '[*]C(=O)OC1CC1(F)F', 'polar'),
    ('acyloxy-C1CCC1-H', '[*]C(=O)OC1CCC1', 'polar'),
    ('acyloxy-C1CCC1-F', '[*]C(=O)OC1CCC1F', 'polar'),
    ('acyloxy-C1CCC1-Cl', '[*]C(=O)OC1CCC1Cl', 'polar'),
    ('acyloxy-C1CCC1-Br', '[*]C(=O)OC1CCC1Br', 'polar'),
    ('acyloxy-C1CCC1-OH', '[*]C(=O)OC1CCC1O', 'polar'),
    ('acyloxy-C1CCC1-OMe', '[*]C(=O)OC1CCC1OC', 'polar'),
    ('acyloxy-C1CCC1-NH2', '[*]C(=O)OC1CCC1N', 'polar'),
    ('acyloxy-C1CCC1-CN', '[*]C(=O)OC1CCC1C#N', 'polar'),
    ('acyloxy-C1CCC1-CF3', '[*]C(=O)OC1CCC1C(F)(F)F', 'polar'),
    ('acyloxy-C1CCC1-Me', '[*]C(=O)OC1CCC1C', 'polar'),
    ('acyloxy-C1CCC1-cyclopropyl', '[*]C(=O)OC1CCC1C1CC1', 'polar'),
    ('acyloxy-C1CCC1-morpholino', '[*]C(=O)OC1CCC1N1CCOCC1', 'polar'),
    ('acyloxy-C1CCC1-piperidino', '[*]C(=O)OC1CCC1N1CCCCC1', 'polar'),
    ('acyloxy-C1CCC1-F2', '[*]C(=O)OC1CCC1(F)F', 'polar'),
    ('acyloxy-C(F)(F)-F', '[*]C(=O)OC(F)(F)F', 'polar'),
    ('acyloxy-C(F)(F)-Cl', '[*]C(=O)OC(F)(F)Cl', 'polar'),
    ('acyloxy-C(F)(F)-Br', '[*]C(=O)OC(F)(F)Br', 'polar'),
    ('acyloxy-C(F)(F)-OH', '[*]C(=O)OC(F)(F)O', 'polar'),
    ('acyloxy-C(F)(F)-OMe', '[*]C(=O)OC(F)(F)OC', 'polar'),
    ('acyloxy-C(F)(F)-NH2', '[*]C(=O)OC(F)(F)N', 'polar'),
    ('acyloxy-C(F)(F)-CN', '[*]C(=O)OC(F)(F)C#N', 'polar'),
    ('acyloxy-C(F)(F)-CF3', '[*]C(=O)OC(F)(F)C(F)(F)F', 'polar'),
    ('acyloxy-C(F)(F)-cyclopropyl', '[*]C(=O)OC(F)(F)C1CC1', 'polar'),
    ('acyloxy-C(F)(F)-morpholino', '[*]C(=O)OC(F)(F)N1CCOCC1', 'polar'),
    ('acyloxy-C(F)(F)-piperidino', '[*]C(=O)OC(F)(F)N1CCCCC1', 'polar'),
    ('acyloxy-C=C-H', '[*]C(=O)OC=C', 'polar'),
    ('acyloxy-C=C-F', '[*]C(=O)OC=CF', 'polar'),
    ('acyloxy-C=C-Cl', '[*]C(=O)OC=CCl', 'polar'),
    ('acyloxy-C=C-Br', '[*]C(=O)OC=CBr', 'polar'),
    ('acyloxy-C=C-OH', '[*]C(=O)OC=CO', 'polar'),
    ('acyloxy-C=C-OMe', '[*]C(=O)OC=COC', 'polar'),
    ('acyloxy-C=C-NH2', '[*]C(=O)OC=CN', 'polar'),
    ('acyloxy-C=C-CN', '[*]C(=O)OC=CC#N', 'polar'),
    ('acyloxy-C=C-CF3', '[*]C(=O)OC=CC(F)(F)F', 'polar'),
    ('acyloxy-C=C-Me', '[*]C(=O)OC=CC', 'polar'),
    ('acyloxy-C=C-cyclopropyl', '[*]C(=O)OC=CC1CC1', 'polar'),
    ('acyloxy-C=C-morpholino', '[*]C(=O)OC=CN1CCOCC1', 'polar'),
    ('acyloxy-C=C-piperidino', '[*]C(=O)OC=CN1CCCCC1', 'polar'),
    ('acyloxy-C=C-F2', '[*]C(=O)OC=C(F)F', 'polar'),
    ('acyloxy-C#C-H', '[*]C(=O)OC#C', 'polar'),
    ('acyloxy-C#C-F', '[*]C(=O)OC#CF', 'polar'),
    ('acyloxy-C#C-Cl', '[*]C(=O)OC#CCl', 'polar'),
    ('acyloxy-C#C-Br', '[*]C(=O)OC#CBr', 'polar'),
    ('acyloxy-C#C-OH', '[*]C(=O)OC#CO', 'polar'),
    ('acyloxy-C#C-OMe', '[*]C(=O)OC#COC', 'polar'),
    ('acyloxy-C#C-NH2', '[*]C(=O)OC#CN', 'polar'),
    ('acyloxy-C#C-CN', '[*]C(=O)OC#CC#N', 'polar'),
    ('acyloxy-C#C-CF3', '[*]C(=O)OC#CC(F)(F)F', 'polar'),
    ('acyloxy-C#C-Me', '[*]C(=O)OC#CC', 'polar'),
    ('acyloxy-C#C-cyclopropyl', '[*]C(=O)OC#CC1CC1', 'polar'),
    ('acyloxy-C#C-morpholino', '[*]C(=O)OC#CN1CCOCC1', 'polar'),
    ('acyloxy-C#C-piperidino', '[*]C(=O)OC#CN1CCCCC1', 'polar'),
    ('acylamino-C-F', '[*]C(=O)NCF', 'polar'),
    ('acylamino-C-Cl', '[*]C(=O)NCCl', 'polar'),
    ('acylamino-C-Br', '[*]C(=O)NCBr', 'polar'),
    ('acylamino-C-OH', '[*]C(=O)NCO', 'polar'),
    ('acylamino-C-OMe', '[*]C(=O)NCOC', 'polar'),
    ('acylamino-C-NH2', '[*]C(=O)NCN', 'polar'),
    ('acylamino-C-CN', '[*]C(=O)NCC#N', 'polar'),
    ('acylamino-C-CF3', '[*]C(=O)NCC(F)(F)F', 'polar'),
    ('acylamino-C-cyclopropyl', '[*]C(=O)NCC1CC1', 'polar'),
    ('acylamino-C-morpholino', '[*]C(=O)NCN1CCOCC1', 'polar'),
    ('acylamino-C-piperidino', '[*]C(=O)NCN1CCCCC1', 'polar'),
    ('acylamino-C-F2', '[*]C(=O)NC(F)F', 'polar'),
    ('acylamino-CC-F', '[*]C(=O)NCCF', 'polar'),
    ('acylamino-CC-Cl', '[*]C(=O)NCCCl', 'polar'),
    ('acylamino-CC-Br', '[*]C(=O)NCCBr', 'polar'),
    ('acylamino-CC-OMe', '[*]C(=O)NCCOC', 'polar'),
    ('acylamino-CC-NH2', '[*]C(=O)NCCN', 'polar'),
    ('acylamino-CC-CN', '[*]C(=O)NCCC#N', 'polar'),
    ('acylamino-CC-CF3', '[*]C(=O)NCCC(F)(F)F', 'polar'),
    ('acylamino-CC-cyclopropyl', '[*]C(=O)NCCC1CC1', 'polar'),
    ('acylamino-CC-morpholino', '[*]C(=O)NCCN1CCOCC1', 'polar'),
    ('acylamino-CC-piperidino', '[*]C(=O)NCCN1CCCCC1', 'polar'),
    ('acylamino-CC-F2', '[*]C(=O)NCC(F)F', 'polar'),
    ('acylamino-CCC-F', '[*]C(=O)NCCCF', 'polar'),
    ('acylamino-CCC-Cl', '[*]C(=O)NCCCCl', 'polar'),
    ('acylamino-CCC-Br', '[*]C(=O)NCCCBr', 'polar'),
    ('acylamino-CCC-OH', '[*]C(=O)NCCCO', 'polar'),
    ('acylamino-CCC-OMe', '[*]C(=O)NCCCOC', 'polar'),
    ('acylamino-CCC-NH2', '[*]C(=O)NCCCN', 'polar'),
    ('acylamino-CCC-CN', '[*]C(=O)NCCCC#N', 'polar'),
    ('acylamino-CCC-CF3', '[*]C(=O)NCCCC(F)(F)F', 'polar'),
    ('acylamino-CCC-Me', '[*]C(=O)NCCCC', 'polar'),
    ('acylamino-CCC-cyclopropyl', '[*]C(=O)NCCCC1CC1', 'polar'),
    ('acylamino-CCC-morpholino', '[*]C(=O)NCCCN1CCOCC1', 'polar'),
    ('acylamino-CCC-piperidino', '[*]C(=O)NCCCN1CCCCC1', 'polar'),
    ('acylamino-CCC-F2', '[*]C(=O)NCCC(F)F', 'polar'),
    ('acylamino-C(C)-F', '[*]C(=O)NC(C)F', 'polar'),
    ('acylamino-C(C)-Cl', '[*]C(=O)NC(C)Cl', 'polar'),
    ('acylamino-C(C)-Br', '[*]C(=O)NC(C)Br', 'polar'),
    ('acylamino-C(C)-OH', '[*]C(=O)NC(C)O', 'polar'),
    ('acylamino-C(C)-OMe', '[*]C(=O)NC(C)OC', 'polar'),
    ('acylamino-C(C)-NH2', '[*]C(=O)NC(C)N', 'polar'),
    ('acylamino-C(C)-CN', '[*]C(=O)NC(C)C#N', 'polar'),
    ('acylamino-C(C)-CF3', '[*]C(=O)NC(C)C(F)(F)F', 'polar'),
    ('acylamino-C(C)-cyclopropyl', '[*]C(=O)NC(C)C1CC1', 'polar'),
    ('acylamino-C(C)-morpholino', '[*]C(=O)NC(C)N1CCOCC1', 'polar'),
    ('acylamino-C(C)-piperidino', '[*]C(=O)NC(C)N1CCCCC1', 'polar'),
    ('acylamino-C(C)-F2', '[*]C(=O)NC(C)(F)F', 'polar'),
    ('acylamino-C(C)C-F', '[*]C(=O)NC(C)CF', 'polar'),
    ('acylamino-C(C)C-Cl', '[*]C(=O)NC(C)CCl', 'polar'),
    ('acylamino-C(C)C-Br', '[*]C(=O)NC(C)CBr', 'polar'),
    ('acylamino-C(C)C-OH', '[*]C(=O)NC(C)CO', 'polar'),
    ('acylamino-C(C)C-OMe', '[*]C(=O)NC(C)COC', 'polar'),
    ('acylamino-C(C)C-NH2', '[*]C(=O)NC(C)CN', 'polar'),
    ('acylamino-C(C)C-CN', '[*]C(=O)NC(C)CC#N', 'polar'),
    ('acylamino-C(C)C-CF3', '[*]C(=O)NC(C)CC(F)(F)F', 'polar'),
    ('acylamino-C(C)C-Me', '[*]C(=O)NC(C)CC', 'polar'),
    ('acylamino-C(C)C-cyclopropyl', '[*]C(=O)NC(C)CC1CC1', 'polar'),
    ('acylamino-C(C)C-morpholino', '[*]C(=O)NC(C)CN1CCOCC1', 'polar'),
    ('acylamino-C(C)C-piperidino', '[*]C(=O)NC(C)CN1CCCCC1', 'polar'),
    ('acylamino-C(C)C-F2', '[*]C(=O)NC(C)C(F)F', 'polar'),
    ('acylamino-C1CC1-F', '[*]C(=O)NC1CC1F', 'polar'),
    ('acylamino-C1CC1-Cl', '[*]C(=O)NC1CC1Cl', 'polar'),
    ('acylamino-C1CC1-Br', '[*]C(=O)NC1CC1Br', 'polar'),
    ('acylamino-C1CC1-OH', '[*]C(=O)NC1CC1O', 'polar'),
    ('acylamino-C1CC1-OMe', '[*]C(=O)NC1CC1OC', 'polar'),
    ('acylamino-C1CC1-NH2', '[*]C(=O)NC1CC1N', 'polar'),
    ('acylamino-C1CC1-CN', '[*]C(=O)NC1CC1C#N', 'polar'),
    ('acylamino-C1CC1-CF3', '[*]C(=O)NC1CC1C(F)(F)F', 'polar'),
    ('acylamino-C1CC1-Me', '[*]C(=O)NC1CC1C', 'polar'),
    ('acylamino-C1CC1-cyclopropyl', '[*]C(=O)NC1CC1C1CC1', 'polar'),
    ('acylamino-C1CC1-morpholino', '[*]C(=O)NC1CC1N1CCOCC1', 'polar'),
    ('acylamino-C1CC1-piperidino', '[*]C(=O)NC1CC1N1CCCCC1', 'polar'),
    ('acylamino-C1CC1-F2', '[*]C(=O)NC1CC1(F)F', 'polar'),
    ('acylamino-C1CCC1-F', '[*]C(=O)NC1CCC1F', 'polar'),
    ('acylamino-C1CCC1-Cl', '[*]C(=O)NC1CCC1Cl', 'polar'),
    ('acylamino-C1CCC1-Br', '[*]C(=O)NC1CCC1Br', 'polar'),
    ('acylamino-C1CCC1-OH', '[*]C(=O)NC1CCC1O', 'polar'),
    ('acylamino-C1CCC1-OMe', '[*]C(=O)NC1CCC1OC', 'polar'),
    ('acylamino-C1CCC1-NH2', '[*]C(=O)NC1CCC1N', 'polar'),
    ('acylamino-C1CCC1-CN', '[*]C(=O)NC1CCC1C#N', 'polar'),
    ('acylamino-C1CCC1-CF3', '[*]C(=O)NC1CCC1C(F)(F)F', 'polar'),
    ('acylamino-C1CCC1-Me', '[*]C(=O)NC1CCC1C', 'polar'),
    ('acylamino-C1CCC1-cyclopropyl', '[*]C(=O)NC1CCC1C1CC1', 'polar'),
    ('acylamino-C1CCC1-morpholino', '[*]C(=O)NC1CCC1N1CCOCC1', 'polar'),
    ('acylamino-C1CCC1-piperidino', '[*]C(=O)NC1CCC1N1CCCCC1', 'polar'),
    ('acylamino-C1CCC1-F2', '[*]C(=O)NC1CCC1(F)F', 'polar'),
    ('acylamino-C(F)(F)-F', '[*]C(=O)NC(F)(F)F', 'polar'),
    ('acylamino-C(F)(F)-Cl', '[*]C(=O)NC(F)(F)Cl', 'polar'),
    ('acylamino-C(F)(F)-Br', '[*]C(=O)NC(F)(F)Br', 'polar'),
    ('acylamino-C(F)(F)-OH', '[*]C(=O)NC(F)(F)O', 'polar'),
    ('acylamino-C(F)(F)-OMe', '[*]C(=O)NC(F)(F)OC', 'polar'),
    ('acylamino-C(F)(F)-NH2', '[*]C(=O)NC(F)(F)N', 'polar'),
    ('acylamino-C(F)(F)-CN', '[*]C(=O)NC(F)(F)C#N', 'polar'),
    ('acylamino-C(F)(F)-CF3', '[*]C(=O)NC(F)(F)C(F)(F)F', 'polar'),
    ('acylamino-C(F)(F)-cyclopropyl', '[*]C(=O)NC(F)(F)C1CC1', 'polar'),
    ('acylamino-C(F)(F)-morpholino', '[*]C(=O)NC(F)(F)N1CCOCC1', 'polar'),
    ('acylamino-C(F)(F)-piperidino', '[*]C(=O)NC(F)(F)N1CCCCC1', 'polar'),
    ('acylamino-C=C-H', '[*]C(=O)NC=C', 'polar'),
    ('acylamino-C=C-F', '[*]C(=O)NC=CF', 'polar'),
    ('acylamino-C=C-Cl', '[*]C(=O)NC=CCl', 'polar'),
    ('acylamino-C=C-Br', '[*]C(=O)NC=CBr', 'polar'),
    ('acylamino-C=C-OH', '[*]C(=O)NC=CO', 'polar'),
    ('acylamino-C=C-OMe', '[*]C(=O)NC=COC', 'polar'),
    ('acylamino-C=C-NH2', '[*]C(=O)NC=CN', 'polar'),
    ('acylamino-C=C-CN', '[*]C(=O)NC=CC#N', 'polar'),
    ('acylamino-C=C-CF3', '[*]C(=O)NC=CC(F)(F)F', 'polar'),
    ('acylamino-C=C-Me', '[*]C(=O)NC=CC', 'polar'),
    ('acylamino-C=C-cyclopropyl', '[*]C(=O)NC=CC1CC1', 'polar'),
    ('acylamino-C=C-morpholino', '[*]C(=O)NC=CN1CCOCC1', 'polar'),
    ('acylamino-C=C-piperidino', '[*]C(=O)NC=CN1CCCCC1', 'polar'),
    ('acylamino-C=C-F2', '[*]C(=O)NC=C(F)F', 'polar'),
    ('acylamino-C#C-H', '[*]C(=O)NC#C', 'polar'),
    ('acylamino-C#C-F', '[*]C(=O)NC#CF', 'polar'),
    ('acylamino-C#C-Cl', '[*]C(=O)NC#CCl', 'polar'),
    ('acylamino-C#C-Br', '[*]C(=O)NC#CBr', 'polar'),
    ('acylamino-C#C-OH', '[*]C(=O)NC#CO', 'polar'),
    ('acylamino-C#C-OMe', '[*]C(=O)NC#COC', 'polar'),
    ('acylamino-C#C-NH2', '[*]C(=O)NC#CN', 'polar'),
    ('acylamino-C#C-CN', '[*]C(=O)NC#CC#N', 'polar'),
    ('acylamino-C#C-CF3', '[*]C(=O)NC#CC(F)(F)F', 'polar'),
    ('acylamino-C#C-Me', '[*]C(=O)NC#CC', 'polar'),
    ('acylamino-C#C-cyclopropyl', '[*]C(=O)NC#CC1CC1', 'polar'),
    ('acylamino-C#C-morpholino', '[*]C(=O)NC#CN1CCOCC1', 'polar'),
    ('acylamino-C#C-piperidino', '[*]C(=O)NC#CN1CCCCC1', 'polar'),
    ('bicyclo[2.2.2]octan-1-yl', '[*]C12CCC(CC1)CC2', 'hydrophobic'),
    ('adamantan-1-yl', '[*]C12CC3CC(CC(C3)C1)C2', 'hydrophobic'),
    ('adamantan-2-yl', '[*]C1C2CC3CC1CC(C2)C3', 'hydrophobic'),
    ('spiro[4.4]nonan-1-yl', '[*]C1CCCC12CCCC2', 'hydrophobic'),
    ('2-oxa-6-azaspiro[3.3]heptyl', '[*]N1CCC12COC2', 'bioisostere'),
    ('1-oxaspiro[4.4]nonan-3-yl', '[*]C1COCC12CCCC2', 'bioisostere'),
    ('5-azaspiro[2.4]heptyl', '[*]N1CCCC12CC2', 'bioisostere'),
    ('2-azaspiro[3.4]octan-2-yl', '[*]N1CCC12CCCC2', 'bioisostere'),
    ('2-azaspiro[3.5]nonan-2-yl', '[*]N1CCC12CCCCC2', 'bioisostere'),
    ('6-azaspiro[2.5]octyl', '[*]N1CCCCC12CC2', 'bioisostere'),
    ('2,2-dimethylcyclopropyl', '[*]C1CC1(C)C', 'hydrophobic'),
    ('2,2-difluorocyclopropyl', '[*]C1CC1(F)F', 'bioisostere'),
    ('1-methylcyclobutyl', '[*]C1(C)CCC1', 'hydrophobic'),
    ('1-fluorocyclobutyl', '[*]C1(F)CCC1', 'hydrophobic'),
    ('3,3-difluorocyclobutyl', '[*]C1CC(F)(F)C1', 'bioisostere'),
    ('3,3-difluorocyclopentyl', '[*]C1CCC(F)(F)C1', 'bioisostere'),
    ('4,4-difluorocyclohexyl', '[*]C1CCC(F)(F)CC1', 'bioisostere'),
    ('3-methylcyclobutyl_2', '[*]C1CC(C)C1', 'hydrophobic'),
    ('4-fluorocyclohexyl', '[*]C1CCC(F)CC1', 'hydrophobic'),
    ('4-hydroxycyclohexyl', '[*]C1CCC(O)CC1', 'polar'),
    ('4-tBu-cyclohexyl', '[*]C1CCC(C(C)(C)C)CC1', 'hydrophobic'),
    ('1-Me-pyrazol-3-yl_2', '[*]c1ccnn1C', 'aromatic'),
    ('1-Me-imidazol-4-yl', '[*]c1cn(C)cn1', 'aromatic'),
    ('5-Me-1-Me-pyrazol-3-yl', '[*]c1cc(C)nn1C', 'aromatic'),
    ('5-Me-1-Me-pyrazol-4-yl', '[*]c1c(C)nn(c1)C', 'aromatic'),
    ('5-Me-1-Me-imidazol-4-yl', '[*]c1c(C)n(C)cn1', 'aromatic'),
    ('5-Et-1-Me-pyrazol-3-yl', '[*]c1cc(CC)nn1C', 'aromatic'),
    ('5-Et-1-Me-pyrazol-4-yl', '[*]c1c(CC)nn(c1)C', 'aromatic'),
    ('5-Et-1-Me-imidazol-4-yl', '[*]c1c(CC)n(C)cn1', 'aromatic'),
    ('5-F-1-Me-pyrazol-3-yl', '[*]c1cc(F)nn1C', 'aromatic'),
    ('5-F-1-Me-pyrazol-4-yl', '[*]c1c(F)nn(c1)C', 'aromatic'),
    ('5-F-1-Me-imidazol-4-yl', '[*]c1c(F)n(C)cn1', 'aromatic'),
    ('5-Cl-1-Me-pyrazol-3-yl', '[*]c1cc(Cl)nn1C', 'aromatic'),
    ('5-Cl-1-Me-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)C', 'aromatic'),
    ('5-Cl-1-Me-imidazol-4-yl', '[*]c1c(Cl)n(C)cn1', 'aromatic'),
    ('5-CF3-1-Me-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1C', 'aromatic'),
    ('5-CF3-1-Me-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)C', 'aromatic'),
    ('5-CF3-1-Me-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(C)cn1', 'aromatic'),
    ('5-CN-1-Me-pyrazol-3-yl', '[*]c1cc(C#N)nn1C', 'aromatic'),
    ('5-CN-1-Me-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)C', 'aromatic'),
    ('5-CN-1-Me-imidazol-4-yl', '[*]c1c(C#N)n(C)cn1', 'aromatic'),
    ('5-OH-1-Me-pyrazol-3-yl', '[*]c1cc(O)nn1C', 'aromatic'),
    ('5-OH-1-Me-pyrazol-4-yl', '[*]c1c(O)nn(c1)C', 'aromatic'),
    ('5-OH-1-Me-imidazol-4-yl', '[*]c1c(O)n(C)cn1', 'aromatic'),
    ('5-OMe-1-Me-pyrazol-3-yl', '[*]c1cc(OC)nn1C', 'aromatic'),
    ('5-OMe-1-Me-pyrazol-4-yl', '[*]c1c(OC)nn(c1)C', 'aromatic'),
    ('5-OMe-1-Me-imidazol-4-yl', '[*]c1c(OC)n(C)cn1', 'aromatic'),
    ('5-NH2-1-Me-pyrazol-3-yl', '[*]c1cc(N)nn1C', 'aromatic'),
    ('5-NH2-1-Me-pyrazol-4-yl', '[*]c1c(N)nn(c1)C', 'aromatic'),
    ('5-NH2-1-Me-imidazol-4-yl', '[*]c1c(N)n(C)cn1', 'aromatic'),
    ('1-Et-pyrazol-3-yl', '[*]c1ccnn1CC', 'aromatic'),
    ('1-Et-pyrazol-4-yl', '[*]c1cnn(c1)CC', 'aromatic'),
    ('1-Et-imidazol-4-yl', '[*]c1cn(CC)cn1', 'aromatic'),
    ('5-Me-1-Et-pyrazol-3-yl', '[*]c1cc(C)nn1CC', 'aromatic'),
    ('5-Me-1-Et-pyrazol-4-yl', '[*]c1c(C)nn(c1)CC', 'aromatic'),
    ('5-Me-1-Et-imidazol-4-yl', '[*]c1c(C)n(CC)cn1', 'aromatic'),
    ('5-Et-1-Et-pyrazol-3-yl', '[*]c1cc(CC)nn1CC', 'aromatic'),
    ('5-Et-1-Et-pyrazol-4-yl', '[*]c1c(CC)nn(c1)CC', 'aromatic'),
    ('5-Et-1-Et-imidazol-4-yl', '[*]c1c(CC)n(CC)cn1', 'aromatic'),
    ('5-F-1-Et-pyrazol-3-yl', '[*]c1cc(F)nn1CC', 'aromatic'),
    ('5-F-1-Et-pyrazol-4-yl', '[*]c1c(F)nn(c1)CC', 'aromatic'),
    ('5-F-1-Et-imidazol-4-yl', '[*]c1c(F)n(CC)cn1', 'aromatic'),
    ('5-Cl-1-Et-pyrazol-3-yl', '[*]c1cc(Cl)nn1CC', 'aromatic'),
    ('5-Cl-1-Et-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)CC', 'aromatic'),
    ('5-Cl-1-Et-imidazol-4-yl', '[*]c1c(Cl)n(CC)cn1', 'aromatic'),
    ('5-CF3-1-Et-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1CC', 'aromatic'),
    ('5-CF3-1-Et-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)CC', 'aromatic'),
    ('5-CF3-1-Et-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(CC)cn1', 'aromatic'),
    ('5-CN-1-Et-pyrazol-3-yl', '[*]c1cc(C#N)nn1CC', 'aromatic'),
    ('5-CN-1-Et-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)CC', 'aromatic'),
    ('5-CN-1-Et-imidazol-4-yl', '[*]c1c(C#N)n(CC)cn1', 'aromatic'),
    ('5-OH-1-Et-pyrazol-3-yl', '[*]c1cc(O)nn1CC', 'aromatic'),
    ('5-OH-1-Et-pyrazol-4-yl', '[*]c1c(O)nn(c1)CC', 'aromatic'),
    ('5-OH-1-Et-imidazol-4-yl', '[*]c1c(O)n(CC)cn1', 'aromatic'),
    ('5-OMe-1-Et-pyrazol-3-yl', '[*]c1cc(OC)nn1CC', 'aromatic'),
    ('5-OMe-1-Et-pyrazol-4-yl', '[*]c1c(OC)nn(c1)CC', 'aromatic'),
    ('5-OMe-1-Et-imidazol-4-yl', '[*]c1c(OC)n(CC)cn1', 'aromatic'),
    ('5-NH2-1-Et-pyrazol-3-yl', '[*]c1cc(N)nn1CC', 'aromatic'),
    ('5-NH2-1-Et-pyrazol-4-yl', '[*]c1c(N)nn(c1)CC', 'aromatic'),
    ('5-NH2-1-Et-imidazol-4-yl', '[*]c1c(N)n(CC)cn1', 'aromatic'),
    ('1-iPr-pyrazol-3-yl', '[*]c1ccnn1C(C)C', 'aromatic'),
    ('1-iPr-pyrazol-4-yl', '[*]c1cnn(c1)C(C)C', 'aromatic'),
    ('1-iPr-imidazol-4-yl', '[*]c1cn(C(C)C)cn1', 'aromatic'),
    ('5-Me-1-iPr-pyrazol-3-yl', '[*]c1cc(C)nn1C(C)C', 'aromatic'),
    ('5-Me-1-iPr-pyrazol-4-yl', '[*]c1c(C)nn(c1)C(C)C', 'aromatic'),
    ('5-Me-1-iPr-imidazol-4-yl', '[*]c1c(C)n(C(C)C)cn1', 'aromatic'),
    ('5-Et-1-iPr-pyrazol-3-yl', '[*]c1cc(CC)nn1C(C)C', 'aromatic'),
    ('5-Et-1-iPr-pyrazol-4-yl', '[*]c1c(CC)nn(c1)C(C)C', 'aromatic'),
    ('5-Et-1-iPr-imidazol-4-yl', '[*]c1c(CC)n(C(C)C)cn1', 'aromatic'),
    ('5-F-1-iPr-pyrazol-3-yl', '[*]c1cc(F)nn1C(C)C', 'aromatic'),
    ('5-F-1-iPr-pyrazol-4-yl', '[*]c1c(F)nn(c1)C(C)C', 'aromatic'),
    ('5-F-1-iPr-imidazol-4-yl', '[*]c1c(F)n(C(C)C)cn1', 'aromatic'),
    ('5-Cl-1-iPr-pyrazol-3-yl', '[*]c1cc(Cl)nn1C(C)C', 'aromatic'),
    ('5-Cl-1-iPr-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)C(C)C', 'aromatic'),
    ('5-Cl-1-iPr-imidazol-4-yl', '[*]c1c(Cl)n(C(C)C)cn1', 'aromatic'),
    ('5-CF3-1-iPr-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1C(C)C', 'aromatic'),
    ('5-CF3-1-iPr-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)C(C)C', 'aromatic'),
    ('5-CF3-1-iPr-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(C(C)C)cn1', 'aromatic'),
    ('5-CN-1-iPr-pyrazol-3-yl', '[*]c1cc(C#N)nn1C(C)C', 'aromatic'),
    ('5-CN-1-iPr-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)C(C)C', 'aromatic'),
    ('5-CN-1-iPr-imidazol-4-yl', '[*]c1c(C#N)n(C(C)C)cn1', 'aromatic'),
    ('5-OH-1-iPr-pyrazol-3-yl', '[*]c1cc(O)nn1C(C)C', 'aromatic'),
    ('5-OH-1-iPr-pyrazol-4-yl', '[*]c1c(O)nn(c1)C(C)C', 'aromatic'),
    ('5-OH-1-iPr-imidazol-4-yl', '[*]c1c(O)n(C(C)C)cn1', 'aromatic'),
    ('5-OMe-1-iPr-pyrazol-3-yl', '[*]c1cc(OC)nn1C(C)C', 'aromatic'),
    ('5-OMe-1-iPr-pyrazol-4-yl', '[*]c1c(OC)nn(c1)C(C)C', 'aromatic'),
    ('5-OMe-1-iPr-imidazol-4-yl', '[*]c1c(OC)n(C(C)C)cn1', 'aromatic'),
    ('5-NH2-1-iPr-pyrazol-3-yl', '[*]c1cc(N)nn1C(C)C', 'aromatic'),
    ('5-NH2-1-iPr-pyrazol-4-yl', '[*]c1c(N)nn(c1)C(C)C', 'aromatic'),
    ('5-NH2-1-iPr-imidazol-4-yl', '[*]c1c(N)n(C(C)C)cn1', 'aromatic'),
    ('1-cPr-pyrazol-3-yl', '[*]c1ccnn1C1CC1', 'aromatic'),
    ('1-cPr-pyrazol-4-yl', '[*]c1cnn(c1)C1CC1', 'aromatic'),
    ('1-cPr-imidazol-4-yl', '[*]c1cn(C1CC1)cn1', 'aromatic'),
    ('5-Me-1-cPr-pyrazol-3-yl', '[*]c1cc(C)nn1C1CC1', 'aromatic'),
    ('5-Me-1-cPr-pyrazol-4-yl', '[*]c1c(C)nn(c1)C1CC1', 'aromatic'),
    ('5-Me-1-cPr-imidazol-4-yl', '[*]c1c(C)n(C1CC1)cn1', 'aromatic'),
    ('5-Et-1-cPr-pyrazol-3-yl', '[*]c1cc(CC)nn1C1CC1', 'aromatic'),
    ('5-Et-1-cPr-pyrazol-4-yl', '[*]c1c(CC)nn(c1)C1CC1', 'aromatic'),
    ('5-Et-1-cPr-imidazol-4-yl', '[*]c1c(CC)n(C1CC1)cn1', 'aromatic'),
    ('5-F-1-cPr-pyrazol-3-yl', '[*]c1cc(F)nn1C1CC1', 'aromatic'),
    ('5-F-1-cPr-pyrazol-4-yl', '[*]c1c(F)nn(c1)C1CC1', 'aromatic'),
    ('5-F-1-cPr-imidazol-4-yl', '[*]c1c(F)n(C1CC1)cn1', 'aromatic'),
    ('5-Cl-1-cPr-pyrazol-3-yl', '[*]c1cc(Cl)nn1C1CC1', 'aromatic'),
    ('5-Cl-1-cPr-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)C1CC1', 'aromatic'),
    ('5-Cl-1-cPr-imidazol-4-yl', '[*]c1c(Cl)n(C1CC1)cn1', 'aromatic'),
    ('5-CF3-1-cPr-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1C1CC1', 'aromatic'),
    ('5-CF3-1-cPr-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)C1CC1', 'aromatic'),
    ('5-CF3-1-cPr-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(C1CC1)cn1', 'aromatic'),
    ('5-CN-1-cPr-pyrazol-3-yl', '[*]c1cc(C#N)nn1C1CC1', 'aromatic'),
    ('5-CN-1-cPr-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)C1CC1', 'aromatic'),
    ('5-CN-1-cPr-imidazol-4-yl', '[*]c1c(C#N)n(C1CC1)cn1', 'aromatic'),
    ('5-OH-1-cPr-pyrazol-3-yl', '[*]c1cc(O)nn1C1CC1', 'aromatic'),
    ('5-OH-1-cPr-pyrazol-4-yl', '[*]c1c(O)nn(c1)C1CC1', 'aromatic'),
    ('5-OH-1-cPr-imidazol-4-yl', '[*]c1c(O)n(C1CC1)cn1', 'aromatic'),
    ('5-OMe-1-cPr-pyrazol-3-yl', '[*]c1cc(OC)nn1C1CC1', 'aromatic'),
    ('5-OMe-1-cPr-pyrazol-4-yl', '[*]c1c(OC)nn(c1)C1CC1', 'aromatic'),
    ('5-OMe-1-cPr-imidazol-4-yl', '[*]c1c(OC)n(C1CC1)cn1', 'aromatic'),
    ('5-NH2-1-cPr-pyrazol-3-yl', '[*]c1cc(N)nn1C1CC1', 'aromatic'),
    ('5-NH2-1-cPr-pyrazol-4-yl', '[*]c1c(N)nn(c1)C1CC1', 'aromatic'),
    ('5-NH2-1-cPr-imidazol-4-yl', '[*]c1c(N)n(C1CC1)cn1', 'aromatic'),
    ('1-Bn-pyrazol-3-yl', '[*]c1ccnn1Cc1ccccc1', 'aromatic'),
    ('1-Bn-pyrazol-4-yl', '[*]c1cnn(c1)Cc1ccccc1', 'aromatic'),
    ('1-Bn-imidazol-4-yl', '[*]c1cn(Cc1ccccc1)cn1', 'aromatic'),
    ('5-Me-1-Bn-pyrazol-3-yl', '[*]c1cc(C)nn1Cc1ccccc1', 'aromatic'),
    ('5-Me-1-Bn-pyrazol-4-yl', '[*]c1c(C)nn(c1)Cc1ccccc1', 'aromatic'),
    ('5-Me-1-Bn-imidazol-4-yl', '[*]c1c(C)n(Cc1ccccc1)cn1', 'aromatic'),
    ('5-Et-1-Bn-pyrazol-3-yl', '[*]c1cc(CC)nn1Cc1ccccc1', 'aromatic'),
    ('5-Et-1-Bn-pyrazol-4-yl', '[*]c1c(CC)nn(c1)Cc1ccccc1', 'aromatic'),
    ('5-Et-1-Bn-imidazol-4-yl', '[*]c1c(CC)n(Cc1ccccc1)cn1', 'aromatic'),
    ('5-F-1-Bn-pyrazol-3-yl', '[*]c1cc(F)nn1Cc1ccccc1', 'aromatic'),
    ('5-F-1-Bn-pyrazol-4-yl', '[*]c1c(F)nn(c1)Cc1ccccc1', 'aromatic'),
    ('5-F-1-Bn-imidazol-4-yl', '[*]c1c(F)n(Cc1ccccc1)cn1', 'aromatic'),
    ('5-Cl-1-Bn-pyrazol-3-yl', '[*]c1cc(Cl)nn1Cc1ccccc1', 'aromatic'),
    ('5-Cl-1-Bn-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)Cc1ccccc1', 'aromatic'),
    ('5-Cl-1-Bn-imidazol-4-yl', '[*]c1c(Cl)n(Cc1ccccc1)cn1', 'aromatic'),
    ('5-CF3-1-Bn-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1Cc1ccccc1', 'aromatic'),
    ('5-CF3-1-Bn-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)Cc1ccccc1', 'aromatic'),
    ('5-CF3-1-Bn-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(Cc1ccccc1)cn1', 'aromatic'),
    ('5-CN-1-Bn-pyrazol-3-yl', '[*]c1cc(C#N)nn1Cc1ccccc1', 'aromatic'),
    ('5-CN-1-Bn-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)Cc1ccccc1', 'aromatic'),
    ('5-CN-1-Bn-imidazol-4-yl', '[*]c1c(C#N)n(Cc1ccccc1)cn1', 'aromatic'),
    ('5-OH-1-Bn-pyrazol-3-yl', '[*]c1cc(O)nn1Cc1ccccc1', 'aromatic'),
    ('5-OH-1-Bn-pyrazol-4-yl', '[*]c1c(O)nn(c1)Cc1ccccc1', 'aromatic'),
    ('5-OH-1-Bn-imidazol-4-yl', '[*]c1c(O)n(Cc1ccccc1)cn1', 'aromatic'),
    ('5-OMe-1-Bn-pyrazol-3-yl', '[*]c1cc(OC)nn1Cc1ccccc1', 'aromatic'),
    ('5-OMe-1-Bn-pyrazol-4-yl', '[*]c1c(OC)nn(c1)Cc1ccccc1', 'aromatic'),
    ('5-OMe-1-Bn-imidazol-4-yl', '[*]c1c(OC)n(Cc1ccccc1)cn1', 'aromatic'),
    ('5-NH2-1-Bn-pyrazol-3-yl', '[*]c1cc(N)nn1Cc1ccccc1', 'aromatic'),
    ('5-NH2-1-Bn-pyrazol-4-yl', '[*]c1c(N)nn(c1)Cc1ccccc1', 'aromatic'),
    ('5-NH2-1-Bn-imidazol-4-yl', '[*]c1c(N)n(Cc1ccccc1)cn1', 'aromatic'),
    ('1-4-FBn-pyrazol-3-yl', '[*]c1ccnn1Cc1ccc(F)cc1', 'aromatic'),
    ('1-4-FBn-pyrazol-4-yl', '[*]c1cnn(c1)Cc1ccc(F)cc1', 'aromatic'),
    ('1-4-FBn-imidazol-4-yl', '[*]c1cn(Cc1ccc(F)cc1)cn1', 'aromatic'),
    ('5-Me-1-4-FBn-pyrazol-3-yl', '[*]c1cc(C)nn1Cc1ccc(F)cc1', 'aromatic'),
    ('5-Me-1-4-FBn-pyrazol-4-yl', '[*]c1c(C)nn(c1)Cc1ccc(F)cc1', 'aromatic'),
    ('5-Me-1-4-FBn-imidazol-4-yl', '[*]c1c(C)n(Cc1ccc(F)cc1)cn1', 'aromatic'),
    ('5-Et-1-4-FBn-pyrazol-3-yl', '[*]c1cc(CC)nn1Cc1ccc(F)cc1', 'aromatic'),
    ('5-Et-1-4-FBn-pyrazol-4-yl', '[*]c1c(CC)nn(c1)Cc1ccc(F)cc1', 'aromatic'),
    ('5-Et-1-4-FBn-imidazol-4-yl', '[*]c1c(CC)n(Cc1ccc(F)cc1)cn1', 'aromatic'),
    ('5-F-1-4-FBn-pyrazol-3-yl', '[*]c1cc(F)nn1Cc1ccc(F)cc1', 'aromatic'),
    ('5-F-1-4-FBn-pyrazol-4-yl', '[*]c1c(F)nn(c1)Cc1ccc(F)cc1', 'aromatic'),
    ('5-F-1-4-FBn-imidazol-4-yl', '[*]c1c(F)n(Cc1ccc(F)cc1)cn1', 'aromatic'),
    ('5-Cl-1-4-FBn-pyrazol-3-yl', '[*]c1cc(Cl)nn1Cc1ccc(F)cc1', 'aromatic'),
    ('5-Cl-1-4-FBn-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)Cc1ccc(F)cc1', 'aromatic'),
    ('5-Cl-1-4-FBn-imidazol-4-yl', '[*]c1c(Cl)n(Cc1ccc(F)cc1)cn1', 'aromatic'),
    ('5-CF3-1-4-FBn-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1Cc1ccc(F)cc1', 'aromatic'),
    ('5-CF3-1-4-FBn-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)Cc1ccc(F)cc1', 'aromatic'),
    ('5-CF3-1-4-FBn-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(Cc1ccc(F)cc1)cn1', 'aromatic'),
    ('5-CN-1-4-FBn-pyrazol-3-yl', '[*]c1cc(C#N)nn1Cc1ccc(F)cc1', 'aromatic'),
    ('5-CN-1-4-FBn-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)Cc1ccc(F)cc1', 'aromatic'),
    ('5-CN-1-4-FBn-imidazol-4-yl', '[*]c1c(C#N)n(Cc1ccc(F)cc1)cn1', 'aromatic'),
    ('5-OH-1-4-FBn-pyrazol-3-yl', '[*]c1cc(O)nn1Cc1ccc(F)cc1', 'aromatic'),
    ('5-OH-1-4-FBn-pyrazol-4-yl', '[*]c1c(O)nn(c1)Cc1ccc(F)cc1', 'aromatic'),
    ('5-OH-1-4-FBn-imidazol-4-yl', '[*]c1c(O)n(Cc1ccc(F)cc1)cn1', 'aromatic'),
    ('5-OMe-1-4-FBn-pyrazol-3-yl', '[*]c1cc(OC)nn1Cc1ccc(F)cc1', 'aromatic'),
    ('5-OMe-1-4-FBn-pyrazol-4-yl', '[*]c1c(OC)nn(c1)Cc1ccc(F)cc1', 'aromatic'),
    ('5-OMe-1-4-FBn-imidazol-4-yl', '[*]c1c(OC)n(Cc1ccc(F)cc1)cn1', 'aromatic'),
    ('5-NH2-1-4-FBn-pyrazol-3-yl', '[*]c1cc(N)nn1Cc1ccc(F)cc1', 'aromatic'),
    ('5-NH2-1-4-FBn-pyrazol-4-yl', '[*]c1c(N)nn(c1)Cc1ccc(F)cc1', 'aromatic'),
    ('5-NH2-1-4-FBn-imidazol-4-yl', '[*]c1c(N)n(Cc1ccc(F)cc1)cn1', 'aromatic'),
    ('1-CH2CN-pyrazol-3-yl', '[*]c1ccnn1CC#N', 'aromatic'),
    ('1-CH2CN-pyrazol-4-yl', '[*]c1cnn(c1)CC#N', 'aromatic'),
    ('1-CH2CN-imidazol-4-yl', '[*]c1cn(CC#N)cn1', 'aromatic'),
    ('5-Me-1-CH2CN-pyrazol-3-yl', '[*]c1cc(C)nn1CC#N', 'aromatic'),
    ('5-Me-1-CH2CN-pyrazol-4-yl', '[*]c1c(C)nn(c1)CC#N', 'aromatic'),
    ('5-Me-1-CH2CN-imidazol-4-yl', '[*]c1c(C)n(CC#N)cn1', 'aromatic'),
    ('5-Et-1-CH2CN-pyrazol-3-yl', '[*]c1cc(CC)nn1CC#N', 'aromatic'),
    ('5-Et-1-CH2CN-pyrazol-4-yl', '[*]c1c(CC)nn(c1)CC#N', 'aromatic'),
    ('5-Et-1-CH2CN-imidazol-4-yl', '[*]c1c(CC)n(CC#N)cn1', 'aromatic'),
    ('5-F-1-CH2CN-pyrazol-3-yl', '[*]c1cc(F)nn1CC#N', 'aromatic'),
    ('5-F-1-CH2CN-pyrazol-4-yl', '[*]c1c(F)nn(c1)CC#N', 'aromatic'),
    ('5-F-1-CH2CN-imidazol-4-yl', '[*]c1c(F)n(CC#N)cn1', 'aromatic'),
    ('5-Cl-1-CH2CN-pyrazol-3-yl', '[*]c1cc(Cl)nn1CC#N', 'aromatic'),
    ('5-Cl-1-CH2CN-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)CC#N', 'aromatic'),
    ('5-Cl-1-CH2CN-imidazol-4-yl', '[*]c1c(Cl)n(CC#N)cn1', 'aromatic'),
    ('5-CF3-1-CH2CN-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1CC#N', 'aromatic'),
    ('5-CF3-1-CH2CN-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)CC#N', 'aromatic'),
    ('5-CF3-1-CH2CN-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(CC#N)cn1', 'aromatic'),
    ('5-CN-1-CH2CN-pyrazol-3-yl', '[*]c1cc(C#N)nn1CC#N', 'aromatic'),
    ('5-CN-1-CH2CN-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)CC#N', 'aromatic'),
    ('5-CN-1-CH2CN-imidazol-4-yl', '[*]c1c(C#N)n(CC#N)cn1', 'aromatic'),
    ('5-OH-1-CH2CN-pyrazol-3-yl', '[*]c1cc(O)nn1CC#N', 'aromatic'),
    ('5-OH-1-CH2CN-pyrazol-4-yl', '[*]c1c(O)nn(c1)CC#N', 'aromatic'),
    ('5-OH-1-CH2CN-imidazol-4-yl', '[*]c1c(O)n(CC#N)cn1', 'aromatic'),
    ('5-OMe-1-CH2CN-pyrazol-3-yl', '[*]c1cc(OC)nn1CC#N', 'aromatic'),
    ('5-OMe-1-CH2CN-pyrazol-4-yl', '[*]c1c(OC)nn(c1)CC#N', 'aromatic'),
    ('5-OMe-1-CH2CN-imidazol-4-yl', '[*]c1c(OC)n(CC#N)cn1', 'aromatic'),
    ('5-NH2-1-CH2CN-pyrazol-3-yl', '[*]c1cc(N)nn1CC#N', 'aromatic'),
    ('5-NH2-1-CH2CN-pyrazol-4-yl', '[*]c1c(N)nn(c1)CC#N', 'aromatic'),
    ('5-NH2-1-CH2CN-imidazol-4-yl', '[*]c1c(N)n(CC#N)cn1', 'aromatic'),
    ('1-CH2OH-pyrazol-3-yl', '[*]c1ccnn1CO', 'aromatic'),
    ('1-CH2OH-pyrazol-4-yl', '[*]c1cnn(c1)CO', 'aromatic'),
    ('1-CH2OH-imidazol-4-yl', '[*]c1cn(CO)cn1', 'aromatic'),
    ('5-Me-1-CH2OH-pyrazol-3-yl', '[*]c1cc(C)nn1CO', 'aromatic'),
    ('5-Me-1-CH2OH-pyrazol-4-yl', '[*]c1c(C)nn(c1)CO', 'aromatic'),
    ('5-Me-1-CH2OH-imidazol-4-yl', '[*]c1c(C)n(CO)cn1', 'aromatic'),
    ('5-Et-1-CH2OH-pyrazol-3-yl', '[*]c1cc(CC)nn1CO', 'aromatic'),
    ('5-Et-1-CH2OH-pyrazol-4-yl', '[*]c1c(CC)nn(c1)CO', 'aromatic'),
    ('5-Et-1-CH2OH-imidazol-4-yl', '[*]c1c(CC)n(CO)cn1', 'aromatic'),
    ('5-F-1-CH2OH-pyrazol-3-yl', '[*]c1cc(F)nn1CO', 'aromatic'),
    ('5-F-1-CH2OH-pyrazol-4-yl', '[*]c1c(F)nn(c1)CO', 'aromatic'),
    ('5-F-1-CH2OH-imidazol-4-yl', '[*]c1c(F)n(CO)cn1', 'aromatic'),
    ('5-Cl-1-CH2OH-pyrazol-3-yl', '[*]c1cc(Cl)nn1CO', 'aromatic'),
    ('5-Cl-1-CH2OH-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)CO', 'aromatic'),
    ('5-Cl-1-CH2OH-imidazol-4-yl', '[*]c1c(Cl)n(CO)cn1', 'aromatic'),
    ('5-CF3-1-CH2OH-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1CO', 'aromatic'),
    ('5-CF3-1-CH2OH-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)CO', 'aromatic'),
    ('5-CF3-1-CH2OH-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(CO)cn1', 'aromatic'),
    ('5-CN-1-CH2OH-pyrazol-3-yl', '[*]c1cc(C#N)nn1CO', 'aromatic'),
    ('5-CN-1-CH2OH-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)CO', 'aromatic'),
    ('5-CN-1-CH2OH-imidazol-4-yl', '[*]c1c(C#N)n(CO)cn1', 'aromatic'),
    ('5-OH-1-CH2OH-pyrazol-3-yl', '[*]c1cc(O)nn1CO', 'aromatic'),
    ('5-OH-1-CH2OH-pyrazol-4-yl', '[*]c1c(O)nn(c1)CO', 'aromatic'),
    ('5-OH-1-CH2OH-imidazol-4-yl', '[*]c1c(O)n(CO)cn1', 'aromatic'),
    ('5-OMe-1-CH2OH-pyrazol-3-yl', '[*]c1cc(OC)nn1CO', 'aromatic'),
    ('5-OMe-1-CH2OH-pyrazol-4-yl', '[*]c1c(OC)nn(c1)CO', 'aromatic'),
    ('5-OMe-1-CH2OH-imidazol-4-yl', '[*]c1c(OC)n(CO)cn1', 'aromatic'),
    ('5-NH2-1-CH2OH-pyrazol-3-yl', '[*]c1cc(N)nn1CO', 'aromatic'),
    ('5-NH2-1-CH2OH-pyrazol-4-yl', '[*]c1c(N)nn(c1)CO', 'aromatic'),
    ('5-NH2-1-CH2OH-imidazol-4-yl', '[*]c1c(N)n(CO)cn1', 'aromatic'),
    ('1-allyl-pyrazol-3-yl', '[*]c1ccnn1CC=C', 'aromatic'),
    ('1-allyl-pyrazol-4-yl', '[*]c1cnn(c1)CC=C', 'aromatic'),
    ('1-allyl-imidazol-4-yl', '[*]c1cn(CC=C)cn1', 'aromatic'),
    ('5-Me-1-allyl-pyrazol-3-yl', '[*]c1cc(C)nn1CC=C', 'aromatic'),
    ('5-Me-1-allyl-pyrazol-4-yl', '[*]c1c(C)nn(c1)CC=C', 'aromatic'),
    ('5-Me-1-allyl-imidazol-4-yl', '[*]c1c(C)n(CC=C)cn1', 'aromatic'),
    ('5-Et-1-allyl-pyrazol-3-yl', '[*]c1cc(CC)nn1CC=C', 'aromatic'),
    ('5-Et-1-allyl-pyrazol-4-yl', '[*]c1c(CC)nn(c1)CC=C', 'aromatic'),
    ('5-Et-1-allyl-imidazol-4-yl', '[*]c1c(CC)n(CC=C)cn1', 'aromatic'),
    ('5-F-1-allyl-pyrazol-3-yl', '[*]c1cc(F)nn1CC=C', 'aromatic'),
    ('5-F-1-allyl-pyrazol-4-yl', '[*]c1c(F)nn(c1)CC=C', 'aromatic'),
    ('5-F-1-allyl-imidazol-4-yl', '[*]c1c(F)n(CC=C)cn1', 'aromatic'),
    ('5-Cl-1-allyl-pyrazol-3-yl', '[*]c1cc(Cl)nn1CC=C', 'aromatic'),
    ('5-Cl-1-allyl-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)CC=C', 'aromatic'),
    ('5-Cl-1-allyl-imidazol-4-yl', '[*]c1c(Cl)n(CC=C)cn1', 'aromatic'),
    ('5-CF3-1-allyl-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1CC=C', 'aromatic'),
    ('5-CF3-1-allyl-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)CC=C', 'aromatic'),
    ('5-CF3-1-allyl-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(CC=C)cn1', 'aromatic'),
    ('5-CN-1-allyl-pyrazol-3-yl', '[*]c1cc(C#N)nn1CC=C', 'aromatic'),
    ('5-CN-1-allyl-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)CC=C', 'aromatic'),
    ('5-CN-1-allyl-imidazol-4-yl', '[*]c1c(C#N)n(CC=C)cn1', 'aromatic'),
    ('5-OH-1-allyl-pyrazol-3-yl', '[*]c1cc(O)nn1CC=C', 'aromatic'),
    ('5-OH-1-allyl-pyrazol-4-yl', '[*]c1c(O)nn(c1)CC=C', 'aromatic'),
    ('5-OH-1-allyl-imidazol-4-yl', '[*]c1c(O)n(CC=C)cn1', 'aromatic'),
    ('5-OMe-1-allyl-pyrazol-3-yl', '[*]c1cc(OC)nn1CC=C', 'aromatic'),
    ('5-OMe-1-allyl-pyrazol-4-yl', '[*]c1c(OC)nn(c1)CC=C', 'aromatic'),
    ('5-OMe-1-allyl-imidazol-4-yl', '[*]c1c(OC)n(CC=C)cn1', 'aromatic'),
    ('5-NH2-1-allyl-pyrazol-3-yl', '[*]c1cc(N)nn1CC=C', 'aromatic'),
    ('5-NH2-1-allyl-pyrazol-4-yl', '[*]c1c(N)nn(c1)CC=C', 'aromatic'),
    ('5-NH2-1-allyl-imidazol-4-yl', '[*]c1c(N)n(CC=C)cn1', 'aromatic'),
    ('1-tBu-pyrazol-3-yl', '[*]c1ccnn1C(C)(C)C', 'aromatic'),
    ('1-tBu-pyrazol-4-yl', '[*]c1cnn(c1)C(C)(C)C', 'aromatic'),
    ('1-tBu-imidazol-4-yl', '[*]c1cn(C(C)(C)C)cn1', 'aromatic'),
    ('5-Me-1-tBu-pyrazol-3-yl', '[*]c1cc(C)nn1C(C)(C)C', 'aromatic'),
    ('5-Me-1-tBu-pyrazol-4-yl', '[*]c1c(C)nn(c1)C(C)(C)C', 'aromatic'),
    ('5-Me-1-tBu-imidazol-4-yl', '[*]c1c(C)n(C(C)(C)C)cn1', 'aromatic'),
    ('5-Et-1-tBu-pyrazol-3-yl', '[*]c1cc(CC)nn1C(C)(C)C', 'aromatic'),
    ('5-Et-1-tBu-pyrazol-4-yl', '[*]c1c(CC)nn(c1)C(C)(C)C', 'aromatic'),
    ('5-Et-1-tBu-imidazol-4-yl', '[*]c1c(CC)n(C(C)(C)C)cn1', 'aromatic'),
    ('5-F-1-tBu-pyrazol-3-yl', '[*]c1cc(F)nn1C(C)(C)C', 'aromatic'),
    ('5-F-1-tBu-pyrazol-4-yl', '[*]c1c(F)nn(c1)C(C)(C)C', 'aromatic'),
    ('5-F-1-tBu-imidazol-4-yl', '[*]c1c(F)n(C(C)(C)C)cn1', 'aromatic'),
    ('5-Cl-1-tBu-pyrazol-3-yl', '[*]c1cc(Cl)nn1C(C)(C)C', 'aromatic'),
    ('5-Cl-1-tBu-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)C(C)(C)C', 'aromatic'),
    ('5-Cl-1-tBu-imidazol-4-yl', '[*]c1c(Cl)n(C(C)(C)C)cn1', 'aromatic'),
    ('5-CF3-1-tBu-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1C(C)(C)C', 'aromatic'),
    ('5-CF3-1-tBu-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)C(C)(C)C', 'aromatic'),
    ('5-CF3-1-tBu-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(C(C)(C)C)cn1', 'aromatic'),
    ('5-CN-1-tBu-pyrazol-3-yl', '[*]c1cc(C#N)nn1C(C)(C)C', 'aromatic'),
    ('5-CN-1-tBu-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)C(C)(C)C', 'aromatic'),
    ('5-CN-1-tBu-imidazol-4-yl', '[*]c1c(C#N)n(C(C)(C)C)cn1', 'aromatic'),
    ('5-OH-1-tBu-pyrazol-3-yl', '[*]c1cc(O)nn1C(C)(C)C', 'aromatic'),
    ('5-OH-1-tBu-pyrazol-4-yl', '[*]c1c(O)nn(c1)C(C)(C)C', 'aromatic'),
    ('5-OH-1-tBu-imidazol-4-yl', '[*]c1c(O)n(C(C)(C)C)cn1', 'aromatic'),
    ('5-OMe-1-tBu-pyrazol-3-yl', '[*]c1cc(OC)nn1C(C)(C)C', 'aromatic'),
    ('5-OMe-1-tBu-pyrazol-4-yl', '[*]c1c(OC)nn(c1)C(C)(C)C', 'aromatic'),
    ('5-OMe-1-tBu-imidazol-4-yl', '[*]c1c(OC)n(C(C)(C)C)cn1', 'aromatic'),
    ('5-NH2-1-tBu-pyrazol-3-yl', '[*]c1cc(N)nn1C(C)(C)C', 'aromatic'),
    ('5-NH2-1-tBu-pyrazol-4-yl', '[*]c1c(N)nn(c1)C(C)(C)C', 'aromatic'),
    ('5-NH2-1-tBu-imidazol-4-yl', '[*]c1c(N)n(C(C)(C)C)cn1', 'aromatic'),
    ('1-CHF2-pyrazol-3-yl', '[*]c1ccnn1C(F)F', 'aromatic'),
    ('1-CHF2-pyrazol-4-yl', '[*]c1cnn(c1)C(F)F', 'aromatic'),
    ('1-CHF2-imidazol-4-yl', '[*]c1cn(C(F)F)cn1', 'aromatic'),
    ('5-Me-1-CHF2-pyrazol-3-yl', '[*]c1cc(C)nn1C(F)F', 'aromatic'),
    ('5-Me-1-CHF2-pyrazol-4-yl', '[*]c1c(C)nn(c1)C(F)F', 'aromatic'),
    ('5-Me-1-CHF2-imidazol-4-yl', '[*]c1c(C)n(C(F)F)cn1', 'aromatic'),
    ('5-Et-1-CHF2-pyrazol-3-yl', '[*]c1cc(CC)nn1C(F)F', 'aromatic'),
    ('5-Et-1-CHF2-pyrazol-4-yl', '[*]c1c(CC)nn(c1)C(F)F', 'aromatic'),
    ('5-Et-1-CHF2-imidazol-4-yl', '[*]c1c(CC)n(C(F)F)cn1', 'aromatic'),
    ('5-F-1-CHF2-pyrazol-3-yl', '[*]c1cc(F)nn1C(F)F', 'aromatic'),
    ('5-F-1-CHF2-pyrazol-4-yl', '[*]c1c(F)nn(c1)C(F)F', 'aromatic'),
    ('5-F-1-CHF2-imidazol-4-yl', '[*]c1c(F)n(C(F)F)cn1', 'aromatic'),
    ('5-Cl-1-CHF2-pyrazol-3-yl', '[*]c1cc(Cl)nn1C(F)F', 'aromatic'),
    ('5-Cl-1-CHF2-pyrazol-4-yl', '[*]c1c(Cl)nn(c1)C(F)F', 'aromatic'),
    ('5-Cl-1-CHF2-imidazol-4-yl', '[*]c1c(Cl)n(C(F)F)cn1', 'aromatic'),
    ('5-CF3-1-CHF2-pyrazol-3-yl', '[*]c1cc(C(F)(F)F)nn1C(F)F', 'aromatic'),
    ('5-CF3-1-CHF2-pyrazol-4-yl', '[*]c1c(C(F)(F)F)nn(c1)C(F)F', 'aromatic'),
    ('5-CF3-1-CHF2-imidazol-4-yl', '[*]c1c(C(F)(F)F)n(C(F)F)cn1', 'aromatic'),
    ('5-CN-1-CHF2-pyrazol-3-yl', '[*]c1cc(C#N)nn1C(F)F', 'aromatic'),
    ('5-CN-1-CHF2-pyrazol-4-yl', '[*]c1c(C#N)nn(c1)C(F)F', 'aromatic'),
    ('5-CN-1-CHF2-imidazol-4-yl', '[*]c1c(C#N)n(C(F)F)cn1', 'aromatic'),
    ('5-OH-1-CHF2-pyrazol-3-yl', '[*]c1cc(O)nn1C(F)F', 'aromatic'),
    ('5-OH-1-CHF2-pyrazol-4-yl', '[*]c1c(O)nn(c1)C(F)F', 'aromatic'),
    ('5-OH-1-CHF2-imidazol-4-yl', '[*]c1c(O)n(C(F)F)cn1', 'aromatic'),
    ('5-OMe-1-CHF2-pyrazol-3-yl', '[*]c1cc(OC)nn1C(F)F', 'aromatic'),
    ('5-OMe-1-CHF2-pyrazol-4-yl', '[*]c1c(OC)nn(c1)C(F)F', 'aromatic'),
    ('5-OMe-1-CHF2-imidazol-4-yl', '[*]c1c(OC)n(C(F)F)cn1', 'aromatic'),
    ('5-NH2-1-CHF2-pyrazol-3-yl', '[*]c1cc(N)nn1C(F)F', 'aromatic'),
    ('5-NH2-1-CHF2-pyrazol-4-yl', '[*]c1c(N)nn(c1)C(F)F', 'aromatic'),
    ('5-NH2-1-CHF2-imidazol-4-yl', '[*]c1c(N)n(C(F)F)cn1', 'aromatic'),
    ('1,4-dioxanyl', '[*]C1COCCO1', 'polar'),
    ('C-tetrahydropyranyl', '[*]CC1CCOCC1', 'polar'),
    ('C-1,4-dioxanyl', '[*]CC1COCCO1', 'polar'),
    ('CC-tetrahydrofuranyl', '[*]CCC1CCCO1', 'polar'),
    ('CC-tetrahydropyranyl', '[*]CCC1CCOCC1', 'polar'),
    ('CC-1,4-dioxanyl', '[*]CCC1COCCO1', 'polar'),
    ('CCC-piperazinyl', '[*]CCCN1CCNCC1', 'basic'),
    ('CCC-azepanyl', '[*]CCCN1CCCCCC1', 'basic'),
    ('CCC-tetrahydrofuranyl', '[*]CCCC1CCCO1', 'polar'),
    ('CCC-tetrahydropyranyl', '[*]CCCC1CCOCC1', 'polar'),
    ('CCC-1,4-dioxanyl', '[*]CCCC1COCCO1', 'polar'),
    ('C(C)-piperazinyl', '[*]C(C)N1CCNCC1', 'basic'),
    ('C(C)-azepanyl', '[*]C(C)N1CCCCCC1', 'basic'),
    ('C(C)-tetrahydrofuranyl', '[*]C(C)C1CCCO1', 'polar'),
    ('C(C)-tetrahydropyranyl', '[*]C(C)C1CCOCC1', 'polar'),
    ('C(C)-1,4-dioxanyl', '[*]C(C)C1COCCO1', 'polar'),
    ('CO-piperazinyl', '[*]CON1CCNCC1', 'basic'),
    ('CO-azepanyl', '[*]CON1CCCCCC1', 'basic'),
    ('CO-tetrahydrofuranyl', '[*]COC1CCCO1', 'polar'),
    ('CO-tetrahydropyranyl', '[*]COC1CCOCC1', 'polar'),
    ('CO-1,4-dioxanyl', '[*]COC1COCCO1', 'polar'),
    ('CCO-piperazinyl', '[*]CCON1CCNCC1', 'basic'),
    ('CCO-azepanyl', '[*]CCON1CCCCCC1', 'basic'),
    ('CCO-tetrahydrofuranyl', '[*]CCOC1CCCO1', 'polar'),
    ('CCO-tetrahydropyranyl', '[*]CCOC1CCOCC1', 'polar'),
    ('CCO-1,4-dioxanyl', '[*]CCOC1COCCO1', 'polar'),
    ('3-OMe-benzyl', '[*]Cc1cccc(OC)c1', 'aromatic'),
    ('2-F-benzyl', '[*]Cc1ccccc1F', 'aromatic'),
    ('2-Cl-benzyl', '[*]Cc1ccccc1Cl', 'aromatic'),
    ('3,4-diF-benzyl', '[*]Cc1ccc(F)c(F)c1', 'aromatic'),
    ('3,4-diCl-benzyl', '[*]Cc1ccc(Cl)c(Cl)c1', 'aromatic'),
    ('2,4-diF-benzyl', '[*]Cc1ccc(F)cc1F', 'aromatic'),
    ('3,5-diF-benzyl', '[*]Cc1cc(F)cc(F)c1', 'aromatic'),
    ('pentafluorobenzyl', '[*]Cc1c(F)c(F)c(F)c(F)c1F', 'aromatic')
]

LIBRARY: List[Frag] = []
_seen_names: set = set()
for _name, _smi, _cat in _FRAGMENT_ROWS + _EXTENDED_ROWS:
    if _name in _seen_names:
        raise ValueError(f"Duplicate fragment name: {_name}")
    _seen_names.add(_name)
    _mol = Chem.MolFromSmiles(_smi)
    if _mol is None:
        raise ValueError(f"Invalid fragment SMILES {_name}: {_smi}")
    LIBRARY.append(Frag(_name, _smi, _cat, _merge_goals(_cat)))

BUILTIN_LIBRARY: List[Frag] = list(LIBRARY)

def infer_fragment_size_class(frag_or_smiles) -> str:
    if isinstance(frag_or_smiles, Frag):
        if frag_or_smiles.size_class and frag_or_smiles.size_class != "auto":
            return frag_or_smiles.size_class
        heavy = frag_or_smiles.heavy
    else:
        m = Chem.MolFromSmiles(str(frag_or_smiles).replace("[*]", "[H]"))
        heavy = m.GetNumHeavyAtoms() if m else 99
    if heavy <= 2:
        return "small"
    if heavy <= 5:
        return "medium"
    if heavy <= 10:
        return "large"
    return "extended"


def validate_fragment_smiles(smi: str) -> Tuple[bool, str]:
    if not isinstance(smi, str) or not smi.strip():
        return False, "empty"
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return False, "RDKit parse failed"
    ndummy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    if ndummy != 1:
        return False, f"needs exactly one [*], found {ndummy}"
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return False, f"sanitize failed: {e}"
    return True, "ok"


def infer_charge_class(smi: str) -> str:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return "unknown"
    q = Chem.GetFormalCharge(mol)
    if q > 0:
        return "basic_or_cationic"
    if q < 0:
        return "acidic_or_anionic"
    if any(x in smi for x in ["C(=O)O", "S(=O)(=O)O", "n[nH]nn"]):
        return "acidic_possible"
    if any(x in smi for x in ["N", "n1", "n2"]):
        return "basic_or_hbonding_possible"
    return "neutral"


def annotate_library(lib: List[Frag]) -> None:
    for f in lib:
        if f.size_class == "auto":
            f.size_class = infer_fragment_size_class(f)
        if f.charge_class == "auto":
            f.charge_class = infer_charge_class(f.smiles)


annotate_library(BUILTIN_LIBRARY)

AVOID_SMARTS = {
    "nitro": "[N+](=O)[O-]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "reactive_acylhalide": "[CX3](=O)[F,Cl,Br,I]",
    "azide": "[N-]=[N+]=N",
    "michael_acceptor": "[CX3]=[CX3][CX3]=O",
    "epoxide": "C1OC1",
}


# ---------------------------------------------------------------------------
# Molecule drawing
# ---------------------------------------------------------------------------

def draw_mol_svg(mol: Chem.Mol, highlight: Optional[List[int]] = None, size=(560, 460)) -> str:
    highlight = list(highlight or [])
    d = rdMolDraw2D.MolDraw2DSVG(*size)
    o = d.drawOptions()
    o.addAtomIndices = True
    o.annotationFontScale = 0.8
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        mol,
        highlightAtoms=highlight,
        highlightAtomColors={i: (1.0, 0.6, 0.6) for i in highlight},
    )
    d.FinishDrawing()
    return d.GetDrawingText()


def attachable_atom_indices(mol: Chem.Mol, carbon_only: bool = False) -> List[int]:
    return [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetTotalNumHs() > 0 and (not carbon_only or a.GetAtomicNum() == 6)
    ]


# ---------------------------------------------------------------------------
# Analog generation
# ---------------------------------------------------------------------------

def attach(parent: Chem.Mol, atom_idx: int, frag_smiles: str) -> Optional[Chem.Mol]:
    """Attach a [*]-fragment to parent atom by replacing one implicit H."""
    atom_idx = int(atom_idx)
    if parent.GetAtomWithIdx(atom_idx).GetTotalNumHs() == 0:
        return None
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        return None
    dummies = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    dummy_idx = dummies[0]
    nbrs = frag.GetAtomWithIdx(dummy_idx).GetNeighbors()
    if len(nbrs) != 1:
        return None
    nbr_idx = nbrs[0].GetIdx()
    combo = Chem.CombineMols(parent, frag)
    rw = Chem.RWMol(combo)
    off = parent.GetNumAtoms()
    rw.AddBond(atom_idx, nbr_idx + off, Chem.BondType.SINGLE)
    rw.RemoveAtom(dummy_idx + off)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
        AllChem.Compute2DCoords(m)
    except Exception:
        return None
    return m


def attach_to_sites(
    parent: Chem.Mol, atom_indices: List[int], frag_smiles: str
) -> Optional[Chem.Mol]:
    m = Chem.Mol(parent)
    for atom_idx in atom_indices:
        m = attach(m, int(atom_idx), frag_smiles)
        if m is None:
            return None
    return m


def _frag_heavy(f) -> int:
    if hasattr(f, "heavy"):
        try:
            return int(f.heavy)
        except Exception:
            pass
    smi = getattr(f, "smiles", "")
    mol = Chem.MolFromSmiles(smi)
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [0, 1]) if mol else 999


def generate_analogs(
    parent: Chem.Mol,
    selected_atoms: List[int],
    chosen_frags: List[Frag],
    site_groups: Optional[List[Tuple]] = None,
    weights: Optional[Dict] = None,
    avoid_opts: Optional[Dict] = None,
    max_MW: float = 600.0,
    max_analogs: int = 100,
    rank_by: str = "Balanced (100-pt weights)",
    parent_name: str = "parent",
) -> pd.DataFrame:
    """
    Generate analogs by attaching fragments to selected sites.
    Returns a DataFrame with property columns.
    """
    if weights is None:
        weights = {k: 1 / 6 for k in ["potency", "selectivity", "solubility", "metabolic", "synthesis", "novelty"]}
    if avoid_opts is None:
        avoid_opts = {k: True for k in AVOID_SMARTS}

    avoid_q = {
        k: Chem.MolFromSmarts(v)
        for k, v in AVOID_SMARTS.items()
        if avoid_opts.get(k, False)
    }

    if site_groups is None:
        site_groups = [(s,) for s in selected_atoms]

    parent_can = Chem.MolToSmiles(parent)
    pfp = AllChem.GetMorganFingerprintAsBitVect(parent, 2, nBits=2048)

    rows, seen = [], {parent_can}
    filter_counts = {"attach_failed": 0, "duplicate": 0, "MW": 0, "formal_charge": 0, "SMARTS": 0, "SA": 0}

    def _passes(mol):
        if Descriptors.MolWt(mol) > max_MW:
            return False, "MW"
        if abs(Chem.GetFormalCharge(mol)) > 1:
            return False, "formal_charge"
        for q in avoid_q.values():
            if q and mol.HasSubstructMatch(q):
                return False, "SMARTS"
        return True, "passed"

    for sites in site_groups:
        for f in chosen_frags:
            m = attach_to_sites(parent, list(sites), f.smiles)
            if m is None:
                filter_counts["attach_failed"] += 1
                continue
            can = Chem.MolToSmiles(m)
            if can in seen:
                filter_counts["duplicate"] += 1
                continue
            ok, reason = _passes(m)
            if not ok:
                filter_counts[reason] = filter_counts.get(reason, 0) + 1
                continue
            seen.add(can)
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            site_label = ",".join(map(str, sites))
            rows.append(dict(
                smiles=can,
                change=(
                    f"concerted@{site_label}+{f.name}"
                    if len(sites) > 1
                    else f"@{site_label}+{f.name}"
                ),
                mode=("concerted" if len(sites) > 1 else "individual"),
                sites=site_label,
                n_sites=len(sites),
                fragment_name=f.name,
                fragment_smiles=f.smiles,
                fragment_category=f.category,
                fragment_size_class=infer_fragment_size_class(f),
                fragment_heavy_atoms=_frag_heavy(f),
                MW=round(Descriptors.MolWt(m), 1),
                logP=round(Crippen.MolLogP(m), 2),
                TPSA=round(rdMolDescriptors.CalcTPSA(m), 1),
                HBD=rdMolDescriptors.CalcNumHBD(m),
                HBA=rdMolDescriptors.CalcNumHBA(m),
                QED=round(QED.qed(m), 3),
                ESOL=round(esol_logS(m), 2),
                SA=round(sa_score(m), 2),
                sim=round(TanimotoSimilarity(pfp, fp), 3),
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ranking
    def _norm(s):
        s = pd.Series(s).astype(float)
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5

    balanced = (
        weights.get("solubility", 0) * _norm(df.ESOL)
        + weights.get("synthesis", 0) * (1 - _norm(df.SA))
        + weights.get("novelty", 0) * (1 - _norm(df.sim))
        + weights.get("metabolic", 0) * (1 - _norm((df.logP - 2.5).abs()))
        + (weights.get("potency", 0) + weights.get("selectivity", 0)) * _norm(df.QED)
    )
    df["balanced"] = balanced.round(3)
    df["binding_proxy"] = (_norm(df.logP.clip(upper=4)) + _norm(df.QED)).round(3)

    rank_col_map = {
        "Balanced (100-pt weights)": ("balanced", False),
        "Similarity to parent": ("sim", False),
        "Solubility (ESOL)": ("ESOL", False),
        "ADMET (QED)": ("QED", False),
        "Synthetic feasibility": ("SA", True),
        "Binding proxy (heuristic)": ("binding_proxy", False),
    }
    col, asc = rank_col_map.get(rank_by, ("balanced", False))
    df = df.sort_values(col, ascending=asc).head(max_analogs).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Pocket residue → fragment suggestion
# ---------------------------------------------------------------------------

AA_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}
AA3_TO_ONE = {v: k for k, v in AA_ONE_TO_THREE.items()}

AA_TOKEN_TO_ONE = {
    **{v: k for k, v in AA_ONE_TO_THREE.items()},
    **{k: k for k in AA_ONE_TO_THREE},
    **{v.upper(): k for k, v in AA_ONE_TO_THREE.items()},
}

AA_TAGS: Dict[str, List[str]] = {
    "D": ["acidic_negative", "hbond_acceptor"],
    "E": ["acidic_negative", "hbond_acceptor"],
    "K": ["basic_positive", "hbond_donor"],
    "R": ["basic_positive", "hbond_donor"],
    "H": ["basic_positive", "hbond_donor", "hbond_acceptor", "aromatic"],
    "S": ["polar_hbond", "hbond_donor", "hbond_acceptor"],
    "T": ["polar_hbond", "hbond_donor", "hbond_acceptor"],
    "N": ["polar_hbond", "hbond_donor", "hbond_acceptor"],
    "Q": ["polar_hbond", "hbond_donor", "hbond_acceptor"],
    "Y": ["polar_hbond", "hbond_donor", "aromatic", "hydrophobic"],
    "C": ["polar_hbond", "hbond_donor", "sulfur_polarizable"],
    "A": ["hydrophobic"], "V": ["hydrophobic"], "L": ["hydrophobic"],
    "I": ["hydrophobic"], "M": ["hydrophobic", "sulfur_polarizable"],
    "P": ["hydrophobic", "shape_constraint"],
    "F": ["hydrophobic", "aromatic"],
    "W": ["hydrophobic", "aromatic", "hbond_donor"],
    "G": ["small_flexible"],
}

TAG_TO_DESIGN: Dict[str, Dict] = {
    "acidic_negative": {
        "pocket_property": "acidic / negative residue",
        "ligand_strategy": "add basic/cationic or H-bond donor groups",
        "fragment_names": ["amino", "dimethylamino", "piperazine", "morpholine", "piperidine"],
    },
    "basic_positive": {
        "pocket_property": "basic / positive residue",
        "ligand_strategy": "add acidic/anionic or strong H-bond acceptor groups",
        "fragment_names": ["carboxyl", "sulfonamide", "tetrazole", "cyano", "methylsulfonyl"],
    },
    "polar_hbond": {
        "pocket_property": "polar H-bond residue",
        "ligand_strategy": "add H-bond donor/acceptor groups",
        "fragment_names": ["hydroxyl", "hydroxymethyl", "amide(C(=O)NH2)", "methoxy", "methylsulfonyl"],
    },
    "hbond_donor": {
        "pocket_property": "residue can donate H-bond",
        "ligand_strategy": "add ligand H-bond acceptor groups",
        "fragment_names": ["methoxy", "cyano", "methylsulfonyl", "pyridin-3-yl"],
    },
    "hbond_acceptor": {
        "pocket_property": "residue can accept H-bond",
        "ligand_strategy": "add ligand H-bond donor groups",
        "fragment_names": ["hydroxyl", "amino", "amide(C(=O)NH2)", "sulfonamide"],
    },
    "hydrophobic": {
        "pocket_property": "hydrophobic residue",
        "ligand_strategy": "add small hydrophobic, aromatic, or halogen groups",
        "fragment_names": ["methyl", "ethyl", "isopropyl", "cyclopropyl", "chloro", "trifluoromethyl", "phenyl"],
    },
    "aromatic": {
        "pocket_property": "aromatic residue",
        "ligand_strategy": "add π-stacking or hydrophobic groups",
        "fragment_names": ["phenyl", "pyridin-3-yl", "thiophen-2-yl", "pyrazol-1-yl", "chloro"],
    },
    "sulfur_polarizable": {
        "pocket_property": "sulfur / polarizable residue",
        "ligand_strategy": "add soft hydrophobic/halogen groups",
        "fragment_names": ["chloro", "trifluoromethyl", "thiophen-2-yl", "phenyl", "methylsulfonyl"],
    },
    "shape_constraint": {
        "pocket_property": "shape-constraining residue",
        "ligand_strategy": "add compact conformationally restricted groups",
        "fragment_names": ["cyclopropyl", "oxetan-3-yl", "methyl"],
    },
    "small_flexible": {
        "pocket_property": "small/flexible residue",
        "ligand_strategy": "space may tolerate small growth",
        "fragment_names": ["methyl", "fluoro", "hydroxyl", "cyano"],
    },
}


def parse_pocket_residues(text: str) -> List[str]:
    found = []
    for raw in re.findall(r"[A-Za-z]{1,14}\d*", text or ""):
        letters = re.sub(r"\d+", "", raw).upper()
        if letters in AA_TOKEN_TO_ONE:
            found.append(AA_TOKEN_TO_ONE[letters])
    return found


def suggest_fragments_from_residues(
    residue_codes: List[str],
    active_library: List[Frag],
    max_suggestions: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Frag]]:
    tag_counts: Dict[str, int] = {}
    for aa in residue_codes:
        for tag in AA_TAGS.get(aa, []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    total_tags = sum(tag_counts.values()) or 1
    ratios = {k: round(100 * v / total_tags, 1) for k, v in sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)}

    ratio_df = pd.DataFrame([{"pocket_property_tag": k, "ratio_%": v} for k, v in ratios.items()])

    frag_score: Dict[str, float] = {}
    strategy_rows = []
    for tag, ratio in ratios.items():
        info = TAG_TO_DESIGN.get(tag)
        if not info:
            continue
        strategy_rows.append({
            "property_tag": tag,
            "ratio_%": ratio,
            "pocket_property": info["pocket_property"],
            "ligand_strategy": info["ligand_strategy"],
            "suggested_fragment_examples": ", ".join(info["fragment_names"][:5]),
        })
        for fname in info["fragment_names"]:
            frag_score[fname] = frag_score.get(fname, 0.0) + ratio

    strategy_df = pd.DataFrame(strategy_rows).sort_values("ratio_%", ascending=False) if strategy_rows else pd.DataFrame()

    lib_by_name = {f.name: f for f in active_library}
    ordered = sorted(frag_score.items(), key=lambda kv: kv[1], reverse=True)
    pocket_frags: List[Frag] = []
    seen: set = set()
    for name, _ in ordered:
        if name in lib_by_name and name not in seen:
            pocket_frags.append(lib_by_name[name])
            seen.add(name)
    pocket_frags = pocket_frags[:max_suggestions]
    return strategy_df, ratio_df, pocket_frags


# ---------------------------------------------------------------------------
# PDB / structure helpers
# ---------------------------------------------------------------------------

AA3_STRUCT = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL SEC PYL ASX GLX HID HIE HIP".split()
)
EXCLUDE_HET = set(
    "HOH WAT DOD NA CL K MG MN ZN CA FE CU CO NI CD HG SO4 PO4 HPO4 ACT ACE EDO GOL PEG DMS DMSO MPD TRS BME MSE".split()
)


def _safe_file_token(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x).strip() or "file")


def _pdb_xyz(line: str) -> np.ndarray:
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)


def _pdb_is_h(line: str) -> bool:
    elem = line[76:78].strip().upper()
    name = line[12:16].strip().upper()
    return elem == "H" or name.startswith("H")


def _pdb_reskey(line: str) -> Tuple:
    return (line[21].strip() or "_", line[17:20].strip().upper(), line[22:26].strip(), line[26].strip())


# ---------------------------------------------------------------------------
# PubChem PUG REST API
# ---------------------------------------------------------------------------

def search_pubchem(query: str) -> Dict:
    query = query.strip()
    if not query:
        return {"found": False, "error": "empty query"}

    try:
        def _get_json(url):
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())

        def _pick_smiles(prop_dict):
            for k in ("IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES", "SMILES"):
                v = prop_dict.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        data = _get_json(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{urllib.request.quote(query)}/cids/JSON"
        )
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return {"found": False, "error": f"'{query}' not found in PubChem"}
        cid = cids[0]

        p = {}
        smiles = ""
        for prop_block in [
            "IUPACName,MolecularFormula,MolecularWeight,IsomericSMILES,CanonicalSMILES,ConnectivitySMILES",
            "IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,ConnectivitySMILES",
            "IUPACName,MolecularFormula,MolecularWeight,ConnectivitySMILES",
        ]:
            try:
                prop_data = _get_json(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/"
                    f"property/{prop_block}/JSON"
                )
                props = prop_data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    p = props[0]
                    smiles = _pick_smiles(p)
                    if smiles:
                        break
            except Exception:
                continue

        if not smiles:
            try:
                pc_data = _get_json(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON"
                )
                pc = pc_data.get("PC_Compounds", [{}])[0]
                for prop in pc.get("props", []):
                    urn = prop.get("urn", {})
                    label = str(urn.get("label", "")).lower()
                    name2 = str(urn.get("name", "")).lower()
                    if "smiles" in label or "smiles" in name2:
                        value = prop.get("value", {})
                        cand = value.get("sval") or value.get("string") or ""
                        if cand.strip():
                            smiles = cand.strip()
                            break
            except Exception:
                pass

        return {
            "found": True,
            "cid": cid,
            "smiles": smiles,
            "iupac": p.get("IUPACName", query),
            "formula": p.get("MolecularFormula", ""),
            "mw": float(p.get("MolecularWeight", 0) or 0),
            "img_url": (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                f"{cid}/PNG?record_type=2d&image_size=300x300"
            ),
            "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        }
    except Exception as e:
        return {"found": False, "error": str(e)}


def pubchem_cid_to_smiles(cid: int) -> Optional[str]:
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/"
        f"property/CanonicalSMILES/JSON"
    )
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RCSB PDB Search API
# ---------------------------------------------------------------------------

def search_rcsb(query: str, max_results: int = 10) -> List[Dict]:
    query = query.strip()
    if not query:
        return []

    search_body = json.dumps({
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query},
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    })
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    try:
        req = urllib.request.Request(
            search_url, data=search_body.encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        pdb_ids = [r["identifier"] for r in data.get("result_set", [])][:max_results]
    except Exception:
        pdb_ids = []

    if not pdb_ids:
        return []

    results = []
    for pdb_id in pdb_ids:
        summary_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        try:
            req = urllib.request.Request(summary_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                entry = json.loads(resp.read())
            struct = entry.get("struct", {})
            exptl = entry.get("exptl", [{}])[0] if entry.get("exptl") else {}
            refine = entry.get("refine", [{}])[0] if entry.get("refine") else {}
            src = entry.get("rcsb_entity_source_organism", [{}])
            organism = src[0].get("ncbi_scientific_name", "") if src else ""
            results.append({
                "id": pdb_id.upper(),
                "title": struct.get("title", ""),
                "resolution": f"{refine.get('ls_d_res_high', '?')} Å",
                "method": exptl.get("method", ""),
                "organism": organism,
            })
        except Exception:
            results.append({
                "id": pdb_id.upper(),
                "title": "(details unavailable)",
                "resolution": "?",
                "method": "",
                "organism": "",
            })
    return results


def download_pdb(pdb_id: str, out_dir: Path) -> str:
    pdb_id = pdb_id.strip().upper()
    assert re.fullmatch(r"[A-Za-z0-9]{4}", pdb_id), "PDB ID must be 4 characters."
    out = out_dir / f"{pdb_id}.pdb"
    if not out.exists() or out.stat().st_size < 1000:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(url, out)
    return str(out)


def cif_to_pdb_if_needed(path: str) -> str:
    path = Path(path)
    if path.suffix.lower() not in [".cif", ".mmcif"]:
        return str(path)
    out = path.with_suffix(".pdb")
    try:
        import gemmi  # type: ignore
        st = gemmi.read_structure(str(path))
        st.write_pdb(str(out))
        return str(out)
    except Exception as e:
        raise RuntimeError("Could not convert CIF to PDB. Install gemmi or provide PDB.") from e


def detect_ligand_candidates(pdb_path: str) -> Tuple[pd.DataFrame, Dict]:
    groups: Dict = {}
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            resn = line[17:20].strip().upper()
            if resn in EXCLUDE_HET or resn in AA3_STRUCT:
                continue
            key = _pdb_reskey(line)
            groups.setdefault(key, {"atoms": 0, "heavy": 0})
            groups[key]["atoms"] += 1
            groups[key]["heavy"] += 0 if _pdb_is_h(line) else 1
    rows = []
    for (chain, resn, resi, icode), val in groups.items():
        rows.append({"chain": chain, "resname": resn, "resnum": resi, "icode": icode,
                     "atoms": val["atoms"], "heavy_atoms": val["heavy"]})
    df = pd.DataFrame(rows).sort_values(["heavy_atoms", "atoms"], ascending=False) if rows else pd.DataFrame()
    return df, groups


def split_protein_ligand(
    pdb_path: str,
    ligand_resname: str = "",
    work_dir: Optional[Path] = None,
) -> Tuple[str, Optional[str], pd.DataFrame]:
    work_dir = work_dir or Path(".")
    work_dir.mkdir(parents=True, exist_ok=True)
    ligand_resname = ligand_resname.strip().upper()
    candidates, groups = detect_ligand_candidates(pdb_path)
    chosen_key = None
    if ligand_resname:
        for key in groups:
            if key[1] == ligand_resname:
                chosen_key = key
                break
        if chosen_key is None:
            raise ValueError(f"Residue {ligand_resname} not found in HETATM candidates.")
    elif groups:
        chosen_key = max(groups, key=lambda k: groups[k]["heavy"])

    protein_lines, ligand_lines = [], []
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("ATOM"):
                protein_lines.append(line)
            elif line.startswith("HETATM") and chosen_key and _pdb_reskey(line) == chosen_key:
                ligand_lines.append(line)

    protein_out = str(work_dir / "protein_only.pdb")
    with open(protein_out, "w") as f:
        f.writelines(protein_lines)
        f.write("END\n")

    ligand_out = None
    if ligand_lines:
        ligand_out = str(work_dir / "reference_ligand.pdb")
        with open(ligand_out, "w") as f:
            f.writelines(ligand_lines)
            f.write("END\n")

    return protein_out, ligand_out, candidates


def combine_protein_ligand_pdb(protein_pdb: str, ligand_pdb: str, out_pdb: str) -> str:
    protein_lines = []
    with open(protein_pdb, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM"):
                protein_lines.append(line)
    ligand_lines = []
    with open(ligand_pdb, "r", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                newline = "HETATM" + line[6:17] + "LIG A 900" + line[26:]
                ligand_lines.append(newline)
    with open(out_pdb, "w") as f:
        f.writelines(protein_lines)
        f.writelines(ligand_lines)
        f.write("END\n")
    return out_pdb


def sdf_first_mol_to_pdb(sdf_path: str, out_pdb: str) -> str:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"No readable molecule in SDF: {sdf_path}")
    Chem.MolToPDBFile(mol, str(out_pdb))
    return str(out_pdb)


# ---------------------------------------------------------------------------
# Pocket distance-shell analysis
# ---------------------------------------------------------------------------

def read_complex_atoms_for_pocket(pdb_path: str) -> Tuple[List[Dict], List[Dict]]:
    protein, ligand = [], []
    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if _pdb_is_h(line):
                continue
            try:
                xyz = _pdb_xyz(line)
            except Exception:
                continue
            rec = {
                "record": line[:6].strip(),
                "atom_name": line[12:16].strip(),
                "resname": line[17:20].strip().upper(),
                "chain": line[21].strip() or "_",
                "resnum": line[22:26].strip(),
                "icode": line[26].strip(),
                "xyz": xyz,
            }
            if line.startswith("ATOM"):
                protein.append(rec)
            else:
                ligand.append(rec)
    return protein, ligand


def analyze_complex_distance_shell(
    complex_pdb: str,
    pocket_cutoff: float = 6.0,
    contact_cutoff: float = 4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
    protein, ligand = read_complex_atoms_for_pocket(complex_pdb)
    if not protein or not ligand:
        raise ValueError("Complex must contain protein ATOM records and ligand HETATM records.")
    lig_xyz = np.array([a["xyz"] for a in ligand], dtype=float)
    by_res: Dict = {}
    for a in protein:
        dmin = float(np.linalg.norm(lig_xyz - a["xyz"], axis=1).min())
        key = (a["chain"], a["resname"], a["resnum"], a["icode"])
        if key not in by_res or dmin < by_res[key]["min_dist_to_ligand_A"]:
            by_res[key] = {
                "key": key, "resname": a["resname"], "chain": a["chain"],
                "resnum": a["resnum"], "icode": a["icode"], "min_dist_to_ligand_A": dmin,
            }
    prot_rows = []
    for r in by_res.values():
        r["is_pocket_residue"] = r["min_dist_to_ligand_A"] <= pocket_cutoff
        r["is_contacted"] = r["min_dist_to_ligand_A"] <= contact_cutoff
        r["is_noncontact_growth_residue"] = r["is_pocket_residue"] and not r["is_contacted"]
        one = AA3_TO_ONE.get(r["resname"], "")
        r["aa_one"] = one
        r["property_tags"] = ",".join(AA_TAGS.get(one, [])) if one else ""
        r["residue_label"] = f"{r['resname']}{r['resnum']}:{r['chain']}"
        prot_rows.append(r)
    df = pd.DataFrame(prot_rows).sort_values("min_dist_to_ligand_A")
    return (
        df[df["is_pocket_residue"]].copy(),
        df[df["is_contacted"]].copy(),
        df[df["is_noncontact_growth_residue"]].copy(),
        ligand,
    )


# ---------------------------------------------------------------------------
# ACD docking command builder
# ---------------------------------------------------------------------------

def build_acd_dock_cmd(
    receptor: str,
    smiles: str,
    center: str = "auto",
    name: str = "ligand",
    ph: float = 7.4,
    output_dir: str = "docking_out",
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    use_pkanet: bool = False,
    neutral: bool = False,
    save_poses: bool = True,
    extra_args: str = "",
) -> List[str]:
    cmd = ["acd", "dock", "--receptor", receptor, "--smiles", smiles,
           "--center", center, "--name", _safe_file_token(name),
           "--ph", str(ph), "-o", output_dir]
    if center == "manual":
        cmd.extend(["--cx", str(cx), "--cy", str(cy), "--cz", str(cz)])
    if use_pkanet:
        cmd.append("--pkanet")
    if neutral:
        cmd.append("--neutral")
    if save_poses:
        cmd.append("--save-poses")
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args.strip()))
    return cmd


def build_acd_batch_cmd(
    receptor: str,
    ligands_smi: str,
    output_dir: str = "docking_out",
    center: str = "auto",
    exhaustiveness: int = 8,
    num_poses: int = 10,
    ph: float = 7.4,
    box_x: float = 16.0,
    box_y: float = 16.0,
    box_z: float = 16.0,
    use_pkanet: bool = False,
    neutral: bool = False,
    extra_args: str = "",
) -> List[str]:
    cmd = ["acd", "batch", "--receptor", receptor, "--ligands", ligands_smi,
           "--output", output_dir, "--center", center,
           "-e", str(exhaustiveness), "-n", str(num_poses), "--ph", str(ph)]
    if use_pkanet:
        cmd.append("--pkanet")
    if neutral:
        cmd.append("--neutral")
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args.strip()))
    return cmd


def run_command(cmd: List[str], log_path: Optional[str] = None) -> Tuple[int, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    output = (proc.stdout or "") + (proc.stderr or "")
    if log_path:
        Path(log_path).write_text(output)
    return proc.returncode, output


def best_score_for_compound(out_dir: str, compound: str) -> Optional[float]:
    token = _safe_file_token(compound)
    best = None
    for csv_path in glob.glob(str(Path(out_dir) / "**" / "*.csv"), recursive=True):
        if token not in Path(csv_path).name and token not in str(Path(csv_path).parent.name):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        score_cols = [c for c in df.columns if re.search(r"score|energy|affinity|binding", c, re.I)]
        if not score_cols:
            continue
        sc = score_cols[0]
        vals = pd.to_numeric(df[sc], errors="coerce").dropna()
        if vals.empty:
            continue
        m = float(vals.min())
        best = m if best is None else min(best, m)
    return best


def parse_acd_score_csvs(out_dir: str) -> Optional[Dict]:
    best_rows = []
    for csv_path in glob.glob(str(Path(out_dir) / "**" / "*.csv"), recursive=True):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        score_cols = [c for c in df.columns if re.search(r"score|energy|affinity|binding", c, re.I)]
        if not score_cols:
            continue
        sc = score_cols[0]
        df[sc] = pd.to_numeric(df[sc], errors="coerce")
        df = df.dropna(subset=[sc])
        if df.empty:
            continue
        idx = df[sc].idxmin()
        row = df.loc[idx].to_dict()
        row["_score_csv"] = csv_path
        row["_score_col"] = sc
        best_rows.append(row)
    if not best_rows:
        return None
    best_rows.sort(key=lambda r: float(r.get(r.get("_score_col", ""), 9999)))
    return best_rows[0]


def find_pose_sdf(out_dir: str) -> Optional[str]:
    sdfs = sorted(glob.glob(str(Path(out_dir) / "**" / "*.sdf"), recursive=True))
    ranked = [p for p in sdfs if re.search(r"out|pose|dock|result", Path(p).name, re.I)] + sdfs
    return ranked[0] if ranked else None


def find_pose_sdfs_for_compound(out_dir: str, compound: str) -> List[str]:
    token = _safe_file_token(compound)
    sdfs = sorted(glob.glob(str(Path(out_dir) / "**" / "*.sdf"), recursive=True))
    matched = [s for s in sdfs if token in Path(s).name or token in str(Path(s).parent.name)]
    return matched if matched else sdfs[:1]


# ---------------------------------------------------------------------------
# RMSD calculation (docked pose vs crystal ligand)
# ---------------------------------------------------------------------------

def _read_heavy_coords_from_pdb(pdb_path: str) -> np.ndarray:
    coords = []
    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if _pdb_is_h(line):
                continue
            try:
                coords.append(_pdb_xyz(line))
            except Exception:
                continue
    return np.array(coords, dtype=float) if coords else np.empty((0, 3))


def _read_heavy_coords_from_sdf(sdf_path: str, mol_index: int = 0) -> Tuple[Optional[Chem.Mol], np.ndarray]:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    idx = 0
    for mol in suppl:
        if mol is None:
            idx += 1
            continue
        if idx == mol_index:
            conf = mol.GetConformer()
            coords = np.array([
                list(conf.GetAtomPosition(i))
                for i in range(mol.GetNumAtoms())
                if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
            ], dtype=float)
            return mol, coords
        idx += 1
    return None, np.empty((0, 3))


def compute_rmsd_rdkit(mol_ref: Chem.Mol, mol_probe: Chem.Mol) -> Optional[float]:
    try:
        from rdkit.Chem import AllChem as _AC
        rmsd = _AC.GetBestRMS(Chem.RemoveHs(mol_ref), Chem.RemoveHs(mol_probe))
        return float(rmsd)
    except Exception:
        pass
    try:
        ref_conf = mol_ref.GetConformer()
        prb_conf = mol_probe.GetConformer()
        ref_c = np.array([list(ref_conf.GetAtomPosition(i))
                          for i in range(mol_ref.GetNumAtoms())
                          if mol_ref.GetAtomWithIdx(i).GetAtomicNum() > 1])
        prb_c = np.array([list(prb_conf.GetAtomPosition(i))
                          for i in range(mol_probe.GetNumAtoms())
                          if mol_probe.GetAtomWithIdx(i).GetAtomicNum() > 1])
        if ref_c.shape == prb_c.shape and len(ref_c) > 0:
            return float(np.sqrt(np.mean(np.sum((ref_c - prb_c) ** 2, axis=1))))
    except Exception:
        pass
    return None


def compute_rmsd_coords(ref_coords: np.ndarray, probe_coords: np.ndarray) -> Optional[float]:
    if ref_coords.shape != probe_coords.shape or len(ref_coords) == 0:
        return None
    return float(np.sqrt(np.mean(np.sum((ref_coords - probe_coords) ** 2, axis=1))))


def _parse_vina_scores_from_pdbqt(pdbqt_path: str) -> List[float]:
    """Extract affinity scores from REMARK VINA RESULT lines in PDBQT."""
    scores = []
    try:
        with open(pdbqt_path, "r", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("REMARK VINA RESULT:"):
                    try:
                        scores.append(float(line.split()[3]))
                    except Exception:
                        pass
    except Exception:
        pass
    return scores


def parse_docked_poses(
    sdf_path: str,
    ref_mol: Optional[Chem.Mol] = None,
    ref_pdb_path: Optional[str] = None,
) -> List[Dict]:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    ref_mol_noH = None
    if ref_mol is not None:
        try:
            ref_mol_noH = Chem.RemoveHs(ref_mol)
        except Exception:
            pass
    elif ref_pdb_path and os.path.exists(str(ref_pdb_path)):
        try:
            ref_mol_noH = Chem.MolFromPDBFile(str(ref_pdb_path), removeHs=True)
        except Exception:
            pass

    # Try reading scores from companion PDBQT (ACD output stores scores there)
    pdbqt_path = str(sdf_path).replace(".sdf", ".pdbqt").replace("_out.sdf", "_out.pdbqt")
    vina_scores = _parse_vina_scores_from_pdbqt(pdbqt_path)

    poses = []
    idx = 0
    for mol in suppl:
        if mol is None:
            idx += 1
            continue

        # 1. Try SDF property (some pipelines inject score here)
        score = None
        for prop_name in mol.GetPropsAsDict():
            if re.search(r"score|energy|affinity|minimizedAffinity|binding", prop_name, re.I):
                try:
                    score = float(mol.GetProp(prop_name))
                except Exception:
                    pass
                if score is not None:
                    break

        # 2. Fallback: use PDBQT REMARK VINA RESULT scores (ACD default output)
        if score is None and idx < len(vina_scores):
            score = vina_scores[idx]

        # 3. Last resort: try to parse from mol title/name
        if score is None:
            try:
                title = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                m = re.search(r"(-?[0-9]+\.[0-9]+)", title)
                if m:
                    score = float(m.group(1))
            except Exception:
                pass

        rmsd = None
        if ref_mol_noH is not None:
            try:
                probe_noH = Chem.RemoveHs(mol)
                rmsd = compute_rmsd_rdkit(ref_mol_noH, probe_noH)
            except Exception:
                pass

        poses.append({
            "pose_index": idx,
            "score": score,
            "rmsd_vs_crystal": round(rmsd, 3) if rmsd is not None else None,
            "mol": mol,
        })
        idx += 1

    return poses


def summarize_docking_for_compound(
    out_dir: str,
    compound: str,
    smiles: str,
    ref_mol: Optional[Chem.Mol] = None,
    ref_pdb_path: Optional[str] = None,
) -> Dict:
    result = {
        "compound": compound,
        "smiles": smiles,
        "top_BE": None,
        "top_RMSD": None,
        "minRMSD_BE": None,
        "minRMSD_RMSD": None,
        "n_poses": 0,
        "status": "no_sdf",
    }
    sdfs = find_pose_sdfs_for_compound(out_dir, compound)
    if not sdfs:
        return result

    all_poses = []
    for sdf in sdfs:
        all_poses.extend(parse_docked_poses(sdf, ref_mol=ref_mol, ref_pdb_path=ref_pdb_path))

    if not all_poses:
        result["status"] = "no_poses"
        return result

    result["n_poses"] = len(all_poses)
    result["status"] = "ok"

    scored = [p for p in all_poses if p["score"] is not None]
    if scored:
        top = min(scored, key=lambda p: p["score"])
        result["top_BE"] = round(top["score"], 2)
        result["top_RMSD"] = top["rmsd_vs_crystal"]

    with_rmsd = [p for p in all_poses if p["rmsd_vs_crystal"] is not None]
    if with_rmsd:
        best_rmsd = min(with_rmsd, key=lambda p: p["rmsd_vs_crystal"])
        result["minRMSD_RMSD"] = best_rmsd["rmsd_vs_crystal"]
        result["minRMSD_BE"] = round(best_rmsd["score"], 2) if best_rmsd["score"] is not None else None

    return result


# ---------------------------------------------------------------------------
# 2D molecule image generation (PNG bytes for tables/CSV)
# ---------------------------------------------------------------------------

def mol_to_png_base64(smiles: str, size: Tuple[int, int] = (250, 180)) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        AllChem.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DCairo(*size)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        import base64
        return base64.b64encode(d.GetDrawingText()).decode()
    except Exception:
        return ""


def mol_to_png_bytes(smiles: str, size: Tuple[int, int] = (250, 180)) -> bytes:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return b""
    try:
        AllChem.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DCairo(*size)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return b""


# ---------------------------------------------------------------------------
# PLIP / distance-contact cIFP
# ---------------------------------------------------------------------------

def _parse_pdb_heavy_atoms(pdb_path: str) -> Tuple[List[Dict], List[Dict]]:
    protein, ligand = [], []
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            elem = line[76:78].strip().upper() or line[12:16].strip()[0:1].upper()
            if elem == "H":
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except Exception:
                continue
            rec = {
                "record": line[:6].strip(),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21].strip() or "_",
                "resnum": line[22:26].strip(),
                "x": x, "y": y, "z": z,
            }
            if line.startswith("ATOM"):
                protein.append(rec)
            else:
                ligand.append(rec)
    return protein, ligand


def distance_contact_cifp(complex_pdb: str, cutoff: float = 4.0) -> List[str]:
    protein, ligand = _parse_pdb_heavy_atoms(complex_pdb)
    feats: set = set()
    c2 = float(cutoff) ** 2
    for pa in protein:
        for la in ligand:
            dx = pa["x"] - la["x"]
            dy = pa["y"] - la["y"]
            dz = pa["z"] - la["z"]
            if dx * dx + dy * dy + dz * dz <= c2:
                feats.add(f"CONTACT:{pa['chain']}:{pa['resname']}:{pa['resnum']}")
                break
    return sorted(feats)


def run_plip(complex_pdb: str, out_dir: str, name: str) -> Tuple[Optional[str], Optional[str]]:
    plip_exe = shutil.which("plipcmd") or shutil.which("plip")
    if plip_exe is None:
        return None, "PLIP executable not found"
    od = Path(out_dir) / name
    od.mkdir(parents=True, exist_ok=True)
    cmd = [plip_exe, "-f", str(complex_pdb), "-x", "-o", str(od)]
    res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (od / "plip_log.txt").write_text(res.stdout)
    xmls = list(od.glob("*.xml")) + list(od.glob("**/*.xml"))
    if res.returncode != 0 or not xmls:
        return None, f"PLIP failed. Return={res.returncode}."
    return str(xmls[0]), None


def parse_plip_xml(xml_path: str) -> List[str]:
    feats: set = set()
    if not xml_path or not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []
    type_map = {
        "hydrophobic_interaction": "HYDROPHOBIC",
        "hydrogen_bond": "HBOND",
        "water_bridge": "WATERBRIDGE",
        "salt_bridge": "SALTBRIDGE",
        "pi_stack": "PISTACK",
        "pi_cation_interaction": "PICATION",
        "halogen_bond": "HALOGEN",
        "metal_complex": "METAL",
    }
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        itype = next((v for k, v in type_map.items() if k in tag), None)
        if itype is None:
            continue
        vals = {c.tag.split("}")[-1].lower(): (c.text or "").strip() for c in elem.iter() if c.text}
        resnr = vals.get("resnr") or vals.get("resnum") or "NA"
        restype = vals.get("restype") or vals.get("resname") or "RES"
        reschain = vals.get("reschain") or vals.get("chain") or "_"
        feats.add(f"{itype}:{reschain}:{restype}:{resnr}")
    return sorted(feats)


# ---------------------------------------------------------------------------
# 3D ligand file generation
# ---------------------------------------------------------------------------

def build_3d_mol(smiles: str, seed: int = 42, mmff: bool = True) -> Tuple[Optional[Chem.Mol], str]:
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        return None, "invalid_smiles"
    mol = Chem.AddHs(mol0)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None, "embed_failed"
    if mmff:
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            if props:
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=1000)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        except Exception:
            pass
    return mol, "ok"


def generate_3d_ligand_files(
    ligand_table: pd.DataFrame,
    out_dir: Path,
    formats: List[str] = ["SDF"],
    mmff: bool = True,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    obabel = shutil.which("obabel")
    combined_sdf = out_dir / "all_ligands_3d.sdf"
    writer = Chem.SDWriter(str(combined_sdf))
    rows = []
    for i, r in ligand_table.iterrows():
        compound = _safe_file_token(r.get("compound", f"ligand_{i+1}"))
        smiles = str(r.get("smiles", "")).strip()
        mol, status = build_3d_mol(smiles, seed=42 + int(i), mmff=mmff)
        sdf_p = pdb_p = mol2_p = None
        if mol:
            mol.SetProp("_Name", compound)
            if "SDF" in formats:
                sdf_p = str(out_dir / f"{compound}.sdf")
                w = Chem.SDWriter(sdf_p)
                w.write(mol)
                w.close()
            if "PDB" in formats:
                pdb_p = str(out_dir / f"{compound}.pdb")
                Chem.MolToPDBFile(mol, pdb_p)
            if "MOL2" in formats and obabel and sdf_p:
                mol2_p = str(out_dir / f"{compound}.mol2")
                subprocess.run([obabel, sdf_p, "-O", mol2_p], capture_output=True, check=False)
                if not os.path.exists(mol2_p):
                    mol2_p = None
            writer.write(mol)
        rows.append({"compound": compound, "smiles": smiles, "status": status,
                     "sdf": sdf_p, "pdb": pdb_p, "mol2": mol2_p})
    writer.close()
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 3D Viewer helper — interacting residues for py3Dmol
# ---------------------------------------------------------------------------

def get_interacting_residues(receptor_pdb: str, lig_mol, cutoff: float = 3.5) -> list:
    """Return list of {chain, resi, resn} dicts for residues within cutoff of ligand."""
    try:
        from rdkit.Chem import AllChem as _AC
        conf    = lig_mol.GetConformer()
        lig_xyz = np.array([
            [conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(lig_mol.GetNumAtoms())
        ])
        protein, _ = read_complex_atoms_for_pocket(receptor_pdb)
        seen = {}
        for a in protein:
            d = float(np.linalg.norm(lig_xyz - a["xyz"], axis=1).min())
            if d <= cutoff:
                key = (a["chain"], a["resnum"])
                if key not in seen:
                    seen[key] = a["resname"]
        return [{"chain": k[0], "resi": k[1], "resn": v} for k, v in seen.items()]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# RMSD calculation — heavy atoms via MCS (adapted from AnyonCanDock)
# ---------------------------------------------------------------------------

def calc_rmsd_heavy(pose_mol, crystal_pdb_path: str):
    """RMSD of pose vs co-crystal ligand using maximum common substructure."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
        import numpy as _np
        if not os.path.exists(crystal_pdb_path):
            return None
        cryst = None
        for sanitize, removeHs, proxBonding in [
            (True,  True, True), (False, True, True),
            (True,  True, False), (False, True, False),
        ]:
            try:
                cryst = Chem.MolFromPDBFile(
                    crystal_pdb_path, sanitize=sanitize,
                    removeHs=removeHs, proximityBonding=proxBonding)
                if cryst is not None and cryst.GetNumConformers() > 0:
                    if not sanitize:
                        try: Chem.SanitizeMol(cryst)
                        except Exception: pass
                    break
                cryst = None
            except Exception:
                cryst = None
        if cryst is None or cryst.GetNumConformers() == 0:
            return None
        pose = Chem.RemoveHs(pose_mol, sanitize=False)
        try: Chem.SanitizeMol(pose)
        except Exception: pass
        if pose.GetNumConformers() == 0:
            return None
        n_smaller = min(pose.GetNumAtoms(), cryst.GetNumAtoms())
        mcs = rdFMCS.FindMCS(
            [pose, cryst], timeout=10,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            completeRingsOnly=False, matchValences=False,
        )
        if mcs.numAtoms < 3 or mcs.numAtoms < 0.6 * n_smaller:
            return None
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is None:
            return None
        pose_matches  = pose.GetSubstructMatches(mcs_mol,  uniquify=False)
        cryst_matches = cryst.GetSubstructMatches(mcs_mol, uniquify=False)
        if not pose_matches or not cryst_matches:
            return None
        pc = pose.GetConformer()
        cc = cryst.GetConformer()
        def _rmsd(pm, cm):
            sq = sum(
                (pc.GetAtomPosition(pi).x - cc.GetAtomPosition(ci).x) ** 2 +
                (pc.GetAtomPosition(pi).y - cc.GetAtomPosition(ci).y) ** 2 +
                (pc.GetAtomPosition(pi).z - cc.GetAtomPosition(ci).z) ** 2
                for pi, ci in zip(pm, cm)
            )
            return float(_np.sqrt(sq / len(pm)))
        return min(_rmsd(pm, cm) for pm in pose_matches for cm in cryst_matches)
    except Exception:
        return None
