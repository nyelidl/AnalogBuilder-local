"""
app.py — ⌬+⌬ Analog Builder · Streamlit web application
Redesigned UX: warm tones, student-friendly, two separate tracks,
sidebar progress indicator, advanced options collapsed.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shlex
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

import core
import streamlit.components.v1 as components

# Module 2: ChemBERTa pocket-aware fragment ranker
try:
    import pocket_reference as _pr
    _PR_OK = True
except ImportError:
    _pr = None
    _PR_OK = False

# ADMET-AI (local only — pip install admet-ai)
try:
    from admet_ai import ADMETModel as _ADMETModel
    _ADMET_AI_OK = True
except ImportError:
    _ADMETModel = None
    _ADMET_AI_OK = False

# pKaNET local (local only — requires pkanet.py in same folder)
try:
    import pkanet as _pkanet_local
    _PKANET_LOCAL_OK = True
except ImportError:
    _pkanet_local = None
    _PKANET_LOCAL_OK = False

# Void / unoccupied space analyzer
try:
    import void_analyzer as _va
    _VA_OK = True
except ImportError:
    _va = None
    _VA_OK = False

# PLIP interaction analyzer
try:
    import plip_analyzer as _pa
    _PA_OK = True
except ImportError:
    _pa = None
    _PA_OK = False

# Optional in-browser molecule sketcher
try:
    from streamlit_ketcher import st_ketcher
    _KETCHER_OK = True
except Exception:
    _KETCHER_OK = False

# Logo
LOGO_URL = "https://raw.githubusercontent.com/nyelidl/AnalogBuilder/main/.fig/AB.svg"
LB_URL   = "https://raw.githubusercontent.com/nyelidl/AnalogBuilder/main/.fig/LB.svg"
SB_URL   = "https://raw.githubusercontent.com/nyelidl/AnalogBuilder/main/.fig/SB.svg"

# ─────────────────────────────────────────────────────────────────────────────
# Tier logic helper
# ─────────────────────────────────────────────────────────────────────────────

# ── LOCAL VERSION: no tier limits ───────────────────────────────────────────
_LOCAL_MODE = True   # set False to restore cloud tier restrictions

def analog_tier(n: int) -> str:
    """Return tier string — LOCAL: always 'docking' (no limits)."""
    if _LOCAL_MODE:
        return "docking"      # unlimited docking regardless of analog count
    # Cloud tier logic (kept for reference):
    if n <= 20:
        return "docking"
    elif n <= 200:
        return "pkanet"
    else:
        return "smi_only"


# ─────────────────────────────────────────────────────────────────────────────
# Page config + global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="⌬+⌬ Analog Builder (Local)",
    page_icon=LOGO_URL,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Warm base ── */
[data-testid="stAppViewContainer"] {
    background: #FAF7F2;
}
[data-testid="stSidebar"] {
    background: #F0EAE0;
    border-right: 1px solid #E0D6C8;
}

/* ── Typography ── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #2C2C2C;
}
h1 { font-size: 1.6rem !important; font-weight: 700; color: #2C2C2C; }
h2 { font-size: 1.2rem !important; font-weight: 600; color: #2C2C2C; }
h3 { font-size: 1.0rem !important; font-weight: 600; color: #3D7A74; }

/* ── Primary button: amber ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #E8A020 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #C88010 !important;
}

/* ── Secondary button ── */
[data-testid="stButton"] > button {
    border-radius: 8px !important;
    border: 1px solid #C8B89A !important;
    background: #FAF7F2 !important;
    color: #2C2C2C !important;
}

/* ── Mode cards — clickable ── */
/* ── Mode cards — card = stMarkdown div + button fused visually ── */
/* Equal height columns */
div[data-testid="stHorizontalBlock"]:has(.mode-card-col) {
    align-items: stretch !important;
}
/* The stVerticalBlock inside each column: flex so card+button fill height */
div[data-testid="stHorizontalBlock"]:has(.mode-card-col)
    > div[data-testid="stColumn"]
    > div[data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}
/* stMarkdownContainer must also stretch to fill available height */
div[data-testid="stHorizontalBlock"]:has(.mode-card-col)
    > div[data-testid="stColumn"]
    > div[data-testid="stVerticalBlock"]
    > div[data-testid="stMarkdownContainer"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 !important;
}
/* Card content div fills the stMarkdownContainer */
div[data-testid="stHorizontalBlock"]:has(.mode-card-col)
    > div[data-testid="stColumn"]
    > div[data-testid="stVerticalBlock"]
    > div[data-testid="stMarkdownContainer"]
    > .mode-card-col {
    flex: 1 !important;
}
/* Card content div */
.mode-card-col {
    background: #FFFFFF;
    border: 2px solid #E0D6C8;
    border-bottom: none;
    border-radius: 14px 14px 0 0;
    padding: 2rem 1.5rem 1.2rem;
    text-align: center;
    flex: 1;
    min-height: 220px;
    box-sizing: border-box;
    transition: border-color 0.2s, box-shadow 0.2s;
}
/* Hover: target the stMarkdown wrapper so hovering card highlights border */
div[data-testid="stHorizontalBlock"]:has(.mode-card-col)
    > div[data-testid="stColumn"]
    > div[data-testid="stVerticalBlock"]:hover .mode-card-col {
    border-color: #E8A020;
}
/* stMarkdown wrapper that contains mode-card-col: remove its own border/bg */
div[data-testid="stHorizontalBlock"]:has(.mode-card-col)
    > div[data-testid="stColumn"]
    > div[data-testid="stVerticalBlock"]
    > div[data-testid="stMarkdownContainer"] {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
/* Button container: no gap between card div and button */
div[data-testid="stHorizontalBlock"]:has(.mode-card-col)
    > div[data-testid="stColumn"]
    > div[data-testid="stVerticalBlock"]
    > div[data-testid="stButton"] {
    margin-top: 0 !important;
}
/* Button styling: bottom half of card */
div[data-testid="stHorizontalBlock"]:has(.mode-card-col)
    > div[data-testid="stColumn"]
    > div[data-testid="stVerticalBlock"]
    > div[data-testid="stButton"] > button {
    background: #E8A020 !important;
    color: #fff !important;
    border: 2px solid #E8A020 !important;
    border-top: none !important;
    border-radius: 0 0 14px 14px !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    font-size: 0.95rem !important;
}
div[data-testid="stHorizontalBlock"]:has(.mode-card-col)
    > div[data-testid="stColumn"]
    > div[data-testid="stVerticalBlock"]
    > div[data-testid="stButton"] > button:hover {
    background: #C88010 !important;
    border-color: #C88010 !important;
}

/* ── Step progress in sidebar ── */
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border-radius: 8px;
    margin-bottom: 4px;
    font-size: 0.88rem;
    color: #6B5E4E;
    cursor: pointer;
}
.step-item.active {
    background: #E8A020;
    color: #fff;
    font-weight: 600;
}
.step-item.done {
    color: #3D7A74;
    font-weight: 500;
}
.step-dot {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #E0D6C8;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; flex-shrink: 0;
    color: #6B5E4E;
}
.step-item.active .step-dot { background: rgba(255,255,255,0.3); color: #fff; }
.step-item.done .step-dot { background: #3D7A74; color: #fff; }

/* ── Hint text ── */
.hint { color: #8B7355; font-size: 0.82rem; margin-top: -6px; margin-bottom: 10px; }

/* ── Info cards ── */
.info-card {
    background: #FFF8EE;
    border-left: 3px solid #E8A020;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0 1rem;
    font-size: 0.88rem;
    color: #5A4A35;
}

/* ── Tier badge ── */
.tier-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 8px;
    vertical-align: middle;
}
.tier-docking  { background: #E6F4EA; color: #1E7E34; border: 1px solid #B7DEC0; }
.tier-pkanet   { background: #FFF3CD; color: #856404; border: 1px solid #FFE083; }
.tier-smi-only { background: #F8D7DA; color: #842029; border: 1px solid #F5C2C7; }

/* ── Metric row ── */
.metric-row {
    display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0;
}
.metric-box {
    background: #fff;
    border: 1px solid #E0D6C8;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    min-width: 120px;
    text-align: center;
}
.metric-box .val { font-size: 1.5rem; font-weight: 700; color: #E8A020; }
.metric-box .lbl { font-size: 0.78rem; color: #8B7355; margin-top: 2px; }

/* ── Dataframe tweaks ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Expander (Advanced) ── */
[data-testid="stExpander"] summary {
    font-size: 0.85rem;
    color: #8B7355;
}

/* ── Fixed page footer ── */
.page-footer {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #F0EAE0;
    border-top: 1px solid #E0D6C8;
    padding: 6px 20px;
    font-size: 0.75rem;
    color: #A89070;
    z-index: 999;
    display: flex;
    align-items: center;
    gap: 6px;
}
.page-footer a { color: #A89070; text-decoration: none; }
.page-footer a:hover { color: #E8A020; }

/* ── Mode tab bar ── */
.main .block-container div[data-testid="stHorizontalBlock"]:first-of-type
    button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    color: #8B7355 !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.2rem !important;
    box-shadow: none !important;
}
.main .block-container div[data-testid="stHorizontalBlock"]:first-of-type
    button:hover {
    color: #2C2C2C !important;
    background: transparent !important;
}
.main .block-container div[data-testid="stHorizontalBlock"]:first-of-type
    button[kind="primary"] {
    background: transparent !important;
    color: #2C2C2C !important;
    border-bottom: 3px solid #E8A020 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="page-footer">' +
    '⌬+⌬ Analog Builder &nbsp;—&nbsp;' +
    '<a href="mailto:kowith@ccs.tsukuba.ac.jp">kowith@ccs.tsukuba.ac.jp</a>' +
    '</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "mode": None,
    "step": 1,
    "parent_smiles": "",
    "parent_name": "compound",
    "parent_mol": None,
    "receptor_path": None,
    "protein_path": None,
    "complex_path": None,
    "ref_ligand_path": None,
    "selected_atoms": set(),
    "concerted": False,
    "allow_heteroatom_H": False,
    "risk": "Moderate",
    "n_analogs": 20,
    "rank_by": "Overall drug-likeness (recommended)",
    "rank_code": "Balanced (100-pt weights)",
    "weights": {"potency": 30, "selectivity": 10, "solubility": 25,
                "metabolic": 15, "synthesis": 10, "novelty": 10},
    "categories_on": {k: True for k in core.CATEGORY_BASE_GOALS},
    "max_MW": 600.0,
    "avoid_nitro": True,
    "avoid_aldehyde": True,
    "avoid_reactive": True,
    "avoid_toxic": True,
    "custom_frags_text": "",
    "pocket_residue_text": "",
    "accept_pocket_suggestions": True,
    "max_pocket_frags": 6,
    "pocket_frags": [],
    "analogs_df": None,
    "docking_ligands": None,
    "_void_subpockets": [],
    "_void_mode": "",
    "_void_size_filter": None,
    "struct_mode": "A",
    "_plip_df": None,            # PLIP interaction table (parent ligand)
    "_plip_tag_counts": {},      # residue tags from PLIP
    "_plip_rec": None,           # unified recommendation dict
    "_plip_parent_feats": [],    # cIFP features of parent
    "_analog_plip": {},          # {compound: [features]} for analogs
    "_cifp_comparison": None,    # compare_cifp DataFrame          # "A" = co-crystal complex | "B" = ligand+protein
    "modeB_docked": False,       # True after Mode B docking completed
    "modeB_complex_path": None,  # path to pseudo-complex (protein + docked pose)
    "docking_summary": None,
    "cifp_results": None,
    "work_dir": None,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_work_dir() -> Path:
    if st.session_state.work_dir is None:
        st.session_state.work_dir = Path(tempfile.mkdtemp(prefix="analog_"))
    return Path(st.session_state.work_dir)


def go(step: int):
    st.session_state.step = step
    st.rerun()


def hint(text: str):
    st.markdown(f'<p class="hint">💡 {text}</p>', unsafe_allow_html=True)


def info_card(text: str):
    st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3D Viewer helpers (adapted from ACD / anyonecandock)
# ---------------------------------------------------------------------------

def _viewer_bg() -> str:
    """Return background colour for py3Dmol canvas."""
    try:
        theme = st.get_option("theme.base")
        return "#1a1a2e" if theme == "dark" else "#f8f6f2"
    except Exception:
        return "#f8f6f2"


def show3d(view, height: int = 480):
    """Render a py3Dmol view via components.html (no stmol dependency)."""
    import re as _re
    try:
        raw  = view._make_html()
        resp = _re.sub(r'(width\s*[:=]\s*)["\'\']?\d+px?["\'\']?', r'\g<1>100%', raw)
        components.html(
            f'<div style="width:100%;overflow:hidden">{resp}</div>',
            height=height, scrolling=False,
        )
    except Exception as _e:
        st.warning(f"3D viewer render error: {_e}")


def render_complex_3d(
    receptor_pdb: str,
    pose_mol,                   # RDKit Mol with 3D conformer
    height: int = 500,
    cutoff: float = 4.0,
    show_labels: bool = True,
    show_surface: bool = False,
    ref_ligand_pdb: str = "",   # co-crystal / reference ligand (magenta)
    key_prefix: str = "v3d",
):
    """
    Render protein–ligand complex in 3D using py3Dmol.

    Protein  → cartoon, spectrum colouring, 45% opacity
    Ligand   → cyan sticks
    Pocket residues (≤ cutoff Å) → orange sticks + optional labels
    Reference ligand → magenta sticks (if provided)
    """
    try:
        import py3Dmol
        from rdkit import Chem

        v  = py3Dmol.view(width="100%", height=height)
        v.setBackgroundColor(_viewer_bg())
        mi = 0

        # ── Protein ─────────────────────────────────────────────────────
        if receptor_pdb and os.path.exists(receptor_pdb):
            v.addModel(open(receptor_pdb).read(), "pdb")
            v.setStyle({"model": mi}, {
                "cartoon": {"color": "spectrum", "opacity": 0.45}
            })
            if show_surface:
                v.addSurface(py3Dmol.SAS,
                             {"opacity": 0.40, "color": "white"},
                             {"model": mi})
            mi += 1

        # ── Reference / co-crystal ligand (magenta) ──────────────────────
        if ref_ligand_pdb and os.path.exists(ref_ligand_pdb):
            v.addModel(open(ref_ligand_pdb).read(), "pdb")
            v.setStyle({"model": mi}, {
                "stick": {"colorscheme": "magentaCarbon", "radius": 0.18}
            })
            mi += 1

        # ── Docked / query ligand (cyan) ──────────────────────────────────
        if pose_mol is not None:
            mol_block = Chem.MolToMolBlock(pose_mol)
            v.addModel(mol_block, "mol")
            lig_m = mi
            v.setStyle({"model": lig_m}, {
                "stick": {"colorscheme": "cyanCarbon", "radius": 0.30}
            })
            v.addSphere({"center": {"x": 0, "y": 0, "z": 0}, "radius": 0.01,
                         "model": lig_m, "opacity": 0})  # anchor for zoomTo

            # ── Pocket residues (orange sticks + labels) ─────────────────
            if receptor_pdb and os.path.exists(receptor_pdb):
                interacting = core.get_interacting_residues(
                    receptor_pdb, pose_mol, cutoff=cutoff)
                for rb in interacting:
                    sel = {"model": 0, "resi": rb["resi"]}
                    if rb["chain"] and rb["chain"].strip():
                        sel["chain"] = rb["chain"]
                    v.setStyle(sel, {
                        "stick": {"colorscheme": "orangeCarbon", "radius": 0.20}
                    })
                    if show_labels:
                        chain_str = rb["chain"] if rb["chain"].strip() else ""
                        v.addLabel(
                            f"{rb['resn']}{rb['resi']}{chain_str}",
                            {"fontSize": 11, "fontColor": "yellow",
                             "backgroundColor": "black",
                             "backgroundOpacity": 0.65,
                             "inFront": True, "showBackground": True},
                            sel,
                        )
            v.zoomTo({"model": lig_m})
        else:
            v.zoomTo()

        show3d(v, height=height)

    except ImportError:
        st.info("Install `py3Dmol` to enable 3D viewer: `pip install py3Dmol`")
    except Exception as _e:
        st.warning(f"3D viewer error: {_e}")


def tier_badge_html(tier: str) -> str:
    if tier == "docking":
        return '<span class="tier-badge tier-docking">✅ Full: Docking + pKaNET + SMI</span>'
    elif tier == "pkanet":
        return '<span class="tier-badge tier-pkanet">⚡ pKaNET + SMI (no docking)</span>'
    else:
        return '<span class="tier-badge tier-smi-only">📄 SMI only</span>'


def show_mol(mol, highlight=None, size=(400, 300), use_container_width=True, atom_indices=False):
    if mol is None:
        return
    try:
        AllChem.Compute2DCoords(mol)
        if atom_indices:
            from rdkit.Chem.Draw import rdMolDraw2D as _d2d
            d = _d2d.MolDraw2DCairo(*size)
            o = d.drawOptions()
            o.addAtomIndices = True
            o.annotationFontScale = 0.7
            _d2d.PrepareAndDrawMolecule(
                d, mol,
                highlightAtoms=list(highlight or []),
                highlightAtomColors={i: (1.0, 0.6, 0.6) for i in (highlight or [])},
            )
            d.FinishDrawing()
            png = d.GetDrawingText()
        else:
            png = Draw.MolsToGridImage(
                [mol], molsPerRow=1,
                subImgSize=size,
                highlightAtomLists=[list(highlight or [])],
                returnPNG=True,
            )
        st.image(png, use_container_width=use_container_width)
    except Exception:
        st.caption("(Could not render structure)")


def _load_receptor_widget(key_prefix: str = "") -> None:
    """Reusable receptor loader widget (search / PDB ID / upload)."""
    rec_src = st.radio(
        "Load receptor from",
        ["🔍 Search RCSB", "#️⃣ PDB ID", "📁 Upload file"],
        horizontal=True,
        key=f"rec_src_{key_prefix}",
    )

    if rec_src == "🔍 Search RCSB":
        rcsb_query = st.text_input(
            "Search RCSB PDB", value="",
            placeholder="e.g. EGFR kinase, JAK2, insulin receptor",
            key=f"rcsb_search_q_{key_prefix}",
        )
        hint("Search by protein name, gene, UniProt ID, or keyword.")
        if st.button("Search RCSB", key=f"rcsb_search_btn_{key_prefix}") and rcsb_query.strip():
            with st.spinner("Searching RCSB PDB…"):
                st.session_state[f"_rcsb_results_{key_prefix}"] = core.search_rcsb(rcsb_query.strip(), max_results=8)
        rcsb_results = st.session_state.get(f"_rcsb_results_{key_prefix}", [])
        if rcsb_results:
            st.markdown(f"**{len(rcsb_results)} results**")
            for r in rcsb_results:
                cols = st.columns([1, 6, 2])
                with cols[0]:
                    st.markdown(f"**{r['id']}**")
                with cols[1]:
                    st.caption(f"{r['title']}")
                    st.caption(f"{r['resolution']} · {r['method']} · {r['organism']}")
                with cols[2]:
                    if st.button("Use", key=f"rcsb_use_{r['id']}_{key_prefix}"):
                        with st.spinner(f"Downloading {r['id']}…"):
                            try:
                                work = get_work_dir()
                                path = core.download_pdb(r["id"], work)
                                prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                                st.session_state.receptor_path   = path
                                st.session_state.protein_path    = prot
                                st.session_state.ref_ligand_path = lig
                                if lig:  # co-crystal complex
                                    st.session_state.complex_path = path
                                st.success(f"Receptor loaded ({r['id']}) ✅")
                            except Exception as e:
                                st.error(f"Could not download: {e}")

    elif rec_src == "#️⃣ PDB ID":
        pdb_id = st.text_input("4-letter PDB ID", value="", max_chars=4,
                               placeholder="e.g. 1M17", key=f"pdb_id_{key_prefix}")
        hint("Example: 1M17 is EGFR, 6VXX is SARS-CoV-2 spike.")
        if st.button("Load receptor", key=f"load_rec_{key_prefix}") and pdb_id.strip():
            with st.spinner("Downloading from RCSB…"):
                try:
                    work = get_work_dir()
                    path = core.download_pdb(pdb_id.strip().upper(), work)
                    prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                    st.session_state.receptor_path   = path
                    st.session_state.protein_path    = prot
                    st.session_state.ref_ligand_path = lig
                    if lig:
                        st.session_state.complex_path = path
                    st.success(f"Receptor loaded ({pdb_id.upper()}) ✅")
                except Exception as e:
                    st.error(f"Could not download: {e}")

    else:  # Upload file
        up = st.file_uploader("Upload .pdb or .cif file", type=["pdb", "cif"],
                               key=f"rec_upload_{key_prefix}")
        if up:
            work = get_work_dir()
            raw = work / up.name
            raw.write_bytes(up.read())
            try:
                path = core.cif_to_pdb_if_needed(str(raw))
                prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                st.session_state.receptor_path   = path
                st.session_state.protein_path    = prot
                st.session_state.ref_ligand_path = lig
                if lig:
                    st.session_state.complex_path = path
                st.success("Receptor uploaded ✅")
            except Exception as e:
                st.error(f"Could not process file: {e}")

    if st.session_state.receptor_path:
        st.success(f"✅ Receptor: `{Path(st.session_state.receptor_path).name}`")
        if st.session_state.ref_ligand_path:
            st.info("Co-crystal ligand detected (Mode A ready)")


def _render_step1_receptor_and_continue(smiles: str, name: str):
    """Step 1 footer: confirmation + continue button for all tracks."""
    md = st.session_state.mode

    if md == "structure":
        # Receptor loading already handled in the step==1 block above.
        # Just show a compact status line.
        if st.session_state.receptor_path:
            sc = "A" if st.session_state.ref_ligand_path else "B"
            mode_label = "Co-crystal (Mode A)" if sc == "A" else "Apo / docking (Mode B)"
            st.caption(
                f"🧬 Receptor: `{Path(st.session_state.receptor_path).name}` · {mode_label}"
            )

    st.write("")
    if md == "structure" and not st.session_state.receptor_path:
        st.warning("⚠️ Load a protein structure above before continuing.")

    ready = bool(smiles and smiles.strip()) and (
        md != "structure" or bool(st.session_state.receptor_path)
    )
    if st.button("Load compound & continue →", type="primary", disabled=not ready):
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            st.error("That SMILES doesn\'t look right. Check for typos.")
        else:
            AllChem.Compute2DCoords(mol)
            st.session_state.parent_smiles  = smiles.strip()
            st.session_state.parent_name    = name.strip() or "compound"
            st.session_state.parent_mol     = mol
            st.session_state.selected_atoms = set()
            st.session_state.analogs_df     = None
            st.session_state.modeB_docked   = False
            st.session_state.modeB_complex_path = None
            go(2)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – progress tracker
# ─────────────────────────────────────────────────────────────────────────────

LIGAND_STEPS = ["Parent compound", "Choose atoms", "Design options", "View results", "Docking", "Export"]
STRUCT_STEPS = ["Parent + receptor", "Choose atoms", "Pocket guidance", "View results", "Docking & cIFP", "Export"]


def render_sidebar():
    st.sidebar.markdown(
        '<div style="background:#1a472a;color:#86efac;text-align:center;'
        'padding:4px 12px;border-radius:8px;font-size:0.72rem;font-weight:500;'
        'margin-bottom:8px;">🖥️ LOCAL MODE — No docking limits</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<style>[data-testid="stSidebar"] [data-testid="stImage"] {text-align: center;}</style>',
        unsafe_allow_html=True,
    )
    st.sidebar.image(LOGO_URL, width=160)
    st.sidebar.markdown(
        '<p style="text-align:center;font-size:0.78rem;color:#8B7355;margin-top:-8px;">Ligand design for everyone</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    mode = st.session_state.mode
    if mode is None:
        st.sidebar.markdown('<p style="color:#8B7355;font-size:0.85rem;">Select a mode to begin.</p>',
                            unsafe_allow_html=True)
        return

    steps = LIGAND_STEPS if mode == "ligand" else STRUCT_STEPS
    current = st.session_state.step

    _mode_icon = LB_URL if mode == "ligand" else SB_URL
    _mode_name = "Ligand-based" if mode == "ligand" else "Structure-based"
    st.sidebar.markdown(
        f'<p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;'
        f'color:#8B7355;margin-bottom:8px;">'
        f'<img src="{_mode_icon}" width="16" style="vertical-align:middle;margin-right:4px;"/>'
        f'{_mode_name} track</p>',
        unsafe_allow_html=True
    )

    for i, label in enumerate(steps, start=1):
        cls = "active" if i == current else ("done" if i < current else "")
        if st.sidebar.button(
            f"{'●' if i == current else ('✓' if i < current else str(i))}  {label}",
            key=f"nav_{i}",
            use_container_width=True,
            type="primary" if i == current else "secondary",
        ):
            if i < current or (i == current + 1 and _step_complete(current)):
                go(i)

    st.sidebar.divider()

    # Show tier info in sidebar if analogs generated
    df_sb = st.session_state.analogs_df
    if df_sb is not None and not df_sb.empty:
        n_sb = len(df_sb)
        tier_sb = analog_tier(n_sb)
        tier_labels = {
            "docking": "🟢 Docking enabled",
            "pkanet": "🟡 pKaNET + SMI",
            "smi_only": "🔴 SMI download only",
        }
        st.sidebar.markdown(
            f'<div style="background:#FFF8EE;border-radius:8px;padding:0.5rem 0.75rem;'
            f'font-size:0.78rem;color:#5A4A35;margin-bottom:8px;">'
            f'<strong>{n_sb} analogs</strong><br>{tier_labels[tier_sb]}</div>',
            unsafe_allow_html=True,
        )

    if st.sidebar.button("↩ Change mode", use_container_width=True):
        st.session_state.mode = None
        st.session_state.step = 1
        st.session_state.parent_mol = None
        st.session_state.analogs_df = None
        st.rerun()

    st.sidebar.caption(f"Fragment library: {len(core.LIBRARY):,} groups")
    st.sidebar.divider()
    st.sidebar.markdown(
        '<p style="font-size:0.75rem;color:#A89070;line-height:1.5;margin:0;">'
        '⌬+⌬ Analog Builder<br>'
        '<a href="mailto:kowith@ccs.tsukuba.ac.jp" style="color:#A89070;">kowith@ccs.tsukuba.ac.jp</a>'
        '</p>',
        unsafe_allow_html=True,
    )


def _step_complete(step: int) -> bool:
    if step == 1:
        return st.session_state.parent_mol is not None
    if step == 2:
        return len(st.session_state.selected_atoms) > 0
    if step == 3:
        return True
    if step == 4:
        return st.session_state.analogs_df is not None
    return True


render_sidebar()


# ─────────────────────────────────────────────────────────────────────────────
# LANDING – mode picker  (clickable cards)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.mode is None:
    st.markdown(
        f'<div style="text-align:center;margin:40px 0 8px;">'
        f'<img src="{LOGO_URL}" width="260" style="display:inline-block;"/>'
        f'</div>'
        f'<p style="text-align:center;color:#8B7355;margin-bottom:28px;">'
        f'Design new drug candidates by modifying a parent compound. Choose how you want to work:</p>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        # Card content + button in one visual unit
        # CSS wraps both into a single card appearance
        st.markdown(
            f'<div class="mode-card-col" id="card-ligand">'
            f'<img src="{LB_URL}" width="90" style="display:block;margin:0 auto 0.75rem;"/>'
            f'<div style="font-size:1.15rem;font-weight:700;color:#2C2C2C;margin-bottom:0.4rem;">Ligand-based</div>'
            f'<div style="font-size:0.88rem;color:#6B5E4E;line-height:1.55;">'
            f'Start with just a SMILES string.<br>'
            f'Great for exploring substitutions quickly — no protein structure needed.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Select Ligand-based →", key="pick_ligand", use_container_width=True, type="primary"):
            st.session_state.mode = "ligand"
            st.session_state.step = 1
            st.rerun()

    with col_r:
        st.markdown(
            f'<div class="mode-card-col" id="card-structure">'
            f'<img src="{SB_URL}" width="90" style="display:block;margin:0 auto 0.75rem;"/>'
            f'<div style="font-size:1.15rem;font-weight:700;color:#2C2C2C;margin-bottom:0.4rem;">Structure-based</div>'
            f'<div style="font-size:0.88rem;color:#6B5E4E;line-height:1.55;">'
            f'Upload or fetch a protein structure.<br>'
            f'Analogs are guided by the actual binding pocket environment.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Select Structure-based →", key="pick_struct", use_container_width=True, type="primary"):
            st.session_state.mode = "structure"
            st.session_state.step = 1
            st.rerun()

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Mode tab bar
# ─────────────────────────────────────────────────────────────────────────────

mode  = st.session_state.mode
step  = st.session_state.step


def switch_mode(new_mode: str):
    if new_mode == st.session_state.mode:
        return
    st.session_state.mode = new_mode
    new_len = len(LIGAND_STEPS if new_mode == "ligand" else STRUCT_STEPS)
    st.session_state.step = min(st.session_state.step, new_len)
    st.rerun()


tab_l, tab_r = st.columns(2)
with tab_l:
    st.markdown(
        f'<div style="text-align:center;margin-bottom:-8px;">'
        f'<img src="{LB_URL}" width="70" style="opacity:{1.0 if mode=="ligand" else 0.4};"/></div>',
        unsafe_allow_html=True,
    )
    if st.button("Ligand-based", key="tab_ligand", use_container_width=True,
                 type="primary" if mode == "ligand" else "secondary"):
        switch_mode("ligand")
with tab_r:
    st.markdown(
        f'<div style="text-align:center;margin-bottom:-8px;">'
        f'<img src="{SB_URL}" width="70" style="opacity:{1.0 if mode=="structure" else 0.4};"/></div>',
        unsafe_allow_html=True,
    )
    if st.button("Structure-based", key="tab_structure", use_container_width=True,
                 type="primary" if mode == "structure" else "secondary"):
        switch_mode("structure")

st.markdown('<hr style="margin:0.2rem 0 1.2rem 0;border:none;border-top:1px solid #E0D6C8;">',
            unsafe_allow_html=True)

mode  = st.session_state.mode
step  = st.session_state.step
steps = LIGAND_STEPS if mode == "ligand" else STRUCT_STEPS

st.markdown(
    f'<p style="font-size:0.8rem;color:#8B7355;margin-bottom:0;">'
    f'Step {step} of {len(steps)}: <strong>{steps[step-1]}</strong></p>',
    unsafe_allow_html=True
)
st.markdown(f"## {steps[step-1]}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Parent compound
# ─────────────────────────────────────────────────────────────────────────────

if step == 1:
    md = st.session_state.mode   # alias used throughout Step 1
    # ── Structure track: show receptor loader FIRST ───────────────────────────
    if mode == "structure":
        st.markdown("### Step 1A — Load protein structure")
        info_card(
            "Load your protein from RCSB or upload a PDB file. "
            "The app will <strong>auto-detect</strong> whether it contains a bound ligand."
        )
        _load_receptor_widget(key_prefix="struct")

        if st.session_state.receptor_path:
            has_lig = bool(st.session_state.ref_ligand_path)
            if has_lig:
                st.success(
                    f"✅ **Co-crystal ligand detected** in "
                    f"`{Path(st.session_state.receptor_path).name}`"
                )
                _extracted_smi = ""
                try:
                    _lig_mol_tmp = Chem.MolFromPDBFile(
                        st.session_state.ref_ligand_path, removeHs=True)
                    if _lig_mol_tmp:
                        _extracted_smi = Chem.MolToSmiles(_lig_mol_tmp)
                        st.session_state["_modeA_extracted_smiles"] = _extracted_smi
                except Exception:
                    pass

                _col_smi, _col_opt = st.columns([3, 2])
                with _col_smi:
                    if _extracted_smi:
                        st.markdown("**Co-crystal ligand SMILES (auto-extracted):**")
                        st.code(_extracted_smi, language=None)
                        if st.button("Use as parent compound ↑", key="use_extracted"):
                            st.session_state.parent_smiles = _extracted_smi
                            st.rerun()
                with _col_opt:
                    st.markdown("**What would you like to do?**")
                    _dock_choice = st.radio(
                        "Docking option",
                        [
                            "Use this complex as-is — skip docking",
                            "Dock my own ligand into this protein instead",
                        ],
                        index=0,
                        key="complex_dock_choice",
                        label_visibility="collapsed",
                    )
                    if _dock_choice.startswith("Use this complex"):
                        st.session_state.struct_mode = "A"
                        st.caption("✅ Co-crystal pose will be used directly.")
                    else:
                        st.session_state.struct_mode = "B"
                        st.caption("ℹ️ ACD will dock your ligand after Step 2.")
            else:
                st.session_state.struct_mode = "B"
                st.info(
                    f"**Apo structure** — `{Path(st.session_state.receptor_path).name}`  \n"
                    "No bound ligand. ACD docking will run automatically after Step 2."
                )

        st.markdown("---")

    # ── If Mode A and SMILES already extracted, skip full radio ─────────
    _modeA_smi = st.session_state.get("_modeA_extracted_smiles", "")
    _is_modeA_auto = (md == "structure"
                      and st.session_state.get("struct_mode") == "A"
                      and bool(_modeA_smi))

    # Only show Step 1B header when parent SMILES still needs user input
    if md == "structure" and not _is_modeA_auto:
        st.markdown("### Step 1B — Parent compound")

    if _is_modeA_auto:
        # Show compact SMILES confirmation — no radio needed
        st.markdown(
            f'<div style="background:#F0F9F0;border:1.5px solid #C3E6CB;border-radius:10px;'
            f'padding:0.9rem 1.2rem;margin-bottom:0.6rem;">'
            f'<div style="font-size:0.78rem;font-weight:600;color:#1E7E34;margin-bottom:4px;">'
            f'✅ Parent compound — extracted from co-crystal PDB</div>'
            f'<code style="font-size:0.8rem;color:#155724;word-break:break-all;">{_modeA_smi}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("✏️ Use a different SMILES instead", key="modeA_override_smi"):
            st.session_state["_modeA_extracted_smiles"] = ""
            st.rerun()
        smiles_input  = _modeA_smi
        draw_mode     = False
        paste_mode    = False
        pubchem_mode  = False
        # Pre-fill session so continue button works
        if not st.session_state.get("parent_smiles"):
            st.session_state.parent_smiles = _modeA_smi
    else:
        input_options = ["🔍 Search PubChem", "⌨️ Paste SMILES", "✏️ Draw it"]
        input_tab = st.radio(
            "How do you want to enter your compound?",
            input_options,
            horizontal=True,
            label_visibility="collapsed",
        )
        draw_mode    = (input_tab == "✏️ Draw it") and _KETCHER_OK
        paste_mode   = (input_tab == "⌨️ Paste SMILES") or (input_tab == "✏️ Draw it" and not _KETCHER_OK)
        pubchem_mode = (input_tab == "🔍 Search PubChem")

    if pubchem_mode:
        pc_col1, pc_col2 = st.columns([5, 1])
        with pc_col1:
            pc_query = st.text_input("Compound name",
                placeholder="e.g. imatinib, apigenin, caffeine, aspirin…", key="pubchem_query")
        with pc_col2:
            st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
            pc_search = st.button("Search", key="pc_search_btn", type="secondary")

        if pc_search and pc_query.strip():
            with st.spinner(f"Searching PubChem for '{pc_query.strip()}'…"):
                _sr = core.search_pubchem(pc_query.strip())
                st.session_state["_pc_result"] = _sr
                if _sr.get("found") and (_sr.get("smiles") or "").strip():
                    st.session_state["smiles_in_pc"] = _sr["smiles"]
                    _auto_name = (_sr["iupac"] or pc_query)[:20].lower().replace(" ", "_")
                    st.session_state["pc_compound_name"] = _auto_name
                    st.session_state.parent_smiles = _sr["smiles"]
                    st.session_state.parent_name = _auto_name
                    st.rerun()

        _sr = st.session_state.get("_pc_result")
        if _sr and _sr.get("found"):
            _ic, _imgc = st.columns([3, 1])
            with _ic:
                st.markdown(
                    f"**{_sr['iupac']}**  \n"
                    f"`{_sr['formula']}` · {_sr['mw']:.2f} g/mol · "
                    f"[PubChem CID {_sr['cid']}]({_sr['url']})"
                )
            with _imgc:
                st.image(_sr["img_url"], width=140)
            if not (_sr.get("smiles") or "").strip():
                st.warning("This PubChem result did not return a usable SMILES string.")
        elif _sr and not _sr.get("found"):
            st.error(f"Not found: {_sr.get('error', 'Unknown error')}")

        smiles = st.text_input("SMILES string", value=st.session_state.parent_smiles,
            key="smiles_in_pc", help="Auto-filled from PubChem search, or paste your own SMILES here.")
        st.session_state.parent_smiles = smiles
        name = st.text_input("Compound name", value=st.session_state.parent_name, key="pc_compound_name")
        hint("Used to label your output files.")
        _render_step1_receptor_and_continue(smiles, name)

    elif draw_mode:
        hint("Draw your molecule, then click **Apply** in the sketcher to capture it.")
        drawn = st_ketcher(st.session_state.parent_smiles or "", key="ketcher_draw", height=480)
        smiles = drawn or st.session_state.parent_smiles

        prev_col, form_col = st.columns([1, 1], gap="large")
        with prev_col:
            mol_preview = Chem.MolFromSmiles(smiles.strip()) if smiles and smiles.strip() else None
            if mol_preview:
                c_sites = core.attachable_atom_indices(mol_preview, carbon_only=True)
                st.markdown("**Preview** — highlighted atoms can be modified")
                show_mol(mol_preview, highlight=c_sites, atom_indices=True)
                st.caption(f"Captured SMILES: `{smiles}`  ·  {mol_preview.GetNumAtoms()} atoms · {len(c_sites)} modifiable C–H sites")
            else:
                st.markdown(
                    '<div style="background:#F0EAE0;border-radius:12px;height:240px;'
                    'display:flex;align-items:center;justify-content:center;color:#A89070;">'
                    '<span style="font-size:0.9rem;">Draw a molecule and click Apply to see the preview</span></div>',
                    unsafe_allow_html=True)
        with form_col:
            name = st.text_input("Give it a short name", value=st.session_state.parent_name,
                                 placeholder="e.g. compound_1")
            hint("Used to label your output files.")
            _render_step1_receptor_and_continue(smiles, name)

    else:
        col_form, col_mol = st.columns([1, 1], gap="large")
        with col_form:
            if input_tab == "✏️ Draw it" and not _KETCHER_OK:
                st.warning("The drawing tool isn't installed here. Add `streamlit-ketcher` to "
                           "requirements.txt to enable it. For now, paste a SMILES instead.")
            smiles = st.text_area("Paste your compound SMILES", value=st.session_state.parent_smiles,
                height=90, placeholder="e.g. CC1=CC=CC=C1")
            hint("SMILES is a text code for a molecule. Copy it from ChemDraw, PubChem, or any chemistry database.")
            name = st.text_input("Give it a short name", value=st.session_state.parent_name,
                                 placeholder="e.g. compound_1")
            hint("Used to label your output files.")
            _render_step1_receptor_and_continue(smiles, name)

        with col_mol:
            mol_preview = Chem.MolFromSmiles(smiles.strip()) if smiles.strip() else None
            if mol_preview:
                c_sites = core.attachable_atom_indices(mol_preview, carbon_only=True)
                st.markdown("**Preview** — highlighted atoms can be modified")
                show_mol(mol_preview, highlight=c_sites, atom_indices=True)
                st.caption(f"{mol_preview.GetNumAtoms()} atoms · {len(c_sites)} modifiable C–H sites")
            else:
                st.markdown(
                    '<div style="background:#F0EAE0;border-radius:12px;height:320px;'
                    'display:flex;align-items:center;justify-content:center;color:#A89070;">'
                    '<span style="font-size:0.9rem;">Molecule preview appears here</span></div>',
                    unsafe_allow_html=True)
            if mode == "structure" and st.session_state.receptor_path:
                st.markdown("**Receptor status**")
                st.success(f"✅ {Path(st.session_state.receptor_path).name}")
                if st.session_state.ref_ligand_path:
                    st.info("Co-crystal ligand detected — will be used as reference pose")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Choose atoms
# ─────────────────────────────────────────────────────────────────────────────

elif step == 2:
    mol = st.session_state.parent_mol
    if mol is None:
        st.warning("Go back to Step 1 and load a compound first.")
        st.stop()

    attachable = core.attachable_atom_indices(mol, carbon_only=False)
    c_only     = core.attachable_atom_indices(mol, carbon_only=True)

    info_card("Click an atom index in the list to select it. "
              "Selected atoms will be highlighted on the structure — those are the spots where new groups will be added.")

    col_pick, col_view = st.columns([1, 1], gap="large")

    with col_pick:
        st.markdown("### Pick attachment points")
        hint("Stick to C–H sites (carbon atoms) unless you have a specific reason to modify N–H or O–H.")

        new_sel = set()
        for idx in attachable:
            atom  = mol.GetAtomWithIdx(idx)
            atype = "C–H" if atom.GetAtomicNum() == 6 else f"{atom.GetSymbol()}–H"
            label = f"Atom {idx}  ({atype})"
            if idx not in c_only:
                label += "  *(heteroatom)*"
            if st.checkbox(label, value=idx in st.session_state.selected_atoms, key=f"atm_{idx}"):
                new_sel.add(idx)

        st.session_state.selected_atoms = new_sel
        st.write("")
        st.markdown("### Options")
        st.session_state.allow_heteroatom_H = st.checkbox(
            "Allow N–H / O–H / S–H substitution",
            value=st.session_state.allow_heteroatom_H,
        )
        hint("By default only carbon (C–H) sites are modified.")
        st.session_state.concerted = st.checkbox(
            "Concerted mode — attach the same group to all selected atoms at once",
            value=st.session_state.concerted,
        )
        hint("Off: each analog changes one site. On: all selected sites changed together.")

    with col_view:
        st.markdown("### Structure")
        show_mol(mol, highlight=sorted(new_sel), atom_indices=True)
        if new_sel:
            st.caption(f"Selected: atoms {sorted(new_sel)}")
        else:
            st.caption("No atoms selected yet")

    st.write("")
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            go(1)
    with col_next:
        if st.button("Continue →", type="primary", disabled=len(new_sel) == 0):
            go(3)
        if not new_sel:
            st.caption("Select at least one atom to continue.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Design options  (ligand track)
# ─────────────────────────────────────────────────────────────────────────────

elif step == 3 and mode == "ligand":
    info_card("These settings control what kinds of groups are added and how the results are ranked. "
              "The defaults work well — you can leave them and click Generate.")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### How many analogs?")
        st.session_state.n_analogs = st.slider(
            "Number of analogs to generate", 5, 1000,
            st.session_state.n_analogs, step=5,
        )
        # Live tier preview
        _tier_preview = analog_tier(st.session_state.n_analogs)
        st.markdown(
            f'<p style="font-size:0.82rem;margin-top:-4px;">'
            f'With <strong>{st.session_state.n_analogs}</strong> analogs: '
            f'{tier_badge_html(_tier_preview)}</p>',
            unsafe_allow_html=True,
        )
        if _tier_preview == "docking":
            hint("≤ 20 analogs → Docking button available after generation.")
        elif _tier_preview == "pkanet":
            hint("21–200 analogs → Download SMI + pKaNET protonation/3D conversion.")
        else:
            hint("Local mode: no limits — docking available for any number of analogs.")

        st.markdown("### How adventurous?")
        risk_map = {
            "Conservative — small groups only": "Conservative",
            "Moderate — balanced (recommended)": "Moderate",
            "Exploratory — larger groups allowed": "Exploratory",
        }
        risk_label = st.radio("Substitution size", list(risk_map.keys()),
                               index=list(risk_map.values()).index(st.session_state.risk))
        st.session_state.risk = risk_map[risk_label]
        hint("Conservative keeps modifications small and drug-like.")

    with col_b:
        st.markdown("### Rank results by")
        rank_map = {
            "Overall drug-likeness (recommended)": "Balanced (100-pt weights)",
            "Most similar to parent":              "Similarity to parent",
            "Best predicted solubility":           "Solubility (ESOL)",
            "Easiest to synthesise":               "Synthetic feasibility",
        }
        rank_labels = list(rank_map.keys())
        current_label = st.session_state.rank_by if st.session_state.rank_by in rank_labels else rank_labels[0]
        rank_label = st.radio("Sort analogs by", rank_labels,
                               index=rank_labels.index(current_label))
        st.session_state.rank_by   = rank_label
        st.session_state.rank_code = rank_map[rank_label]

        st.markdown("### Fragment categories")
        hint("Uncheck any group types you want to exclude.")
        cat_cols = st.columns(2)
        cats = list(core.CATEGORY_BASE_GOALS.keys())
        new_cats = {}
        for i, cat in enumerate(cats):
            with cat_cols[i % 2]:
                new_cats[cat] = st.checkbox(cat.replace("_", " ").capitalize(),
                    value=st.session_state.categories_on.get(cat, True), key=f"cat_{cat}")
        st.session_state.categories_on = new_cats

        # ── Module 2: pocket-aware ML ranking (ligand track) ──────────────
        if _PR_OK:
            st.markdown("### 🧬 Pocket-aware fragment ranking")
            hint("Enter binding-site residues — ML will auto-rank and use the best fragments. Leave blank to use all fragment categories.")
            lb_residues = st.text_input(
                "Key binding-site residues *(optional)*",
                value=st.session_state.get("lb_pocket_residues", ""),
                placeholder="e.g. ASP315 LYS89 PHE82 LEU83  (one-letter or three-letter OK)",
                key="lb_pocket_input",
            )

            if lb_residues.strip():
                lb_codes = core.parse_pocket_residues(lb_residues)
                if lb_codes:
                    # ── Auto-compute ML scores immediately ──────────────────
                    st.session_state["lb_pocket_residues"] = lb_residues
                    _lb_tag_counts: dict = {}
                    for _aa in lb_codes:
                        for _tag in core.AA_TAGS.get(_aa, []):
                            _lb_tag_counts[_tag] = _lb_tag_counts.get(_tag, 0) + 1
                    st.session_state["_pocket_tag_counts"] = _lb_tag_counts

                    all_frags = [f for f in core.LIBRARY
                                 if st.session_state.categories_on.get(f.category, True)]
                    lb_scored = _pr.score_fragments(all_frags, _lb_tag_counts, alpha=0.7)

                    # Auto-use top-N as generation pool (no checkbox needed)
                    n_ml = st.session_state.n_analogs
                    top_n = lb_scored[:max(n_ml, 50)]  # always keep ≥50 for diversity
                    st.session_state.pocket_frags = [f for f, _ in top_n]

                    # ── Summary info card ───────────────────────────────────
                    dominant = max(_lb_tag_counts, key=_lb_tag_counts.get)
                    _mstatus = _pr.model_status()
                    _badge = "🟢 ChemBERTa" if _mstatus["chemberta_loaded"] else "🟡 Rule-based fallback"
                    info_card(
                        f"<strong>{len(lb_codes)} residues detected</strong> · "
                        f"dominant: <code>{dominant.replace('_', ' ')}</code> · "
                        f"{_badge}<br>"
                        f"Ranked {len(all_frags):,} fragments → using top <strong>{len(top_n)}</strong> "
                        f"as generation pool for Step 4."
                    )

                    # ── Ranked table (top 20 preview) ──────────────────────
                    st.markdown(f"**Top 20 ML-ranked fragments** *(used automatically in Step 4)*")
                    preview_20 = lb_scored[:20]
                    lb_df = pd.DataFrame([{
                        "Rank":     i + 1,
                        "Fragment": f.name,
                        "Category": f.category,
                        "ML score": f"{s:.3f}",
                        "Reason":   _pr.explain_score(f, _lb_tag_counts)["reason"][:60] + "…",
                    } for i, (f, s) in enumerate(preview_20)])

                    def _lb_colour(val):
                        try:
                            v = float(val)
                            if v >= 0.75:   return "background-color:#E6F4EA;color:#1E7E34"
                            elif v >= 0.50: return "background-color:#FFF3CD;color:#856404"
                            else:           return "background-color:#FDE8E8;color:#842029"
                        except Exception:
                            return ""

                    st.dataframe(
                        lb_df.style.map(_lb_colour, subset=["ML score"]),
                        use_container_width=True, hide_index=True, height=340,
                    )

                    with st.expander("🔍 Score breakdown for top fragment"):
                        top_f, top_s = preview_20[0]
                        exp = _pr.explain_score(top_f, _lb_tag_counts)
                        st.markdown(f"**{top_f.name}** — ML score: `{top_s:.3f}`")
                        st.markdown(f"- Category: `{exp.get('category', '—')}`")
                        st.markdown(f"- Category score: `{exp.get('category_score', '—')}`")
                        st.markdown(f"- Model: `{exp.get('model_used', 'rule-based')}`")
                        st.markdown(f"- Dominant pocket tag: `{exp.get('dominant_pocket_tag', '—')}`")
                        st.markdown(f"- Reason: {exp.get('reason', '—')}")
                        top_tags = exp.get('top_pocket_tags') or exp.get('top_matching_tags') or []
                        if top_tags:
                            tags_str = ", ".join(f"`{t}` ({c})" for t, c in top_tags[:3])
                            st.markdown(f"- Top pocket tags: {tags_str}")

                else:
                    st.warning("Could not parse residue codes. Try formats like: ASP315 LYS89 PHE82  or  D K F L")
            else:
                # No residues provided → clear ML selection, use all categories
                st.session_state["_pocket_tag_counts"] = {}
                if st.session_state.get("pocket_frags"):
                    st.session_state.pocket_frags = []
                st.caption("💡 No residues entered — all fragment categories will be used in Step 4.")

    with st.expander("⚙️ Advanced options"):
        adv1, adv2 = st.columns(2)
        with adv1:
            st.markdown("**Structural filters**")
            st.session_state.avoid_nitro    = st.checkbox("Remove nitro groups",    value=st.session_state.avoid_nitro)
            st.session_state.avoid_aldehyde = st.checkbox("Remove aldehydes",       value=st.session_state.avoid_aldehyde)
            st.session_state.avoid_reactive = st.checkbox("Remove reactive groups", value=st.session_state.avoid_reactive)
            st.session_state.avoid_toxic    = st.checkbox("Remove toxic flags",     value=st.session_state.avoid_toxic)
            st.session_state.max_MW = st.number_input("Max molecular weight (Da)", value=st.session_state.max_MW, step=25.0)
        with adv2:
            st.markdown("**Goal weights** (must sum to ~100)")
            w = st.session_state.weights
            new_w = {}
            for k in w:
                new_w[k] = st.slider(k.capitalize(), 0, 100, int(w[k]), step=5, key=f"w_{k}")
            st.session_state.weights = new_w
        st.markdown("**Custom fragments** — one SMILES with `[*]` per line")
        st.session_state.custom_frags_text = st.text_area(
            "Custom fragments", value=st.session_state.custom_frags_text,
            height=80, label_visibility="collapsed", placeholder="[*]C1CC1\n[*]OCC")

    st.write("")
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            go(2)
    with col_next:
        if st.button("Generate analogs →", type="primary"):
            go(4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Pocket guidance  (structure track)
# ─────────────────────────────────────────────────────────────────────────────

elif step == 3 and mode == "structure":
    info_card("We'll analyse which residues are near the binding site and suggest the best functional groups to add.")

    # ── Mode B: dock ligand first if not yet done ────────────────────────────
    if st.session_state.struct_mode == "B" and not st.session_state.modeB_docked:
        st.markdown("### 🔬 Step 3A — Dock ligand with ACD")
        info_card(
            "<strong>Mode B</strong>: Your ligand needs to be docked into the "
            "protein before pocket analysis. ACD (Anyone Can Dock) will run now."
        )
        acd_ok    = bool(shutil.which("acd"))
        obabel_ok = bool(shutil.which("obabel"))
        receptor  = st.session_state.receptor_path
        protein   = st.session_state.protein_path

        bcol1, bcol2 = st.columns(2)
        bcol1.markdown(
            f'<div style="padding:0.6rem 1rem;border-radius:8px;'
            f'background:{"#E6F4EA" if acd_ok else "#FDECEA"};'
            f'color:{"#1E7E34" if acd_ok else "#B00020"};font-size:0.85rem;">'
            f'{"✅ acd available" if acd_ok else "❌ acd not found — pip install anyonecandock"}</div>',
            unsafe_allow_html=True,
        )
        bcol2.markdown(
            f'<div style="padding:0.6rem 1rem;border-radius:8px;'
            f'background:{"#E6F4EA" if obabel_ok else "#FDECEA"};'
            f'color:{"#1E7E34" if obabel_ok else "#B00020"};font-size:0.85rem;">'
            f'{"✅ obabel available" if obabel_ok else "❌ obabel not found — apt install openbabel"}</div>',
            unsafe_allow_html=True,
        )

        if acd_ok and obabel_ok and receptor:
            dock_ph  = st.number_input("pH for protonation", value=7.4, step=0.1, key="modeB_ph")
            use_pkanet = st.checkbox("Use pKaNET protonation", value=True, key="modeB_pkanet")

            if st.button("🚀 Dock ligand now", type="primary", key="modeB_dock_btn"):
                work     = get_work_dir()
                dock_out = str(work / "modeB_docking")
                compound = st.session_state.parent_name or "ligand"
                smiles   = st.session_state.parent_smiles

                with st.spinner(f"Docking {compound} into {Path(receptor).name}…"):
                    cmd = core.build_acd_dock_cmd(
                        receptor=receptor,
                        smiles=smiles,
                        name=compound,
                        ph=float(dock_ph),
                        output_dir=dock_out,
                        use_pkanet=bool(use_pkanet),
                        save_poses=True,
                    )
                    rc, log = core.run_command(cmd)
                    (work / "modeB_acd.log").write_text(log)

                    if rc != 0:
                        st.error(f"ACD docking failed (exit {rc}). Check the log below.")
                        with st.expander("ACD log"):
                            st.text(log[-3000:])
                    else:
                        # Find best pose SDF → convert to PDB → build pseudo-complex
                        sdf_path = core.find_pose_sdf(dock_out)
                        if sdf_path:
                            pose_pdb = str(work / "modeB_pose.pdb")
                            core.sdf_first_mol_to_pdb(sdf_path, pose_pdb)
                            # Build pseudo-complex: protein + docked pose
                            complex_pdb = str(work / "modeB_complex.pdb")
                            if protein and os.path.exists(protein):
                                core.combine_protein_ligand_pdb(protein, pose_pdb, complex_pdb)
                            else:
                                core.combine_protein_ligand_pdb(receptor, pose_pdb, complex_pdb)

                            st.session_state.complex_path       = complex_pdb
                            st.session_state.ref_ligand_path    = pose_pdb
                            st.session_state.modeB_docked       = True
                            st.session_state.modeB_complex_path = complex_pdb

                            # Parse best score
                            best = core.parse_acd_score_csvs(dock_out)
                            be_str = ""
                            if best:
                                sc_col = best.get("_score_col","")
                                be_val = best.get(sc_col)
                                if be_val is not None:
                                    be_str = f"  ·  BE = {float(be_val):.2f} kcal/mol"

                            st.success(f"✅ Docking complete{be_str} — proceeding to pocket analysis")
                            st.rerun()
                        else:
                            st.error("Docking ran but no pose SDF found. Check ACD output.")
                            with st.expander("ACD log"):
                                st.text(log[-3000:])
        elif not receptor:
            st.warning("No receptor loaded. Go back to Step 1 and load a protein PDB.")
        else:
            st.info("ACD or OpenBabel not available. Cannot run docking in Mode B.")

        # Stop here until docking is done
        if not st.session_state.modeB_docked:
            st.divider()
            if st.button("← Back to Step 1", key="modeB_back"):
                go(1)
            st.stop()

    # ── Pocket analysis (Mode A or Mode B after docking) ─────────────────────
    if st.session_state.complex_path and os.path.exists(str(st.session_state.complex_path)):
        col_analysis, col_result = st.columns([1, 1], gap="large")

        with col_analysis:
            st.markdown("### Automatic pocket analysis")
            hint("Click the button to detect residues within 6 Å of the co-crystal ligand.")
            cutoff = st.slider("Pocket distance cutoff (Å)", 4.0, 10.0, 6.0, 0.5)
            if st.button("Analyse pocket", type="primary"):
                with st.spinner("Analysing binding pocket…"):
                    try:
                        pocket_df, contact_df, growth_df, lig_atoms = core.analyze_complex_distance_shell(
                            st.session_state.complex_path, pocket_cutoff=cutoff)
                        residue_codes = [x for x in growth_df["aa_one"].tolist() if x]
                        st.session_state.pocket_residue_text = " ".join(
                            core.AA_ONE_TO_THREE.get(r, r) for r in residue_codes)
                        active_lib = list(core.BUILTIN_LIBRARY)
                        _, _, pocket_frags = core.suggest_fragments_from_residues(
                            residue_codes, active_lib, st.session_state.max_pocket_frags)
                        st.session_state.pocket_frags = pocket_frags
                        # Store tag counts for Module 2 ML scoring
                        _tag_counts: dict = {}
                        for _aa in residue_codes:
                            for _tag in core.AA_TAGS.get(_aa, []):
                                _tag_counts[_tag] = _tag_counts.get(_tag, 0) + 1
                        st.session_state["_pocket_tag_counts"] = _tag_counts
                        st.success(f"Found {len(pocket_df)} pocket residues, {len(growth_df)} growth opportunities")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

            st.markdown("### Or paste residues manually")
            hint("Type residue names like: ASP315 LYS89 TYR102")
            manual = st.text_input("Pocket residues", value=st.session_state.pocket_residue_text,
                placeholder="ASP315 LYS89 TYR102", label_visibility="collapsed")
            if manual != st.session_state.pocket_residue_text:
                st.session_state.pocket_residue_text = manual
                codes = core.parse_pocket_residues(manual)
                if codes:
                    _, _, pf = core.suggest_fragments_from_residues(codes, core.BUILTIN_LIBRARY, 6)
                    st.session_state.pocket_frags = pf
                    # Store tag counts for Module 2 ML scoring
                    _tag_counts_m: dict = {}
                    for _aa in codes:
                        for _tag in core.AA_TAGS.get(_aa, []):
                            _tag_counts_m[_tag] = _tag_counts_m.get(_tag, 0) + 1
                    st.session_state["_pocket_tag_counts"] = _tag_counts_m

        with col_result:
            if st.session_state.pocket_frags:
                st.markdown("### Suggested functional groups")

                # ── Module 2: ML pocket-aware ranking ──────────────────────
                if _PR_OK and st.session_state.get("_pocket_tag_counts"):
                    # Show model status badge
                    _mstatus = _pr.model_status()
                    _badge_col = "🟢" if _mstatus["chemberta_loaded"] else "🟡"
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:#5A4A35;margin-bottom:6px;">'
                        f'{_badge_col} <strong>{_mstatus["mode"]}</strong>'
                        f'{" · cache: " + str(_mstatus["cache_size"]) + " SMILES" if _mstatus["chemberta_loaded"] else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    hint("Ranked by ChemBERTa semantic similarity + PDB co-occurrence (blended).")
                    tag_counts = st.session_state["_pocket_tag_counts"]
                    alpha = st.slider(
                        "Category vs fragment-level weight",
                        0.0, 1.0, 0.7, 0.05,
                        help="Higher = more weight on fragment category fit; lower = more weight on specific fragment PDB frequency",
                        key="pr_alpha",
                    )
                    scored = _pr.score_fragments(
                        st.session_state.pocket_frags, tag_counts, alpha=alpha
                    )
                    fdf = pd.DataFrame([{
                        "Rank":     i + 1,
                        "Group":    f.name,
                        "Category": f.category,
                        "ML score": f"{score:.3f}",
                        "Why":      _pr.explain_score(f, tag_counts)["reason"],
                    } for i, (f, score) in enumerate(scored)])

                    # Colour-code by score
                    def _score_colour(val):
                        try:
                            v = float(val)
                            if v >= 0.75:   return "background-color:#E6F4EA;color:#1E7E34"
                            elif v >= 0.50: return "background-color:#FFF3CD;color:#856404"
                            else:           return "background-color:#FDE8E8;color:#842029"
                        except Exception:
                            return ""

                    styled = fdf.style.map(_score_colour, subset=["ML score"])
                    st.dataframe(styled, use_container_width=True, hide_index=True, height=320)

                    # Update pocket_frags order to ML-ranked order
                    st.session_state.pocket_frags = [f for f, _ in scored]

                    # Expandable explanation for top fragment
                    with st.expander("🔍 Score breakdown for top fragment"):
                        top_f, top_s = scored[0]
                        exp = _pr.explain_score(top_f, tag_counts)
                        st.markdown(f"**{top_f.name}** — ML score: `{top_s:.3f}`")
                        st.markdown(f"- Category: `{exp.get('category', '—')}`")
                        st.markdown(f"- Category score: `{exp.get('category_score', '—')}`")
                        st.markdown(f"- Model: `{exp.get('model_used', 'rule-based')}`")
                        st.markdown(f"- Dominant pocket tag: `{exp.get('dominant_pocket_tag', '—')}`")
                        st.markdown(f"- Reason: {exp.get('reason', '—')}")
                        top_tags = exp.get('top_pocket_tags') or exp.get('top_matching_tags') or []
                        if top_tags:
                            tags_str = ", ".join(f"`{t}` ({c})" for t, c in top_tags[:3])
                            st.markdown(f"- Top pocket tags: {tags_str}")

                else:
                    # Fallback: rule-based display
                    hint("These groups match the chemistry of your binding pocket residues.")
                    fdf = pd.DataFrame([
                        {"Group": f.name, "Category": f.category, "Why": f.notes or f.category}
                        for f in st.session_state.pocket_frags
                    ])
                    st.dataframe(fdf, use_container_width=True, hide_index=True)
                    if not _PR_OK:
                        st.caption("ℹ️ Install `pocket_reference.py` to enable ML scoring.")
            else:
                st.markdown(
                    '<div style="background:#F0EAE0;border-radius:12px;padding:2rem;'
                    'text-align:center;color:#A89070;margin-top:1rem;">'
                    'Suggested groups will appear here after analysis</div>',
                    unsafe_allow_html=True)
    else:
        st.warning("No receptor file found. Go back to Step 1 and load a receptor.")

    # ── PLIP interaction analysis ───────────────────────────────────────────
    if st.session_state.complex_path and os.path.exists(str(st.session_state.complex_path)):
        st.divider()
        st.markdown("### 🔬 Protein–Ligand Interaction Analysis (PLIP)")
        plip_ok = bool(shutil.which("plipcmd") or shutil.which("plip"))

        if not plip_ok:
            st.info(
                "PLIP is not installed — using distance-based contact fingerprint as fallback. "
                "Install with: `pip install plip`"
            )

        col_plip_btn, col_plip_status = st.columns([2, 3])
        with col_plip_btn:
            if st.button("Run PLIP analysis", type="primary", key="plip_btn"):
                work = get_work_dir()
                cpx  = st.session_state.complex_path
                name = st.session_state.parent_name or "ligand"

                with st.spinner("Running PLIP interaction analysis…"):
                    if plip_ok:
                        xml_path, err, log = _pa.run_plip_on_complex(
                            cpx, str(work / "plip_out"), name)
                        if xml_path:
                            plip_df = _pa.parse_plip_to_table(xml_path)
                        else:
                            st.warning(f"PLIP failed: {err} — using distance fallback")
                            plip_df = pd.DataFrame()
                    else:
                        plip_df = pd.DataFrame()

                    # Fallback: distance-based cIFP
                    if plip_df.empty:
                        feats = _pa.distance_cifp_features(cpx, cutoff=4.0)
                        st.session_state["_plip_parent_feats"] = feats
                        st.session_state["_plip_df"] = pd.DataFrame(
                            [{"type":"CONTACT","residue":f,"distance_A":None,"is_key":False}
                             for f in feats])
                        st.session_state["_plip_tag_counts"] = {}
                        st.session_state["_plip_rec"] = None
                        st.success(f"Distance cIFP: {len(feats)} contacts detected")
                    else:
                        # Build tag counts + recommendation
                        tag_counts, res_codes = _pa.plip_to_residue_tags(
                            plip_df, core.AA_TAGS)
                        st.session_state["_plip_df"]         = plip_df
                        st.session_state["_plip_tag_counts"] = tag_counts
                        st.session_state["_plip_parent_feats"] = [
                            f"{r['type']}:{r['chain']}:{r['resname']}:{r['resnum']}"
                            for _, r in plip_df.iterrows()
                        ]
                        # Store for pocket_reference ML scoring
                        st.session_state["_pocket_tag_counts"] = tag_counts
                        st.success(
                            f"PLIP: {len(plip_df)} interactions · "
                            f"{plip_df['is_key'].sum()} key contacts"
                        )
                        st.rerun()

        with col_plip_status:
            plip_df = st.session_state.get("_plip_df")
            if plip_df is not None and not plip_df.empty and "type" in plip_df.columns:
                from collections import Counter
                type_counts = Counter(plip_df["type"].tolist())
                badges = " &nbsp; ".join(
                    f'<span style="background:#E8A020;color:#fff;padding:2px 8px;'
                    f'border-radius:10px;font-size:0.75rem;">{t}: {c}</span>'
                    for t, c in sorted(type_counts.items())
                )
                st.markdown(badges, unsafe_allow_html=True)

        # Show PLIP results
        plip_df = st.session_state.get("_plip_df")
        if plip_df is not None and not plip_df.empty:
            tab_int, tab_rec = st.tabs(["📋 Interaction table", "💡 Design recommendation"])

            with tab_int:
                def _style_plip(val):
                    colors = {
                        "HBOND":"#E6F4EA","SALTBRIDGE":"#FDECEA","PISTACK":"#EDE7F6",
                        "HYDROPHOBIC":"#FFF3CD","PICATION":"#E3F2FD",
                        "HALOGEN":"#FBE9E7","METAL":"#FCE4EC","WATERBRIDGE":"#E0F7FA",
                        "CONTACT":"#F3E5F5",
                    }
                    return f"background-color:{colors.get(val,'#FAFAFA')}"

                show_cols = [c for c in ["type","residue","distance_A","is_key"] if c in plip_df.columns]
                styled = plip_df[show_cols].style.map(
                    _style_plip, subset=["type"] if "type" in show_cols else [])
                st.dataframe(styled, use_container_width=True, hide_index=True)

            with tab_rec:
                rec = st.session_state.get("_plip_rec")
                tag_counts = st.session_state.get("_plip_tag_counts", {})

                if tag_counts and _VA_OK:
                    # Build recommendation using void subpockets
                    void_sps = st.session_state.get("_void_subpockets", [])
                    rec = _pa.unified_recommendation(
                        plip_df, void_sps, tag_counts, core.AA_TAGS)
                    st.session_state["_plip_rec"] = rec

                    st.markdown(f"**{rec['summary']}**")
                    st.markdown("")

                    if rec["preserve"]:
                        st.markdown("#### 🔒 Interactions to preserve")
                        for p in rec["preserve"]:
                            st.markdown(
                                f"- **{p['residue']}** `{p['type']}` — {p['strategy']}"
                            )

                    if rec["grow"]:
                        st.markdown("#### 🌱 Growth vectors (unoccupied space)")
                        for g in rec["grow"]:
                            st.markdown(
                                f"- **Sub-pocket {g['sub_pocket_id']}**"
                                f" [{g['size_class']}, {g['void_volume_A3']:.0f} ų] — "
                                f"{g['strategy']}\n\nExample fragments: *{g['example_frags']}*"
                            )
                    elif not void_sps:
                        st.info(
                            "Run **Unoccupied space analysis** below to see growth vectors."
                        )
                elif tag_counts:
                    st.info("Run **Unoccupied space analysis** below to get full recommendations.")
                else:
                    st.info("Run PLIP analysis above to get design recommendations.")

    # ── 3D Complex Viewer (parent ligand in pocket) ─────────────────────
    if st.session_state.complex_path and os.path.exists(str(st.session_state.complex_path)):
        st.divider()
        st.markdown("### 🧬 Binding Pocket 3D View")
        hint("Parent ligand (cyan) in the protein pocket. Orange = interacting residues. Magenta = reference ligand (if available).")

        v3d_s_col1, v3d_s_col2 = st.columns([1, 3])
        with v3d_s_col1:
            s3_labels  = st.checkbox("Residue labels",  value=True,  key="s3_labels")
            s3_surface = st.checkbox("Protein surface", value=False, key="s3_surface")
            s3_cutoff  = st.slider("Pocket cutoff (Å)", 3.0, 6.0, 4.0, 0.5, key="s3_cutoff")
            s3_height  = st.slider("Height (px)", 300, 600, 460, 50, key="s3_height")
            st.markdown("""
<div style="font-size:0.75rem;line-height:1.8;margin-top:6px;">
<span style="color:#00FFFF">■</span> Ligand &nbsp;
<span style="color:#FF8C00">■</span> Pocket residues<br>
<span style="color:#FF00FF">■</span> Reference &nbsp;
<span style="color:#888">■</span> Protein
</div>""", unsafe_allow_html=True)

        with v3d_s_col2:
            # Load ref ligand mol from PDB if available
            ref_pdb = st.session_state.ref_ligand_path or ""
            pose_mol_s3 = None
            if ref_pdb and os.path.exists(ref_pdb):
                try:
                    from rdkit import Chem as _Chem_s3
                    pose_mol_s3 = _Chem_s3.MolFromPDBFile(ref_pdb, removeHs=False)
                except Exception:
                    pass

            render_complex_3d(
                receptor_pdb   = st.session_state.protein_path or st.session_state.receptor_path or "",
                pose_mol       = pose_mol_s3,
                height         = s3_height,
                cutoff         = s3_cutoff,
                show_labels    = s3_labels,
                show_surface   = s3_surface,
                ref_ligand_pdb = "",
                key_prefix     = "s3_viewer",
            )

    # ── Void / unoccupied space analysis ─────────────────────────────────
    if _VA_OK and st.session_state.complex_path and os.path.exists(str(st.session_state.complex_path)):
        st.divider()
        st.markdown("### 📐 Unoccupied pocket space analysis")
        info_card(
            "Detects sub-pockets <strong>not filled by the current ligand</strong> "
            "and recommends fragment sizes that fit the available space. "
            "Use this to decide which attachment vector to grow into."
        )

        va_cutoff = st.slider("Pocket cutoff (Å)", 4.0, 10.0, 6.0, 0.5, key="va_cutoff")
        va_cluster = st.slider("Sub-pocket cluster distance (Å)", 3.0, 10.0, 6.0, 0.5, key="va_cluster")

        if st.button("Analyse unoccupied space", type="primary", key="va_btn"):
            with st.spinner("Detecting void sub-pockets…"):
                try:
                    from core import read_complex_atoms_for_pocket, analyze_complex_distance_shell
                    prot_atoms, lig_atoms = read_complex_atoms_for_pocket(
                        st.session_state.complex_path)
                    p_df, c_df, g_df, lig_raw = analyze_complex_distance_shell(
                        st.session_state.complex_path,
                        pocket_cutoff=va_cutoff,
                        contact_cutoff=4.0,
                    )
                    subpockets, va_mode = _va.analyze_void(
                        prot_atoms, lig_atoms if lig_atoms else None,
                        p_df, g_df,
                        pocket_cutoff=va_cutoff,
                        cluster_dist=va_cluster,
                    )
                    st.session_state["_void_subpockets"] = subpockets
                    st.session_state["_void_mode"] = va_mode
                    # Store all pocket residue resnums for viewer grey-out
                    try:
                        _all_resnums = [
                            row.get("resnum", "")
                            for _, row in p_df.iterrows()
                        ]
                        st.session_state["_void_all_pocket_resnums"] = [r for r in _all_resnums if r]
                    except Exception:
                        st.session_state["_void_all_pocket_resnums"] = []
                    st.success(
                        f"Mode: **{va_mode}** — "
                        f"found **{len(subpockets)}** unoccupied sub-pocket(s)"
                    )
                except Exception as e:
                    st.error(f"Void analysis failed: {e}")

        # Show results if available
        subpockets = st.session_state.get("_void_subpockets", [])
        va_mode    = st.session_state.get("_void_mode", "")
        if subpockets:
            # Summary text
            st.markdown(_va.void_summary_text(subpockets, va_mode))

            # Detailed table
            sp_df = _va.subpockets_to_df(subpockets)
            st.dataframe(sp_df, use_container_width=True, hide_index=True)

            # Per sub-pocket fragment size filter
            st.markdown("#### Apply size filter to fragment generation")
            sp_options = ["All sizes (no filter)"] + [
                f"Sub-pocket {sp['sub_pocket_id']}: {sp['size_class']} "
                f"(r≤{sp['available_radius_A']:.1f}Å, {sp['void_volume_est_A3']:.0f}ų)"
                for sp in subpockets
            ]
            chosen_sp = st.selectbox(
                "Restrict fragments to fit selected sub-pocket",
                sp_options, index=0, key="va_sp_select",
            )
            if chosen_sp != "All sizes (no filter)":
                sp_idx = int(chosen_sp.split(":")[0].replace("Sub-pocket","").strip()) - 1
                if 0 <= sp_idx < len(subpockets):
                    sel_sp = subpockets[sp_idx]
                    st.session_state["_void_size_filter"] = sel_sp["size_class"]
                    hint(
                        f"Fragments larger than **{sel_sp['size_class']}** will be excluded. "
                        f"Examples that fit: {_va.SIZE_EXAMPLES.get(sel_sp['size_class'], '')}"
                    )
            else:
                st.session_state["_void_size_filter"] = None

            # ── 3D Pocket Viewer (void sub-pockets) ──────────────────────
            if subpockets:
                st.markdown("#### 🧬 Pocket 3D view — sub-pocket map")
                hint(
                    "Protein pocket coloured by sub-pocket. "
                    "Each colour represents a distinct unoccupied region detected by void analysis."
                )

                # Sub-pocket colour palette
                _SP_COLOURS = [
                    "orangeCarbon",   # sub-pocket 1
                    "cyanCarbon",     # sub-pocket 2
                    "magentaCarbon",  # sub-pocket 3
                    "greenCarbon",    # sub-pocket 4
                    "yellowCarbon",   # sub-pocket 5
                ]
                _SP_HEX = ["#FF8C00", "#00FFFF", "#FF00FF", "#00FF44", "#FFD700"]

                _va_v3d_col1, _va_v3d_col2 = st.columns([1, 3])
                with _va_v3d_col1:
                    _va_labels  = st.checkbox("Residue labels", value=True, key="va3d_labels")
                    _va_surface = st.checkbox("Protein surface", value=False, key="va3d_surface")
                    _va_cutoff  = st.slider("Cutoff (Å)", 3.0, 8.0, 6.0, 0.5, key="va3d_cutoff")
                    _va_height  = st.slider("Height (px)", 300, 600, 440, 50, key="va3d_height")

                    # Legend
                    st.markdown("**Sub-pocket legend:**")
                    for sp in subpockets[:5]:
                        sp_id  = sp["sub_pocket_id"] - 1
                        colour = _SP_HEX[sp_id % len(_SP_HEX)]
                        sc     = sp["size_class"]
                        vol    = sp["void_volume_est_A3"]
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:8px;'
                            f'font-size:0.78rem;margin-bottom:3px;">'
                            f'<div style="width:12px;height:12px;border-radius:3px;'
                            f'background:{colour};flex-shrink:0"></div>'
                            f'SP{sp["sub_pocket_id"]} [{sc}, {vol:.0f}ų]</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        '<div style="font-size:0.72rem;color:#888;margin-top:6px;">'
                        '<span style="color:#AAAAAA">■</span> Protein &nbsp; '
                        '<span style="color:#888">·</span> All pocket residues</div>',
                        unsafe_allow_html=True,
                    )

                with _va_v3d_col2:
                    try:
                        import py3Dmol as _p3d
                        _va_view = _p3d.view(width="100%", height=_va_height)
                        _va_view.setBackgroundColor(_viewer_bg())

                        # Protein cartoon
                        rec_pdb = (st.session_state.protein_path
                                   or st.session_state.receptor_path or "")
                        if rec_pdb and os.path.exists(rec_pdb):
                            _va_view.addModel(open(rec_pdb).read(), "pdb")
                            _va_view.setStyle({"model": 0}, {
                                "cartoon": {"color": "lightgray", "opacity": 0.30}
                            })
                            if _va_surface:
                                _va_view.addSurface(
                                    _p3d.SAS,
                                    {"opacity": 0.18, "color": "white"},
                                    {"model": 0}
                                )

                        # Co-crystal / docked ligand (cyan)
                        ref_pdb = st.session_state.ref_ligand_path or ""
                        if ref_pdb and os.path.exists(ref_pdb):
                            _va_view.addModel(open(ref_pdb).read(), "pdb")
                            _va_view.setStyle({"model": 1}, {
                                "stick": {"colorscheme": "cyanCarbon", "radius": 0.30}
                            })

                        # Colour each sub-pocket's residues differently
                        _zoom_sel = []
                        for sp in subpockets[:5]:
                            sp_id    = sp["sub_pocket_id"] - 1
                            _colour  = _SP_COLOURS[sp_id % len(_SP_COLOURS)]
                            _hex     = _SP_HEX[sp_id % len(_SP_HEX)]
                            for member in sp.get("members", []):
                                _rnum  = member["resnum"]
                                _chain = member.get("chain", "")
                                _rname = member.get("resname", "")
                                _sel   = {"resi": _rnum}
                                if _chain and _chain.strip():
                                    _sel["chain"] = _chain
                                _va_view.setStyle(_sel, {
                                    "stick": {"colorscheme": _colour, "radius": 0.22}
                                })
                                if _va_labels:
                                    _va_view.addLabel(
                                        f"{_rname}{_rnum}",
                                        {
                                            "fontSize": 10,
                                            "fontColor": _hex,
                                            "backgroundColor": "black",
                                            "backgroundOpacity": 0.55,
                                            "inFront": True,
                                        },
                                        _sel,
                                    )
                                _zoom_sel.append(_rnum)

                        # All other pocket residues (not in any sub-pocket)
                        _sp_residues = {
                            m["resnum"]
                            for sp in subpockets
                            for m in sp.get("members", [])
                        }
                        _all_pocket = st.session_state.get("_void_all_pocket_resnums", [])
                        for _rnum in _all_pocket:
                            if _rnum not in _sp_residues:
                                _va_view.setStyle(
                                    {"resi": _rnum},
                                    {"stick": {"color": "#555555", "radius": 0.12}}
                                )

                        # Zoom to pocket
                        if _zoom_sel:
                            _va_view.zoomTo({"resi": _zoom_sel})
                        else:
                            _va_view.zoomTo()

                        show3d(_va_view, height=_va_height)

                    except ImportError:
                        st.info("Install `py3Dmol` to enable 3D pocket viewer.")
                    except Exception as _va_e:
                        st.warning(f"3D pocket viewer error: {_va_e}")

    st.divider()
    st.markdown("### Generation settings")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.n_analogs = st.slider(
            "Number of analogs", 5, 1000, st.session_state.n_analogs, step=5)
        _tier_preview = analog_tier(st.session_state.n_analogs)
        st.markdown(
            f'<p style="font-size:0.82rem;margin-top:-4px;">'
            f'With <strong>{st.session_state.n_analogs}</strong> analogs: '
            f'{tier_badge_html(_tier_preview)}</p>',
            unsafe_allow_html=True,
        )
    with c2:
        risk_map = {
            "Conservative":          "Conservative",
            "Moderate (recommended)":"Moderate",
            "Exploratory":           "Exploratory",
        }
        r = st.radio("Substitution size", list(risk_map.keys()),
                     index=list(risk_map.values()).index(st.session_state.risk), horizontal=True)
        st.session_state.risk = risk_map[r]

    with st.expander("⚙️ Advanced options"):
        st.session_state.max_MW      = st.number_input("Max MW (Da)", value=st.session_state.max_MW, step=25.0)
        st.session_state.avoid_nitro = st.checkbox("Remove nitro groups", value=st.session_state.avoid_nitro)
        st.session_state.avoid_toxic = st.checkbox("Remove toxic flags",  value=st.session_state.avoid_toxic)
        st.session_state.max_pocket_frags = st.slider("Max pocket-guided fragments", 3, 20,
                                                       st.session_state.max_pocket_frags)

    st.write("")
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            go(2)
    with col_next:
        if st.button("Generate analogs →", type="primary"):
            go(4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – View results
# ─────────────────────────────────────────────────────────────────────────────

elif step == 4:
    mol = st.session_state.parent_mol

    size_cap = {"Conservative": 4, "Moderate": 8, "Exploratory": 14}[st.session_state.risk]
    active_lib = [
        f for f in core.LIBRARY
        if st.session_state.categories_on.get(f.category, True) and f.heavy <= size_cap
    ]
    # Apply void size filter if user selected a sub-pocket
    _void_size_filter = st.session_state.get("_void_size_filter")
    if _void_size_filter and _VA_OK:
        active_lib = _va.filter_frags_by_size(active_lib, _void_size_filter, allow_smaller=True)
        st.info(
            f"📐 Void filter active: showing fragments ≤ **{_void_size_filter}** size "
            f"({len(active_lib):,} fragments match). "
            f"Change in Step 3 → Unoccupied space analysis."
        )
    pocket_frags = st.session_state.pocket_frags or []
    if mode == "structure" and pocket_frags and st.session_state.accept_pocket_suggestions:
        chosen = pocket_frags + [f for f in active_lib if f.name not in {g.name for g in pocket_frags}]
    else:
        chosen = active_lib

    # ── Custom fragments: bypass all rules, generate directly ───────────────
    _custom_text = st.session_state.custom_frags_text.strip()
    _custom_frags = []
    for smi in _custom_text.splitlines():
        smi = smi.strip()
        if not smi:
            continue
        ok, _ = core.validate_fragment_smiles(smi)
        if ok:
            _custom_frags.append(core.Frag(f"custom_{len(_custom_frags)}", smi, "custom", core.G()))

    _custom_only = bool(_custom_frags)   # True → ignore library, skip all filters
    if _custom_only:
        chosen = _custom_frags
    else:
        chosen.extend(_custom_frags)

    selected = st.session_state.selected_atoms
    allow_het = st.session_state.allow_heteroatom_H
    valid_sites = [
        s for s in sorted(selected)
        if mol.GetAtomWithIdx(s).GetTotalNumHs() > 0
        and (allow_het or mol.GetAtomWithIdx(s).GetAtomicNum() == 6)
    ]

    if not valid_sites:
        st.error("No valid attachment sites. Go back to Step 2 and select atoms.")
        if st.button("← Back to atom selection"):
            go(2)
        st.stop()

    site_groups = [tuple(valid_sites)] if (st.session_state.concerted and len(valid_sites) > 1) \
                  else [(s,) for s in valid_sites]

    tot = sum(st.session_state.weights.values()) or 1
    weights = {k: v / tot for k, v in st.session_state.weights.items()}

    avoid_opts = {
        "nitro":             st.session_state.avoid_nitro,
        "aldehyde":          st.session_state.avoid_aldehyde,
        "reactive_acylhalide": st.session_state.avoid_reactive,
        "azide":             st.session_state.avoid_toxic,
        "michael_acceptor":  st.session_state.avoid_reactive,
        "epoxide":           st.session_state.avoid_toxic,
    }

    if st.session_state.analogs_df is None:
        if _custom_only:
            _spinner_msg = f"Generating analogs from {len(chosen)} custom fragment(s) — all rules bypassed…"
        else:
            _spinner_msg = f"Generating up to {st.session_state.n_analogs} analogs from {len(chosen):,} fragments…"
        with st.spinner(_spinner_msg):
            df = core.generate_analogs(
                mol,
                selected_atoms=list(selected),
                chosen_frags=chosen,
                site_groups=site_groups,
                weights=({k: 1.0 for k in weights} if _custom_only else weights),
                avoid_opts=({k: False for k in avoid_opts} if _custom_only else avoid_opts),
                max_MW=(9999 if _custom_only else st.session_state.max_MW),
                max_analogs=(999999 if _custom_only else st.session_state.n_analogs),
                rank_by=("none" if _custom_only else st.session_state.get("rank_code", "Balanced (100-pt weights)")),
            )
        st.session_state.analogs_df = df
        if _custom_only and df is not None and not df.empty:
            st.info(
                f"✅ Generated **{len(df)} analogs** from {len(chosen)} custom fragment(s). "
                "All property filters and goal weights were bypassed."
            )

    df = st.session_state.analogs_df

    if df is None or df.empty:
        st.error("No analogs were generated. Try relaxing your filters in Step 3.")
        if st.button("← Back to settings"):
            go(3)
        st.stop()

    n_analogs = len(df)
    tier = analog_tier(n_analogs)

    # ── Tier info banner ────────────────────────────────────────────────────
    tier_msgs = {
        "docking":  f"✅ <strong>{n_analogs} analogs generated</strong> — Full docking, pKaNET protonation, and SMI download available. (Local — no limits)",
        "pkanet":   f"✅ <strong>{n_analogs} analogs generated</strong> — Full docking available (Local mode — no tier restrictions).",
        "smi_only": f"✅ <strong>{n_analogs} analogs generated</strong> — Full docking available (Local mode — no tier restrictions).",
    }
    info_card(tier_msgs[tier])

    # ── Metrics row ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box"><div class="val">{n_analogs}</div><div class="lbl">Analogs generated</div></div>
      <div class="metric-box"><div class="val">{df.MW.median():.0f}</div><div class="lbl">Median MW (Da)</div></div>
      <div class="metric-box"><div class="val">{df.QED.median():.2f}</div><div class="lbl">Median QED</div></div>
      <div class="metric-box"><div class="val">{df.sim.median():.2f}</div><div class="lbl">Median similarity</div></div>
      <div class="metric-box"><div class="val">{df.fragment_category.value_counts().index[0]}</div><div class="lbl">Top category</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Filter bar ───────────────────────────────────────────────────────────
    with st.expander("🔍 Filter results"):
        fc1, fc2, fc3 = st.columns(3)
        mw_max  = fc1.slider("Max MW", 200.0, 900.0, float(df.MW.max()), 10.0)
        qed_min = fc2.slider("Min QED", 0.0, 1.0, 0.0, 0.05)
        cats_f  = fc3.multiselect("Category", sorted(df.fragment_category.unique()),
                                   default=sorted(df.fragment_category.unique()))

    df_show = df[(df.MW <= mw_max) & (df.QED >= qed_min) & (df.fragment_category.isin(cats_f))]

    # ── Tabs: table / grid ───────────────────────────────────────────────────
    tab_tbl, tab_grid = st.tabs(["📋 Table", "🖼️ Structure grid"])

    with tab_tbl:
        cols_show = ["change", "fragment_category", "MW", "logP", "QED", "ESOL", "SA", "sim", "smiles"]
        st.dataframe(df_show[[c for c in cols_show if c in df_show.columns]],
                     use_container_width=True, height=380, hide_index=True)

    with tab_grid:
        PAGE_SIZE = 50
        total_show = len(df_show)
        if total_show == 0:
            st.info("No analogs match the current filters.")
        else:
            n_pages = max(1, (total_show + PAGE_SIZE - 1) // PAGE_SIZE)
            page_num = 1   # default

            if n_pages > 1:
                pg_col, info_col = st.columns([2, 3])
                with pg_col:
                    page_num = st.number_input(
                        "Page", min_value=1, max_value=n_pages,
                        value=1, step=1, key="grid_page_input",
                    )
                with info_col:
                    start_idx = (int(page_num) - 1) * PAGE_SIZE
                    end_idx   = min(start_idx + PAGE_SIZE, total_show)
                    st.markdown(
                        f'<p style="font-size:0.82rem;color:#8B7355;padding-top:0.45rem;">'
                        f'Page {int(page_num)} of {n_pages} &nbsp;·&nbsp; '
                        f'showing compounds {start_idx+1}–{end_idx} of {total_show}</p>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(f"{total_show} compounds")

            start_idx = (int(page_num) - 1) * PAGE_SIZE
            end_idx   = min(start_idx + PAGE_SIZE, total_show)
            df_page = df_show.iloc[start_idx:end_idx]

            pairs = []
            for local_i, (abs_i, row) in enumerate(df_page.iterrows()):
                m = Chem.MolFromSmiles(str(row["smiles"]))
                if m is not None:
                    try:
                        AllChem.Compute2DCoords(m)
                        global_n = start_idx + local_i + 1
                        pairs.append((m, f"{global_n}. {row['change']}"))
                    except Exception:
                        pass

            if not pairs:
                st.info("Could not render any structures on this page.")
            else:
                try:
                    png = Draw.MolsToGridImage(
                        [p[0] for p in pairs], legends=[p[1] for p in pairs],
                        molsPerRow=5, subImgSize=(240, 190), returnPNG=True,
                    )
                    st.image(png, use_container_width=True)
                except Exception as e:
                    st.warning(f"Structure grid could not be rendered ({e}). See the Table tab.")

    # ── Navigation — TIER LOGIC ──────────────────────────────────────────────
    st.write("")
    st.divider()

    parent_name = st.session_state.parent_name or "compound"
    smi_lines = "\n".join(f"{r.smiles}\t{parent_name}_A{i+1}" for i, r in df.iterrows())

    col_back, col_regen, *col_actions = st.columns([1, 1, 2, 2, 2])

    with col_back:
        if st.button("← Back"):
            go(3)
    with col_regen:
        if st.button("↺ Regenerate"):
            st.session_state.analogs_df = None
            st.rerun()

    # Tier-dependent action buttons
    if tier == "docking":
        # Full tier: docking + download SMI
        with col_actions[0]:
            next_label = "Docking & cIFP →" if mode == "structure" else "Docking →"
            if st.button(next_label, type="primary", use_container_width=True):
                go(5)
        with col_actions[1]:
            st.download_button(
                "⬇️ Download SMI",
                data=smi_lines.encode(),
                file_name=f"{parent_name}_analogs.smi",
                mime="text/plain",
                use_container_width=True,
            )
        with col_actions[2]:
            if st.button("Export →", use_container_width=True):
                go(len(steps))

    elif tier in ("pkanet", "smi_only"):
        # LOCAL: these tiers don't exist — always show full docking buttons
        with col_actions[0]:
            next_label = "Docking & cIFP →" if mode == "structure" else "Docking →"
            if st.button(next_label, type="primary", use_container_width=True):
                go(5)
        with col_actions[1]:
            st.download_button(
                "⬇️ Download SMI",
                data=smi_lines.encode(),
                file_name=f"{parent_name}_analogs.smi",
                mime="text/plain",
                use_container_width=True,
            )
        with col_actions[2]:
            if st.button("Export →", use_container_width=True):
                go(len(steps))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Docking & cIFP
# ─────────────────────────────────────────────────────────────────────────────

elif step == 5:
    df_analogs = st.session_state.analogs_df
    if df_analogs is None or df_analogs.empty:
        st.warning("No analogs yet. Go back to Step 4.")
        st.stop()

    # LOCAL: no tier gate — docking always available
    n_a = len(df_analogs)
    if n_a > 50:
        st.info(
            f"ℹ️ **{n_a} analogs** queued for docking. "
            "Local mode has no limits — consider batching large sets to manage runtime."
        )
        with col_d:
            parent_name = st.session_state.parent_name or "compound"
            smi_lines = "\n".join(f"{r.smiles}\t{parent_name}_A{i+1}" for i, r in df_analogs.iterrows())
            st.download_button("⬇️ Download SMI", data=smi_lines.encode(),
                               file_name=f"{parent_name}_analogs.smi", mime="text/plain")
        st.stop()

    acd_ok    = bool(shutil.which("acd"))
    obabel_ok = bool(shutil.which("obabel"))
    work      = get_work_dir()
    dock_in   = work / "docking_inputs"
    dock_in.mkdir(parents=True, exist_ok=True)

    info_card("Docking predicts how tightly each designed analog binds to the target protein. "
              "Requires <strong>Anyone Can Dock</strong> (acd) and <strong>OpenBabel</strong>.")

    bcol1, bcol2 = st.columns(2)
    bcol1.markdown(
        f'<div style="padding:0.6rem 1rem;border-radius:8px;'
        f'background:{"#E6F4EA" if acd_ok else "#FDECEA"};'
        f'color:{"#1E7E34" if acd_ok else "#B00020"};font-size:0.85rem;">'
        f'{"✅ acd available" if acd_ok else "❌ acd not found — pip install anyonecandock"}</div>',
        unsafe_allow_html=True)
    bcol2.markdown(
        f'<div style="padding:0.6rem 1rem;border-radius:8px;'
        f'background:{"#E6F4EA" if obabel_ok else "#FDECEA"};'
        f'color:{"#1E7E34" if obabel_ok else "#B00020"};font-size:0.85rem;">'
        f'{"✅ obabel available" if obabel_ok else "❌ obabel not found — apt install openbabel"}</div>',
        unsafe_allow_html=True)

    # Receptor loader (ligand track)
    if mode == "ligand" and not st.session_state.receptor_path:
        st.divider()
        st.markdown("### Choose a target protein")
        info_card("You designed analogs without a structure. To dock them, pick the protein they bind to.")
        rec_src = st.radio("Load receptor from",
            ["🔍 Search RCSB", "#️⃣ PDB ID", "📁 Upload file"],
            horizontal=True, key="dock_rec_src")
        if rec_src == "🔍 Search RCSB":
            dq = st.text_input("Search RCSB PDB", placeholder="e.g. EGFR kinase, JAK2", key="dock_rcsb_q")
            if st.button("Search", key="dock_rcsb_btn") and dq.strip():
                with st.spinner("Searching RCSB PDB…"):
                    st.session_state["_dock_rcsb_results"] = core.search_rcsb(dq.strip(), max_results=6)
            for r in st.session_state.get("_dock_rcsb_results", []):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.caption(f"**{r['id']}** — {r['title']}  ({r['resolution']} · {r['organism']})")
                with c2:
                    if st.button("Use", key=f"dock_rcsb_{r['id']}"):
                        with st.spinner(f"Downloading {r['id']}…"):
                            try:
                                path = core.download_pdb(r["id"], work)
                                prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                                st.session_state.receptor_path   = path
                                st.session_state.protein_path    = prot
                                st.session_state.complex_path    = path
                                st.session_state.ref_ligand_path = lig
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not download: {e}")
        elif rec_src == "#️⃣ PDB ID":
            pdb_id = st.text_input("4-letter PDB ID", value="", max_chars=4,
                                   placeholder="e.g. 1M17", key="dock_pdb_id")
            if st.button("Load receptor", key="dock_load_rec") and pdb_id.strip():
                with st.spinner("Downloading from RCSB…"):
                    try:
                        path = core.download_pdb(pdb_id.strip().upper(), work)
                        prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                        st.session_state.receptor_path   = path
                        st.session_state.protein_path    = prot
                        st.session_state.complex_path    = path
                        st.session_state.ref_ligand_path = lig
                        st.success(f"Receptor loaded ({pdb_id.upper()}) ✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not download: {e}")
        else:
            up = st.file_uploader("Upload .pdb or .cif file", type=["pdb", "cif"], key="dock_upload")
            if up:
                raw = work / up.name
                raw.write_bytes(up.read())
                try:
                    path = core.cif_to_pdb_if_needed(str(raw))
                    prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                    st.session_state.receptor_path   = path
                    st.session_state.protein_path    = prot
                    st.session_state.complex_path    = path
                    st.session_state.ref_ligand_path = lig
                    st.success("Receptor uploaded ✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not process file: {e}")

    receptor = st.session_state.receptor_path
    if receptor:
        st.success(f"Target receptor: `{Path(receptor).name}`")

    st.divider()
    rows = [{"compound": "original_ligand", "smiles": st.session_state.parent_smiles}]
    for i, r in df_analogs.iterrows():
        rows.append({"compound": f"{st.session_state.parent_name}_A{i+1}", "smiles": r.smiles})
    lig_df = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
    st.session_state.docking_ligands = lig_df

    smi_path = dock_in / "compounds.smi"
    with open(smi_path, "w") as fh:
        for _, r in lig_df.iterrows():
            fh.write(f"{r.smiles}\t{r.compound}\n")

    with st.expander(f"Ligand list — {len(lig_df)} compounds (parent + analogs)"):
        st.dataframe(lig_df, use_container_width=True, hide_index=True, height=200)

    st.markdown("### Docking settings")
    d1, d2 = st.columns(2)
    with d1:
        exhaustiveness = st.slider("Exhaustiveness", 1, 32, 8)
        hint("Higher = more thorough but slower.")
        num_poses = st.slider("Poses per compound", 1, 20, 10)
    with d2:
        dock_ph = st.number_input("pH", value=7.4, step=0.1)
        dock_out_dir = str(work / "docking_out")

    with st.expander("⚙️ Advanced docking options"):
        bx = st.number_input("Box X (Å)", value=16.0)
        by = st.number_input("Box Y (Å)", value=16.0)
        bz = st.number_input("Box Z (Å)", value=16.0)

    if acd_ok and obabel_ok and receptor:
        if st.button("Run docking", type="primary"):
            n = len(lig_df)
            progress = st.progress(0.0, text=f"Preparing to dock {n} compounds…")
            status = st.empty()
            live_table = st.empty()
            dock_results = []
            any_fail = False
            full_log = []

            ref_mol = None
            ref_pdb = st.session_state.ref_ligand_path
            if ref_pdb and os.path.exists(str(ref_pdb)):
                try:
                    ref_mol = Chem.MolFromPDBFile(str(ref_pdb), removeHs=True)
                except Exception:
                    pass

            for i, row in lig_df.iterrows():
                compound = str(row["compound"])
                smi = str(row["smiles"])
                progress.progress(i / n, text=f"Docking {i+1} of {n}:  {compound}")
                status.markdown(
                    f'<div style="font-size:0.85rem;color:#8B7355;">'
                    f'⏳ Running AutoDock Vina on <strong>{compound}</strong>…</div>',
                    unsafe_allow_html=True)
                cmd = core.build_acd_dock_cmd(
                    receptor=receptor, smiles=smi, name=compound,
                    ph=dock_ph, output_dir=dock_out_dir, save_poses=True)
                rc, output = core.run_command(cmd)
                full_log.append(f"=== {compound} (exit {rc}) ===\n{output}\n")
                if rc != 0:
                    any_fail = True
                summary = core.summarize_docking_for_compound(
                    out_dir=dock_out_dir, compound=compound, smiles=smi,
                    ref_mol=ref_mol, ref_pdb_path=ref_pdb)
                summary["dock_status"] = "✅" if rc == 0 else "❌"
                dock_results.append(summary)
                preview_df = pd.DataFrame([{
                    "compound": r["compound"], "status": r["dock_status"],
                    "top_BE": r.get("top_BE", "—"), "top_RMSD": r.get("top_RMSD", "—"),
                } for r in dock_results])
                live_table.dataframe(preview_df, use_container_width=True, hide_index=True)

            progress.progress(1.0, text=f"Finished docking {n} compounds")
            status.empty()
            (work / "acd_batch.log").write_text("\n".join(full_log))

            if any_fail:
                st.warning("Docking finished, but some compounds failed.")
            else:
                st.success(f"Docking finished — all {n} compounds ✅")

            st.divider()
            st.markdown("### Docking results")
            if ref_mol:
                st.info("RMSD computed against the co-crystal ligand pose.")

            result_rows = []
            for r in dock_results:
                result_rows.append({
                    "compound": r["compound"], "SMILES": r["smiles"],
                    "status": r["dock_status"], "n_poses": r.get("n_poses", 0),
                    "top_BE (kcal/mol)": r.get("top_BE"),
                    "top_RMSD (Å)": r.get("top_RMSD"),
                    "minRMSD_BE (kcal/mol)": r.get("minRMSD_BE"),
                    "minRMSD (Å)": r.get("minRMSD_RMSD"),
                })
            # ── Build per-compound per-pose RMSD vs co-crystal ─────────────
            cryst_pdb = st.session_state.ref_ligand_path or ""
            _has_crystal = bool(cryst_pdb and os.path.exists(cryst_pdb))
            # Store co-crystal reference BE for plot (from Mode A redocking or manual entry)
            if "_ref_be_for_plot" not in st.session_state:
                st.session_state["_ref_be_for_plot"] = None

            # Compute pose-level RMSD and best scores per compound
            for r in dock_results:
                cmpd_name = r["compound"]
                dock_dir  = dock_out_dir
                # Try to find all pose SDFs for this compound
                pose_sdfs = core.find_pose_sdfs_for_compound(dock_dir, cmpd_name) if hasattr(core, "find_pose_sdfs_for_compound") else []
                pose_rows = []
                if pose_sdfs:
                    try:
                        from rdkit.Chem import SDMolSupplier
                        suppl = SDMolSupplier(pose_sdfs[0], removeHs=False)
                        scores_list = r.get("scores", [])
                        for i, mol in enumerate(suppl):
                            if mol is None:
                                continue
                            pose_num = i + 1
                            be = None
                            if scores_list and i < len(scores_list):
                                be = scores_list[i].get("affinity") or scores_list[i].get("score")
                            rmsd_val = None
                            if _has_crystal:
                                rmsd_val = core.calc_rmsd_heavy(mol, cryst_pdb)
                            pose_rows.append({
                                "Pose": pose_num,
                                "Affinity (kcal/mol)": round(float(be), 2) if be is not None else None,
                                "RMSD vs co-crystal (Å)": round(rmsd_val, 2) if rmsd_val is not None else "—",
                            })
                    except Exception:
                        pass
                r["_pose_rows"] = pose_rows

            # ── Summary table (one row per compound) ────────────────────────
            results_df = pd.DataFrame(result_rows)
            if "top_BE (kcal/mol)" in results_df.columns:
                results_df = results_df.sort_values("top_BE (kcal/mol)", na_position="last").reset_index(drop=True)

            st.markdown("#### Docking summary")
            hint("Sorted by best binding energy (most negative = strongest predicted binding).")

            # Colour coding: BE gradient + RMSD flag
            def _colour_be(val):
                try:
                    v = float(val)
                    if v < -10:   return "background-color:#E6F4EA;color:#1E7E34"
                    elif v < -8:  return "background-color:#FFF3CD;color:#856404"
                    elif v < -6:  return "background-color:#FDE8E8;color:#842029"
                    else:         return "background-color:#F8D7DA;color:#721c24"
                except Exception:
                    return ""

            def _colour_rmsd(val):
                try:
                    v = float(val)
                    if v <= 2.0:  return "color:#1E7E34;font-weight:500"
                    elif v <= 3.0:return "color:#856404"
                    else:         return "color:#842029"
                except Exception:
                    return ""

            be_col   = "top_BE (kcal/mol)"
            rmsd_col = "top_RMSD (Å)"

            styled = results_df.style
            if be_col in results_df.columns:
                styled = styled.map(_colour_be, subset=[be_col])
            if rmsd_col in results_df.columns and _has_crystal:
                styled = styled.map(_colour_rmsd, subset=[rmsd_col])
            if be_col in results_df.columns:
                fmt = {be_col: lambda x: f"{x:.2f}" if x is not None and str(x) != "nan" else "—"}
                if rmsd_col in results_df.columns:
                    fmt[rmsd_col] = lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
                styled = styled.format(fmt, na_rep="—")

            st.dataframe(styled, use_container_width=True, hide_index=True)

            if not _has_crystal:
                st.caption("ℹ️ RMSD vs co-crystal not shown — no reference ligand available (Mode B or no co-crystal PDB).")

            # ── Batch score plot (ACD-style) ─────────────────────────────────
            st.markdown("#### Score plot")
            _plot_df = results_df.dropna(subset=["top_BE (kcal/mol)"]).copy()
            _plot_df = _plot_df.sort_values("top_BE (kcal/mol)").reset_index(drop=True)

            if not _plot_df.empty:
                import matplotlib.pyplot as _plt
                import io as _io

                _n    = len(_plot_df)
                _names  = _plot_df["compound"].tolist()
                _scores = _plot_df["top_BE (kcal/mol)"].tolist()
                _best   = min(_scores)

                # Colour: green = best, blue = others
                _dot_colors = ["#3fb950" if s == _best else "#58a6ff" for s in _scores]

                # RMSD colour ring: green ≤2Å, amber 2-3Å, red >3Å, gray = no crystal
                _rmsd_vals = []
                if _has_crystal and "top_RMSD (Å)" in _plot_df.columns:
                    for v in _plot_df["top_RMSD (Å)"]:
                        try:
                            _rmsd_vals.append(float(v))
                        except Exception:
                            _rmsd_vals.append(None)
                else:
                    _rmsd_vals = [None] * _n

                _rmsd_ring = []
                for rv in _rmsd_vals:
                    if rv is None:     _rmsd_ring.append("#888888")
                    elif rv <= 2.0:    _rmsd_ring.append("#3fb950")
                    elif rv <= 3.0:    _rmsd_ring.append("#d29922")
                    else:              _rmsd_ring.append("#f85149")

                # Reference line = co-crystal BE if redocking was done
                _ref_be = st.session_state.get("_ref_be_for_plot")

                # Dark/light aware colours
                _dark = False
                try:
                    _dark = st.get_option("theme.base") == "dark"
                except Exception:
                    pass

                _bg      = "#0d1117" if _dark else "#ffffff"
                _bg_sub  = "#161b22" if _dark else "#f6f8fa"
                _txt     = "#e6edf3" if _dark else "#1f2328"
                _muted   = "#8b949e" if _dark else "#6e7781"
                _border  = "#30363d" if _dark else "#d0d7de"
                _leg_bg  = "#21262d" if _dark else "#f6f8fa"

                _fig_w = max(5, _n * 0.7 + 1.8)
                _fig, _ax = _plt.subplots(figsize=(_fig_w, 3.8))
                _fig.patch.set_facecolor(_bg)
                _ax.set_facecolor(_bg_sub)

                _xs = list(range(_n))

                # Line connecting dots
                _ax.plot(_xs, _scores, color=_border, linewidth=0.8, zorder=2)

                # Dots with RMSD-coloured edge ring
                _ax.scatter(_xs, _scores,
                            color=_dot_colors,
                            edgecolors=_rmsd_ring,
                            linewidths=2.2,
                            s=100, zorder=3)

                # Reference dashed line
                if _ref_be is not None:
                    _ax.axhline(_ref_be, color="#f85149", linewidth=1.8,
                                linestyle="--",
                                label=f"Co-crystal ref: {_ref_be:.2f} kcal/mol")
                    _ax.legend(facecolor=_leg_bg, edgecolor=_border,
                               labelcolor=_txt, fontsize=8)

                _ax.set_xticks(_xs)
                _ax.set_xticklabels(_names, rotation=40, ha="right", fontsize=7)
                _ax.set_xlim(-0.5, _n - 0.5)
                _ax.set_ylabel("Vina score (kcal/mol)", color=_muted, fontsize=9)
                _ax.set_xlabel("Analog", color=_muted, fontsize=9)
                _ax.tick_params(colors=_muted, labelsize=7)
                for _sp in _ax.spines.values():
                    _sp.set_edgecolor(_border)

                _fig.tight_layout()

                # Render
                _pbuf = _io.BytesIO()
                _fig.savefig(_pbuf, format="png", dpi=150,
                             bbox_inches="tight", facecolor=_fig.get_facecolor())
                _pbuf.seek(0)
                st.image(_pbuf.getvalue(), use_container_width=True)
                _plt.close(_fig)

                # Legend note
                _legend_parts = [
                    "🟢 Best binding energy",
                    "🔵 Other analogs",
                ]
                if _has_crystal:
                    _legend_parts += [
                        "Ring colour: 🟢 RMSD ≤2 Å  🟡 2–3 Å  🔴 >3 Å  ⚫ no crystal"
                    ]
                st.caption("  ·  ".join(_legend_parts))

            # ── Per-pose detail table (expandable per compound) ─────────────
            st.markdown("#### Per-pose scores")
            hint("Expand each compound to see all poses with binding energy and RMSD vs co-crystal.")
            for r in dock_results:
                pose_rows = r.get("_pose_rows", [])
                if not pose_rows:
                    continue
                cmpd_name = r["compound"]
                best_be   = r.get("top_BE")
                be_str    = f"{best_be:.2f} kcal/mol" if best_be else "—"
                with st.expander(f"{cmpd_name}  ·  best BE: {be_str}  ·  {len(pose_rows)} poses"):
                    pose_df = pd.DataFrame(pose_rows)
                    if "Affinity (kcal/mol)" in pose_df.columns:
                        pose_df = pose_df.sort_values("Affinity (kcal/mol)", na_position="last")

                    pstyled = pose_df.style.map(_colour_be, subset=["Affinity (kcal/mol)"])
                    if _has_crystal and "RMSD vs co-crystal (Å)" in pose_df.columns:
                        pstyled = pstyled.map(_colour_rmsd, subset=["RMSD vs co-crystal (Å)"])
                    st.dataframe(pstyled, use_container_width=True, hide_index=True)

                    # Best pose flag
                    if _has_crystal and "RMSD vs co-crystal (Å)" in pose_df.columns:
                        try:
                            best_rmsd_row = pose_df[pose_df["RMSD vs co-crystal (Å)"] != "—"].copy()
                            best_rmsd_row["_r"] = pd.to_numeric(best_rmsd_row["RMSD vs co-crystal (Å)"], errors="coerce")
                            best_rmsd_row = best_rmsd_row.dropna(subset=["_r"])
                            if not best_rmsd_row.empty:
                                br = best_rmsd_row.loc[best_rmsd_row["_r"].idxmin()]
                                rmsd_v = float(br["_r"])
                                icon = "✅" if rmsd_v <= 2.0 else ("⚠️" if rmsd_v <= 3.0 else "❌")
                                label = "good reproduction" if rmsd_v <= 2.0 else ("moderate" if rmsd_v <= 3.0 else "poor reproduction")
                                st.caption(
                                    f"{icon} Best RMSD: **{rmsd_v:.2f} Å** (Pose {int(br['Pose'])}) — {label}  "
                                    f"{'(≤2 Å = binding mode reproduced)' if rmsd_v <= 2.0 else '(>2 Å = check docking protocol)'}"
                                )
                        except Exception:
                            pass

            st.session_state.docking_summary = results_df

            grid_mols, grid_legs = [], []
            for r in result_rows:
                m = Chem.MolFromSmiles(str(r["SMILES"]))
                if m:
                    try:
                        AllChem.Compute2DCoords(m)
                        be_str   = f"BE={r['top_BE (kcal/mol)']}" if r.get("top_BE (kcal/mol)") else ""
                        rmsd_str = f"RMSD={r['top_RMSD (Å)']}"   if r.get("top_RMSD (Å)") else ""
                        grid_mols.append(m)
                        grid_legs.append(f"{r['compound']}\n{be_str}  {rmsd_str}")
                    except Exception:
                        pass
            if grid_mols:
                try:
                    png = Draw.MolsToGridImage(
                        grid_mols, legends=grid_legs, molsPerRow=4,
                        subImgSize=(280, 210), returnPNG=True)
                    st.image(png, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render structure grid: {e}")

            # ── 3D Viewer ─────────────────────────────────────────────────────────
            st.divider()
            st.markdown("### 🧬 3D Complex Viewer")
            hint("Select a compound to visualise its docked pose in the binding pocket.")

            if result_rows:
                v3d_compounds = [r["compound"] for r in result_rows if r.get("n_poses", 0) > 0]
                if v3d_compounds:
                    v3d_col1, v3d_col2 = st.columns([2, 3])
                    with v3d_col1:
                        sel_compound = st.selectbox(
                            "Compound", v3d_compounds, key="v3d_compound_sel")
                        show_labels  = st.checkbox("Residue labels", value=True, key="v3d_labels")
                        show_surface = st.checkbox("Protein surface", value=False, key="v3d_surface")
                        v3d_cutoff   = st.slider("Pocket cutoff (Å)", 3.0, 6.0, 4.0, 0.5, key="v3d_cutoff")
                        v3d_height   = st.slider("Viewer height (px)", 300, 700, 480, 50, key="v3d_height")

                        # Colour legend
                        st.markdown("""
<div style="font-size:0.78rem;line-height:1.8;margin-top:8px;">
<span style="color:#00FFFF">■</span> Docked ligand &nbsp;
<span style="color:#FF8C00">■</span> Pocket residues<br>
<span style="color:#FF00FF">■</span> Reference ligand &nbsp;
<span style="color:#AAAAAA">■</span> Protein
</div>""", unsafe_allow_html=True)

                    with v3d_col2:
                        # Find best pose SDF for selected compound
                        sdfs = core.find_pose_sdfs_for_compound(dock_out_dir, sel_compound)
                        pose_mol = None
                        if sdfs:
                            from rdkit.Chem import SDMolSupplier
                            suppl = SDMolSupplier(sdfs[0], removeHs=False)
                            for _m in suppl:
                                if _m is not None:
                                    pose_mol = _m
                                    break

                        if pose_mol:
                            render_complex_3d(
                                receptor_pdb  = receptor or "",
                                pose_mol      = pose_mol,
                                height        = v3d_height,
                                cutoff        = v3d_cutoff,
                                show_labels   = show_labels,
                                show_surface  = show_surface,
                                ref_ligand_pdb= st.session_state.ref_ligand_path or "",
                                key_prefix    = f"v3d_{sel_compound}",
                            )
                        else:
                            st.info(f"No pose found for {sel_compound}")
                else:
                    st.info("No docked poses available yet.")

            st.divider()
            csv_rows = [{"compound": r["compound"], "SMILES": r["SMILES"],
                "status": r["status"], "n_poses": r["n_poses"],
                "top_pose_BE_kcal_mol": r["top_BE (kcal/mol)"],
                "top_pose_RMSD_vs_crystal_A": r["top_RMSD (Å)"],
                "min_RMSD_pose_BE_kcal_mol": r["minRMSD_BE (kcal/mol)"],
                "min_RMSD_vs_crystal_A": r["minRMSD (Å)"],
            } for r in result_rows]
            csv_bytes = pd.DataFrame(csv_rows).to_csv(index=False).encode()
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("⬇️ Docking results CSV", data=csv_bytes,
                    file_name=f"{st.session_state.parent_name}_docking_results.csv",
                    mime="text/csv", use_container_width=True)
            with dl2:
                smi_out = "\n".join(f"{r['SMILES']}\t{r['compound']}" for r in result_rows)
                st.download_button("⬇️ Docked compounds SMILES", data=smi_out.encode(),
                    file_name=f"{st.session_state.parent_name}_docked.smi",
                    mime="text/plain", use_container_width=True)
            with st.expander("ACD log"):
                st.text("\n".join(full_log)[-4000:])

    elif not receptor:
        st.info("Load a target receptor above to enable docking.")
    else:
        st.info("ACD or OpenBabel is not installed here. Download the ligand file and dock externally.")
        st.download_button("⬇️ Download compounds.smi", data=smi_path.read_text(),
                           file_name="compounds.smi", mime="text/plain")

    if mode == "structure" and st.session_state.protein_path:
        st.divider()
        st.markdown("### 🔬 Interaction fingerprints (PLIP / cIFP)")
        hint("Compare binding interactions of parent vs each docked analog.")

        plip_available = bool(shutil.which("plipcmd") or shutil.which("plip"))
        if not plip_available:
            st.caption("PLIP not installed — using distance-based cIFP as fallback.")

        if st.button("Run PLIP / cIFP on all poses", type="primary", key="cifp_run_btn"):
            cifp_dir = work / "plip_cifp" / "complexes"
            cifp_dir.mkdir(parents=True, exist_ok=True)
            pose_dir = work / "docking_out" / "selected_pose_pdbs"
            pose_pdbs = list(pose_dir.glob("*.pdb")) if pose_dir.exists() else []

            if not pose_pdbs:
                st.warning("No docked pose PDBs found. Run docking first.")
            else:
                rows_c = []
                analog_feats_map = {}
                prog = st.progress(0.0, text="Running interaction analysis…")

                for idx_p, p in enumerate(pose_pdbs[:30]):
                    prog.progress((idx_p+1)/min(len(pose_pdbs),30),
                                  text=f"Analysing {p.stem}…")
                    cpx = str(cifp_dir / f"{p.stem}_complex.pdb")
                    protein_pdb = st.session_state.protein_path
                    core.combine_protein_ligand_pdb(protein_pdb, str(p), cpx)

                    feats = []
                    method = "distance"
                    if plip_available and _PA_OK:
                        xml_path, err, _ = _pa.run_plip_on_complex(
                            cpx, str(work / "plip_cifp" / "plip_out"), p.stem)
                        if xml_path:
                            plip_tbl = _pa.parse_plip_to_table(xml_path)
                            feats = [
                                f"{r['type']}:{r['chain']}:{r['resname']}:{r['resnum']}"
                                for _, r in plip_tbl.iterrows()
                            ] if not plip_tbl.empty else []
                            method = "PLIP"

                    if not feats and _PA_OK:
                        feats = _pa.distance_cifp_features(cpx, cutoff=4.0)

                    analog_feats_map[p.stem] = feats
                    rows_c.append({
                        "compound":        p.stem,
                        "method":          method,
                        "n_interactions":  len(feats),
                        "features":        ";".join(feats[:10]),
                    })

                prog.progress(1.0, text="Done!")
                cifp_df = pd.DataFrame(rows_c)
                st.session_state.cifp_results   = cifp_df
                st.session_state["_analog_plip"] = analog_feats_map

                # Compare parent vs analogs
                parent_feats = st.session_state.get("_plip_parent_feats", [])
                if parent_feats and analog_feats_map and _PA_OK:
                    cmp_df = _pa.compare_cifp(parent_feats, analog_feats_map)
                    st.session_state["_cifp_comparison"] = cmp_df
                    st.success(f"PLIP/cIFP computed for {len(cifp_df)} poses")
                else:
                    st.session_state["_cifp_comparison"] = None
                    st.success(f"cIFP computed for {len(cifp_df)} poses")
                st.rerun()

        # Show results
        cifp_df = st.session_state.get("cifp_results")
        cmp_df  = st.session_state.get("_cifp_comparison")

        if cifp_df is not None and not cifp_df.empty:
            tab_raw, tab_cmp = st.tabs(["📋 Interaction table", "⚖️ Parent vs Analogs"])

            with tab_raw:
                st.dataframe(cifp_df, use_container_width=True, hide_index=True)

            with tab_cmp:
                if cmp_df is not None and not cmp_df.empty:
                    st.markdown("**Tanimoto similarity to parent interaction fingerprint**")
                    st.caption(
                        "🟢 High tanimoto = similar binding mode  "
                        "· ⚠️ = lost key interaction  "
                        "· n_new = potentially new contacts"
                    )

                    def _cmp_style(val):
                        try:
                            v = float(val)
                            if v >= 0.7:   return "background-color:#E6F4EA;color:#1E7E34"
                            elif v >= 0.4: return "background-color:#FFF3CD;color:#856404"
                            else:          return "background-color:#FDE8E8;color:#842029"
                        except Exception:
                            return ""

                    styled = cmp_df.style.map(_cmp_style, subset=["tanimoto"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    # Highlight warnings
                    warn_df = cmp_df[cmp_df["warning"] != ""]
                    if not warn_df.empty:
                        st.markdown("**⚠️ Analogs with lost key interactions:**")
                        for _, row in warn_df.iterrows():
                            st.warning(f"{row['compound']}: {row['warning']}")
                else:
                    st.info(
                        "Run PLIP analysis in Step 3 first to get parent interaction "
                        "fingerprint for comparison."
                    )

    st.write("")
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            go(4)
    with col_next:
        if st.button("Export results →", type="primary"):
            go(len(steps))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 (ligand) / STEP 6 (structure) – Export
# ─────────────────────────────────────────────────────────────────────────────

elif step == len(steps):
    df = st.session_state.analogs_df
    if df is None or df.empty:
        st.warning("No analogs generated yet. Complete the earlier steps first.")
        st.stop()

    parent_name = st.session_state.parent_name or "compound"
    work = get_work_dir()
    tier = analog_tier(len(df))

    info_card("Your analogs are ready. Download the table, SMILES file, 3D structures, or a full ZIP archive.")

    smi_lines = "\n".join(f"{r.smiles}\t{parent_name}_A{i+1}" for i, r in df.iterrows())

    st.markdown("### Download analogs")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇️ Analog table (CSV)",
            data=df.to_csv(index=False).encode(),
            file_name=f"{parent_name}_analogs.csv", mime="text/csv",
            use_container_width=True)
    with dl2:
        st.download_button("⬇️ SMILES file (.smi)",
            data=smi_lines.encode(),
            file_name=f"{parent_name}_analogs.smi", mime="text/plain",
            use_container_width=True)

    # ── 3D SDF — available for docking and pkanet tiers ─────────────────────
    if tier in ("docking", "pkanet"):
        st.divider()
        if tier == "pkanet":
            st.markdown("### pKaNET — Check protonation state & Generate 3D structures")
            info_card(
                "pKaNET predicts the dominant protonation state at your target pH before building 3D conformers. "
                "This gives more accurate structures for downstream docking or MD simulation."
            )
        else:
            st.markdown("### Generate 3D structures")
        hint("Creates a 3D conformer for each analog — useful for visualisation or further docking.")

        g1, g2 = st.columns(2)
        with g1:
            fmt_sel = st.multiselect("Output formats", ["SDF", "PDB", "MOL2"], default=["SDF"])
        with g2:
            mmff_opt = st.checkbox("MMFF geometry optimisation", value=True)

        if tier == "pkanet":
            pk1, pk2 = st.columns(2)
            with pk1:
                pkanet_ph = st.number_input("Target pH for protonation", value=7.4, step=0.1,
                                            help="pKaNET will assign the dominant state at this pH.")
            with pk2:
                st.markdown('<div style="height:1.4rem;"></div>', unsafe_allow_html=True)
                use_pkanet = st.checkbox("Use pKaNET protonation", value=True)
        else:
            use_pkanet = False
            pkanet_ph  = 7.4

        if st.button("Generate 3D structures", type="primary"):
            lig_table = df[["smiles"]].copy()
            lig_table["compound"] = [f"{parent_name}_A{i+1}" for i in range(len(df))]
            out_dir = work / "ligands_3d"
            with st.spinner("Building 3D conformers…"):
                manifest = core.generate_3d_ligand_files(lig_table, out_dir, formats=fmt_sel, mmff=mmff_opt)
            ok_count = int((manifest.status == "ok").sum())
            st.success(f"3D files generated: {ok_count} / {len(manifest)}")
            st.dataframe(manifest, use_container_width=True, hide_index=True)
            combined = out_dir / "all_ligands_3d.sdf"
            if combined.exists():
                st.download_button("⬇️ Download combined SDF",
                    data=combined.read_bytes(),
                    file_name=f"{parent_name}_3d.sdf",
                    mime="chemical/x-mdl-sdfile",
                    use_container_width=True)

    else:
        # smi_only tier — show info about upgrading
        st.divider()
        st.info(
            f"You have **{len(df)} analogs**. "
            "3D generation and pKaNET are available for ≤ 200 analogs, "
            "and docking for ≤ 20 analogs. "
            "Go back to Step 3 to reduce the count if needed."
        )

    # ── Full ZIP ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Full archive")
    hint("Everything in one ZIP — analog table, SMILES, 3D files, docking results, and a session summary.")

    if st.button("Build ZIP archive", use_container_width=True):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{parent_name}_analogs.csv", df.to_csv(index=False))
            z.writestr(f"{parent_name}_analogs.smi", smi_lines)
            for subdir in ["ligands_3d", "docking_out"]:
                p = work / subdir
                if p.exists():
                    for f in p.rglob("*.*"):
                        if f.is_file():
                            z.write(f, arcname=f"{subdir}/{f.relative_to(p)}")
            cifp = st.session_state.cifp_results
            if cifp is not None and not cifp.empty:
                z.writestr("cifp_results.csv", cifp.to_csv(index=False))
            z.writestr("session_info.json", json.dumps({
                "parent_smiles":  st.session_state.parent_smiles,
                "parent_name":    parent_name,
                "mode":           mode,
                "n_analogs":      len(df),
                "tier":           tier,
                "selected_atoms": sorted(st.session_state.selected_atoms),
                "risk":           st.session_state.risk,
                "rank_by":        st.session_state.rank_by,
            }, indent=2))
        buf.seek(0)
        st.download_button("⬇️ Download full ZIP",
            data=buf.getvalue(),
            file_name=f"{parent_name}_analog_builder_results.zip",
            mime="application/zip",
            use_container_width=True)

    st.divider()
    with st.expander("Session summary"):
        st.json({
            "mode":              mode,
            "parent_smiles":     st.session_state.parent_smiles,
            "parent_name":       parent_name,
            "selected_atoms":    sorted(st.session_state.selected_atoms),
            "analogs_generated": len(df),
            "tier":              tier,
            "top_category":      df.fragment_category.value_counts().index[0] if len(df) else "—",
            "receptor_loaded":   bool(st.session_state.receptor_path),
            "docking_run":       st.session_state.docking_ligands is not None,
        })

    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            go(step - 1)
