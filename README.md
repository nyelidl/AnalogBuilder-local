# <img src="https://raw.githubusercontent.com/nyelidl/AnalogBuilder/main/.fig/AB.svg" width="32"> nalog Designer — Local Version

**Analog Designer: ML-guided drug analog generation — Run locally, no limits!**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ligandbuilder.streamlit.app)

> Start with a parent SMILES. Select attachment atoms. Let ChemBERTa rank fragments by pocket chemistry. Generate analogs, dock them all — no analog count limit in local mode.

---

## 🖥️ Run locally

### Linux (Ubuntu / Debian)
```bash
sudo apt install python3.11 python3.11-venv openbabel libcairo2-dev libpangocairo-1.0-0 && \
git clone https://github.com/nyelidl/AnalogBuilder.git && \
cd AnalogBuilder && \
python3.11 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements_local.txt && \
streamlit run app_local.py
```

### macOS
```bash
brew install python@3.11 open-babel cairo pango && \
git clone https://github.com/nyelidl/AnalogBuilder.git && \
cd AnalogBuilder && \
python3.11 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements_local.txt && \
streamlit run app_local.py
```

> **Apple Silicon (M1/M2/M3/M4):** Fully supported — AnyonCanDock auto-downloads the correct `aarch64` Vina binary.

### Windows

> **Recommended:** Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) with Ubuntu and follow the Linux instructions above.

For native Windows:
1. **OpenBabel** — download from [openbabel.org](https://openbabel.org/wiki/Category:Installation) and add to PATH
2. **Cairo & Pango** — via conda: `conda install -c conda-forge cairo pango`

```bash
git clone https://github.com/nyelidl/AnalogBuilder.git
cd AnalogBuilder
python -m venv venv
venv\Scripts\activate
pip install -r requirements_local.txt
streamlit run app_local.py
```

### All platforms

- Python 3.10+ required
- AutoDock Vina 1.2.7 binary is **downloaded automatically** on first launch (Linux, macOS Intel/ARM, Windows)
- **No docking limit** — local mode unlocks docking for any number of analogs
- ADMET-AI and pKaNET run fully offline after install

---

## ✨ What it does

| | |
|---|---|
| <img src="https://raw.githubusercontent.com/nyelidl/AnalogBuilder/main/.fig/LB.svg" width="20"> | **Ligand-based analog generation** — SMILES + optional residue input for ML ranking |
| <img src="https://raw.githubusercontent.com/nyelidl/AnalogBuilder/main/.fig/SB.svg" width="20"> | **Structure-based analog generation** — co-crystal or apo PDB with auto-detect |
| 🤖 | **ChemBERTa ML fragment ranking** — transformer-based pocket-aware scoring, zero-shot |
| 📐 | **Pocket void analysis** — 3D sub-pocket detection with interactive colour-coded viewer |
| 🔬 | **PLIP interaction analysis** — 8 types, design recommendations, cIFP comparison |
| 🚀 | **ACD / AutoDock Vina docking** — unlimited analogs in local mode |
| 📊 | **Batch score plot** — BE vs compound with RMSD ring colour encoding |
| 🧊 | **3D viewer** — py3Dmol pocket view (Step 3) and pose viewer (Step 5) |
| ☠️ | **ADMET-AI predictions** — full Chemprop MPNN + TDC datasets (local only) |
| 🧪 | **pKaNET local protonation** — offline tautomer-aware microstate ranking |
| 📈 | **RMSD vs co-crystal** — MCS-based heavy-atom RMSD for all docked poses |

---

## 🖥️ Local vs Cloud

| Feature | 🖥️ Local | ☁️ Cloud |
|---------|----------|---------|
| Docking limit | **None — unlimited** | ≤ 20 analogs |
| ADMET-AI ML predictions | ✅ Full Chemprop MPNN | ❌ Not available |
| pKaNET protonation | ✅ Offline (`pkanet.py`) | ✅ Cloud API |
| ChemBERTa fragment ranking | ✅ Full (torch installed) | ⚠️ Fallback rule-based (Py 3.14) |
| PLIP interaction analysis | ✅ Full (Python openbabel) | ✅ system `obabel` |
| stmol 3D viewer | ✅ Available | ❌ Removed (dep conflict) |
| cairosvg 2D diagrams | ✅ With libcairo2 | ❌ Not available |
| GPU required | ❌ None | ❌ None |

---

## 🤖 ML Model — ChemBERTa Fragment Ranking

| Property | Value |
|----------|-------|
| Model | `seyonec/ChemBERTa-zinc-base-v1` |
| Architecture | RoBERTa transformer |
| Pre-training data | 77 million SMILES (ZINC database) |
| Pre-training task | Masked language modelling on SMILES |
| Output | 384-dimensional embeddings |
| Inference | CPU-only, ~0.3–0.5 s for 76 fragments |
| Fine-tuning required | ❌ None — zero-shot |

### Scoring pipeline

```
Binding-site residues (from PLIP or user input)
    ↓
1.  Map residue property tags → context SMILES (TAG_CONTEXT_SMILES)
2.  ChemBERTa embeds context SMILES → v_pocket ∈ ℝ³⁸⁴
3.  ChemBERTa embeds each fragment SMILES → v_frag_i ∈ ℝ³⁸⁴
4.  Cosine similarity → score_CB(i) = (v_pocket · v_frag_i + 1) / 2
5.  Blend: score_final = 0.7 × score_CB + 0.3 × score_rule_based
```

**Fallback:** if `transformers`/`torch` unavailable → rule-based 10×8 co-occurrence matrix (PDB-derived). No error, no user action needed — UI shows 🟢 ChemBERTa or 🟡 Rule-based.

---

## 🧪 ADMET-AI Predictions (Local Only)

Full Chemprop MPNN predictions using TDC benchmark datasets — available after `pip install admet-ai`.

| Property group | Predictions |
|----------------|-------------|
| Absorption | Caco-2 permeability, HIA, Bioavailability |
| Distribution | BBB penetration, VDss, PPBR |
| Metabolism | CYP1A2 / 2C9 / 2C19 / 2D6 / 3A4 substrate & inhibitor |
| Excretion | Half-life, clearance |
| Toxicity | hERG, DILI, Ames, FDAMDD, LD50 |

**Fallback:** if ADMET-AI not installed → RDKit rule-based descriptors (MW, clogP, TPSA, HBD/HBA, rotatable bonds) always available offline.

---

## 🔬 PLIP Interaction Analysis — 8 Types

| Type | Chemical basis | Key? |
|------|---------------|------|
| HBOND | Hydrogen bond donor–acceptor | ⭐ |
| HYDROPHOBIC | Van der Waals contact | |
| PISTACK | π–π aromatic stacking | |
| PICATION | Cation–π interaction | ⭐ |
| SALTBRIDGE | Ionic / electrostatic | ⭐ |
| HALOGEN | Halogen bond C–X···O/N | |
| WATERBRIDGE | Water-mediated H-bond | |
| METAL | Metal ion coordination | ⭐ |

⭐ Key interactions — loss in an analog raises a warning in the cIFP comparison table.

---

## 🗺️ Workflow — 6 Steps

```
Step 1A  Load protein structure   (structure track)
         └── RCSB search / PDB ID / upload
         └── Auto-detect: complex → skip docking | apo → dock automatically

Step 1B  Define parent compound
         └── SMILES paste · PubChem search · Ketcher 2D editor

Step 2   Select attachment atoms
         └── Click atoms on 2D depiction (multi-site supported)

Step 3   Pocket analysis + ML fragment ranking        ← ML runs here
         ├── ACD docking if apo (Mode B) — unlimited in local
         ├── PLIP → 8 interaction types + design recommendations
         ├── ChemBERTa/rule-based fragment ranking
         ├── Void analysis → 3D colour-coded sub-pocket viewer
         └── 🔒 preserve / 🌱 grow recommendations

Step 4   Generate analogs
         └── 76-fragment library across 8 categories
         └── Filter: MW, QED, reactive groups
         └── LOCAL: Docking → button always available regardless of count

Step 5   Docking + Evaluation   (LOCAL: no analog count limit)
         ├── ACD / AutoDock Vina 1.2.7 (binary auto-downloaded)
         ├── Batch score plot — BE vs compound, RMSD ring colour
         ├── Per-pose table — affinity + RMSD vs co-crystal
         ├── cIFP Tanimoto: parent vs each analog
         ├── ADMET-AI predictions per analog
         └── 3D viewer: select any pose

Step 6   Export
         └── SMILES CSV · docking scores · cIFP table
```

---

## 📐 Pocket Void Analysis

Detects unoccupied space between ligand and pocket boundary.

| Size class | Available radius | Fragment examples |
|------------|-----------------|-------------------|
| Small | r < 2.0 Å | F, Cl, OH, CH₃, CN |
| Medium | 2.0 – 3.2 Å | cyclopropyl, OMe, CF₃, azetidine |
| Large | 3.2 – 4.5 Å | phenyl, piperidine, pyridine |
| Extended | r ≥ 4.5 Å | indole, biphenyl, benzimidazole |

Sub-pocket residues are visualised in the 3D viewer with distinct colours (orange, cyan, magenta, green, yellow).

---

## 🧬 Fragment Library — 76 entries

| Category | Count | Examples |
|----------|------:|---------|
| Aromatic | 16 | phenyl, pyridin-3-yl, thiophen-2-yl, indol-3-yl |
| Polar | 13 | hydroxyl, methoxy, methylsulfonyl, cyano |
| Hydrophobic | 12 | cyclopropyl, tert-butyl, cyclohexyl |
| Basic | 9 | piperidine, morpholine, N-methylpiperazine |
| Halogen | 8 | fluoro, chloro, trifluoromethyl |
| Bioisostere | 7 | oxetan-3-yl, bicyclo[1.1.1]pentan-1-yl |
| Acidic | 6 | carboxyl, sulfonamide, tetrazole |
| Solubility | 5 | methoxyethyl, hydroxymethyl |

All fragments: MW ≤ 250 Da · rotatable bonds ≤ 6 · clogP ∈ [−3, 4.5] · single `[*]` attachment.

---

## 📁 Project structure

```
AnalogBuilder/
├── app_local.py            # Streamlit UI — LOCAL (no tier limits)
├── app.py                  # Streamlit UI — Cloud version
├── core.py                 # Chemistry backend + fragment library
├── pocket_reference.py     # ChemBERTa ML fragment ranker
├── void_analyzer.py        # Pocket void geometry
├── plip_analyzer.py        # PLIP interaction analysis
├── pkanet.py               # pKaNET local protonation (optional)
├── requirements_local.txt  # Python deps — local
├── requirements.txt        # Python deps — cloud
├── packages_local.txt      # System packages reference
├── run_local.sh            # One-command launcher
└── .streamlit/
    └── config.toml         # Disables file watcher
```

---

## 💻 Platform compatibility

| Platform | Vina binary | OpenBabel | ADMET-AI | Status |
|----------|-------------|-----------|----------|--------|
| **Linux x86_64** | ✅ Auto | `apt install openbabel` | ✅ | Fully supported (primary) |
| **macOS Intel** | ✅ Auto | `brew install open-babel` | ✅ | Fully supported |
| **macOS Apple Silicon** (M1–M4) | ✅ Native `aarch64` | `brew install open-babel` | ✅ | Fully supported |
| **Windows x86_64** | ✅ Auto | [Installer](https://openbabel.org/wiki/Category:Installation) | ✅ | Supported (WSL2 recommended) |
| **Streamlit Cloud** | ✅ Auto | via `packages.txt` | ❌ | Cloud version (`app.py`) |

---

## ⚙️ Optional components

### pKaNET local (`pkanet.py`)
Place `pkanet.py` in the same folder as `app_local.py`. The app detects it automatically and offers offline tautomer-aware protonation alongside the Cloud API option.

### ADMET-AI
```bash
pip install admet-ai
```
Downloads Chemprop model weights on first run (~500 MB). Subsequent runs use the local cache.

### ChemBERTa (ML fragment ranking)
```bash
pip install transformers torch
```
Downloads `ChemBERTa-zinc-base-v1` weights from HuggingFace on first run (~90 MB). Falls back to rule-based scoring if unavailable.

### cairosvg (2D diagrams)
```bash
# Ubuntu
sudo apt-get install libcairo2-dev libpango1.0-dev

# macOS
brew install cairo pango

pip install cairosvg
```

---

## 📊 Docking results

After docking, Step 5 shows:

**Batch score plot** — scatter + connecting line sorted by binding energy:
- 🟢 Best BE compound · 🔵 Other analogs
- Ring colour: 🟢 RMSD ≤ 2 Å · 🟡 2–3 Å · 🔴 > 3 Å · ⚫ no reference
- Dashed red line = co-crystal reference BE (if available)

**Per-compound table:**

| Compound | BE (kcal/mol) | RMSD vs crystal (Å) | Poses | cIFP Tanimoto |
|---|---|---|---|---|
| 🟢 < −10 · 🟡 −8–10 · 🔴 > −6 | | 🟢 ≤2 · ⚠️ 2–3 · ❌ >3 | | |

**Per-pose expandable** — all poses per compound with individual RMSD.

**cIFP comparison** — Tanimoto similarity of PLIP interaction fingerprints: parent vs each analog. Lost key interactions (HBOND, SALTBRIDGE, PICATION, METAL) trigger ⚠️ warning.

---

## 📝 Notes and limitations

- ChemBERTa is not fine-tuned for drug–pocket interaction; scoring is zero-shot.
- Fragment library covers 76 common building blocks — not the full synthetic accessibility space.
- Docking scores are screening-level estimates, not free energy predictions.
- RMSD vs co-crystal requires Mode A (co-crystal reference); not available in Mode B.
- Structure-guided suggestions depend on the quality of the input complex or docked pose.

---

## 📄 Citation

If you use Analog Designer in your research, please cite:

> **Analog Designer**
> Hengphasatporn K. et al. Analog Designer: A Streamlit-Based Platform for ML-Guided Drug Analog Generation with Integrated Pocket Analysis, Docking, and Interaction Fingerprinting. *J. Chem. Inf. Model.* (in preparation).
> https://github.com/nyelidl/AnalogBuilder

Please also acknowledge the underlying tools used in your run:

> **AutoDock Vina 1.2.7**
> Eberhardt et al., *J. Chem. Inf. Model.*, 2021
> DOI: https://doi.org/10.1021/acs.jcim.1c00203

> **AnyonCanDock**
> Hengphasatporn K. et al., *J. Chem. Inf. Model.*, 2026
> https://github.com/nyelidl/anyone-docking/

> **ChemBERTa**
> Chithrananda S. et al., *arXiv* 2020, 2010.09885
> https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1

> **PLIP**
> Salentin S. et al., *Nucleic Acids Res.*, 2015, 43, W443–W447
> DOI: https://doi.org/10.1093/nar/gkv315

> **ADMET-AI** *(if used)*
> Swanson K. et al., *Bioinformatics*, 2024, 40, btae416
> DOI: https://doi.org/10.1093/bioinformatics/btae416

> **RDKit**
> Landrum G. (2023). RDKit: Open-source cheminformatics.
> https://www.rdkit.org

> **pKaNET Cloud** *(if used)*
> Hengphasatporn K. et al., *J. Chem. Inf. Model.*, 2026, **66**(4), 1955–1963
> DOI: https://doi.org/10.1021/acs.jcim.5c02852

> **Dimorphite-DL**
> Ropp et al., *J. Cheminform.*, 2019
> DOI: https://doi.org/10.1186/s13321-019-0336-9

---

## 📜 License

MIT License — see `LICENSE` for details.

---

## 🔗 Related tools

| Tool | Description |
|------|-------------|
| [AnyonCanDock](https://github.com/nyelidl/anyone-docking) | Browser-based molecular docking |
| [pKaNET](https://doi.org/10.1021/acs.jcim.5c02852) | Tautomer-aware protonation state prediction |
| [DFDD](https://github.com/nyelidl/DFDD) | Host-guest binding thermodynamics |
