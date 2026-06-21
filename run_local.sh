#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Analog Designer — Local launcher
# Usage: bash run_local.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ⌬+⌬ Analog Designer — Local Mode"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python
python3 --version || { echo "❌ Python 3 not found"; exit 1; }

# Check Streamlit
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements_local.txt
fi

# Check obabel
if command -v obabel &>/dev/null; then
    echo "✅ obabel: $(obabel --version 2>&1 | head -1)"
else
    echo "⚠️  obabel not found — install with:"
    echo "     Ubuntu: sudo apt-get install openbabel"
    echo "     macOS:  brew install open-babel"
fi

# Check ADMET-AI
if python3 -c "from admet_ai import ADMETModel" 2>/dev/null; then
    echo "✅ ADMET-AI: available"
else
    echo "ℹ️  ADMET-AI not installed (optional) — pip install admet-ai"
fi

# Check pKaNET local
if [ -f "pkanet.py" ]; then
    echo "✅ pkanet.py: found"
else
    echo "ℹ️  pkanet.py not found — pKaNET local mode unavailable"
fi

echo ""
echo "🚀 Starting Analog Designer (Local)..."
echo "   Open: http://localhost:8501"
echo ""

streamlit run app_local.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none
