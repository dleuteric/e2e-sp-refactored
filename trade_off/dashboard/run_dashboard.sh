#!/bin/bash
# Quick launcher for ez-SMAD MCDA Dashboard

echo "=================================="
echo "ez-SMAD MCDA Dashboard Launcher"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found in current directory"
    echo "Please run this script from trade_off/dashboard/"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import dash, dash_bootstrap_components, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install dash dash-bootstrap-components plotly -q
fi

echo "✓ Dependencies OK"
echo ""
echo "Starting dashboard..."
echo "=================================="
echo ""
echo "📊 Dashboard URL: http://127.0.0.1:8050/"
echo "   Press Ctrl+C to stop"
echo ""
echo "=================================="
echo ""

# Launch the dashboard
python app.py
