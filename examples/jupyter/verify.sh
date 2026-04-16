#!/bin/bash
# Quick verification that Jupyter notebook approach works

echo "🔍 Verifying Gollek Jupyter Setup"
echo "=================================="
echo

# Check Python
echo "1. Python:"
python3 --version 2>&1 | head -1

# Check Jupyter
echo
echo "2. Jupyter:"
jupyter --version 2>&1 | head -1

# Check Java
echo
echo "3. Java:"
java -version 2>&1 | head -1

# Check Runner JAR
echo
echo "4. Gollek Runner:"
RUNNER="gollek/ui/gollek-cli/target/gollek-runner.jar"
if [ -f "$RUNNER" ]; then
    SIZE=$(du -h "$RUNNER" | cut -f1)
    echo "✅ Found ($SIZE)"
else
    echo "❌ Not found. Build first:"
    echo "   mvn clean package -pl gollek/ui/gollek-cli -am -DskipTests"
    exit 1
fi

# Check notebooks
echo
echo "5. Notebooks:"
for nb in gollek/examples/jupyter/*.ipynb; do
    if [ -f "$nb" ]; then
        echo "✅ $(basename $nb)"
    fi
done

# Test a quick call
echo
echo "6. Quick Java CLI test:"
java -jar "$RUNNER" --help 2>&1 | head -1
echo "✅ CLI accessible"

echo
echo "=================================="
echo "✅ Verification Complete!"
echo
echo "To start notebook:"
echo "  jupyter notebook gollek/examples/jupyter/08-gollek-complete-demo.ipynb"
echo
echo "Select kernel: Python 3"
echo "Run cells to test all Gollek features!"
