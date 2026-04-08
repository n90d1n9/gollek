#!/bin/bash
# Gollek SDK - Interactive Setup Script
# Usage: curl -sSL https://raw.githubusercontent.com/bhangun/gollek/main/scripts/install.sh | bash

set -e

# Visual Header
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Gollek - Complete Interactive Setup                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo "This script will prepare your environment for Gollek development."
echo ""

prompt_install() {
    local reply
    # Use /dev/tty so interactive prompt works even if script is piped via curl
    read -p "$1 (y/n) " -n 1 -r reply < /dev/tty
    echo
    if [[ $reply =~ ^[Yy]$ ]]; then
        return 0
    fi
    return 1
}

# 1. Java Verification
echo "🔍 Checking Java environment..."
if ! command -v java &> /dev/null || ! java -version 2>&1 | grep -q 'version "25'; then
    if prompt_install "Java 25 is required but not found or not currently active. Install it now via SDKMAN?"; then
        echo "Installing Java 25..."
        curl -sSL https://raw.githubusercontent.com/bhangun/gollek/main/scripts/install-jdk.sh | bash
        
        export SDKMAN_DIR="$HOME/.sdkman"
        if [ -s "$SDKMAN_DIR/bin/sdkman-init.sh" ]; then
            source "$SDKMAN_DIR/bin/sdkman-init.sh"
        fi
    else
        echo "⏭️ Skipping Java installation."
    fi
else
    java_version=$(java -version 2>&1 | head -n 1)
    echo "✓ Found $java_version"
fi

# 2. Gollek CLI Installation
echo ""
echo "🚀 Checking Gollek CLI..."
if ! command -v gollek &> /dev/null; then
    if prompt_install "Gollek CLI not found. Install it now?"; then
        curl -sSL https://raw.githubusercontent.com/bhangun/gollek/main/scripts/install-cli.sh | bash
    else
        echo "⏭️ Skipping Gollek CLI installation."
    fi
else
    echo "✓ Gollek CLI is already available: $(gollek --version 2>/dev/null || echo 'installed')"
fi

# 3. JBang Installation
echo ""
echo "📦 Checking jbang..."
if ! command -v jbang &> /dev/null; then
    if prompt_install "jbang not found. Install it now?"; then
        curl -sSL https://sh.jbang.dev | bash -s - app setup
        export PATH="$PATH:$HOME/.jbang/bin"
    else
        echo "⏭️ Skipping jbang installation."
    fi
else
    echo "✓ jbang $(jbang --version 2>/dev/null | head -n 1) is already available."
fi

# 4. Jupyter Notebook Support
echo ""
echo "📓 Checking Jupyter kernel integration..."
if prompt_install "Do you want to setup Jupyter integration for Gollek?"; then
    echo "Setting up Jupyter integration..."
    curl -sSL https://raw.githubusercontent.com/bhangun/gollek/main/sdk/integration/install-jupyter-jbang.sh | bash
else
    echo "⏭️ Skipping Jupyter integration."
fi

# 5. Summary & Usage
echo ""
echo "✅ Gollek environment setup complete!"
echo ""
echo "🚀 Next Steps (Run any of these to start):"
echo ""
echo "1. Run the Hello Gollek example via jbang:"
echo "   jbang https://raw.githubusercontent.com/bhangun/gollek/main/sdk/integration/jbang-templates/examples/hello_gollek.java"
echo ""
echo "2. Run the Batch Processing example:"
echo "   jbang https://raw.githubusercontent.com/bhangun/gollek/main/sdk/integration/jbang-templates/examples/batch_process.java"
echo ""
echo "3. Use the newly installed Gollek CLI:"
echo "   gollek chat"
echo ""
echo "📖 View full documentation at: https://github.com/bhangun/gollek/tree/main/website/gollek-ai.github.io/docs"
echo "⚠️  Note: If you installed tools purely in this session, please open a NEW TERMINAL to ensure all paths are updated."
echo ""
