#!/bin/bash
# Gollek CLI Installation Script
# Usage: curl -sSL https://raw.githubusercontent.com/bhangun/gollek/main/scripts/install-cli.sh | bash

set -e

echo "🚀 Installing Gollek CLI..."

if command -v brew &> /dev/null; then
    echo "📦 Using Homebrew to install gollek-cli..."
    brew tap bhangun/gollek
    brew install gollek
else
    echo "📦 Using official Gollek release installer..."
    curl -fsSL https://raw.githubusercontent.com/bhangun/gollek/main/scripts/install.sh | bash
fi

echo "✅ Gollek CLI installation complete!"
echo "Try running 'gollek --version' to verify."
