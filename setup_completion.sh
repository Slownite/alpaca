#!/bin/bash
# Setup bash completion for alpaca CLI

echo "Setting up tab completion for alpaca CLI..."

# Check if argcomplete is installed
if ! python3 -c "import argcomplete" 2>/dev/null; then
    echo "Error: argcomplete is not installed. Install it with: pip install argcomplete"
    exit 1
fi

# Check if alpaca is available
if ! which alpaca >/dev/null 2>&1; then
    echo "Error: alpaca command not found. Make sure it's installed and in PATH."
    exit 1
fi

# Enable completion for this session
eval "$(register-python-argcomplete alpaca)"

# Add to shell config for permanent setup
COMPLETION_LINE='eval "$(register-python-argcomplete alpaca)"'

if [ -n "$BASH_VERSION" ]; then
    # Bash
    SHELL_CONFIG="$HOME/.bashrc"
    echo "Adding completion to $SHELL_CONFIG"
    
    if ! grep -q "register-python-argcomplete alpaca" "$SHELL_CONFIG" 2>/dev/null; then
        echo "" >> "$SHELL_CONFIG"
        echo "# alpaca CLI tab completion" >> "$SHELL_CONFIG"
        echo "$COMPLETION_LINE" >> "$SHELL_CONFIG"
        echo "Added completion to $SHELL_CONFIG"
    else
        echo "Completion already configured in $SHELL_CONFIG"
    fi
elif [ -n "$ZSH_VERSION" ]; then
    # Zsh
    SHELL_CONFIG="$HOME/.zshrc"
    echo "Adding completion to $SHELL_CONFIG"
    
    if ! grep -q "register-python-argcomplete alpaca" "$SHELL_CONFIG" 2>/dev/null; then
        echo "" >> "$SHELL_CONFIG"
        echo "# alpaca CLI tab completion" >> "$SHELL_CONFIG"
        echo "$COMPLETION_LINE" >> "$SHELL_CONFIG"
        echo "Added completion to $SHELL_CONFIG"
    else
        echo "Completion already configured in $SHELL_CONFIG"
    fi
fi

echo ""
echo "Setup complete! Tab completion is now enabled for the alpaca command."
echo "Try typing 'alpaca serve <TAB>' to see model completions."
echo ""
echo "Note: You may need to restart your shell or run 'source ~/.bashrc' (or ~/.zshrc) for the changes to take effect."