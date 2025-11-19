#!/bin/bash

# Script to run the webscraper in a tmux session with wandb monitoring
# Usage: ./run_scraper_tmux.sh

SESSION_NAME="truth-social-scraper"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRAPER_SCRIPT="${SCRIPT_DIR}/webscraper/data_collector/post_collector.py"

# Truth Social credentials (from setup.sh or environment)
TRUTHSOCIAL_USERNAME="${TRUTHSOCIAL_USERNAME:-aditya_murthy}"
TRUTHSOCIAL_PASSWORD="${TRUTHSOCIAL_PASSWORD:-Truthsocialus3r}"

# Load .env file if it exists
if [ -f "${SCRIPT_DIR}/.env" ]; then
    # Extract WANDB_API_KEY from .env file (handles comments and whitespace)
    if [ -z "$WANDB_API_KEY" ]; then
        WANDB_API_KEY=$(grep -E "^WANDB_API_KEY=" "${SCRIPT_DIR}/.env" | cut -d '=' -f2 | tr -d ' ' | tr -d '"' | tr -d "'")
        if [ -n "$WANDB_API_KEY" ]; then
            export WANDB_API_KEY
        fi
    fi
    # Extract TRUTHSOCIAL credentials if not already set
    if [ -z "$TRUTHSOCIAL_USERNAME" ]; then
        TRUTHSOCIAL_USERNAME=$(grep -E "^TRUTHSOCIAL_USERNAME=" "${SCRIPT_DIR}/.env" | cut -d '=' -f2 | tr -d ' ' | tr -d '"' | tr -d "'")
        if [ -n "$TRUTHSOCIAL_USERNAME" ]; then
            export TRUTHSOCIAL_USERNAME
        fi
    fi
    if [ -z "$TRUTHSOCIAL_PASSWORD" ]; then
        TRUTHSOCIAL_PASSWORD=$(grep -E "^TRUTHSOCIAL_PASSWORD=" "${SCRIPT_DIR}/.env" | cut -d '=' -f2 | tr -d ' ' | tr -d '"' | tr -d "'")
        if [ -n "$TRUTHSOCIAL_PASSWORD" ]; then
            export TRUTHSOCIAL_PASSWORD
        fi
    fi
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first."
    exit 1
fi

# Check if the scraper script exists
if [ ! -f "$SCRAPER_SCRIPT" ]; then
    echo "Error: Scraper script not found at $SCRAPER_SCRIPT"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists."
    echo "You can attach to it with: tmux attach -t $SESSION_NAME"
    echo "Or kill it with: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Check wandb authentication
echo "Checking wandb authentication..."
if [ -z "$WANDB_API_KEY" ]; then
    # Check if wandb is logged in via wandb settings
    if command -v python3 &> /dev/null; then
        WANDB_STATUS=$(python3 -c "import os; from pathlib import Path; settings_file = Path.home() / '.netrc'; print('found' if settings_file.exists() and 'api.wandb.ai' in settings_file.read_text() if settings_file.exists() else 'not found')" 2>/dev/null || echo "unknown")
    else
        WANDB_STATUS="unknown"
    fi
    
    if [ "$WANDB_STATUS" != "found" ]; then
        echo ""
        echo "⚠️  WARNING: Wandb authentication not detected!"
        echo ""
        echo "To enable wandb monitoring, choose one of the following:"
        echo ""
        echo "Option 1: Set WANDB_API_KEY environment variable (recommended for tmux):"
        echo "  export WANDB_API_KEY='your-api-key-here'"
        echo "  (Get your API key from: https://wandb.ai/authorize)"
        echo ""
        echo "Option 2: Login interactively before running:"
        echo "  wandb login"
        echo ""
        echo "Option 3: Continue without wandb (will use offline mode)"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting. Set up wandb authentication and try again."
            exit 1
        fi
    else
        echo "✓ Wandb authentication found"
    fi
else
    echo "✓ WANDB_API_KEY environment variable is set"
fi

# Create new tmux session and run the scraper
echo "Creating tmux session '$SESSION_NAME'..."

# Build command with environment variables
CMD="python3 $SCRAPER_SCRIPT --max_likes 1000 --max_comments 500"

# Build environment variable exports
ENV_EXPORTS=""
if [ -n "$WANDB_API_KEY" ]; then
    ENV_EXPORTS="export WANDB_API_KEY='$WANDB_API_KEY' && "
fi
if [ -n "$TRUTHSOCIAL_USERNAME" ]; then
    ENV_EXPORTS="${ENV_EXPORTS}export TRUTHSOCIAL_USERNAME='$TRUTHSOCIAL_USERNAME' && "
fi
if [ -n "$TRUTHSOCIAL_PASSWORD" ]; then
    ENV_EXPORTS="${ENV_EXPORTS}export TRUTHSOCIAL_PASSWORD='$TRUTHSOCIAL_PASSWORD' && "
fi

# Create tmux session and run the command with environment variables
FULL_CMD="${ENV_EXPORTS}${CMD}; echo ''; echo 'Scraper finished. Press any key to close this window...'; read"
tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR" "$FULL_CMD"

echo "Scraper is now running in tmux session '$SESSION_NAME'"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from the session (while attached):"
echo "  Press Ctrl+B, then D"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "To view the session without attaching:"
echo "  tmux capture-pane -t $SESSION_NAME -p"
echo ""
echo "Monitor progress on wandb.ai dashboard"

