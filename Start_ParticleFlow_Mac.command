#!/bin/bash
cd "$(dirname "$0")"
clear
printf '%s\n' '=========================================='
printf '%s\n' '  Particle Flow Video Analysis Tool'
printf '%s\n' '=========================================='
printf '\n'

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found on this Mac."
  echo "Please install Python 3.10 or newer, then run this file again."
  read -r -p "Press Enter to close..."
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "First-time setup: creating a private Python environment..."
  python3 -m venv .venv || exit 1
  echo "Installing required packages. This may take a few minutes..."
  .venv/bin/python -m pip install --upgrade pip || exit 1
  .venv/bin/python -m pip install -r requirements.txt || exit 1
fi

echo "Starting Particle Flow Analysis Tool..."
echo "Your browser should open automatically."
echo "Keep this Terminal window open while using the tool."
echo
.venv/bin/python -m streamlit run app.py
