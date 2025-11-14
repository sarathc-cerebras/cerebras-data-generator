#!/bin/bash
set -e

# Get the absolute path of the directory containing this script.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/../.venv/bin/activate"
cd $SCRIPT_DIR/../packages/synth-gen-client; python -m build