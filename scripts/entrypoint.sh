#!/bin/bash
set -euo pipefail

# TRT engines are automatically built from ONNX on first startup
# by the C++ binary (cached by TRT version + model hash).
# No Python or manual conversion needed.

exec "$@"
