#!/usr/bin/env bash
# 一键启动 Auto Stock 控制台（项目根目录执行）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
cd "$ROOT/src"
echo "Starting control panel at http://127.0.0.1:8000 ..."
exec python -m auto_select_stock.ops_dashboard "$@"
