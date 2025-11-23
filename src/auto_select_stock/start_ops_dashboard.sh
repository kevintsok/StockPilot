#!/usr/bin/env bash
# 一键启动 Auto Stock 控制台（包内路径执行）
set -exuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
echo "Starting control panel at http://127.0.0.1:8000 ..."
exec python -m auto_select_stock.ops_dashboard "$@"
