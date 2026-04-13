@echo off
REM Auto-optimize pipeline: optimize -> backtest -> top10 -> plot -> push
REM Called by hourly cron job

cd /d D:\Projects\auto-select-stock

echo =========================================
echo [%date% %time%] Starting optimization pipeline
echo =========================================

call wsl -d Ubuntu bash -c "source /home/julian/miniconda3/etc/profile.d/conda.sh && conda activate fin && cd /mnt/d/Projects/auto-select-stock && PYTHONPATH=./src python auto_optimize.py"
echo [Step1] Optimization done

call wsl -d Ubuntu bash -c "source /home/julian/miniconda3/etc/profile.d/conda.sh && conda activate fin && cd /mnt/d/Projects/auto-select-stock && PYTHONPATH=./src python -m auto_select_stock.cli backtest-strategies --start 2025-04-01 --end 2026-03-27 --checkpoint models/price_transformer_2025-train20250331-val20260327.pt --cost-bps 0 --slippage-bps 0 --strategies-dir src/auto_select_stock/predict/strategies/configs --output models/price_transformer_2025-train20250331-val20260327_strategies_comparison.json"
echo [Step2] Backtest done

call wsl -d Ubuntu bash -c "source /home/julian/miniconda3/etc/profile.d/conda.sh && conda activate fin && cd /mnt/d/Projects/auto-select-stock && PYTHONPATH=./src python plot_strategies.py"
echo [Step3] Plot done

echo =========================================
echo [%date% %time%] Pipeline complete
echo =========================================
