#!/usr/bin/env python3
"""
Minimal standalone server for sector rotation dashboard.
Serves the HTML template and sector API endpoints without requiring torch.
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import math

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "stock.db"
TEMPLATE_PATH = Path(__file__).resolve().parent / "src/auto_select_stock/web/templates/ops_sector_rotation.html"

def get_sector_list():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT s.code, s.name, s.category, sd.date, sd.close, sd.pct_change "
        "FROM sector s "
        "LEFT JOIN sector_daily sd ON s.code = sd.sector_code "
        "AND sd.date = (SELECT MAX(date) FROM sector_daily WHERE sector_code = s.code) "
        "ORDER BY s.category, s.name"
    )
    rows = []
    for r in cur.fetchall():
        rows.append({
            "code": r[0], "name": r[1], "category": r[2],
            "last_date": r[3], "last_close": r[4], "pct_change": r[5]
        })
    conn.close()
    return rows

def get_sector_data(sector_code, start_date=None, end_date=None):
    conn = sqlite3.connect(DB_PATH)
    sql = "SELECT date, open, high, low, close, volume, amount, turnover_rate, pct_change FROM sector_daily WHERE sector_code = ?"
    params = [sector_code]
    if start_date:
        sql += " AND date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND date <= ?"
        params.append(end_date)
    sql += " ORDER BY date ASC"
    cur = conn.execute(sql, params)
    rows = []
    for r in cur.fetchall():
        rows.append({
            "date": r[0], "open": r[1], "high": r[2], "low": r[3],
            "close": r[4], "volume": r[5], "amount": r[6],
            "turnover_rate": r[7], "pct_change": r[8]
        })
    conn.close()
    return rows

def get_sector_close_series(sector_code, start_date=None, end_date=None):
    """Get close prices as dict {date: close}"""
    conn = sqlite3.connect(DB_PATH)
    sql = "SELECT date, close FROM sector_daily WHERE sector_code = ?"
    params = [sector_code]
    if start_date:
        sql += " AND date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND date <= ?"
        params.append(end_date)
    sql += " ORDER BY date ASC"
    cur = conn.execute(sql, params)
    rows = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()
    return rows

def compute_correlation_matrix(sector_codes, start_date=None, end_date=None):
    """Compute Pearson correlation of returns for given sectors."""
    series = {}
    for code in sector_codes:
        data = get_sector_close_series(code, start_date, end_date)
        if len(data) < 2:
            continue
        dates = sorted(data.keys())
        returns = []
        for i in range(1, len(dates)):
            prev = data[dates[i-1]]
            curr = data[dates[i]]
            if prev and curr and prev != 0:
                returns.append((dates[i], (curr - prev) / prev))
        if returns:
            series[code] = returns

    if not series:
        return {"error": "No data available"}

    codes = list(series.keys())
    n = len(codes)
    corr = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Compute pairwise correlations
    for i in range(n):
        for j in range(i+1, n):
            code1, code2 = codes[i], codes[j]
            # Get common dates
            dates1 = set(r[0] for r in series[code1])
            dates2 = set(r[0] for r in series[code2])
            common = sorted(dates1 & dates2)
            if len(common) < 2:
                corr[i][j] = corr[j][i] = 0.0
                continue
            rets1 = [series[code1][next(i for i, r in enumerate(series[code1]) if r[0] == d)][1] for d in common]
            rets2 = [series[code2][next(i for i, r in enumerate(series[code2]) if r[0] == d)][1] for d in common]
            mean1 = sum(rets1) / len(rets1)
            mean2 = sum(rets2) / len(rets2)
            cov = sum((rets1[k] - mean1) * (rets2[k] - mean2) for k in range(len(rets1)))
            std1 = math.sqrt(sum((r - mean1)**2 for r in rets1))
            std2 = math.sqrt(sum((r - mean2)**2 for r in rets2))
            if std1 > 0 and std2 > 0:
                corr[i][j] = corr[j][i] = cov / (std1 * std2)
            else:
                corr[i][j] = corr[j][i] = 0.0

    return {"correlation": corr, "sectors": codes}

def compute_momentum(sector_codes, lookback_days=20):
    """Rank sectors by momentum (cumulative return over lookback period)."""
    if not sector_codes:
        # Get all sectors if none specified
        sectors = get_sector_list()
        sector_codes = [s["code"] for s in sectors]

    conn = sqlite3.connect(DB_PATH)
    results = []
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days*2)).strftime("%Y-%m-%d")

    for code in sector_codes:
        cur = conn.execute(
            "SELECT date, close FROM sector_daily WHERE sector_code = ? AND date >= ? ORDER BY date ASC",
            (code, start)
        )
        rows = cur.fetchall()
        if len(rows) < 2:
            continue
        # Get last available date before lookback and last date
        cutoff_idx = max(0, len(rows) - lookback_days - 1)
        if cutoff_idx >= len(rows):
            continue
        start_price = rows[cutoff_idx][1]
        end_price = rows[-1][1]
        if start_price and end_price and start_price > 0:
            momentum = (end_price - start_price) / start_price * 100
            results.append({"code": code, "momentum": momentum})

    conn.close()
    results.sort(key=lambda x: x["momentum"], reverse=True)
    return {"momentum": results[:100]}

def detect_cycles(sector_code, threshold_pct=5.0):
    """Detect continuous up/down cycles exceeding threshold."""
    data = get_sector_data(sector_code)
    if not data:
        return {"error": f"No data for sector {sector_code}"}

    closes = [(d["date"], d["close"]) for d in data if d["close"]]
    if len(closes) < 2:
        return {"error": "Insufficient data"}

    # Compute cumulative returns from start
    base = closes[0][1]
    cumret = [(d, (c - base) / base * 100) for d, c in closes]

    cycles = []
    in_up = None
    start_date = None
    peak = None
    trough = None

    for i, (date, ret) in enumerate(cumret):
        if in_up is None:
            in_up = ret > threshold_pct
            start_date = date
            peak = ret if in_up else None
            trough = ret if not in_up else None
        elif in_up:
            if ret > threshold_pct:
                peak = ret
            else:
                # Cycle ended
                cycles.append({"type": "up", "start": start_date, "end": date,
                               "magnitude": peak, "duration_days": (i)})
                in_up = False
                start_date = date
                trough = ret
        else:
            if ret < -threshold_pct:
                trough = ret
            else:
                cycles.append({"type": "down", "start": start_date, "end": date,
                               "magnitude": abs(trough) if trough else 0, "duration_days": (i)})
                in_up = True
                start_date = date
                peak = ret

    # Compute statistics
    up_cycles = [c for c in cycles if c["type"] == "up"]
    down_cycles = [c for c in cycles if c["type"] == "down"]

    stats = {
        "sector": sector_code,
        "threshold_pct": threshold_pct,
        "total_cycles": len(cycles),
        "up_cycles": len(up_cycles),
        "down_cycles": len(down_cycles),
        "avg_up_duration": sum(c["duration_days"] for c in up_cycles) / max(1, len(up_cycles)),
        "avg_down_duration": sum(c["duration_days"] for c in down_cycles) / max(1, len(down_cycles)),
        "avg_up_magnitude": sum(c["magnitude"] for c in up_cycles) / max(1, len(up_cycles)),
        "avg_down_magnitude": sum(abs(c["magnitude"]) for c in down_cycles) / max(1, len(down_cycles)),
        "recent_cycles": cycles[-10:] if len(cycles) > 10 else cycles
    }
    return stats

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/sector-rotation":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            try:
                with open(TEMPLATE_PATH, "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"<html><body><h1>Template not found</h1></body></html>")
            return

        if path == "/sector-report" or path == "/report":
            report_path = Path(__file__).resolve().parent / "sector_report.html"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            try:
                with open(report_path, "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"<html><body><h1>Report not found</h1></body></html>")
            return

        if path == "/sector-list":
            try:
                self.send_json(get_sector_list())
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            return

        if path == "/sector-data":
            sector = (query.get("sector") or [""])[0]
            if not sector:
                self.send_error(400, "missing sector")
                return
            start = (query.get("start") or [None])[0]
            end = (query.get("end") or [None])[0]
            try:
                self.send_json(get_sector_data(sector, start, end))
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            return

        if path == "/sector-compare":
            sectors_raw = query.get("sectors", [])
            sectors_param = " ".join(sectors_raw).strip()
            sectors = [s for s in sectors_param.replace(",", " ").split() if s]
            start = (query.get("start") or [None])[0]
            end = (query.get("end") or [None])[0]
            if not sectors:
                self.send_json({"error": "missing sectors"}, 400)
                return
            try:
                # Get normalized index values for comparison
                result = {}
                for code in sectors:
                    data = get_sector_close_series(code, start, end)
                    if data:
                        dates = sorted(data.keys())
                        base = data[dates[0]] if dates else 1
                        result[code] = [{"date": d, "value": data[d] / base * 100 if base else 100} for d in dates]
                self.send_json(result)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            return

        if path == "/sector-analysis":
            analysis_type = (query.get("type") or ["correlation"])[0]
            # Handle multiple sectors query params
            sectors_raw = query.get("sectors", [])
            sectors_param = " ".join(sectors_raw).strip()
            sectors = [s for s in sectors_param.replace(",", " ").split() if s]
            sector = (query.get("sector") or [""])[0].strip()
            start = (query.get("start") or [None])[0]
            end = (query.get("end") or [None])[0]
            lookback = int((query.get("lookback") or ["20"])[0])

            try:
                if analysis_type == "correlation":
                    self.send_json(compute_correlation_matrix(sectors, start, end))
                elif analysis_type == "momentum":
                    self.send_json(compute_momentum(sectors if sectors else None, lookback))
                elif analysis_type == "cycles":
                    if not sector:
                        self.send_json({"error": "missing sector"}, 400)
                        return
                    self.send_json(detect_cycles(sector))
                else:
                    self.send_json({"error": f"unknown analysis type: {analysis_type}"}, 400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            return

        # Serve static data files
        if path.startswith("/data/"):
            filename = path[6:]  # Remove "/data/"
            file_path = DATA_DIR / filename
            if file_path.exists() and file_path.is_file():
                content = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        self.send_error(404, "Not found")

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("Serving sector rotation dashboard on http://127.0.0.1:8000/sector-rotation")
    print("Full report at http://127.0.0.1:8000/sector-report")
    server.serve_forever()
