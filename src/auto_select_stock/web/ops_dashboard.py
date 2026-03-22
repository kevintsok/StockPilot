"""
Simple local control panel server to run dataset fetching, financial downloads, training, and dashboard rendering
from a single web page. Start with:
    python -m auto_select_stock.ops_dashboard
Then open http://127.0.0.1:8000

Security: Set OPS_DASHBOARD_TOKEN env var to enable token-based authentication.
"""

import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Token-based auth - set OPS_DASHBOARD_TOKEN env var to protect all endpoints
OPS_DASHBOARD_TOKEN = os.getenv("OPS_DASHBOARD_TOKEN", "")

# Template directory (set by ops_handlers)
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _load_template(name: str) -> str:
    """Load HTML template from templates directory."""
    path = TEMPLATE_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<html><body><h1>Template not found: {}</h1></body></html>".format(name)


def check_auth(handler) -> bool:
    """Check if request has valid auth token. Returns True if no token configured."""
    if not OPS_DASHBOARD_TOKEN:
        return True  # No auth configured
    auth_header = handler.headers.get("Authorization", "")
    return auth_header == f"Bearer {OPS_DASHBOARD_TOKEN}"


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler that delegates to ops_handlers module."""

    def log_message(self, fmt, *args):
        return  # keep quiet in console

    def _require_auth(self) -> bool:
        """Check auth, send 401 if unauthorized."""
        if not check_auth(self):
            self.send_error(401, "Unauthorized")
            return False
        return True

    def _write_ok(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            return

    def do_GET(self):
        if not self._require_auth():
            return

        from urllib.parse import parse_qs, urlparse

        path = self.path
        parsed = urlparse(path)
        query = parsed.query

        # Import here to avoid circular imports and lazy load
        from .ops_handlers import (
            LOG_DIR, LOG_PATHS, RUNNING,
            _collect_stats, _dashboard_rows, _load_stock_detail,
            _read_log_tail, _purge_dead_processes,
        )
        from http import HTTPStatus

        _purge_dead_processes()

        if path == "/" or path.startswith("/index"):
            self._write_ok(_load_template("ops_dashboard.html").encode("utf-8"), "text/html; charset=utf-8")
            return
        if path.startswith("/state"):
            import json, time
            payload = {"running": RUNNING, "logs": LOG_PATHS, "ts": time.time()}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._write_ok(body, "application/json; charset=utf-8")
            return
        if path.startswith("/log"):
            qs = parse_qs(query)
            log_param = qs.get("path", [None])[0]
            if not log_param:
                self.send_error(HTTPStatus.BAD_REQUEST, "missing path")
                return
            log_path = Path(log_param)
            try:
                log_path = log_path.resolve()
            except Exception:
                self.send_error(HTTPStatus.BAD_REQUEST, "bad path")
                return
            if LOG_DIR not in log_path.parents and LOG_DIR != log_path.parent:
                self.send_error(HTTPStatus.BAD_REQUEST, "log path not allowed")
                return
            content = _read_log_tail(log_path)
            body = content.encode("utf-8", errors="replace")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if path.startswith("/stats"):
            import json, time
            stats = _collect_stats()
            body = json.dumps(stats, ensure_ascii=False).encode("utf-8")
            self._write_ok(body, "application/json; charset=utf-8")
            return
        if path.startswith("/dashboard-data"):
            import json, time
            try:
                snapshot = _dashboard_rows()
                rows = snapshot["rows"]
                cols = snapshot["columns"]
                ts = float(snapshot.get("ts", time.time()))
                payload = {
                    "rows": rows,
                    "columns": cols,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                    "count": len(rows),
                    "source": snapshot.get("source", "realtime"),
                    "error": snapshot.get("error"),
                }
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._write_ok(body, "application/json; charset=utf-8")
            except Exception as exc:
                err = f"failed to load dashboard data: {exc}"
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, err)
            return
        if path.startswith("/dashboard"):
            self._write_ok(_load_template("ops_realtime_dashboard.html").encode("utf-8"), "text/html; charset=utf-8")
            return
        if path.startswith("/stock-detail"):
            import json
            qs = parse_qs(query)
            symbol = (qs.get("symbol") or [""])[0].strip()
            if not symbol:
                self.send_error(HTTPStatus.BAD_REQUEST, "missing symbol")
                return
            try:
                detail = _load_stock_detail(symbol)
                body = json.dumps(detail, ensure_ascii=False).encode("utf-8")
                self._write_ok(body, "application/json; charset=utf-8")
            except Exception as exc:
                err = f"failed to load detail for {symbol}: {exc}"
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, err)
            return
        if path.startswith("/screener"):
            self._write_ok(_load_template("ops_screener.html").encode("utf-8"), "text/html; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        if not self._require_auth():
            return

        import json
        from urllib.parse import parse_qs, urlparse
        from http import HTTPStatus

        path = self.path
        parsed = urlparse(path)
        query = parsed.query

        from .ops_handlers import (
            _build_cmd, _purge_dead_processes, _screener_rows,
            _stop_action, _wipe_data, _wipe_dataset_cache,
            _wipe_feature_cache, _wipe_financial_data,
            _wipe_price_data,
        )
        from .screener import _criteria_to_str

        _purge_dead_processes()

        if path == "/run":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
                action = data.get("action")
                payload = data.get("payload") or {}
                cmd = _build_cmd(action, payload)
                from .ops_handlers import _start_process
                log_path = _start_process(cmd, action)
                resp_payload = {"status": "started", "cmd": cmd, "log": log_path}
                body = json.dumps(resp_payload, ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = f"Error: {exc}".encode("utf-8")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return
        if path.startswith("/stop"):
            qs = parse_qs(query)
            action = qs.get("action", [None])[0]
            if not action:
                self.send_error(HTTPStatus.BAD_REQUEST, "missing action")
                return
            msg = _stop_action(action)
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if path == "/wipe-data":
            msg = _wipe_data()
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if path == "/wipe-price":
            msg = _wipe_price_data()
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if path == "/wipe-financial":
            msg = _wipe_financial_data()
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if path == "/wipe-feature-cache":
            msg = _wipe_feature_cache()
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if path == "/wipe-dataset-cache":
            msg = _wipe_dataset_cache()
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if path == "/screener":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
                query_str = data.get("query", "")
                result = _screener_rows(query_str)
                rows_data = [
                    {
                        "symbol": r.symbol,
                        "name": r.name,
                        "lookback_pct_change": r.lookback_pct_change,
                        "last_close": r.last_close,
                        "last_date": r.last_date,
                        "roe": r.roe,
                        "eps": r.eps,
                        "turnover_rate": r.turnover_rate,
                    }
                    for r in (result.get("rows") or [])
                ]
                criteria = result.get("criteria")
                payload = {
                    "rows": rows_data,
                    "criteria_str": _criteria_to_str(criteria) if criteria else "",
                }
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._write_ok(body, "application/json; charset=utf-8")
            except Exception as exc:
                err = f"screener error: {exc}"
                body = json.dumps({"error": err}, ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


def run(server_address=("127.0.0.1", 8000)):
    from .ops_handlers import ensure_logs
    from http.server import ThreadingHTTPServer

    ensure_logs()
    httpd = ThreadingHTTPServer(server_address, DashboardHandler)
    print(f"Serving control panel on http://{server_address[0]}:{server_address[1]}")
    if OPS_DASHBOARD_TOKEN:
        print("WARNING: Authentication is ENABLED. Set Authorization header with Bearer token.")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
