from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
REPLAY_DIR = ROOT / "replays"
HTML_PATH = ROOT / "dashboard" / "index.html"


def _load_run(path: Path) -> dict:
    rows = []
    final_payload = {}
    if not path.exists():
        return {"name": path.name, "rows": rows, "summary": {}}

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    days = []
    for row in rows:
        if row.get("kind") == "final_score":
            final_payload = row.get("final", {}) or {}
            continue
        obs = row.get("observation", {}) or {}
        day_result = row.get("day_result", {}) or {}
        cost_breakdown = obs.get("cost_breakdown", {}) or {}
        inventory = obs.get("inventory", []) or []
        expiring_soon = 0
        zeroish = 0
        total_inventory_kg = 0.0
        for inv in inventory:
            total_inventory_kg += float(inv.get("total_kg", 0) or 0)
            batches = inv.get("batches", []) or []
            soon = sum(
                float(b.get("quantity_kg", 0) or 0)
                for b in batches
                if int(b.get("expires_in_days", 9999) or 9999) <= 1
            )
            if soon > 0:
                expiring_soon += 1
            if float(inv.get("total_kg", 0) or 0) <= 0.05:
                zeroish += 1

        reviews = obs.get("recent_reviews", []) or []
        review_values = [float(r.get("stars", 0) or 0) for r in reviews if r.get("stars") is not None]
        avg_review = sum(review_values) / len(review_values) if review_values else 0.0
        pending_orders = obs.get("pending_orders", []) or []
        pending_qty = sum(float(po.get("quantity_kg", 0) or 0) for po in pending_orders)
        notes = str(obs.get("notes", "") or "")
        mode = ""
        if "mode=" in notes:
            mode = notes.split("mode=", 1)[1].split()[0]

        actions = row.get("actions", []) or []
        action_counts: dict[str, int] = {}
        for action in actions:
            tool = action.get("tool", "?")
            action_counts[tool] = action_counts.get(tool, 0) + 1

        days.append({
            "day": row.get("day"),
            "scenario": row.get("scenario"),
            "seed": row.get("seed"),
            "cash": float(obs.get("cash", 0) or 0),
            "days_remaining": int(obs.get("days_remaining", 0) or 0),
            "reputation_band": obs.get("reputation_band", ""),
            "customer_trend": obs.get("customer_trend", ""),
            "weather_today": obs.get("weather_today", ""),
            "staff_level": int(obs.get("staff_level", 0) or 0),
            "active_menu_count": len(obs.get("active_menu", []) or []),
            "pending_order_count": len(pending_orders),
            "pending_order_qty": pending_qty,
            "alerts": obs.get("alerts", []) or [],
            "alert_count": len(obs.get("alerts", []) or []),
            "mode": mode,
            "covers": int(day_result.get("total_covers", 0) or 0),
            "revenue": float(
                day_result.get("total_revenue", day_result.get("revenue", 0)) or 0
            ),
            "total_costs": float(
                obs.get("yesterday_total_costs", day_result.get("total_costs", 0)) or 0
            ),
            "staff_cost": float(cost_breakdown.get("staff", 0) or 0),
            "fixed_cost": float(cost_breakdown.get("fixed", 0) or 0),
            "marketing_cost": float(cost_breakdown.get("marketing", 0) or 0),
            "waste_cost": float(cost_breakdown.get("waste", 0) or 0),
            "walkout_band": day_result.get("walkout_band", "n/a"),
            "avg_wait_minutes": float(day_result.get("avg_wait_minutes", 0) or 0),
            "peak_wait_minutes": float(day_result.get("peak_wait_minutes", 0) or 0),
            "table_utilization_peak": float(day_result.get("table_utilization_peak", 0) or 0),
            "substitution_count": int(day_result.get("substitution_count", day_result.get("substitutions", 0)) or 0),
            "stockouts": sorted((day_result.get("dishes_unavailable_at", {}) or {}).keys()),
            "action_counts": action_counts,
            "expiring_soon_ingredients": expiring_soon,
            "zero_inventory_ingredients": zeroish,
            "total_inventory_kg": total_inventory_kg,
            "avg_review": avg_review,
        })

    summary = {}
    if days:
        last = days[-1]
        score_block = final_payload.get("score", {}) if isinstance(final_payload.get("score"), dict) else {}
        summary = {
            "scenario": last["scenario"],
            "seed": last["seed"],
            "last_day": last["day"],
            "cash": last["cash"],
            "reputation_band": last["reputation_band"],
            "covers": last["covers"],
            "revenue": last["revenue"],
            "walkout_band": last["walkout_band"],
            "pending_order_count": last["pending_order_count"],
            "pending_order_qty": last["pending_order_qty"],
            "staff_level": last["staff_level"],
            "active_menu_count": last["active_menu_count"],
            "total_costs": last["total_costs"],
            "alert_count": last["alert_count"],
            "mode": last["mode"],
            "final_score": score_block.get("total_score", final_payload.get("total_score")),
            "net_profit": score_block.get("net_profit", final_payload.get("net_profit")),
            "walkout_penalty": score_block.get("walkout_penalty", final_payload.get("walkout_penalty")),
            "reputation_penalty": score_block.get("reputation_penalty", final_payload.get("reputation_penalty")),
            "waste_penalty": score_block.get("waste_penalty", final_payload.get("waste_penalty")),
        }

    return {"name": path.name, "rows": days, "summary": summary, "final": final_payload}


def _latest_run() -> Path | None:
    if not REPLAY_DIR.exists():
        return None
    files = sorted(REPLAY_DIR.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        if path.stat().st_size > 0:
            return path
    return files[0] if files else None


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PATH.read_text(encoding="utf-8"))
            return

        if parsed.path == "/api/runs":
            REPLAY_DIR.mkdir(parents=True, exist_ok=True)
            runs = sorted(
                REPLAY_DIR.glob("run_*.jsonl"),
                key=lambda p: (p.stat().st_size > 0, p.stat().st_mtime),
                reverse=True,
            )
            self._send_json({
                "runs": [
                    {"name": p.name, "mtime": p.stat().st_mtime, "size": p.stat().st_size}
                    for p in runs
                ]
            })
            return

        if parsed.path == "/api/run":
            query = parse_qs(parsed.query)
            name = query.get("name", [None])[0]
            path = REPLAY_DIR / name if name else _latest_run()
            if path is None or not path.exists():
                self._send_json({"error": "No replay found"}, status=404)
                return
            self._send_json(_load_run(path))
            return

        self._send_json({"error": "Not found"}, status=404)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "8765"))
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Dashboard running at http://{host}:{port}")
    print(f"Watching replay dir: {REPLAY_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
