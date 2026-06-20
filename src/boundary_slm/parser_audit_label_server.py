from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlencode

from boundary_slm.parser_audit_labeler import (
    VALID_OPTIONS,
    apply_label,
    completed,
    next_indices,
    progress,
    read_csv,
    render_row,
    write_csv_atomic,
    write_progress,
)


def page(title: str, body: str) -> bytes:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.45;margin:32px;max-width:1100px}"
        ".meta{background:#f6f8fa;border:1px solid #d0d7de;padding:16px;border-radius:6px}"
        ".excerpt{white-space:pre-wrap;border:1px solid #d0d7de;padding:18px;border-radius:6px;background:#fff}"
        ".buttons button{margin:4px;padding:10px 14px;font-size:16px}"
        "textarea{width:100%;min-height:72px} code{background:#f6f8fa;padding:2px 4px}"
        "</style></head><body>"
        f"{body}</body></html>"
    ).encode("utf-8")


def row_html(row: dict[str, str], index: int, total: int, status: dict[str, Any]) -> str:
    rendered = render_row(row, index, total).split("\n")
    meta = "\n".join(html.escape(line) for line in rendered[:9])
    excerpt = html.escape("\n".join(rendered[9:]).strip())
    option_buttons = "\n".join(
        f"<button type='submit' name='answer' value='{letter}'>{letter}</button>" for letter in sorted(VALID_OPTIONS)
    )
    return f"""
    <h1>Parser Audit Labeler</h1>
    <p><a href='/progress'>Progress JSON</a></p>
    <div class='meta'><pre>{meta}</pre></div>
    <h2>Response Excerpt</h2>
    <div class='excerpt'>{excerpt}</div>
    <h2>Human Label</h2>
    <form method='post' action='/label'>
      <input type='hidden' name='row_index' value='{index}'>
      <label>Notes, optional</label><br>
      <textarea name='notes'></textarea>
      <div class='buttons'>
        {option_buttons}
        <button type='submit' name='answer' value='U'>Unanswered</button>
        <button type='submit' name='answer' value='S'>Skip</button>
      </div>
    </form>
    <p>
      Completed: {status['completed_rows']} / {status['total_rows']};
      high-risk completed: {status['completed_high_risk_rows']} / {status['high_risk_rows']}.
    </p>
    """


def complete_html(status: dict[str, Any]) -> str:
    return f"""
    <h1>Parser Audit Complete</h1>
    <p>All first-pass parser-audit rows are labeled.</p>
    <pre>{html.escape(json.dumps(status, indent=2, sort_keys=True))}</pre>
    <p>Next commands:</p>
    <pre>PYTHONPATH=main/src python3 -m boundary_slm.parser_audit_impact
PYTHONPATH=main/src python3 -m boundary_slm.mmlu_scoring_robustness</pre>
    """


def make_handler(sample_csv: Path, progress_json: Path, high_risk_first: bool) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, *, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            rows = read_csv(sample_csv)
            status = progress(rows)
            if self.path.startswith("/progress"):
                self._send(json.dumps(status, indent=2, sort_keys=True).encode("utf-8"), content_type="application/json")
                return
            indices = next_indices(rows, high_risk_first=high_risk_first, limit=1)
            if not indices:
                self._send(page("Parser Audit Complete", complete_html(status)))
                return
            index = indices[0]
            self._send(page("Parser Audit Labeler", row_html(rows[index], index, len(rows), status)))

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/label":
                self._send(page("Not Found", "<h1>Not found</h1>"), status=404)
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            payload = self.rfile.read(length).decode("utf-8")
            fields = parse_qs(payload)
            rows = read_csv(sample_csv)
            index = int(fields.get("row_index", ["-1"])[0])
            answer = fields.get("answer", [""])[0].strip().upper()
            notes = fields.get("notes", [""])[0]
            if index < 0 or index >= len(rows):
                self._send(page("Bad Request", "<h1>Bad row index</h1>"), status=400)
                return
            if answer == "S":
                self.send_response(303)
                self.send_header("Location", "/")
                self.end_headers()
                return
            try:
                if answer == "U":
                    rows[index] = apply_label(rows[index], human_prediction="", human_answered=False, human_notes=notes)
                elif answer in VALID_OPTIONS:
                    rows[index] = apply_label(rows[index], human_prediction=answer, human_answered=True, human_notes=notes)
                else:
                    raise ValueError("answer must be A-J, U, or S")
            except ValueError as exc:
                self._send(page("Bad Request", f"<h1>Bad label</h1><p>{html.escape(str(exc))}</p>"), status=400)
                return
            write_csv_atomic(sample_csv, rows)
            write_progress(progress_json, rows)
            self.send_response(303)
            self.send_header("Location", "/" + ("?" + urlencode({"saved": index}) if index >= 0 else ""))
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def serve(
    *,
    sample_csv: Path,
    progress_json: Path,
    host: str,
    port: int,
    high_risk_first: bool = True,
) -> None:
    if not sample_csv.exists():
        raise FileNotFoundError(sample_csv)
    handler = make_handler(sample_csv, progress_json, high_risk_first)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Serving parser-audit labeler at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local browser UI for first-pass parser-audit labeling.")
    parser.add_argument("--sample-csv", type=Path, default=Path("main/analysis/parser_audit/parser_audit_sample.csv"))
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=Path("main/analysis/parser_audit/parser_audit_interactive_progress.json"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--original-order", action="store_true", help="Do not prioritize high-risk rows first.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    serve(
        sample_csv=args.sample_csv,
        progress_json=args.progress_json,
        host=args.host,
        port=args.port,
        high_risk_first=not args.original_order,
    )


if __name__ == "__main__":
    main()
