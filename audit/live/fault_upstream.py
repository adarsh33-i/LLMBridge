"""Fault-injecting OpenAI-compatible upstream used by the live bridge tests.

Routes are prefixed so one process can act as several providers:

    POST /ok/v1/chat/completions      normal SSE stream or JSON completion
    POST /stall/v1/chat/completions   five chunks, then the socket dies
    POST /fail/v1/chat/completions    500 with an error body
    POST /slow/v1/chat/completions    three seconds of nothing, then a normal answer
    POST /tool/v1/chat/completions    JSON completion carrying a tool call
    POST /toolstall/v1/chat/completions  streams text, opens a tool call, dies mid-arguments
    POST /toolok/v1/chat/completions  streams one complete tool call (a fallback for toolstall)
    POST /restart/v1/chat/completions streams a continuation that opens like a fresh answer
    GET  /counts                      how many upstream calls each route saw

The stall route is the one that matters: it is what a provider does when it drops a
connection mid-answer, and it is the only input that can trigger a mid-stream heal.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COUNTS: Counter[str] = Counter()
LOCK = threading.Lock()

PARTIAL = [
    "The gateway receives the request in the format ",
    "the client already speaks, translates it, ",
    "and forwards it to the model. When the model ",
    "stops halfway through an answer, the bridge ",
    "keeps the text it has and asks the next ",
]
TAIL = [
    "model to continue from that exact point, ",
    "so the reader sees a single answer, ",
    "with no gap and no error message. ",
    "Each handover is written to the ledger, ",
    "where its join can be checked later.",
]


def sse(content: str, finish: str | None = None) -> bytes:
    payload = {
        "id": "chatcmpl-stub",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "stub-model",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def _route(self) -> str:
        return self.path.strip("/").split("/")[0]

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/counts":
            with LOCK:
                self._json(200, dict(COUNTS))
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        route = self._route()
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}
        with LOCK:
            COUNTS[route] += 1

        if route == "fail":
            self._json(500, {"error": {"message": "upstream exploded", "type": "server_error"}})
            return
        if route == "slow":
            time.sleep(3)
        if route == "tool":
            self._json(200, {
                "id": "chatcmpl-tool",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "stub-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Checking the file.",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
            })
            return

        if route == "restart" and body.get("stream"):
            # A fallback that ignores the prefill: the seam audit must flag this.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ("Sure, here's the rest: ", "the bridge keeps the text it has ", "and continues."):
                self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}, "finish_reason": None}]}).encode() + b"\n\n")
                self.wfile.flush(); time.sleep(0.03)
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 12}}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            self.close_connection = True
            return

        if route in ("toolstall", "toolok") and body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def tool_sse(index, call_id=None, name=None, arguments=None, finish=None):
                call = {"index": index, "function": {}}
                if call_id: call["id"] = call_id
                if name: call["function"]["name"] = name
                if arguments is not None: call["function"]["arguments"] = arguments
                return ("data: " + json.dumps({"choices": [{"delta": {"tool_calls": [call]}, "finish_reason": finish}]}) + "\n\n").encode()

            if route == "toolstall":
                self.wfile.write(b'data: ' + json.dumps({"choices": [{"delta": {"content": "Let me check that file. "}, "finish_reason": None}]}).encode() + b"\n\n")
                self.wfile.flush(); time.sleep(0.05)
                self.wfile.write(tool_sse(0, "call_stall", "read_file", '{"path": "/etc/ho'))
                self.wfile.flush(); time.sleep(0.05)
                self.close_connection = True
                try:
                    self.connection.close()
                except OSError:
                    pass
                return
            self.wfile.write(tool_sse(0, "call_fb", "read_file", '{"path": '))
            self.wfile.flush(); time.sleep(0.05)
            self.wfile.write(tool_sse(0, arguments='"/etc/hosts"}'))
            self.wfile.flush()
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 9}}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return

        if route == "echo":
            self._json(200, {
                "id": "chatcmpl-echo", "object": "chat.completion",
                "created": int(time.time()), "model": "stub-model",
                "choices": [{"index": 0, "message": {"role": "assistant",
                             "content": json.dumps(body)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
            return

        if not body.get("stream"):
            text = "".join(TAIL)
            if route == "stall":
                text = "".join(PARTIAL)
            self._json(200, {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "stub-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": len(text.split()),
                          "total_tokens": 20 + len(text.split())},
            })
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        chunks = PARTIAL if route == "stall" else TAIL
        if route == "repeat":
            # A fallback that re-emits the tail it was handed: the classic stitching
            # artifact the seam audit is meant to flag.
            chunks = [PARTIAL[-1]] + TAIL
        for piece in chunks:
            self.wfile.write(sse(piece))
            self.wfile.flush()
            time.sleep(0.05)
        if route == "stall":
            # Close mid-answer without a [DONE] sentinel.
            self.close_connection = True
            try:
                self.connection.close()
            except OSError:
                pass
            return
        self.wfile.write(sse("", "stop"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8790), Handler)
    server.daemon_threads = True
    print("fault upstream on 127.0.0.1:8790", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
