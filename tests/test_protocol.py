import base64
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from llmbridge.protocol import (
    ProtocolError,
    anthropic_to_openai,
    openai_to_anthropic,
    stream_events,
)


class ProtocolTests(unittest.TestCase):
    def test_translates_messages_tools_and_model_override(self):
        request = anthropic_to_openai(
            {
                "model": "claude-client-model",
                "max_tokens": 200,
                "system": "Be concise",
                "messages": [
                    {"role": "user", "content": "Find a file"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Choose the matching tool"},
                            {
                                "type": "tool_use",
                                "id": "call-1",
                                "name": "find",
                                "input": {"path": "."},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-1",
                                "content": "Found it",
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "name": "find",
                        "description": "Find files",
                        "input_schema": {"type": "object"},
                    }
                ],
                "stream": True,
            },
            "nemotron",
        )
        self.assertEqual(request["model"], "nemotron")
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertEqual(request["messages"][2]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(request["messages"][3]["tool_call_id"], "call-1")
        self.assertEqual(request["tools"][0]["function"]["name"], "find")
        self.assertEqual(request["stream_options"], {"include_usage": True})

    def test_translates_assistant_tool_call_response(self):
        result = openai_to_anthropic(
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-2",
                                    "function": {
                                        "name": "find",
                                        "arguments": '{"path":"."}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            "nemotron",
        )
        self.assertEqual(result["stop_reason"], "tool_use")
        self.assertEqual(result["content"][0]["input"], {"path": "."})
        self.assertEqual(result["usage"]["input_tokens"], 10)

    def test_rejects_missing_max_tokens(self):
        with self.assertRaises(ProtocolError):
            anthropic_to_openai({"model": "nemotron", "messages": []})

    def test_translates_plain_text_document_block(self):
        request = anthropic_to_openai(
            {
                "model": "client-model",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "text",
                                    "media_type": "text/plain",
                                    "data": "The policy covers fire damage.",
                                },
                                "title": "Policy",
                                "context": "Customer's policy document",
                            },
                            {"type": "text", "text": "What does it cover?"},
                        ],
                    }
                ],
            }
        )

        content = request["messages"][0]["content"]
        self.assertIn("Document title: Policy", content[0]["text"])
        self.assertIn("The policy covers fire damage.", content[0]["text"])
        self.assertEqual(content[1]["text"], "What does it cover?")

    def test_translates_base64_text_document(self):
        encoded = base64.b64encode("col1,col2\n1,2".encode()).decode()
        request = anthropic_to_openai(
            {
                "model": "client-model",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "text/csv",
                                    "data": encoded,
                                },
                            }
                        ],
                    }
                ],
            }
        )

        self.assertIn("col1,col2\n1,2", request["messages"][0]["content"][0]["text"])

    def test_extracts_text_from_base64_pdf_document(self):
        encoded = base64.b64encode(b"%PDF-test").decode()
        reader = SimpleNamespace(
            is_encrypted=False,
            pages=[
                SimpleNamespace(extract_text=lambda: "First page"),
                SimpleNamespace(extract_text=lambda: "Second page"),
            ],
        )
        with patch("llmbridge.protocol.PdfReader", return_value=reader):
            request = anthropic_to_openai(
                {
                    "model": "client-model",
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": encoded,
                                    },
                                }
                            ],
                        }
                    ],
                }
            )

        text = request["messages"][0]["content"][0]["text"]
        self.assertIn("[Page 1]\nFirst page", text)
        self.assertIn("[Page 2]\nSecond page", text)

    def test_extracts_text_from_docx_document(self):
        document = io.BytesIO()
        with ZipFile(document, "w") as archive:
            archive.writestr(
                "word/document.xml",
                (
                    '<w:document xmlns:w="urn:test"><w:body><w:p>'
                    '<w:r><w:t>Policy text</w:t></w:r>'
                    "</w:p></w:body></w:document>"
                ),
            )
        encoded = base64.b64encode(document.getvalue()).decode()

        request = anthropic_to_openai(
            {
                "model": "client-model",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": (
                                        "application/vnd.openxmlformats-officedocument."
                                        "wordprocessingml.document"
                                    ),
                                    "data": encoded,
                                },
                            }
                        ],
                    }
                ],
            }
        )

        self.assertIn("Policy text", request["messages"][0]["content"][0]["text"])

    def test_rejects_office_documents_above_expanded_size_limit(self):
        document = io.BytesIO()
        with ZipFile(document, "w") as archive:
            archive.writestr("word/document.xml", "<document>content</document>")
        encoded = base64.b64encode(document.getvalue()).decode()

        with patch("llmbridge.protocol._MAX_OFFICE_DOCUMENT_BYTES", 8):
            with self.assertRaisesRegex(ProtocolError, "extraction limit"):
                anthropic_to_openai(
                    {
                        "model": "client-model",
                        "max_tokens": 100,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "document",
                                        "source": {
                                            "type": "base64",
                                            "media_type": (
                                                "application/vnd.openxmlformats-officedocument."
                                                "wordprocessingml.document"
                                            ),
                                            "data": encoded,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                )

    def test_extracts_text_from_xlsx_document(self):
        document = io.BytesIO()
        with ZipFile(document, "w") as archive:
            archive.writestr(
                "xl/sharedStrings.xml",
                (
                    '<sst xmlns="urn:test"><si><t>Name</t></si>'
                    "<si><t>Alice</t></si></sst>"
                ),
            )
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                (
                    '<worksheet xmlns="urn:test"><sheetData><row>'
                    '<c r="A1" t="s"><v>0</v></c>'
                    '<c r="B1" t="s"><v>1</v></c>'
                    "</row></sheetData></worksheet>"
                ),
            )
        encoded = base64.b64encode(document.getvalue()).decode()

        request = anthropic_to_openai(
            {
                "model": "client-model",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": (
                                        "application/vnd.openxmlformats-officedocument."
                                        "spreadsheetml.sheet"
                                    ),
                                    "data": encoded,
                                },
                            }
                        ],
                    }
                ],
            }
        )

        self.assertIn("Name\tAlice", request["messages"][0]["content"][0]["text"])

    def test_extracts_text_from_pptx_document(self):
        document = io.BytesIO()
        with ZipFile(document, "w") as archive:
            archive.writestr(
                "ppt/slides/slide1.xml",
                (
                    '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>'
                    "Quarterly results</a:t></p:sld>"
                ),
            )
        encoded = base64.b64encode(document.getvalue()).decode()

        request = anthropic_to_openai(
            {
                "model": "client-model",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": (
                                        "application/vnd.openxmlformats-officedocument."
                                        "presentationml.presentation"
                                    ),
                                    "data": encoded,
                                },
                            }
                        ],
                    }
                ],
            }
        )

        self.assertIn(
            "Quarterly results",
            request["messages"][0]["content"][0]["text"],
        )

    def test_extracts_html_and_rtf_as_text(self):
        for media_type, data, expected in (
            ("text/html", "<p>Visible</p><script>ignore</script>", "Visible"),
            ("application/rtf", r"{\rtf1\ansi Hello}", "Hello"),
        ):
            with self.subTest(media_type=media_type):
                encoded = base64.b64encode(data.encode()).decode()
                request = anthropic_to_openai(
                    {
                        "model": "client-model",
                        "max_tokens": 100,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "document",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": encoded,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                )
                self.assertIn(
                    expected,
                    request["messages"][0]["content"][0]["text"],
                )

    def test_rejects_unsupported_document_sources_and_media_types(self):
        for source in (
            {"type": "url", "url": "https://example.test/file.pdf"},
            {"type": "file", "file_id": "file-123"},
            {
                "type": "base64",
                "media_type": "application/octet-stream",
                "data": base64.b64encode(b"bytes").decode(),
            },
        ):
            with self.subTest(source_type=source["type"], media_type=source.get("media_type")):
                with self.assertRaises(ProtocolError):
                    anthropic_to_openai(
                        {
                            "model": "client-model",
                            "max_tokens": 100,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "document", "source": source}
                                    ],
                                }
                            ],
                        }
                    )

    def test_rejects_invalid_base64_document(self):
        with self.assertRaisesRegex(ProtocolError, "valid base64"):
            anthropic_to_openai(
                {
                    "model": "client-model",
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "text/plain",
                                        "data": "not base64!",
                                    },
                                }
                            ],
                        }
                    ],
                }
            )

    def test_translates_streamed_text_and_finish_reason(self):
        upstream_events = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ]
        events = b"".join(stream_events(upstream_events, "nemotron")).decode()
        self.assertIn("event: message_start", events)
        self.assertIn('"type": "text_delta"', events)
        self.assertIn('"text": "Hello"', events)
        self.assertIn('"stop_reason": "end_turn"', events)
        self.assertIn("event: message_stop", events)

    def test_rejects_stream_ending_without_finish_reason(self):
        upstream_events = [
            b'data: {"choices":[{"delta":{"content":"Partial"},"finish_reason":null}]}\n',
            b"data: [DONE]\n",
        ]
        events = stream_events(upstream_events, "nemotron")
        next(events)
        next(events)
        next(events)
        with self.assertRaisesRegex(ProtocolError, "before a completion reason"):
            next(events)

    def test_translates_streamed_tool_call_across_chunks(self):
        upstream_events = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"find","arguments":"{\\\"path\\\":"}}]},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\\".\\\"}"}}]},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
            b"data: [DONE]\n",
        ]
        events = b"".join(stream_events(upstream_events, "nemotron")).decode()
        payloads = [
            json.loads(line[6:])
            for line in events.splitlines()
            if line.startswith("data: ")
        ]
        tool_start = next(item for item in payloads if item.get("type") == "content_block_start")
        partial_json = "".join(
            item["delta"]["partial_json"]
            for item in payloads
            if item.get("type") == "content_block_delta"
        )
        message_delta = next(item for item in payloads if item.get("type") == "message_delta")
        self.assertEqual(tool_start["content_block"]["id"], "call-1")
        self.assertEqual(tool_start["content_block"]["name"], "find")
        self.assertEqual(json.loads(partial_json), {"path": "."})
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_buffers_tool_arguments_until_id_and_name_arrive(self):
        upstream_events = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"arguments":"{\\\"x\\\":"}}]},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"find","arguments":"1}"}}]},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
            b"data: [DONE]\n",
        ]
        events = b"".join(stream_events(upstream_events, "nemotron")).decode()
        payloads = [
            json.loads(line[6:])
            for line in events.splitlines()
            if line.startswith("data: ")
        ]
        tool_start = next(
            item for item in payloads if item.get("type") == "content_block_start"
        )
        partial_json = "".join(
            item["delta"]["partial_json"]
            for item in payloads
            if item.get("type") == "content_block_delta"
        )
        self.assertEqual(tool_start["content_block"]["name"], "find")
        self.assertEqual(json.loads(partial_json), {"x": 1})


if __name__ == "__main__":
    unittest.main()
