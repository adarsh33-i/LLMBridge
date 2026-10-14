import base64
import importlib.util
import io
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from llmbridge.protocol import (
    ProtocolError,
    _ocr_pdf_page,
    anthropic_to_openai,
    openai_to_anthropic,
    stream_events,
)


class ProtocolTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("pymupdf"),
        "PyMuPDF is only installed with the optional OCR extra",
    )
    def test_ocr_renderer_rasterizes_pdf_page_for_tesseract(self):
        import pymupdf

        document = pymupdf.open()
        document.new_page()
        pdf_bytes = document.tobytes()
        document.close()
        tesseract = SimpleNamespace(returncode=0, stdout=b"Recognized text")

        with patch("llmbridge.protocol.subprocess.run", return_value=tesseract) as run:
            text = _ocr_pdf_page(pdf_bytes, 0, time.monotonic() + 30)

        self.assertEqual(text, "Recognized text")
        self.assertEqual(run.call_args.kwargs["timeout"], 20)
        self.assertEqual(run.call_args.args[0], ["tesseract", "stdin", "stdout"])

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

    def test_rejects_assistant_reasoning_and_signature_blocks_explicitly(self):
        for block_type in ("thinking", "redacted_thinking", "signature"):
            with self.subTest(block_type=block_type):
                with self.assertRaisesRegex(ProtocolError, block_type):
                    anthropic_to_openai(
                        {
                            "model": "client-model",
                            "max_tokens": 100,
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": block_type,
                                            "thinking": "private reasoning",
                                            "signature": "opaque-signature",
                                            "data": "opaque-data",
                                        }
                                    ],
                                },
                                {"role": "user", "content": "Continue"},
                            ],
                        }
                    )

    def test_preserves_text_citation_details_as_inline_annotations(self):
        request = anthropic_to_openai(
            {
                "model": "client-model",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "The policy covers fire damage.",
                                "citations": [
                                    {
                                        "type": "page_location",
                                        "document_title": "Policy",
                                        "start_page_number": 2,
                                        "end_page_number": 3,
                                        "cited_text": "Coverage includes fire damage.",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )

        text = request["messages"][0]["content"][0]["text"]
        self.assertIn("[Citation: document: Policy; page: 2; end page: 3", text)
        self.assertIn('quoted text: "Coverage includes fire damage."', text)

    def test_marks_tool_result_errors_in_upstream_content(self):
        request = anthropic_to_openai(
            {
                "model": "client-model",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-1",
                                "is_error": True,
                                "content": "Permission denied",
                            }
                        ],
                    }
                ],
            }
        )

        self.assertEqual(
            request["messages"][0]["content"],
            "Tool returned an error:\nPermission denied",
        )

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

    def test_rejects_unknown_upstream_finish_reason(self):
        with self.assertRaisesRegex(ProtocolError, "Unsupported upstream completion reason"):
            openai_to_anthropic(
                {
                    "choices": [
                        {
                            "message": {"content": "Filtered"},
                            "finish_reason": "content_filter",
                        }
                    ]
                },
                "nemotron",
            )

    def test_rejects_unknown_stream_finish_reason(self):
        events = [
            b'data: {"choices":[{"delta":{},"finish_reason":"content_filter"}]}\n',
        ]
        with self.assertRaisesRegex(ProtocolError, "Unsupported upstream completion reason"):
            list(stream_events(events, "nemotron"))

    def test_rejects_missing_max_tokens(self):
        with self.assertRaises(ProtocolError):
            anthropic_to_openai({"model": "nemotron", "messages": []})

    def test_ignores_cache_control_metadata_with_warning(self):
        request = {
            "model": "client-model",
            "max_tokens": 100,
            "cache_control": {"type": "ephemeral"},
            "system": [
                {"type": "text", "text": "Rules", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Question",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Tool output",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "lookup",
                    "cache_control": {"type": "ephemeral"},
                    "input_schema": {
                        "type": "object",
                        "properties": {"cache_control": {"type": "string"}},
                    },
                }
            ],
        }

        with self.assertLogs("llmbridge.protocol", level="WARNING") as logs:
            translated = anthropic_to_openai(request)

        self.assertIn("prompt caching is disabled", logs.output[0])
        self.assertNotIn("cache_control", translated)
        self.assertNotIn("cache_control", translated["messages"][0]["content"][0])
        tool_message = next(
            message for message in translated["messages"] if message["role"] == "tool"
        )
        self.assertNotIn("cache_control", json.dumps(tool_message))
        self.assertNotIn("cache_control", translated["tools"][0])
        self.assertEqual(
            translated["tools"][0]["function"]["parameters"]["properties"][
                "cache_control"
            ],
            {"type": "string"},
        )
        self.assertIn(
            "cache_control",
            request["messages"][0]["content"][0],
            "Converting a request must not mutate the caller's object",
        )

    def test_does_not_warn_when_cache_control_is_a_schema_property(self):
        request = {
            "model": "client-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Question"}],
            "tools": [
                {
                    "name": "lookup",
                    "input_schema": {
                        "type": "object",
                        "properties": {"cache_control": {"type": "string"}},
                    },
                }
            ],
        }

        with self.assertNoLogs("llmbridge.protocol", level="WARNING"):
            translated = anthropic_to_openai(request)

        self.assertIn(
            "cache_control",
            translated["tools"][0]["function"]["parameters"]["properties"],
        )

    def test_accepts_cache_control_property_inside_tool_schema(self):
        request = anthropic_to_openai(
            {
                "model": "client-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Question"}],
                "tools": [
                    {
                        "name": "lookup",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "cache_control": {"type": "string"}
                            },
                        },
                    }
                ],
            }
        )

        self.assertEqual(
            request["tools"][0]["function"]["parameters"]["properties"][
                "cache_control"
            ],
            {"type": "string"},
        )

    def test_maps_enabled_thinking_to_configured_upstream_effort(self):
        request = anthropic_to_openai(
            {
                "model": "client-model",
                "max_tokens": 100,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": "Think this through"}],
            },
            reasoning_effort="high",
        )

        self.assertEqual(request["reasoning_effort"], "high")

    def test_rejects_enabled_thinking_without_route_mapping(self):
        with self.assertRaisesRegex(ProtocolError, "no configured reasoning_effort"):
            anthropic_to_openai(
                {
                    "model": "client-model",
                    "max_tokens": 100,
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "messages": [{"role": "user", "content": "Think this through"}],
                }
            )

    def test_rejects_images_when_route_declares_no_image_support(self):
        with self.assertRaisesRegex(ProtocolError, "does not support images"):
            anthropic_to_openai(
                {
                    "model": "client-model",
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "https://example.test/image.png",
                                    },
                                }
                            ],
                        }
                    ],
                },
                supports_images=False,
            )

    def test_rejects_tool_use_when_route_declares_no_tool_support(self):
        with self.assertRaisesRegex(ProtocolError, "does not support tools"):
            anthropic_to_openai(
                {
                    "model": "client-model",
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "call-1",
                                    "name": "lookup",
                                    "input": {},
                                }
                            ],
                        }
                    ],
                },
                supports_tools=False,
            )

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

    def test_uses_ocr_for_textless_pdf_pages_when_enabled(self):
        encoded = base64.b64encode(b"%PDF-test").decode()
        reader = SimpleNamespace(
            is_encrypted=False,
            pages=[SimpleNamespace(extract_text=lambda: "")],
        )
        with (
            patch.dict(os.environ, {"LLMBRIDGE_ENABLE_OCR": "1"}),
            patch("llmbridge.protocol.PdfReader", return_value=reader),
            patch("llmbridge.protocol._ocr_pdf_page", return_value="Recognized text") as ocr,
        ):
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

        ocr.assert_called_once()
        self.assertIn(
            "[Page 1]\nRecognized text",
            request["messages"][0]["content"][0]["text"],
        )

    def test_limits_ocr_pages_per_pdf(self):
        encoded = base64.b64encode(b"%PDF-test").decode()
        reader = SimpleNamespace(
            is_encrypted=False,
            pages=[SimpleNamespace(extract_text=lambda: "") for _ in range(11)],
        )
        with (
            patch.dict(os.environ, {"LLMBRIDGE_ENABLE_OCR": "1"}),
            patch("llmbridge.protocol.PdfReader", return_value=reader),
            patch("llmbridge.protocol._ocr_pdf_page", return_value="Recognized text"),
        ):
            with self.assertRaisesRegex(ProtocolError, "limited to 10 scanned pages"):
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
                                            "media_type": "application/pdf",
                                            "data": encoded,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                )

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

    def test_preserves_docx_paragraph_boundaries(self):
        document = io.BytesIO()
        with ZipFile(document, "w") as archive:
            archive.writestr(
                "word/document.xml",
                (
                    '<w:document xmlns:w="urn:test"><w:body>'
                    "<w:p><w:r><w:t>First paragraph.</w:t></w:r></w:p>"
                    "<w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p>"
                    "</w:body></w:document>"
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

        self.assertIn(
            "First paragraph.\nSecond paragraph.",
            request["messages"][0]["content"][0]["text"],
        )

    def test_preserves_docx_table_rows_and_cell_boundaries(self):
        document = io.BytesIO()
        with ZipFile(document, "w") as archive:
            archive.writestr(
                "word/document.xml",
                (
                    '<w:document xmlns:w="urn:test"><w:body>'
                    "<w:p><w:r><w:t>Summary</w:t></w:r></w:p>"
                    "<w:tbl><w:tr>"
                    "<w:tc><w:p><w:r><w:t>Item</w:t></w:r></w:p></w:tc>"
                    "<w:tc><w:p><w:r><w:t>Amount</w:t></w:r></w:p></w:tc>"
                    "</w:tr><w:tr>"
                    "<w:tc><w:p><w:r><w:t>Books</w:t></w:r></w:p></w:tc>"
                    "<w:tc><w:p><w:r><w:t>12</w:t></w:r></w:p></w:tc>"
                    "</w:tr></w:tbl>"
                    "</w:body></w:document>"
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

        text = request["messages"][0]["content"][0]["text"]
        self.assertIn("Summary\nItem\tAmount\nBooks\t12", text)

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
                    '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:p><a:r><a:t>'
                    "Quarterly results</a:t></a:r></a:p></p:sld>"
                ),
            )
            archive.writestr(
                "ppt/slides/slide2.xml",
                (
                    '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:p><a:r><a:t>'
                    "Forecast</a:t></a:r></a:p></p:sld>"
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
            "[Slide 1]\nQuarterly results",
            request["messages"][0]["content"][0]["text"],
        )
        self.assertIn(
            "[Slide 2]\nForecast",
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

    def test_preserves_html_heading_and_table_structure(self):
        encoded = base64.b64encode(
            "<h2>Results</h2><table><tr><th>Name</th><th>Score</th></tr>"
            "<tr><td>Ada</td><td>10</td></tr></table>".encode()
        ).decode()
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
                                    "media_type": "text/html",
                                    "data": encoded,
                                },
                            }
                        ],
                    }
                ],
            }
        )

        text = request["messages"][0]["content"][0]["text"]
        self.assertIn("Results", text)
        self.assertIn("Name\tScore", text)
        self.assertIn("Ada\t10", text)

    def test_supports_url_document_source(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.getheader.side_effect = lambda name, default=None: {
            "Content-Type": "text/plain",
            "Content-Length": None,
        }.get(name, default)
        mock_response.read.return_value = b"Remote file content"
        mock_connection = MagicMock()
        mock_connection.getresponse.return_value = mock_response
        public_dns = [
            (2, 1, 6, "", ("93.184.216.34", 80)),
        ]
        with (
            patch("llmbridge.protocol.socket.getaddrinfo", return_value=public_dns),
            patch(
                "llmbridge.protocol._PinnedHTTPConnection",
                return_value=mock_connection,
            ) as connection,
        ):
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
                                        "type": "url",
                                        "url": "http://example.test/file.txt",
                                        "media_type": "text/plain",
                                    },
                                }
                            ],
                        }
                    ],
                }
            )

        connection.assert_called_once_with("example.test", 80, "93.184.216.34")
        mock_connection.request.assert_called_once_with(
            "GET", "/file.txt", headers={"Host": "example.test", "Connection": "close"}
        )
        self.assertIn("Remote file content", request["messages"][0]["content"][0]["text"])

    def test_rejects_oversized_url_document(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.getheader.side_effect = lambda name, default=None: default
        mock_response.read.return_value = b"too large"
        mock_connection = MagicMock()
        mock_connection.getresponse.return_value = mock_response
        public_dns = [
            (2, 1, 6, "", ("93.184.216.34", 80)),
        ]
        with (
            patch("llmbridge.protocol._MAX_REMOTE_DOCUMENT_BYTES", 4),
            patch("llmbridge.protocol.socket.getaddrinfo", return_value=public_dns),
            patch(
                "llmbridge.protocol._PinnedHTTPConnection",
                return_value=mock_connection,
            ),
        ):
            with self.assertRaisesRegex(ProtocolError, "10 MiB download limit"):
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
                                            "type": "url",
                                            "url": "http://example.test/file.txt",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                )

    def test_rejects_private_url_document_destination(self):
        with (
            patch(
                "llmbridge.protocol.socket.getaddrinfo",
                return_value=[(2, 1, 6, "", ("127.0.0.1", 80))],
            ),
            patch("llmbridge.protocol._PinnedHTTPConnection") as connection,
        ):
            with self.assertRaisesRegex(ProtocolError, "public internet"):
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
                                            "type": "url",
                                            "url": "http://example.test/file.txt",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                )

        connection.assert_not_called()

    def test_rejects_redirect_to_private_url_document_destination(self):
        response = MagicMock()
        response.status = 302
        response.getheader.return_value = "http://127.0.0.1/private"
        connection_instance = MagicMock()
        connection_instance.getresponse.return_value = response
        public_dns = [(2, 1, 6, "", ("93.184.216.34", 80))]
        private_dns = [(2, 1, 6, "", ("127.0.0.1", 80))]
        with (
            patch(
                "llmbridge.protocol.socket.getaddrinfo",
                side_effect=[public_dns, private_dns],
            ),
            patch(
                "llmbridge.protocol._PinnedHTTPConnection",
                return_value=connection_instance,
            ) as connection,
        ):
            with self.assertRaisesRegex(ProtocolError, "public internet"):
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
                                            "type": "url",
                                            "url": "http://example.test/file.txt",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                )

        connection.assert_called_once()

    def test_supports_file_document_source_from_local_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "document.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("Local file content")
            with patch.dict(os.environ, {"LLMBRIDGE_DOCUMENT_ROOTS": directory}):
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
                                            "type": "file",
                                            "file_id": path,
                                            "media_type": "text/plain",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                )

        self.assertIn("Local file content", request["messages"][0]["content"][0]["text"])

    def test_rejects_local_document_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as handle:
                handle.write("Private file content")
                handle.flush()
                with patch.dict(os.environ, {"LLMBRIDGE_DOCUMENT_ROOTS": directory}):
                    with self.assertRaisesRegex(ProtocolError, "outside"):
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
                                                    "type": "file",
                                                    "file_id": handle.name,
                                                },
                                            }
                                        ],
                                    }
                                ],
                            }
                        )

    def test_rejects_local_document_symlink_escape(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            target = os.path.join(outside, "secret.txt")
            link = os.path.join(allowed, "linked.txt")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("Private file content")
            os.symlink(target, link)
            with patch.dict(os.environ, {"LLMBRIDGE_DOCUMENT_ROOTS": allowed}):
                with self.assertRaisesRegex(ProtocolError, "outside"):
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
                                                "type": "file",
                                                "file_id": link,
                                            },
                                        }
                                    ],
                                }
                            ],
                        }
                    )

    def test_local_document_read_errors_are_protocol_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, "missing.txt")
            with patch.dict(os.environ, {"LLMBRIDGE_DOCUMENT_ROOTS": directory}):
                with self.assertRaisesRegex(ProtocolError, "resolve"):
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
                                                "type": "file",
                                                "file_id": missing,
                                            },
                                        }
                                    ],
                                }
                            ],
                        }
                    )

    def test_resolves_anthropic_managed_file_id(self):
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.headers.get.return_value = None
        mock_response.headers.get_content_type.return_value = "text/plain"
        mock_response.read.return_value = b"Anthropic file content"
        with (
            patch.dict(os.environ, {"LLMBRIDGE_ANTHROPIC_FILES_API_KEY": "secret"}),
            patch("llmbridge.protocol.urlopen", return_value=mock_response) as fetch,
        ):
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
                                        "type": "file",
                                        "file_id": "file_abc123",
                                    },
                                }
                            ],
                        }
                    ],
                }
            )

        upstream_request = fetch.call_args.args[0]
        self.assertEqual(
            upstream_request.full_url,
            "https://api.anthropic.com/v1/files/file_abc123/content",
        )
        self.assertEqual(upstream_request.get_header("X-api-key"), "secret")
        self.assertIn(
            "Anthropic file content",
            request["messages"][0]["content"][0]["text"],
        )

    def test_anthropic_file_id_requires_files_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                ProtocolError, "LLMBRIDGE_ANTHROPIC_FILES_API_KEY"
            ):
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
                                            "type": "file",
                                            "file_id": "file_abc123",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                )

    def test_rejects_unsupported_document_media_types(self):
        source = {
            "type": "base64",
            "media_type": "application/octet-stream",
            "data": base64.b64encode(b"bytes").decode(),
        }
        with self.assertRaises(ProtocolError):
            anthropic_to_openai(
                {
                    "model": "client-model",
                    "max_tokens": 100,
                    "messages": [
                        {"role": "user", "content": [{"type": "document", "source": source}]}
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
