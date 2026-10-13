import base64
import binascii
import html.parser
import io
import json
import re
from typing import Any
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from xml.etree import ElementTree


class ProtocolError(ValueError):
    pass


_OFFICE_DOCUMENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
}
_TEXT_DOCUMENT_TYPES = {
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "application/toml",
    "application/rtf",
}
_MAX_OFFICE_DOCUMENT_BYTES = 25 * 1024 * 1024


class _HTMLTextExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in (
            "br",
            "div",
            "li",
            "p",
            "tr",
        ):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in ("div", "li", "p", "tr"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def _xml_text(xml_data: bytes) -> list[str]:
    root = ElementTree.fromstring(xml_data)
    return [
        element.text or ""
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] in ("t", "v")
    ]


def _office_document_text(data: bytes, document_type: str) -> str:
    try:
        with ZipFile(io.BytesIO(data)) as archive:
            if (
                sum(info.file_size for info in archive.infolist())
                > _MAX_OFFICE_DOCUMENT_BYTES
            ):
                raise ProtocolError(
                    "Expanded Office document exceeds the extraction limit"
                )

            expanded_bytes = 0

            def read_entry(path: str) -> bytes:
                nonlocal expanded_bytes
                remaining_bytes = _MAX_OFFICE_DOCUMENT_BYTES - expanded_bytes
                with archive.open(path) as source:
                    content = source.read(remaining_bytes + 1)
                if len(content) > remaining_bytes:
                    raise ProtocolError(
                        "Expanded Office document exceeds the extraction limit"
                    )
                expanded_bytes += len(content)
                return content

            if document_type == "docx":
                xml_paths = ["word/document.xml"]
            elif document_type == "xlsx":
                shared_strings = (
                    _xml_text(read_entry("xl/sharedStrings.xml"))
                    if "xl/sharedStrings.xml" in archive.namelist()
                    else []
                )
                xml_paths = sorted(
                    path
                    for path in archive.namelist()
                    if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", path)
                )
                worksheet_rows: list[str] = []
                for path in xml_paths:
                    root = ElementTree.fromstring(read_entry(path))
                    for row in root.iter():
                        if row.tag.rsplit("}", 1)[-1] != "row":
                            continue
                        values = []
                        for cell in row:
                            if cell.tag.rsplit("}", 1)[-1] != "c":
                                continue
                            value_element = next(
                                (
                                    child
                                    for child in cell
                                    if child.tag.rsplit("}", 1)[-1] == "v"
                                ),
                                None,
                            )
                            inline_text = _xml_text(ElementTree.tostring(cell))
                            value = (
                                shared_strings[int(value_element.text)]
                                if cell.get("t") == "s"
                                and value_element is not None
                                and value_element.text is not None
                                else " ".join(inline_text)
                                if cell.get("t") == "inlineStr"
                                else value_element.text
                                if value_element is not None and value_element.text
                                else ""
                            )
                            values.append(value)
                        worksheet_rows.append("\t".join(values))
                return "\n".join(worksheet_rows)
            else:
                xml_paths = sorted(
                    (
                        path
                        for path in archive.namelist()
                        if re.fullmatch(r"ppt/slides/slide\d+\.xml", path)
                    ),
                    key=lambda path: int(re.search(r"slide(\d+)", path).group(1)),
                )
            if not xml_paths:
                raise ProtocolError(
                    f"The {document_type.upper()} document contains no readable content"
                )
            return "\n".join(
                " ".join(_xml_text(read_entry(path)))
                for path in xml_paths
            )
    except ProtocolError:
        raise
    except (
        BadZipFile,
        IndexError,
        KeyError,
        ElementTree.ParseError,
        ValueError,
        OSError,
    ) as error:
        raise ProtocolError(
            f"Could not read the supplied {document_type.upper()} document"
        ) from error


def _convert_document_bytes(data: bytes, media_type: str) -> str:
    if media_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ProtocolError("Encrypted PDF documents are not supported")
            pages = [page.extract_text() or "" for page in reader.pages]
        except (PdfReadError, ValueError, OSError) as error:
            raise ProtocolError("Could not read the supplied PDF document") from error
        text = "\n\n".join(
            f"[Page {page_number}]\n{page_text}"
            for page_number, page_text in enumerate(pages, start=1)
            if page_text.strip()
        )
        if not text:
            raise ProtocolError(
                "PDF contains no extractable text; scanned PDFs require OCR"
            )
        return text

    office_type = _OFFICE_DOCUMENT_TYPES.get(media_type)
    if office_type:
        return _office_document_text(data, office_type)

    if media_type.startswith("text/") or media_type in _TEXT_DOCUMENT_TYPES:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ProtocolError(
                f"Document {media_type} content must be UTF-8 encoded"
            ) from error
        if media_type in ("text/html", "text/htm"):
            extractor = _HTMLTextExtractor()
            extractor.feed(text)
            extractor.close()
            return "".join(extractor.parts)
        if media_type in ("text/rtf", "application/rtf"):
            return re.sub(
                r"\\[a-zA-Z]+-?\d* ?|\\[^a-zA-Z]|[{}]",
                "",
                text,
            )
        return text

    raise ProtocolError(f"Unsupported document media type: {media_type}")


def _document_text(block: dict[str, Any]) -> str:
    source = block.get("source")
    if not isinstance(source, dict):
        raise ProtocolError("Document blocks need a source object")

    source_type = source.get("type")
    media_type = source.get("media_type", "text/plain")
    if not isinstance(media_type, str):
        raise ProtocolError("Document media_type must be a string")

    if source_type == "text":
        data = source.get("data")
        if not isinstance(data, str):
            raise ProtocolError("Text documents need a data string")
        text = _convert_document_bytes(data.encode("utf-8"), media_type)
    elif source_type == "base64":
        encoded_data = source.get("data")
        if not isinstance(encoded_data, str):
            raise ProtocolError("Base64 documents need a data string")
        try:
            data = base64.b64decode(encoded_data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ProtocolError("Document data is not valid base64") from error
        text = _convert_document_bytes(data, media_type)
    elif source_type == "url":
        raise ProtocolError(
            "URL documents are not supported by OpenAI-compatible chat routes; "
            "send the document content as text or base64 instead"
        )
    elif source_type == "file":
        raise ProtocolError(
            "Anthropic file IDs cannot be resolved by OpenAI-compatible chat routes; "
            "send the document content directly instead"
        )
    else:
        raise ProtocolError(f"Unsupported document source type: {source_type}")

    title = block.get("title")
    context = block.get("context")
    if title is not None and not isinstance(title, str):
        raise ProtocolError("Document title must be a string")
    if context is not None and not isinstance(context, str):
        raise ProtocolError("Document context must be a string")

    prefix = []
    if title:
        prefix.append(f"Document title: {title}")
    if context:
        prefix.append(f"Document context: {context}")
    if prefix:
        return "\n".join(prefix) + "\n\n" + text
    return text


def _text_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ProtocolError("Message content must be text or a list of content blocks")

    openai_content: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise ProtocolError("Content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise ProtocolError("Text content blocks need a text string")
            openai_content.append({"type": "text", "text": text})
        elif block_type == "document":
            openai_content.append({"type": "text", "text": _document_text(block)})
        elif block_type == "image":
            source = block.get("source", {})
            if not isinstance(source, dict):
                raise ProtocolError("Image source must be an object")
            if source.get("type") == "url":
                image_url = source.get("url")
                if not isinstance(image_url, str) or not image_url:
                    raise ProtocolError("Image URL must be a non-empty string")
            elif source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data")
                if not isinstance(media_type, str) or not isinstance(data, str):
                    raise ProtocolError("Base64 images need media_type and data strings")
                image_url = f"data:{media_type};base64,{data}"
            else:
                raise ProtocolError("Image source must be a URL or base64 data")
            openai_content.append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )
        else:
            raise ProtocolError(f"Unsupported content block type: {block_type}")
    return openai_content


def anthropic_to_openai(
    request: dict[str, Any], upstream_model: str | None = None
) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ProtocolError("Request body must be a JSON object")
    model = upstream_model or request.get("model")
    messages = request.get("messages")
    if not isinstance(request.get("model"), str) or not request["model"]:
        raise ProtocolError("A client model name is required")
    if not isinstance(model, str) or not model:
        raise ProtocolError("A model name is required")
    if not isinstance(messages, list) or not messages:
        raise ProtocolError("At least one message is required")
    if (
        not isinstance(request.get("max_tokens"), int)
        or isinstance(request["max_tokens"], bool)
        or request["max_tokens"] < 1
    ):
        raise ProtocolError("max_tokens must be a positive integer")

    openai_messages: list[dict[str, Any]] = []
    system = request.get("system")
    if isinstance(system, str):
        openai_messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        system_parts: list[str] = []
        for block in system:
            if not isinstance(block, dict):
                raise ProtocolError("System content blocks must be objects")
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if not isinstance(text, str):
                    raise ProtocolError("Text content blocks need a text string")
                system_parts.append(text)
            elif block_type == "document":
                system_parts.append(_document_text(block))
            else:
                raise ProtocolError(f"Unsupported system content block type: {block_type}")
        system_text = "\n\n".join(system_parts)
        if system_text:
            openai_messages.append({"role": "system", "content": system_text})
    elif system is not None:
        raise ProtocolError("system must be a string or list of text blocks")

    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in (
            "user",
            "assistant",
        ):
            raise ProtocolError("Each message must have role user or assistant")
        role = message["role"]
        blocks = message.get("content", "")
        if isinstance(blocks, str):
            openai_messages.append({"role": role, "content": blocks})
            continue
        if not isinstance(blocks, list):
            raise ProtocolError("Message content must be text or a list")

        text_blocks: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                raise ProtocolError("Content blocks must be objects")
            block_type = block.get("type")
            if block_type in ("text", "image", "document"):
                text_blocks.extend(_text_content([block]))
            elif role == "assistant" and block_type == "tool_use":
                tool_name = block.get("name")
                tool_input = block.get("input", {})
                if not isinstance(tool_name, str) or not tool_name:
                    raise ProtocolError("tool_use blocks require a name")
                if not isinstance(tool_input, dict):
                    raise ProtocolError("tool_use input must be an object")
                tool_id = block.get("id", f"call_{uuid4().hex}")
                if not isinstance(tool_id, str) or not tool_id:
                    raise ProtocolError("tool_use blocks require a string id")
                tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input),
                        },
                    }
                )
            elif role == "user" and block_type == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str) or not tool_use_id:
                    raise ProtocolError("tool_result blocks require tool_use_id")
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": _text_content(block.get("content", "")),
                    }
                )
            elif role == "assistant" and block_type in (
                "thinking",
                "redacted_thinking",
                "signature",
            ):
                continue
            else:
                raise ProtocolError(f"Unsupported content block type: {block_type}")

        if text_blocks or tool_calls:
            converted: dict[str, Any] = {
                "role": role,
                "content": text_blocks or None,
            }
            if tool_calls:
                converted["tool_calls"] = tool_calls
            openai_messages.append(converted)

    payload: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": request["max_tokens"],
        "stream": bool(request.get("stream", False)),
    }
    for key in ("temperature", "top_p"):
        if key in request:
            payload[key] = request[key]
    if "stop_sequences" in request:
        if not isinstance(request["stop_sequences"], list) or not all(
            isinstance(value, str) for value in request["stop_sequences"]
        ):
            raise ProtocolError("stop_sequences must be a list of strings")
        payload["stop"] = request["stop_sequences"]

    if "tools" in request:
        if not isinstance(request["tools"], list):
            raise ProtocolError("tools must be a list")
        for tool in request["tools"]:
            if (
                not isinstance(tool, dict)
                or not isinstance(tool.get("name"), str)
                or not tool["name"]
                or not isinstance(tool.get("input_schema", {}), dict)
            ):
                raise ProtocolError("Each tool needs a name and object input_schema")
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in request["tools"]
        ]
    tool_choice = request.get("tool_choice")
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "tool":
            if not isinstance(tool_choice.get("name"), str) or not tool_choice["name"]:
                raise ProtocolError("tool_choice of type tool requires a name")
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }
        elif choice_type == "any":
            payload["tool_choice"] = "required"
        elif choice_type == "auto":
            payload["tool_choice"] = "auto"
        else:
            raise ProtocolError(f"Unsupported tool_choice type: {choice_type}")
    elif tool_choice in ("auto", "none", "required"):
        payload["tool_choice"] = tool_choice
    elif tool_choice is not None:
        raise ProtocolError("tool_choice must be a supported string or object")

    if payload["stream"]:
        payload["stream_options"] = {"include_usage": True}
    return payload


def openai_to_anthropic(
    response: dict[str, Any], requested_model: str
) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ProtocolError("Upstream response must be a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProtocolError("Upstream response contains no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProtocolError("Upstream response choice must be an object")
    message = choice.get("message", {})
    if not isinstance(message, dict):
        raise ProtocolError("Upstream response message must be an object")
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if text is not None:
        if not isinstance(text, str):
            raise ProtocolError("Upstream response content must be text")
        if text:
            content.append({"type": "text", "text": text})
    tool_calls = message.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        raise ProtocolError("Upstream tool_calls must be a list")
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise ProtocolError("Upstream tool call must be an object")
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            raise ProtocolError("Upstream tool function must be an object")
        raw_arguments = function.get("arguments", "{}")
        try:
            arguments = (
                raw_arguments
                if isinstance(raw_arguments, dict)
                else json.loads(raw_arguments or "{}")
            )
        except (json.JSONDecodeError, TypeError) as error:
            raise ProtocolError("Upstream returned invalid tool-call JSON") from error
        if not isinstance(arguments, dict):
            raise ProtocolError("Upstream tool-call arguments must be a JSON object")
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id", f"toolu_{uuid4().hex}"),
                "name": function.get("name", ""),
                "input": arguments,
            }
        )

    finish_reason = choice.get("finish_reason")
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
    }.get(finish_reason, "end_turn")
    usage = response.get("usage", {})
    return {
        "id": response.get("id", f"msg_{uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def stream_events(response_lines: Any, requested_model: str):
    message_id = f"msg_{uuid4().hex}"
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": requested_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    open_blocks: set[int] = set()
    tool_ids: dict[int, str] = {}
    tool_names: dict[int, str] = {}
    pending_tool_arguments: dict[int, list[str]] = {}
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    for raw_line in response_lines:
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise ProtocolError("Upstream returned a non-UTF-8 streaming event") from error
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as error:
            raise ProtocolError("Upstream returned an invalid streaming event") from error
        if not isinstance(chunk, dict):
            raise ProtocolError("Upstream streaming event must be a JSON object")
        if chunk.get("error"):
            raise ProtocolError("Upstream reported an error during streaming")
        if chunk.get("usage"):
            if not isinstance(chunk["usage"], dict):
                raise ProtocolError("Upstream stream usage must be an object")
            usage = chunk["usage"]
        choices = chunk.get("choices", [])
        if not isinstance(choices, list):
            raise ProtocolError("Upstream stream choices must be a list")
        for choice in choices:
            if not isinstance(choice, dict):
                raise ProtocolError("Upstream stream choice must be an object")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise ProtocolError("Upstream stream delta must be an object")
            text = delta.get("content")
            if text is not None and not isinstance(text, str):
                raise ProtocolError("Upstream stream content must be text")
            if text:
                if 0 not in open_blocks:
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    open_blocks.add(0)
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            tool_deltas = delta.get("tool_calls", [])
            if not isinstance(tool_deltas, list):
                raise ProtocolError("Upstream stream tool_calls must be a list")
            for tool in tool_deltas:
                if not isinstance(tool, dict):
                    raise ProtocolError("Upstream stream tool call must be an object")
                tool_index = tool.get("index", 0)
                if (
                    not isinstance(tool_index, int)
                    or isinstance(tool_index, bool)
                    or tool_index < 0
                ):
                    raise ProtocolError("Upstream stream tool index must be non-negative")
                block_index = tool_index + 1
                function = tool.get("function", {})
                if not isinstance(function, dict):
                    raise ProtocolError("Upstream stream tool function must be an object")
                name = function.get("name", "")
                if not isinstance(name, str):
                    raise ProtocolError("Upstream stream tool name must be text")
                tool_id = tool.get("id")
                if tool_id is not None and not isinstance(tool_id, str):
                    raise ProtocolError("Upstream stream tool id must be text")
                if tool_id:
                    tool_ids[tool_index] = tool_ids.get(tool_index, "") + tool_id
                if name:
                    tool_names[tool_index] = tool_names.get(tool_index, "") + name
                arguments = function.get("arguments")
                if arguments is not None and not isinstance(arguments, str):
                    raise ProtocolError("Upstream stream tool arguments must be text")
                if block_index not in open_blocks and (
                    tool_index not in tool_ids or tool_index not in tool_names
                ):
                    if arguments:
                        pending_tool_arguments.setdefault(tool_index, []).append(arguments)
                    continue
                if block_index not in open_blocks:
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_ids[tool_index],
                                "name": tool_names[tool_index],
                                "input": {},
                            },
                        },
                    )
                    open_blocks.add(block_index)
                    for pending_arguments in pending_tool_arguments.pop(
                        tool_index, []
                    ):
                        yield _tool_arguments_event(block_index, pending_arguments)
                if arguments:
                    yield _tool_arguments_event(block_index, arguments)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    if pending_tool_arguments or set(tool_ids) != set(tool_names):
        raise ProtocolError("Upstream stream ended with incomplete tool-call metadata")
    if finish_reason is None:
        raise ProtocolError("Upstream stream ended before a completion reason was received")

    for block_index in sorted(open_blocks):
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
    }.get(finish_reason, "end_turn")
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage.get("completion_tokens", 0)},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


def _tool_arguments_event(block_index: int, arguments: str) -> bytes:
    return _sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "input_json_delta", "partial_json": arguments},
        },
    )


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
