"""Streaming multipart reader for uploads, with limits enforced as the bytes arrive.

Starlette's form parser spools every file part to disk without a size cap and only applies its
part limit to text fields, so size checks could only run after the whole body was received (and
not at all for chunked requests). This reader feeds ``python-multipart`` chunk by chunk and stops
with 413 as soon as one file passes ``max_file_bytes``, a text field passes
``max_field_bytes`` or the body passes ``max_total_bytes``. Parts are kept in memory (an upload is
decoded from memory anyway), so nothing is ever written to a temporary file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Request
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header

from app.errors import BadRequestError, InvalidInputError, PayloadTooLargeError

MULTIPART = "multipart/form-data"
URLENCODED = "application/x-www-form-urlencoded"
MALFORMED = "The upload is not a valid multipart form."


@dataclass
class FormPart:
    name: str
    filename: str | None  # None for a plain text field
    content_type: str | None
    data: bytearray = field(default_factory=bytearray)

    @property
    def is_file(self) -> bool:
        return self.filename is not None

    def text(self) -> str:
        return bytes(self.data).decode("utf-8", errors="replace")


@dataclass(frozen=True)
class FormLimits:
    max_file_bytes: int
    max_field_bytes: int
    max_total_bytes: int
    max_files: int
    max_fields: int
    file_limit_label: str  # e.g. "40 MB" (shown to the user)
    field_limit_label: str
    # Parts held to the text-field limit even when sent as a file (e.g. the JSON manifest).
    small_parts: frozenset[str] = frozenset()


class _Reader:
    def __init__(self, limits: FormLimits) -> None:
        self.limits = limits
        self.parts: dict[str, FormPart] = {}
        self.current: FormPart | None = None
        self.headers: dict[bytes, bytes] = {}
        self.header_name = bytearray()
        self.header_value = bytearray()
        self.files = 0
        self.fields = 0

    # python-multipart callbacks ---------------------------------------------------------------

    def on_part_begin(self) -> None:
        self.current = None
        self.headers = {}

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self.header_name += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self.header_value += data[start:end]

    def on_header_end(self) -> None:
        self.headers[bytes(self.header_name).lower()] = bytes(self.header_value)
        self.header_name.clear()
        self.header_value.clear()

    def on_headers_finished(self) -> None:
        _disposition, options = parse_options_header(self.headers.get(b"content-disposition", b""))
        raw_name = options.get(b"name")
        if raw_name is None:
            raise BadRequestError("Every form part needs a name.")
        name = raw_name.decode("utf-8", errors="replace")
        raw_filename = options.get(b"filename")
        filename = (
            raw_filename.decode("utf-8", errors="replace") if raw_filename is not None else None
        )
        if filename is not None:
            self.files += 1
            if self.files > self.limits.max_files:
                raise InvalidInputError(f"Too many files (at most {self.limits.max_files}).")
        else:
            self.fields += 1
            if self.fields > self.limits.max_fields:
                raise InvalidInputError(f"Too many form fields (at most {self.limits.max_fields}).")
        if name in self.parts:
            raise InvalidInputError(f"Form field '{name}' was sent more than once.")
        content_type = self.headers.get(b"content-type")
        self.current = FormPart(
            name=name,
            filename=filename,
            content_type=content_type.decode("latin-1") if content_type else None,
        )
        self.parts[name] = self.current

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        part = self.current
        if part is None:  # pragma: no cover - python-multipart always sends headers first
            raise BadRequestError(MALFORMED)
        part.data += data[start:end]
        small = not part.is_file or part.name in self.limits.small_parts
        if small and len(part.data) > self.limits.max_field_bytes:
            raise PayloadTooLargeError(
                f"Form field '{part.name}' is larger than the {self.limits.field_limit_label} "
                "limit."
            )
        if part.is_file and len(part.data) > self.limits.max_file_bytes:
            raise PayloadTooLargeError(
                f"File '{part.name}' is larger than the {self.limits.file_limit_label} limit."
            )

    def on_part_end(self) -> None:
        self.current = None


async def read_upload_form(request: Request, limits: FormLimits) -> dict[str, FormPart]:
    """Parse the request body into named parts; raises 400/413/422 ``ApiError``s."""
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    kind = content_type.decode("latin-1").lower()
    if kind == URLENCODED:
        return await _read_urlencoded(request, limits)
    if kind != MULTIPART:
        return {}  # no form at all: the caller reports the missing manifest
    boundary = params.get(b"boundary")
    if not boundary:
        raise BadRequestError("The multipart upload has no boundary.")

    reader = _Reader(limits)
    callbacks = {
        "on_part_begin": reader.on_part_begin,
        "on_part_data": reader.on_part_data,
        "on_part_end": reader.on_part_end,
        "on_header_field": reader.on_header_field,
        "on_header_value": reader.on_header_value,
        "on_header_end": reader.on_header_end,
        "on_headers_finished": reader.on_headers_finished,
    }
    try:
        parser = MultipartParser(boundary, callbacks)  # type: ignore[arg-type]
    except (MultipartParseError, ValueError) as exc:
        raise BadRequestError(MALFORMED) from exc
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > limits.max_total_bytes:
                raise PayloadTooLargeError(
                    f"The upload is larger than the {limits.max_total_bytes // (1024 * 1024)} "
                    "MB request limit."
                )
            if chunk:
                parser.write(chunk)
        parser.finalize()
    except MultipartParseError as exc:
        raise BadRequestError(MALFORMED) from exc
    if reader.current is not None:
        raise BadRequestError(MALFORMED)  # the body ended in the middle of a part
    return reader.parts


async def _read_urlencoded(request: Request, limits: FormLimits) -> dict[str, FormPart]:
    form = await request.form(max_files=0, max_fields=limits.max_fields)
    try:
        parts: dict[str, FormPart] = {}
        for name, value in form.multi_items():
            if name in parts:
                raise InvalidInputError(f"Form field '{name}' was sent more than once.")
            data = bytearray(str(value).encode("utf-8"))
            if len(data) > limits.max_field_bytes:
                raise PayloadTooLargeError(
                    f"Form field '{name}' is larger than the {limits.field_limit_label} limit."
                )
            parts[name] = FormPart(name=name, filename=None, content_type=None, data=data)
        return parts
    finally:
        await form.close()
