from __future__ import annotations

import functools
import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown

from ._core.env import DEFAULT_BASE_URL
from ._http.client import Client
from .errors import TurboOcrError
from .markdown import render_to_markdown
from .models import OcrResponse, PdfMode, PdfResponse


class OutputFormat(StrEnum):
    json = "json"
    blocks = "blocks"
    text = "text"
    markdown = "markdown"


app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.callback()
def _root() -> None:
    pass


def _exit_on_sdk_error[F](fn: F) -> F:
    @functools.wraps(fn)  # type: ignore[arg-type]
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return fn(*args, **kwargs)  # type: ignore[operator]
        except TurboOcrError as exc:
            typer.echo(f"{type(exc).__name__}: {exc}", err=True)
            raise typer.Exit(code=2) from exc

    return wrapper  # type: ignore[return-value]


def _build_client(base_url: str, api_key: str | None) -> Client:
    return Client(base_url=base_url, api_key=api_key)


@app.command()
@_exit_on_sdk_error
def ocr(
    image: Annotated[Path, typer.Argument(exists=True, readable=True)],
    base_url: Annotated[str, typer.Option(envvar="TURBO_OCR_BASE_URL")] = DEFAULT_BASE_URL,
    api_key: Annotated[str | None, typer.Option(envvar="TURBO_OCR_API_KEY")] = None,
    layout: bool = True,
    reading_order: bool = True,
    include_blocks: bool = True,
    tables: Annotated[
        bool, typer.Option(help="table recognition → HTML (needs TABLE_BACKEND)")
    ] = False,
    formulas: Annotated[
        bool, typer.Option(help="formula recognition → LaTeX (needs FORMULA_BACKEND)")
    ] = False,
    output: Annotated[OutputFormat, typer.Option()] = OutputFormat.markdown,
) -> None:
    with _build_client(base_url, api_key) as client:
        response = client.recognize_image(
            image,
            layout=layout,
            reading_order=reading_order,
            include_blocks=include_blocks,
            tables=tables or None,
            formulas=formulas or None,
        )
    _emit(response, output)


@app.command()
@_exit_on_sdk_error
def pdf(
    pdf_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    base_url: Annotated[str, typer.Option(envvar="TURBO_OCR_BASE_URL")] = DEFAULT_BASE_URL,
    api_key: Annotated[str | None, typer.Option(envvar="TURBO_OCR_API_KEY")] = None,
    dpi: int = 100,
    mode: PdfMode = PdfMode.auto,
    layout: bool = True,
    reading_order: bool = True,
    include_blocks: bool = True,
    tables: Annotated[
        bool, typer.Option(help="table recognition → HTML (needs TABLE_BACKEND)")
    ] = False,
    formulas: Annotated[
        bool, typer.Option(help="formula recognition → LaTeX (needs FORMULA_BACKEND)")
    ] = False,
    autorotate: Annotated[bool, typer.Option(help="straighten rotated pages first")] = False,
    output: Annotated[OutputFormat, typer.Option()] = OutputFormat.markdown,
) -> None:
    with _build_client(base_url, api_key) as client:
        response = client.recognize_pdf(
            pdf_file,
            dpi=dpi,
            mode=mode,
            layout=layout,
            reading_order=reading_order,
            include_blocks=include_blocks,
            tables=tables or None,
            formulas=formulas or None,
            autorotate=autorotate or None,
        )
    _emit(response, output)


@app.command()
@_exit_on_sdk_error
def markdown(
    document: Annotated[Path, typer.Argument(exists=True, readable=True)],
    base_url: Annotated[str, typer.Option(envvar="TURBO_OCR_BASE_URL")] = DEFAULT_BASE_URL,
    api_key: Annotated[str | None, typer.Option(envvar="TURBO_OCR_API_KEY")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="write .md here instead of stdout")
    ] = None,
    dpi: Annotated[
        int | None, typer.Option(help="PDF rasterization DPI (server default 100)")
    ] = None,
    mode: Annotated[PdfMode | None, typer.Option(help="PDF reader strategy")] = None,
) -> None:
    """Server-side Markdown export: image → /ocr/markdown, PDF → /ocr/pdf?markdown=1."""
    with _build_client(base_url, api_key) as client:
        if document.suffix.lower() == ".pdf":
            md = client.pdf_markdown(document, dpi=dpi, mode=mode)
        else:
            md = client.page_markdown(document)
    assert isinstance(md, str)
    if out is not None:
        out.write_text(md, encoding="utf-8")
        console.print(f"wrote {len(md):,} chars -> {out}")
    else:
        console.print(Markdown(md))


@app.command()
@_exit_on_sdk_error
def capabilities(
    base_url: Annotated[str, typer.Option(envvar="TURBO_OCR_BASE_URL")] = DEFAULT_BASE_URL,
    api_key: Annotated[str | None, typer.Option(envvar="TURBO_OCR_API_KEY")] = None,
) -> None:
    """What the running server has loaded (layout / tables / formulas / autorotate)."""
    with _build_client(base_url, api_key) as client:
        caps = client.capabilities()
    console.print_json(caps.model_dump_json())


@app.command(name="searchable-pdf")
@_exit_on_sdk_error
def searchable_pdf(
    pdf_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output PDF path")],
    base_url: Annotated[str, typer.Option(envvar="TURBO_OCR_BASE_URL")] = DEFAULT_BASE_URL,
    api_key: Annotated[str | None, typer.Option(envvar="TURBO_OCR_API_KEY")] = None,
    dpi: int = 200,
    mode: PdfMode = PdfMode.ocr,
    font_path: Annotated[
        str | None,
        typer.Option(help="optional custom TTF; default bundled glyphless font covers all BMP"),
    ] = None,
) -> None:
    with _build_client(base_url, api_key) as client:
        overlay_bytes = client.make_searchable_pdf(
            pdf_file, dpi=dpi, mode=mode, font_path=font_path
        )
    out.write_bytes(overlay_bytes)
    console.print(f"wrote {len(overlay_bytes):,} bytes -> {out}")


@app.command()
@_exit_on_sdk_error
def blocks(
    image: Annotated[Path, typer.Argument(exists=True, readable=True)],
    base_url: Annotated[str, typer.Option(envvar="TURBO_OCR_BASE_URL")] = DEFAULT_BASE_URL,
    api_key: Annotated[str | None, typer.Option(envvar="TURBO_OCR_API_KEY")] = None,
) -> None:
    response: OcrResponse | PdfResponse
    with _build_client(base_url, api_key) as client:
        suffix = image.suffix.lower()
        if suffix == ".pdf":
            response = client.recognize_pdf(image, include_blocks=True)
        else:
            response = client.recognize_image(image, include_blocks=True)
    console.print(_blocks_dump(response))


@app.command()
def health(
    base_url: Annotated[str, typer.Option(envvar="TURBO_OCR_BASE_URL")] = DEFAULT_BASE_URL,
    api_key: Annotated[str | None, typer.Option(envvar="TURBO_OCR_API_KEY")] = None,
    ready: bool = False,
) -> None:
    with _build_client(base_url, api_key) as client:
        status = client.health(ready=ready)
    console.print_json(status.model_dump_json())
    raise typer.Exit(code=0 if status.ok else 1)


def _blocks_dump(response: OcrResponse | PdfResponse) -> str:
    if isinstance(response, PdfResponse):
        payload: object = {
            "pages": [
                {"page": p.page, "blocks": [b.model_dump(by_alias=True) for b in p.blocks]}
                for p in response.pages
            ]
        }
    else:
        payload = {"blocks": [b.model_dump(by_alias=True) for b in response.blocks]}
    return json.dumps(payload, indent=2, ensure_ascii=False)


_Response = OcrResponse | PdfResponse
_Emitter = Callable[[_Response], None]


def _emit_json(response: _Response) -> None:
    console.print_json(response.model_dump_json())


def _emit_blocks(response: _Response) -> None:
    console.print(_blocks_dump(response))


def _emit_text(response: _Response) -> None:
    console.print(response.text)


def _emit_markdown(response: _Response) -> None:
    console.print(Markdown(render_to_markdown(response).markdown))


_EMITTERS: dict[OutputFormat, _Emitter] = {
    OutputFormat.json: _emit_json,
    OutputFormat.blocks: _emit_blocks,
    OutputFormat.text: _emit_text,
    OutputFormat.markdown: _emit_markdown,
}


def _emit(response: _Response, output: OutputFormat) -> None:
    _EMITTERS[output](response)


if __name__ == "__main__":
    app()
