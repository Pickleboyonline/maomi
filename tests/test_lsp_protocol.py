"""Protocol-level integration tests for the Maomi LSP server.

Starts the real Maomi LSP server as a subprocess via pygls's BaseLanguageClient,
sends actual LSP protocol messages, and verifies responses.
"""

import asyncio
import pytest
import pytest_asyncio
from lsprotocol import types
from pygls.lsp.client import BaseLanguageClient


DOC_URI = "file:///test/test.mao"


@pytest_asyncio.fixture
async def client():
    """Start maomi LSP server and return connected client."""
    c = BaseLanguageClient("test-client", "0.1.0")
    await c.start_io("uv", "run", "--extra", "lsp", "--extra", "run", "maomi", "lsp")

    response = await c.initialize_async(types.InitializeParams(
        capabilities=types.ClientCapabilities(),
        root_uri=None,
    ))
    assert response is not None
    assert response.capabilities is not None
    c.initialized(types.InitializedParams())

    yield c

    try:
        await c.shutdown_async(None)
        c.exit(None)
        await asyncio.sleep(0.2)
    except Exception:
        pass
    try:
        await c.stop()
    except Exception:
        pass


def _open_doc(client, source, uri=DOC_URI):
    """Send didOpen notification."""
    client.text_document_did_open(types.DidOpenTextDocumentParams(
        text_document=types.TextDocumentItem(
            uri=uri,
            language_id="maomi",
            version=1,
            text=source,
        )
    ))


@pytest.mark.asyncio
async def test_server_initializes(client):
    """Server responds to initialize with capabilities."""
    # The fixture already asserts this -- just verify client is alive
    assert client is not None


@pytest.mark.asyncio
async def test_hover_on_variable(client):
    source = "fn f(x: f32) -> f32 { x }"
    _open_doc(client, source)
    await asyncio.sleep(0.5)

    result = await client.text_document_hover_async(types.HoverParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
        position=types.Position(line=0, character=22),
    ))
    assert result is not None
    assert "f32" in result.contents.value


@pytest.mark.asyncio
async def test_hover_on_function_name(client):
    source = "fn f(x: f32) -> f32 { x }"
    _open_doc(client, source)
    await asyncio.sleep(0.5)

    result = await client.text_document_hover_async(types.HoverParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
        position=types.Position(line=0, character=3),
    ))
    assert result is not None
    assert "fn f" in result.contents.value


@pytest.mark.asyncio
async def test_hover_returns_none_on_whitespace(client):
    source = "fn f(x: f32) -> f32 { x }"
    _open_doc(client, source)
    await asyncio.sleep(0.5)

    result = await client.text_document_hover_async(types.HoverParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
        position=types.Position(line=0, character=20),
    ))
    assert result is None


@pytest.mark.asyncio
async def test_completion_includes_keywords_and_builtins(client):
    source = "fn f(x: f32) -> f32 { x }"
    _open_doc(client, source)
    await asyncio.sleep(0.5)

    result = await client.text_document_completion_async(types.CompletionParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
        position=types.Position(line=0, character=23),
    ))
    assert result is not None
    labels = {item.label for item in result.items}
    assert "fn" in labels
    assert "exp" in labels
    assert "x" in labels


@pytest.mark.asyncio
async def test_goto_definition_variable(client):
    source = "fn f(x: f32) -> f32 { x }"
    _open_doc(client, source)
    await asyncio.sleep(0.5)

    result = await client.text_document_definition_async(types.DefinitionParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
        position=types.Position(line=0, character=22),
    ))
    assert result is not None
    # Should point back to param definition on line 0
    if isinstance(result, list):
        assert result[0].range.start.line == 0
    else:
        assert result.range.start.line == 0


@pytest.mark.asyncio
async def test_goto_definition_function_call(client):
    source = "fn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) }"
    _open_doc(client, source)
    await asyncio.sleep(0.5)

    result = await client.text_document_definition_async(types.DefinitionParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
        position=types.Position(line=1, character=25),
    ))
    assert result is not None
    if isinstance(result, list):
        assert result[0].range.start.line == 0
    else:
        assert result.range.start.line == 0


@pytest.mark.asyncio
async def test_find_references(client):
    source = "fn f(x: f32) -> f32 { x + x }"
    _open_doc(client, source)
    await asyncio.sleep(0.5)

    result = await client.text_document_references_async(types.ReferenceParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
        position=types.Position(line=0, character=22),
        context=types.ReferenceContext(include_declaration=True),
    ))
    assert result is not None
    assert len(result) == 3  # param decl + 2 uses


@pytest.mark.asyncio
async def test_rename(client):
    source = "fn f(x: f32) -> f32 { x + x }"
    _open_doc(client, source)
    await asyncio.sleep(0.5)

    result = await client.text_document_rename_async(types.RenameParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
        position=types.Position(line=0, character=22),
        new_name="a",
    ))
    assert result is not None
    assert DOC_URI in result.changes
    assert len(result.changes[DOC_URI]) == 3


@pytest.mark.asyncio
async def test_document_symbols(client):
    source = "struct P { x: f32 }\nfn f(p: P) -> f32 { p.x }"
    _open_doc(client, source)
    await asyncio.sleep(0.5)

    result = await client.text_document_document_symbol_async(types.DocumentSymbolParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
    ))
    assert result is not None
    names = {s.name for s in result}
    assert "P" in names
    assert "f" in names


@pytest.mark.asyncio
async def test_document_change_and_rehover(client):
    """Verify document changes are picked up."""
    source_v1 = "fn f(x: f32) -> f32 { x }"
    _open_doc(client, source_v1)
    await asyncio.sleep(0.5)

    # Change document
    source_v2 = "fn f(x: f32, y: f32) -> f32 { x + y }"
    client.text_document_did_change(types.DidChangeTextDocumentParams(
        text_document=types.VersionedTextDocumentIdentifier(uri=DOC_URI, version=2),
        content_changes=[types.TextDocumentContentChangeWholeDocument(text=source_v2)],
    ))
    await asyncio.sleep(0.5)

    # Hover should now show new content
    result = await client.text_document_hover_async(types.HoverParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
        position=types.Position(line=0, character=3),
    ))
    assert result is not None
    # Should reflect the new signature with y parameter
    assert "y" in result.contents.value


@pytest.mark.asyncio
async def test_hover_with_struct(client):
    source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
    _open_doc(client, source)
    await asyncio.sleep(0.5)

    result = await client.text_document_hover_async(types.HoverParams(
        text_document=types.TextDocumentIdentifier(uri=DOC_URI),
        position=types.Position(line=1, character=5),
    ))
    assert result is not None
    assert "Point" in result.contents.value
