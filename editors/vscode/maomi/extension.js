const vscode = require("vscode");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

let client;

function activate(context) {
  const config = vscode.workspace.getConfiguration("maomi");
  const command = config.get("serverPath", "maomi");

  const serverOptions = {
    command: command,
    args: ["lsp"],
    transport: TransportKind.stdio,
  };

  const clientOptions = {
    documentSelector: [{ scheme: "file", language: "maomi" }],
  };

  client = new LanguageClient(
    "maomi",
    "Maomi Language Server",
    serverOptions,
    clientOptions
  );

  const runDisposable = vscode.commands.registerCommand('maomi.run', async (uri, fnName) => {
    const terminal = vscode.window.createTerminal('Maomi Run');
    terminal.show();
    const filePath = vscode.Uri.parse(uri).fsPath;
    terminal.sendText(`uv run maomi run "${filePath}" --fn ${fnName}`);
  });
  context.subscriptions.push(runDisposable);
  const matchBraceDisposable = vscode.commands.registerCommand('maomi.matchBrace', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !client) return;
    const pos = editor.selection.active;
    const result = await client.sendRequest('maomi/matchingBrace', {
      textDocument: { uri: editor.document.uri.toString() },
      position: { line: pos.line, character: pos.character },
    });
    if (result) {
      const newPos = new vscode.Position(result.line, result.character);
      editor.selection = new vscode.Selection(newPos, newPos);
      editor.revealRange(new vscode.Range(newPos, newPos));
    }
  });
  context.subscriptions.push(matchBraceDisposable);

  client.start();
  context.subscriptions.push(client);
}

function deactivate() {
  if (client) {
    return client.stop();
  }
}

module.exports = { activate, deactivate };
