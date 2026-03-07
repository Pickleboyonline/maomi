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

  client.start();
  context.subscriptions.push(client);
}

function deactivate() {
  if (client) {
    return client.stop();
  }
}

module.exports = { activate, deactivate };
