## Windows Subsystem for Linux (WSL2) Commands/Issues

### Opening Files

- Open Visual Studio Code: `code .`

- Open Windows file explorer: `explorer.exe .`

### Missing Packages

- If Jupyter claims ipykernel not installed (even though `conda list ipykernel` shows valid install):

  - Close all integrated terminals
  - Exit vscode
  - Restart WSL2 shell (*not* from Anaconda prompt)
  - Open VSCode from WSL2 shell
  - Open ipykernel interactive terminal before opening an integrated terminal
  - [[Source](https://github.com/microsoft/vscode-jupyter/issues/1290#issuecomment-738614258)]
