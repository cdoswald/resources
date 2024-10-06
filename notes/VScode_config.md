## Visual Studio Code Configuration

- Run Selected Text in Active Terminal: 
`Open File > Preferences > Keyboard Shortcuts > workbench.action.terminal.runSelectedText`

- Open User Settings JSON File:
`Ctrl + Shift + P > Preferences: Open User Settings (JSON)'

- Terminal Integrated Profile (Windows):
```
	"terminal.integrated.profiles.windows": {
		"<terminal_name>": {
			"source": "<source_name>",
			"path": "<path_to_exe_file_if_no_source_name>",
			"args": [
				"<arg1>",
				"<arg2>"
			],
			"icon": "<icon_name>""
		}
	}
```

- Anaconda Terminal (via PowerShell):
```
	"terminal.integrated.profiles.windows": {
        "Anaconda Prompt (PS)": {
            "path":<path_to_powershell_exe>,
            "args": [
                "powershell",
                "-NoExit",
                "-ExecutionPolicy <policy>",
                "-Scope CurrentUser",
                "-NoProfile",
                "-File ~anaconda3\\shell\\condabin\\conda-hook.ps1"
            ]
        }
	}
```