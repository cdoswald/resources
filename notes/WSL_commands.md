## Windows Subsystem for Linux (WSL) Commands

### Opening Files

Open Visual Studio Code: `code .`

Open Windows file explorer: `explorer.exe .`

### Opening Remote Folder

To edit files located in Windows filesystem:

1. Open WSL terminal from Powershell using `wsl`

2. Mount Windows filesystem: `/mnt/c/Users/<username>/<path>`

3. Launch VSCode from remote folder: `code .`

Or from Windows (Anaconda) command prompt: `code --remote wsl+Ubuntu <path in WSL>`

### Cloning GitHub Repository Using PAT

1. Generate PAT via GitHub settings > developer settings

2. Run the following in PowerShell/Bash: `git clone https://<tokenhere>@github.com/<user>/<repo>.git`

### System Time

1. Sync: `sudo hwclock -s`

### Setting Up Consistent Line Endings for Git

