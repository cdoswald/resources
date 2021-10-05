## Frequently Used Git Commands

### Config

View username: `git config --global user.name`

Set username: `git config --global user.name "<username>"`

View email: `git config --global user.email`

Set email: `git config --global user.email "<email>"`

### Status

View current status: `git status`

### Add

Stage specific file: `git add <file_name>`

Stage all files: `git add --all`

Unstage specific file: `git rm --cached <file_name>`

Unstage all files (recursively): `git rm -r --cached`

### Diff

View diff prior to add: `git diff <optional_file_name>`

View diff after add: `git diff --cached <optional_file_name>`

### Commit

Commit with message: `git commit -m "<message>"`

### Rename

Rename file: `git mv <old_file_name> <new_file_name>`