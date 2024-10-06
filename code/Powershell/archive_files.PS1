# Specify file paths
$repo_root_dir = "$PSScriptRoot\"
$project_server_root_dir = ""
$move_folders_list = "", "", ""

# Move files to _Archive folder on project server
Write-Host "Archiving old data files on project server (start time: $(Get-Date))..."
$current_date = Get-Date -Format "yyyy-MM-dd"
$archived_folder_path = Join-Path -Path $project_server_root_dir -ChildPath "_Archive\Archived $current_date"
if (Test-Path $archived_folder_path) {
	$version = 2
	$versioned_archived_folder_path = "$archived_folder_path (v$version)"
	while (Test-Path $versioned_archived_folder_path){
		$version += 1
		$versioned_archived_folder_path = "$archived_folder_path (v$version)"
	}
	$archived_folder_path = $versioned_archived_folder_path
}
Foreach ($folder in $move_folders_list) {
	$move_folder_path = Join-Path -Path $project_server_root_dir -ChildPath $folder
	Move-Item -Path $move_folder_path -Destination $archived_folder_path
}

# Copy files from repository to project server
Write-Host "Copying new data files to project server (start time: $(Get-Date))..."
Foreach ($folder in $move_folders_list) {
	$copy_folder_path = Join-Path -Path $repo_root_dir -ChildPath $folder
	$new_folder_path = Join-Path -Path $project_server_root_dir -ChildPath $folder
	Copy-Item -Path $copy_folder_path -Destination $new_folder_path -Recurse
}
Write-Host "Finished copying new data files to project server (end time: $(Get-Date))--press any key to continue..."
[void][System.Console]::ReadKey($true)