# Create Windows Task Scheduler task for StockPilot BC Push
# Runs daily at 9:00 AM

$taskName = "StockPilotBCPush"
$description = "StockPilot BC Model Top-5 Push - runs daily at 9:00 AM"
$scriptPath = "D:\Projects\auto-select-stock\push_bc_recommend.py"
$wslCommand = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fin && cd /mnt/d/Projects/auto-select-stock && PYTHONPATH=./src python push_bc_recommend.py"

# Delete existing task if present
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Deleted existing task: $taskName"
}

# Create action - runs bash with the WSL command
$action = New-ScheduledTaskAction -Execute "bash.exe" -Argument "-c `"$wslCommand`""

# Create trigger - daily at 9:00 AM
$trigger = New-ScheduledTaskTrigger -Daily -At "9:00AM"

# Create principal - run whether user is logged on or not (if possible)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive -RunLevel Limited

# Create settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register the task
Register-ScheduledTask -TaskName $taskName -Description $description -Action $action -Trigger $trigger -Principal $principal -Settings $settings

Write-Host "Created scheduled task: $taskName"
Write-Host "  Runs: Daily at 9:00 AM"
Write-Host "  Script: $scriptPath"
