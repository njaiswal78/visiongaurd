# Install VisionGuard as a Windows background service (Task Scheduler).
# Runs when you log in, keeps monitoring and alerting on Telegram.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = $ScriptDir
$TaskName = "VisionGuard"
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    $PythonExe = (Get-Command python3 -ErrorAction SilentlyContinue).Source
}
if (-not $PythonExe) {
    Write-Host "Error: python or python3 not found in PATH"
    exit 1
}

$BotScript = Join-Path $ProjectDir "run_telegram_bot.py"
if (-not (Test-Path $BotScript)) {
    Write-Host "Error: run_telegram_bot.py not found at $BotScript"
    exit 1
}

# Create logs directory
$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Remove existing task if present
$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create scheduled task - runs at logon
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$BotScript`"" -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "VisionGuard - AI camera monitoring with Telegram alerts"

Write-Host "Installed. VisionGuard will start when you log in."
Write-Host ""
Write-Host "Commands:"
Write-Host "  Start:   Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Stop:    Stop-ScheduledTask -TaskName $TaskName"
Write-Host "  Status:  Get-ScheduledTask -TaskName $TaskName"
Write-Host "  Uninstall: .\uninstall-service.ps1"

# Start now
Start-ScheduledTask -TaskName $TaskName
Write-Host ""
Write-Host "Service started."
