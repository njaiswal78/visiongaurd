# Uninstall VisionGuard Windows scheduled task.

$TaskName = "VisionGuard"
$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($Task) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "VisionGuard service uninstalled."
} else {
    Write-Host "VisionGuard service was not installed."
}
