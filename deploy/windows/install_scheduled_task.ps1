<#
.SYNOPSIS
    Installs StreamFusion Homelab Daemon as an unattended Windows Scheduled Task.

.DESCRIPTION
    Configures Windows Task Scheduler to run StreamFusion 24/7 background supervisor
    at system startup or user logon with highest available privileges.
#>

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TaskName = "StreamFusionDaemon"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at $PythonExe. Please initialize .venv first."
    exit 1
}

Write-Host "Registering Windows Scheduled Task: $TaskName" -ForegroundColor Cyan
Write-Host "Project Directory: $ProjectRoot"
Write-Host "Python Executable: $PythonExe"

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m stream_fusion.cli daemon run --foreground" `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -AtStartup
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "StreamFusion 24/7 Homelab Ingestion and Multimodal Pipeline Daemon" `
    -Force

Write-Host "[OK] Task $TaskName successfully registered. It will start automatically on boot." -ForegroundColor Green
Write-Host "To start it immediately, run: Start-ScheduledTask -TaskName '$TaskName'"
