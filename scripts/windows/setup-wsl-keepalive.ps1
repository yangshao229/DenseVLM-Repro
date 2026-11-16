<#
.SYNOPSIS
  Register a logon task that starts this WSL distro headless and keeps it alive.

.DESCRIPTION
  tailscaled runs inside the WSL distro. When every WSL instance stops, the
  distro's tailnet node (100.86.118.15) goes offline and remote access from the
  Mac dies. This task starts `wsl -d <Distro> -u root /home/lenovo/bin/wsl-keepalive.sh`
  at logon; that script starts tailscaled (systemd, already enabled) and then
  blocks on `sleep infinity` so the VM is not torn down by the idle timeout.
  It runs through wscript.exe so no console window flashes.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-wsl-keepalive.ps1
#>
param(
    [string]$Distro = "Ubuntu",
    [string]$TaskName = "WSL-Ubuntu-KeepAlive",
    [string]$KeepAlive = "/home/lenovo/bin/wsl-keepalive.sh"
)

$ErrorActionPreference = "Stop"

$available = @(wsl -l -q | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ })
if ($available -notcontains $Distro) {
    throw "WSL distro '$Distro' not found. Available: $($available -join ', ')"
}
Write-Host "target distro: $Distro"

$vbsDir = Join-Path $env:LOCALAPPDATA "DenseVLM"
New-Item -ItemType Directory -Force -Path $vbsDir | Out-Null
$vbs = Join-Path $vbsDir "wsl-keepalive.vbs"

$vbsBody = @"
' Start the WSL distro headless so tailscaled stays reachable on the tailnet.
CreateObject("WScript.Shell").Run "wsl.exe -d $Distro -u root $KeepAlive", 0, False
"@
Set-Content -Path $vbs -Value $vbsBody -Encoding ASCII
Write-Host "wrote $vbs"

$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\wscript.exe" -Argument "`"$vbs`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Force `
    -Description "Keep the $Distro WSL distro running so Tailscale stays online for remote access." | Out-Null

Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 4

Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo |
    Format-List TaskName, State, LastRunTime, LastTaskResult
wsl -l -v
