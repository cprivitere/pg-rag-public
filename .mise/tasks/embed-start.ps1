#!/usr/bin/env pwsh
#MISE description="Start embedding server with logging"
#MISE alias="se"
#MISE depends=["ensure-logs-dir"]

param()

if (-not $env:EMBED_MODEL) {
    Write-Error 'EMBED_MODEL not set — run via `mise start` so the [env] single source exports it'
    exit 1
}
if (-not $env:EMBED_FLAGS) {
    Write-Error 'EMBED_FLAGS not set — run via `mise start` so the [env] single source exports it'
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Embedding server already running on :8081 (PID $($existing.OwningProcess))"
    exit 0
}

$exe = 'llama-server'
$logFile = Join-Path $logDir 'embed.log'
$serverArgs = @('-hf', $env:EMBED_MODEL) + ($env:EMBED_FLAGS -split '\s+') + @('--host','0.0.0.0','--port','8081','-b','4096','--ubatch-size','4096','-np','1',"--log-file","$logFile")

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started embedding server (:8081) - log: $logFile"
