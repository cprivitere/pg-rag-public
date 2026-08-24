#!/usr/bin/env pwsh
#MISE description="Start LLM server (Qwen3.5-9B on :8080) with logging"
#MISE alias="sl"
#MISE depends=["ensure-logs-dir"]

param()

if (-not $env:LLM_MODEL) {
    Write-Error 'LLM_MODEL not set — run via `mise start` so the [env] single source exports it'
    exit 1
}
if (-not $env:LLM_FLAGS) {
    Write-Error 'LLM_FLAGS not set — run via `mise start` so the [env] single source exports it'
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "LLM server already running on :8080 (PID $($existing.OwningProcess))"
    exit 0
}

$exe = 'llama-server'
$logFile = Join-Path $logDir 'llm.log'
# Qwen3.5-9B-MTP: MTP card config — `--spec-type draft-mtp --spec-draft-n-max 6`
# (acceptance ~0.63, ~140 tok/s). Thinking ON with `--reasoning-budget 4096`
# (in LLM_FLAGS): caps deliberation so a bigger context can't starve `content` —
# unbounded thinking over >10k tokens emitted empty; the budget guarantees it.
# CONTEXT_BUDGET is 68000 chars (~16k tok) → -c 32768 covers prompt + 8192 output.
# VRAM is flat (~8 GB) from 16K to native 262144, so more is free if needed.
$serverArgs = @('-hf', $env:LLM_MODEL) + ($env:LLM_FLAGS -split '\s+') + @('--host','0.0.0.0','--port','8080','-np','1',"--log-file","$logFile")

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started LLM server (:8080) - log: $logFile"