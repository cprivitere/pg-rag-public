#!/usr/bin/env pwsh
#MISE description="Start LLM server on :8080 with logging"
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
# Model + flags come from mise.toml [env] (LLM_MODEL/LLM_FLAGS) — single source.

#
# Per-model launch rationale (flags, context, KV, VRAM, golden tallies):
# docs/LLM_MODEL_WIRING.md. Swap models by editing the two [env] vars, then
# `mise llm-stop` → `mise llm-start`. Launchers append only host/port/-np/log-file
# assembly — no model literals here, so this comment never goes stale on swap.
$serverArgs = @('-hf', $env:LLM_MODEL) + ($env:LLM_FLAGS -split '\s+') + @('--host','0.0.0.0','--port','8080','-np','1',"--log-file","$logFile")

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started LLM server (:8080) - log: $logFile"