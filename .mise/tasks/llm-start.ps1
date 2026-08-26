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
# Current: gemma-4-12B-it-qat — MTP draft (auto-discovered, no --model-draft),
# native thinking `--reasoning-budget 1024` (NOT 4096: gemma-4's verbose
# thinking empties `content`), `--no-mmproj` (text-only RAG), `-c 32768` (KV
# shaved; reasoning@1024 + ≤19k ctx + ≤8k ans ≈ ≤28.5k < 32k at typical ~4.2
# chars/token; densest ~2.4 chars/token (config.py note) can approach the
# 32k edge — accepted residual, never observed in goldens), Q8_0 KV (never
# Q4 V-axis: facts). Measured 8,019 MiB. RAG context (CONTEXT_BUDGET=80000 chars in
# config.py) is hard-capped by _fit_context; the densest-content worst case
# is an accepted config.py residual, never observed in goldens.
$serverArgs = @('-hf', $env:LLM_MODEL) + ($env:LLM_FLAGS -split '\s+') + @('--host','0.0.0.0','--port','8080','-np','1',"--log-file","$logFile")

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started LLM server (:8080) - log: $logFile"