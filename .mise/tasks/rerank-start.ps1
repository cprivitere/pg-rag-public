#!/usr/bin/env pwsh
#MISE description="Start reranker server — bge-reranker-v2-m3 cross-encoder (:8082) with logging"
#MISE alias="sr"
#MISE depends=["ensure-logs-dir"]

param(
    # Model comes from the mise env single source (RERANK_MODEL). `-m` accepts
    # a local path; `-hf` a repo:quant ref. Empty + no env => loud error.
    [string]$Model = '',
    # -Ctx is capped by the model: llama-server clamps the slot ctx to the
    # model's training ctx (8192) regardless of the requested -c, so 32768
    # above already allocates 8192. vram_sweep confirmed ctx variants are
    # byte-identical (all 309 MB) — only -Batch/-ubatch are a real VRAM lever,
    # and lowering batch to 4096 measured +134 MB (worse). Both left as-is.
    [int]$Ctx = 32768,
    [int]$Batch = 8192
)

if (-not $Model) { $Model = $env:RERANK_MODEL }
if (-not $Model) {
    Write-Error 'RERANK_MODEL not set — run via `mise start` so the [env] single source exports it'
    exit 1
}
if (-not $env:RERANK_FLAGS) {
    Write-Error 'RERANK_FLAGS not set — run via `mise start` so the [env] single source exports it'
    exit 1
}
$rerankFlags = @($env:RERANK_FLAGS -split '\s+')

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-NetTCPConnection -LocalPort 8082 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Reranker server already running on :8082 (PID $($existing.OwningProcess))"
    exit 0
}

$exe = 'llama-server'
$logFile = Join-Path $logDir 'rerank.log'
$common = @('--alias','bge-reranker-v2-m3','--host','0.0.0.0','--port','8082') + $rerankFlags + @('-c',"$Ctx",'-b',"$Batch",'-ub',"$Batch","--log-file","$logFile")
if (Test-Path -LiteralPath $Model) {
    $serverArgs = @('-m',$Model) + $common
} else {
    $serverArgs = @('-hf',$Model) + $common
}

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started reranker server (:8082) - log: $logFile"
