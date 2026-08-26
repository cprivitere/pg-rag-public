#!/usr/bin/env pwsh
#MISE description="Stop Gradio chat (kills process on port 7860)"
#MISE alias="xc"

param()

$p = $null
try {
    $portProc = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction Stop
    if ($portProc) {
        $p = Get-Process -Id $portProc.OwningProcess -ErrorAction Stop
    }
} catch { }

if (-not $p) {
    $p = Get-Process -Name uv -ErrorAction SilentlyContinue | Where-Object {
        try {
            $_ | Get-NetTCPConnection -LocalPort 7860 -ErrorAction Stop
            $true
        } catch { $false }
    }
}

if (-not $p) {
    $p = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
        try {
            $_ | Get-NetTCPConnection -LocalPort 7860 -ErrorAction Stop
            $true
        } catch { $false }
    }
}

if ($p) {
    foreach ($proc in @($p)) {
        taskkill /PID $proc.Id /T /F 2>$null | Out-Null
    }
    "Stopped Chat"
} else {
    "Not running"
}