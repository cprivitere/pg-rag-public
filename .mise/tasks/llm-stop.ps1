#!/usr/bin/env pwsh
#MISE description="Stop LLM server (kills process on port 8080)"
#MISE alias="xl"

param()

$p = $null
try {
    $portProc = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction Stop
    if ($portProc) {
        $p = Get-Process -Id $portProc.OwningProcess -ErrorAction Stop
    }
} catch { }

if (-not $p) {
    $p = Get-Process -Name llama-server -ErrorAction SilentlyContinue | Where-Object {
        try {
            $_ | Get-NetTCPConnection -LocalPort 8080 -ErrorAction Stop
            $true
        } catch { $false }
    }
}

if ($p) {
    foreach ($proc in @($p)) {
        taskkill /PID $proc.Id /T /F 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            "Failed to stop PID $($proc.Id) (taskkill exit $LASTEXITCODE)"
            exit 1
        }
    }
    "Stopped LLM"
} else {
    "Not running"
}