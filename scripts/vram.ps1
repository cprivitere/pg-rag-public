<#
.SYNOPSIS
  Report VRAM (dedicated GPU memory) usage per process on Windows, including AMD cards.
  Uses the DXGI/WDDM GPU performance counters -- the same source as Task Manager's
  "Dedicated GPU memory" column. No nvidia-smi / NVML required.

.DESCRIPTION
  "Dedicated usage" = physical VRAM currently resident for a process on a GPU segment.
  Because llama.cpp reserves large buffers (weights + kv cache + compute) up front, a
  llama-server will show a big steady number even when idle -- that is real allocation,
  not a miscount.

.PARAMETER Filter
  Case-insensitive substring matched against the process name. Default 'llama'.
.PARAMETER All
  Show every process currently holding VRAM, ignoring -Filter.
.PARAMETER Raw
  Print bytes instead of human-readable units.
.PARAMETER Samples
  Number of counter samples to average (damps brief WDDM fluctuations). Default 3.
.PARAMETER Repeat
  If > 0, re-run every N seconds (Ctrl-C to stop). Useful for watching VRAM while a
  model loads or streams.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File vram.ps1
  powershell -ExecutionPolicy Bypass -File vram.ps1 -All
  powershell -ExecutionPolicy Bypass -File vram.ps1 -Filter llm -Repeat 5
#>
param(
  [string]$Filter = 'llama',
  [switch]$All,
  [switch]$Raw,
  [int]$Samples = 3,
  [int]$Repeat = 0
)

$ErrorActionPreference = 'Stop'

function ConvertTo-Human([double]$Bytes) {
  if ($Raw) { return ('{0:N0} B' -f $Bytes) }
  if ($Bytes -ge 1GB) { return ('{0:N2} GB' -f ($Bytes / 1GB)) }
  if ($Bytes -ge 1MB) { return ('{0:N0} MB' -f ($Bytes / 1MB)) }
  return ('{0:N0} KB' -f ($Bytes / 1KB))
}

# instance format: pid_<pid>_luid_<hi>_<lo>_phys_<n>
$instRe = '^pid_(\d+)_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_(\d+)$'

function Get-DedicatedByPid {
  $byPid = @{}
  for ($i = 0; $i -lt $Samples; $i++) {
    $cs = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
    foreach ($s in $cs) {
      if ($s.CookedValue -le 0) { continue }
      if ($s.InstanceName -notmatch $instRe) { continue }
      $pidKey = [int]$Matches[1]
      $luid   = $Matches[2]
      if (-not $byPid.ContainsKey($pidKey)) { $byPid[$pidKey] = @{} }
      if (-not $byPid[$pidKey].ContainsKey($luid)) { $byPid[$pidKey][$luid] = 0 }
      $byPid[$pidKey][$luid] += $s.CookedValue   # accumulate across samples, per-pid per-luid
    }
    if ($i -lt ($Samples - 1)) { Start-Sleep -Milliseconds 150 }
  }
  # average across samples: tracked per pid per luid
  return $byPid
}

function Get-AdapterTotals {
  $agg = @{}
  for ($i = 0; $i -lt $Samples; $i++) {
    $cs = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
    foreach ($s in $cs) {
      if ($s.InstanceName -notmatch '^luid_(0x[0-9a-fA-F]+)_(0x[0-9a-fA-F]+)_phys_(\d+)$') { continue }
      if ($s.CookedValue -le 0) { continue }
      $key = $Matches[1] + '_' + $Matches[2]
      $agg[$key] = $agg[$key] + $s.CookedValue
    }
    if ($i -lt ($Samples - 1)) { Start-Sleep -Milliseconds 150 }
  }
  return $agg
}

try {
  # ---- per-process ----
  $byPid = Get-DedicatedByPid
  $rows = @(foreach ($pidKey in $byPid.Keys) {
    $sum = ($byPid[$pidKey].Values | Measure-Object -Sum).Sum / $Samples
    $name = '?'
    $p = Get-Process -Id $pidKey -ErrorAction SilentlyContinue
    if ($p) { $name = $p.ProcessName }
    [PSCustomObject]@{ Pid = $pidKey; Name = $name; Dedicated = [long]$sum }
  }) | Sort-Object -Property Dedicated -Descending

  $visible = if ($All) { $rows | Where-Object Dedicated -gt 0 } else { $rows | Where-Object { $_.Name -like "*$Filter*" } }

  if ($visible) {
    $label = if ($All) { 'ALL GPU processes' } else { "matches '*$Filter*'" }
    Write-Output ("=== VRAM (dedicated) per process -- $label ===")
    $visible | ForEach-Object {
      Write-Output ("  PID {0,-7} {1,-20} {2}" -f $_.Pid, $_.Name, (ConvertTo-Human $_.Dedicated))
    }
    $tot = ($visible | Measure-Object Dedicated -Sum).Sum
    Write-Output ("  TOTAL {0}" -f (ConvertTo-Human $tot))
  } else {
    Write-Output ("No process name matching '*{0}' is currently holding VRAM." -f $Filter)
  }

  # ---- per-adapter ----
  Write-Output "=== GPU adapter VRAM (dedicated) ==="
  $adp = Get-AdapterTotals
  foreach ($k in ($adp.Keys | Sort-Object { $adp[$_] } -Descending)) {
    Write-Output ("  LUID {0}  {1}" -f $k, (ConvertTo-Human ($adp[$k] / $Samples)))
  }
  Write-Output ""
  Write-Output "Note: per-process 'Dedicated' figures are COMMITTED memory (same as Task Manager's"
  Write-Output "'Dedicated GPU memory' column) and may over-sum when added up (shared/aliased buffers,"
  Write-Output "committed-but-not-resident). The GPU ADAPTER total above is the ACTUAL VRAM resident on"
  Write-Output "the card right now -- trust it for true VRAM pressure. For a single llama-server both"
  Write-Output "agree closely, since it is the dominant allocation on the adapter."
} catch {
  Write-Error ("GPU memory counter query failed: " + $_.Exception.Message)
  Write-Error "These counters require a WDDM 2.x GPU driver (standard on Win10/11 incl. AMD)."
}

if ($Repeat -gt 0) {
  Write-Output ("Repeating every {0}s (Ctrl-C to stop)..." -f $Repeat)
  Start-Sleep -Seconds $Repeat
  & $PSCommandPath -Filter $Filter -All:$All -Raw:$Raw -Samples $Samples -Repeat $Repeat
}