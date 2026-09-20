# molab sandbox helper for the molab-notebook skill.
#
# Connection info (url + token) is cached per sandbox id, first use, keyed by
# the pasted connection block the user sends. On a fresh sandbox the user
# pastes ONE block that contains the URL and token - this script parses it,
# caches it, and remembers it for later sessions.
#
# Usage:
#   molab-connect.ps1 -Cache                                            # list cached sandboxes
#   molab-connect.ps1 -Set -Url URL -Token TOKEN [-SandboxId SBID]      # manual upsert
#   molab-connect.ps1 -Info                                             # show info + kernel probe
#   molab-connect.ps1 -Pasteline "user-pasted block" [-Save]             # parse (+ cache)
#   molab-connect.ps1 <execute-code.ps1 args...>                         # delegate
#
# Delegation runs the sibling execute-code.ps1 with -Url/-Token filled from
# cache unless the caller supplied them.

[CmdletBinding()]
param(
    [string]$Pasteline,
    [switch]$Save,
    [switch]$Cache,
    [switch]$Set,
    [string]$Url,
    [string]$Token,
    [string]$SandboxId,
    [switch]$Info,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Pass
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$execScript = Join-Path $scriptDir "execute-code.ps1"
if (-not (Test-Path $execScript)) {
    Write-Error "execute-code.ps1 not found next to molab-connect.ps1 ($scriptDir)"
}



function Get-CachePath {
    Join-Path $env:USERPROFILE ".marimo\molab-connections.json"
}

function Get-Cache {
    $p = Get-CachePath
    if (Test-Path $p) {
        try { return (Get-Content $p -Raw | ConvertFrom-Json) } catch { }
    }
    return @()
}

function Set-Cache {
    param($Entries)
    $dir = Split-Path -Parent (Get-CachePath)
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $Entries | ConvertTo-Json -Depth 5 | Set-Content -Path (Get-CachePath) -Encoding UTF8
}

# --- Parse a pasted connection block -----------------------------------------
# Accepts prose like:
#   "Connect to the notebook at: https://sb-....sb.molab.run/"
#   "Use this auth token when calling execute-code.sh: ... --token XXX"
function Parse-Pasteline {
    param([string]$Text)

    $urlMatch = [regex]::Match($Text, "https?://sb-[0-9a-fA-F]+\.sb\.molab\.run/?")
    $tokMatch = [regex]::Match($Text, "--token\s+([0-9a-fA-F]{64})")
    if (-not $tokMatch.Success) {
        $tokMatch = [regex]::Match($Text, "(?i)token[:\s=`"]+([0-9a-fA-F]{64})")
    }
    $url = $urlMatch.Value
    $token = if ($tokMatch.Success) { $tokMatch.Groups[1].Value } else { $null }
    return $url, $token
}

function Sandbox-Id {
    param([string]$Url)
    if ($Url -match "sb-([0-9a-fA-F]+)\.sb\.molab\.run") {
        return "sb-" + $Matches[1]
    }
    return $null
}

function Upsert-Entry {
    param([string]$Sid, [string]$EntryUrl, [string]$EntryToken)
    $sandboxes = @(Get-Cache)
    $entry = [pscustomobject]@{ sandbox = $Sid; url = $EntryUrl; token = $EntryToken; saved = (Get-Date -Format "yyyy-MM-dd") }
    $sandboxes = @($sandboxes | Where-Object { $_.sandbox -ne $Sid }) + $entry
    Set-Cache $sandboxes
}

if ($Pasteline) {
    $url, $token = Parse-Pasteline $Pasteline
    if (-not $url) { Write-Error "No molab URL found in the pasted block." }
    if (-not $token) { Write-Error "No 64-hex-char token found in the pasted block." }
    $sid = Sandbox-Id $url
    Write-Host "parsed: url=$url sandbox=$sid token=$($token.Substring(0,8))..."
    if ($Save) {
        Upsert-Entry -Sid $sid -EntryUrl $url -EntryToken $token
        Write-Host "cached to $(Get-CachePath)"
    }
    exit 0
}

if ($Set) {
    if (-not $Url -or -not $Token) { Write-Error "-Set requires -Url and -Token" }
    $sid = if ($SandboxId) { $SandboxId } else { Sandbox-Id $Url }
    Upsert-Entry -Sid $sid -EntryUrl $Url -EntryToken $Token
    Write-Host "set sandbox $sid"
    exit 0
}

if ($Cache) {
    $sandboxes = @(Get-Cache)
    if ($sandboxes.Count -eq 0) { Write-Host "no cached sandboxes"; exit 0 }
    foreach ($e in $sandboxes) {
        Write-Host ("{0}  {1}  (saved {2})" -f $e.sandbox, $e.url, $e.saved)
    }
    exit 0
}

# --- Resolve connection for delegation ---------------------------------------
$sandboxes = @(Get-Cache)
if ($SandboxId) {
    $entry = $sandboxes | Where-Object { $_.sandbox -eq $SandboxId } | Select-Object -First 1
    if (-not $entry) { Write-Error "No cached sandbox '$SandboxId'. Use -Set or -Pasteline -Save first." }
} else {
    $entry = $sandboxes | Select-Object -Last 1
    if (-not $entry) {
        Write-Error "No cached molab connection. Paste a connect block with -Pasteline -Save first, or use -Set."
    }
}

if (-not $Url) { $Url = $entry.url }
if (-not $Token) { $Token = $entry.token }

if ($Info) {
    Write-Host "sandbox: $($entry.sandbox)"
    Write-Host "url:     $Url"
    Write-Host "token:   $($Token.Substring(0,8))..."
    try {
        & $execScript -Url $Url -Token $Token -Code "import marimo; print('kernel probe OK - marimo', marimo.__version__)"
    } catch {
        Write-Host "probe failed: $_"
        exit 1
    }
    exit 0
}

# --- Delegate to execute-code.ps1 --------------------------------------------
$execArgs = @{
    Url = $Url
}
if ($Token) { $execArgs.Token = $Token }
if ($Pass) {
    # Pass-through extras: -Code, -File, positional script path...
    for ($i = 0; $i -lt $Pass.Count; $i++) {
        $a = $Pass[$i]
        if ($a -match '^-(\w+)$' -and $i + 1 -lt $Pass.Count) {
            $execArgs[$Matches[1]] = $Pass[$i + 1]
            $i++
        } elseif ($a -notmatch '^-') {
            $execArgs["ScriptPath"] = $a
        }
    }
}

& $execScript @execArgs
exit $LASTEXITCODE
