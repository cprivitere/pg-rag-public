#!/usr/bin/env pwsh
#MISE description="Upload data/documents.json to the HF bucket (Nubula/paddock) for the molab notebook"
#MISE alias="up"

# Publishes the locally generated corpus to the bucket the molab notebook
# reads. Explicit on purpose: a rebuild does NOT auto-publish (a bad
# local rebuild must never silently become the molab corpus on next boot).
#
# Auth: requires HF_TOKEN (write scope) in env or `hf auth login` cache.
# The bucket is public-read, so the notebook side needs no token.

param(
    [string]$Source = "",   # default: data/documents.json
    [string]$Dest = ""      # default: buckets/Nubula/paddock/documents.json
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Source)  { $Source = Join-Path $root 'data\documents.json' }
if (-not $Dest)    { $Dest = 'buckets/Nubula/paddock/documents.json' }

if (-not (Test-Path $Source)) {
    Write-Error "Source not found: $Source — run `mise generate-docs` first."
}

$sizeMb = [math]::Round((Get-Item $Source).Length / 1MB, 1)

# Token check BEFORE the ~3 min upload starts.
$pyCheck = @"
import os
from huggingface_hub import HfApi
try:
    who = HfApi(token=os.environ.get('HF_TOKEN') or None).whoami()
    print('authenticated as', who.get('name'))
except Exception as e:
    raise SystemExit('no HF write auth: ' + str(e))
"@
$probe = uv run --with 'huggingface-hub' python -c $pyCheck 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host $probe
    Write-Error "HF_TOKEN missing or invalid. Set it (write scope) or run: hf auth login"
}

Write-Host "Uploading $Source ($sizeMb MB) -> hf://$Dest"
$pyUp = @"
from huggingface_hub import HfFileSystem
import os, shutil
fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))
src = r'$Source'
with open(src, 'rb') as f_in, fs.open('hf://$Dest', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
print('done')
"@
uv run --with 'huggingface-hub' python -c $pyUp
if ($LASTEXITCODE -ne 0) {
    Write-Error "Upload failed."
}

# Verify: read back the remote size and compare.
$pyVerify = @"
from huggingface_hub import HfFileSystem
import os
fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))
info = fs.info('hf://$Dest')
remote = info['size']
local = os.path.getsize(r'$Source')
print(f'remote {remote:,} B / local {local:,} B')
assert remote == local, 'size mismatch after upload'
print('verified')
"@
uv run --with 'huggingface-hub' python -c $pyVerify
if ($LASTEXITCODE -ne 0) {
    Write-Error "Verification failed — remote size differs. Re-run the upload."
} else {
    Write-Host "OK: hf://$Dest matches local $sizeMb MB. Next molab sandbox boot picks it up."
}
