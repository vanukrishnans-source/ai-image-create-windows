# Assemble the portable zip from the PyInstaller onedir output.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$dist = Join-Path $root "dist\AIImageCreate"
$stage = Join-Path $root "dist\pkg\AIImageCreate"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item -Recurse "$dist\*" $stage
Copy-Item (Join-Path $root "README.md") $stage -Force
Copy-Item (Join-Path $root "THIRD_PARTY.md") $stage -Force
Copy-Item (Join-Path $root "LICENSE") $stage -Force
$ver = (Get-Content (Join-Path $root "aic\__init__.py") | Select-String '__version__\s*=\s*"([^"]+)"').Matches.Groups[1].Value
$zip = Join-Path $root "dist\AIImageCreate-$ver-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
$sha = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
$size = (Get-Item $zip).Length
"$sha  AIImageCreate-$ver-win64.zip" | Out-File -Encoding ascii (Join-Path $root "dist\SHA256SUMS")
Write-Host "ZIP $zip ($size bytes)"
Write-Host "SHA256 $sha"
