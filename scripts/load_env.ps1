# Load my_work/.env into the current PowerShell session.
# Usage:  . .\scripts\load_env.ps1

$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (-not (Test-Path $envFile)) {
    Write-Error "Missing $envFile — copy .env.example to .env and add your tokens."
    return
}

Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { return }
    $parts = $line.Split("=", 2)
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    if ($name) {
        Set-Item -Path "Env:$name" -Value $value
    }
}

Write-Host "Loaded environment from $envFile"
