#Requires -Version 5.1
param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Load .env file if present
$EnvFile = Join-Path $ScriptDir ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

# Required environment variables
$RequiredVars = @(
    "CF_TUNNEL_ID",
    "CF_ACCOUNT_TAG",
    "CF_TUNNEL_SECRET",
    "API_DOMAIN",
    "FRONTEND_DOMAIN",
    "IMAGE_REGISTRY"
)

# Validate all required variables are set
foreach ($var in $RequiredVars) {
    $value = [Environment]::GetEnvironmentVariable($var, "Process")
    if ([string]::IsNullOrEmpty($value)) {
        Write-Error "ERROR: Required variable $var is not set. Copy .env.example to .env and fill in the values"
        exit 1
    }
}

# Create generated directory
$GeneratedDir = Join-Path $ScriptDir "generated"
New-Item -ItemType Directory -Force -Path $GeneratedDir | Out-Null

# Process all templates with envsubst (requires: task k8s:install-envsubst)
$TemplatesDir = Join-Path $ScriptDir "templates"
Get-ChildItem -Path $TemplatesDir -Filter "*.yaml" -Recurse | ForEach-Object {
    $relativePath = $_.FullName.Substring($TemplatesDir.Length + 1)
    $outputFile = Join-Path $GeneratedDir $relativePath
    $outputDir = Split-Path -Parent $outputFile
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    # Use envsubst to replace placeholders
    Get-Content $_.FullName | envsubst | Set-Content $outputFile
    Write-Host "Generated: $outputFile"
}

# Apply if -Apply flag provided
if ($Apply) {
    Write-Host "Applying manifests to cluster..."
    kubectl apply -f (Join-Path $GeneratedDir "namespace.yaml")
    kubectl apply -f (Join-Path $GeneratedDir "secrets")
    kubectl apply -f (Join-Path $GeneratedDir "configmaps")
    kubectl apply -f (Join-Path $GeneratedDir "services")
    kubectl apply -f (Join-Path $GeneratedDir "deployments")
    Write-Host "Deployment complete!"
}
