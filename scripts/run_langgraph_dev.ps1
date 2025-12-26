# Script to run langgraph dev with LANGSMITH_API_KEY

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
$rootEnv = Join-Path $projectRoot ".env"
$langgraphDir = Join-Path $projectRoot "src\langgraph_server"
$langgraphEnv = Join-Path $langgraphDir ".env"

# Check root .env
$rootEnvExists = Test-Path $rootEnv
if ($rootEnvExists -eq $false) {
    Write-Host "File .env not found in project root" -ForegroundColor Red
    Write-Host "Create .env file with LANGSMITH_API_KEY"
    exit 1
}

# Read LANGSMITH_API_KEY from root .env
$langsmithKey = $null
$lines = Get-Content $rootEnv
foreach ($line in $lines) {
    $trimmedLine = $line.Trim()
    if ($trimmedLine -like "LANGSMITH_API_KEY=*") {
        $keyValue = $trimmedLine.Substring(19)
        $langsmithKey = $keyValue.Trim()
        if ($langsmithKey.StartsWith('"')) {
            $langsmithKey = $langsmithKey.TrimStart('"').TrimEnd('"')
        }
        if ($langsmithKey.StartsWith("'")) {
            $langsmithKey = $langsmithKey.TrimStart("'").TrimEnd("'")
        }
        break
    }
}

# If not found, check langgraph_server/.env
$langgraphEnvExists = Test-Path $langgraphEnv
if ($langsmithKey -eq $null) {
    if ($langgraphEnvExists) {
        $lines = Get-Content $langgraphEnv
        foreach ($line in $lines) {
            $trimmedLine = $line.Trim()
            if ($trimmedLine -like "LANGSMITH_API_KEY=*") {
                $keyValue = $trimmedLine.Substring(19)
                $langsmithKey = $keyValue.Trim()
                if ($langsmithKey.StartsWith('"')) {
                    $langsmithKey = $langsmithKey.TrimStart('"').TrimEnd('"')
                }
                if ($langsmithKey.StartsWith("'")) {
                    $langsmithKey = $langsmithKey.TrimStart("'").TrimEnd("'")
                }
                break
            }
        }
    }
}

# Set environment variable
if ($langsmithKey -ne $null) {
    $env:LANGSMITH_API_KEY = $langsmithKey
    Write-Host "LANGSMITH_API_KEY set" -ForegroundColor Green
} else {
    Write-Host "LANGSMITH_API_KEY not found in .env files" -ForegroundColor Yellow
    Write-Host "LangGraph will work without monitoring"
}

# Change to langgraph_server directory and run
Set-Location $langgraphDir

Write-Host ""
Write-Host "Starting langgraph dev..." -ForegroundColor Cyan
Write-Host ""

langgraph dev
