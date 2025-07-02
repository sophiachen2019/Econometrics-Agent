# Get the directory where this script is located
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

# Add agent to PYTHONPATH so metagpt module can be found
$env:PYTHONPATH = "$SCRIPT_DIR\agent;$env:PYTHONPATH"

# Add chatpilot to PYTHONPATH so chatpilot module can be found
$env:PYTHONPATH = "$SCRIPT_DIR\chatpilot;$env:PYTHONPATH"

$KEY_FILE = ".webui_secret_key"

$PORT = if ($env:PORT) { $env:PORT } else { "1280" }

if (-not $env:WEBUI_SECRET_KEY -and -not $env:WEBUI_JWT_SECRET_KEY) {
    Write-Host "No WEBUI_SECRET_KEY provided"

    if (-not (Test-Path $KEY_FILE)) {
        Write-Host "Generating WEBUI_SECRET_KEY"
        # Generate a random value to use as a WEBUI_SECRET_KEY
        $randomBytes = New-Object byte[] 12
        (New-Object Security.Cryptography.RNGCryptoServiceProvider).GetBytes($randomBytes)
        $randomString = [Convert]::ToBase64String($randomBytes)
        $randomString | Out-File -FilePath $KEY_FILE -Encoding ASCII
    }

    Write-Host "Loading WEBUI_SECRET_KEY from $KEY_FILE"
    $env:WEBUI_SECRET_KEY = Get-Content $KEY_FILE -Raw
}

# Kill existing chatpilot processes
Get-Process | Where-Object { $_.ProcessName -like "*python*" -and $_.CommandLine -like "*chatpilot*" } | Stop-Process -Force -ErrorAction SilentlyContinue

# Start the application
gunicorn -k uvicorn.workers.UvicornWorker chatpilot.server:app --bind "0.0.0.0:$PORT" --forwarded-allow-ips '*' -w 2 