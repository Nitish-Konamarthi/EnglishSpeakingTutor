$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$defaultModel = Join-Path $projectRoot 'models\llm\Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
$defaultServer = Join-Path $projectRoot 'runtime\llama.cpp\llama-server.exe'
$appHost = '127.0.0.1'
$appPort = if ($env:TUTOR_APP_PORT) { [int]$env:TUTOR_APP_PORT } else { 8765 }
$serverUrl = if ($env:TUTOR_SERVER_URL) { $env:TUTOR_SERVER_URL } else { 'http://127.0.0.1:8080' }
$modelPath = if ($env:TUTOR_LLM_MODEL) { $env:TUTOR_LLM_MODEL } else { $defaultModel }
$serverExe = if ($env:TUTOR_LLAMA_SERVER_EXE) { $env:TUTOR_LLAMA_SERVER_EXE } else { $defaultServer }
$ctxSize = if ($env:TUTOR_CONTEXT_SIZE) { $env:TUTOR_CONTEXT_SIZE } else { '8192' }
$gpuLayers = if ($env:TUTOR_GPU_LAYERS) { $env:TUTOR_GPU_LAYERS } else { '20' }
$readyTimeout = if ($env:TUTOR_SERVER_TIMEOUT_SECONDS) { [int]$env:TUTOR_SERVER_TIMEOUT_SECONDS } else { 300 }
$managedServer = $null
$backend = $null
$stopRequested = $false
$previousShutdownToken = $env:TUTOR_SHUTDOWN_TOKEN
$shutdownToken = $null

function Get-LoadedModel {
    param([string]$BaseUrl)
    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -TimeoutSec 2
        if ($response.data -and $response.data.Count -gt 0) { return [string]$response.data[0].id }
    } catch { return $null }
    return $null
}

function Test-ExpectedModel {
    param([string]$ModelId)
    if (-not $ModelId) { return $false }
    if ($env:TUTOR_MODEL_ID) { return $ModelId -like "*$($env:TUTOR_MODEL_ID)*" }
    $expected = [IO.Path]::GetFileNameWithoutExtension($modelPath)
    return $ModelId -like "*$expected*"
}

function Stop-OwnedProcess {
    param($Process, [string]$Description)
    if ($null -ne $Process) {
        try {
            if (-not $Process.HasExited) {
                Stop-Process -Id $Process.Id -Force -ErrorAction Stop
                Write-Host "Stopped managed $Description (PID $($Process.Id))."
            }
        } catch { Write-Warning "Could not stop managed $Description (PID $($Process.Id)): $($_.Exception.Message)" }
    }
}

function Stop-ManagedBackend {
    param($Process, [string]$BaseUrl, [string]$Token)
    if ($null -eq $Process -or $Process.HasExited) { return }
    try {
        Invoke-RestMethod -Uri "$BaseUrl/api/v1/admin/shutdown" -Method Post -Headers @{ 'X-Tutor-Shutdown-Token' = $Token } -TimeoutSec 3 | Out-Null
    } catch { Write-Warning "Graceful tutor shutdown request failed: $($_.Exception.Message)" }
    $deadline = [DateTime]::UtcNow.AddSeconds(20)
    while (-not $Process.HasExited -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 250 }
    if (-not $Process.HasExited) {
        Write-Warning "Tutor backend did not exit cleanly within 20 seconds; stopping only managed PID $($Process.Id)."
        Stop-OwnedProcess $Process 'tutor backend'
    } else {
        Write-Host "Tutor backend completed its graceful shutdown (PID $($Process.Id))."
    }
}

try {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Project Python environment is missing: $python. Create the Python 3.12 .venv and install dependencies with: .\.venv\Scripts\python.exe -m pip install -e ."
    }
    & $python -c "import fastapi, uvicorn, whisper, soundfile, kokoro, sentence_transformers, sqlite_vec"
    if ($LASTEXITCODE -ne 0) { throw 'A required Python dependency is missing. Run: .\.venv\Scripts\python.exe -m pip install -e .' }

    $serverUri = $null
    if (-not [Uri]::TryCreate($serverUrl, [UriKind]::Absolute, [ref]$serverUri)) { throw "TUTOR_SERVER_URL is not a valid absolute URL: $serverUrl" }
    if ($serverUri.Scheme -ne 'http' -or $serverUri.Host -notin @('127.0.0.1', 'localhost', '::1')) {
        throw "llama-server must use a local HTTP loopback address. Configure TUTOR_SERVER_URL with 127.0.0.1, localhost, or ::1; got '$serverUrl'."
    }
    $serverBase = $serverUri.GetLeftPart([UriPartial]::Authority)
    $serverPort = $serverUri.Port
    $serverHost = $serverUri.Host
    if (-not (Test-Path -LiteralPath $serverExe -PathType Leaf)) { throw "llama-server executable is missing: $serverExe. Set TUTOR_LLAMA_SERVER_EXE to the installed llama-server.exe." }
    if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) { throw "Qwen GGUF model is missing: $modelPath. Set TUTOR_LLM_MODEL to the existing GGUF model file." }

    $loadedModel = Get-LoadedModel $serverBase
    if ($loadedModel) {
        if (-not (Test-ExpectedModel $loadedModel)) {
            $expected = if ($env:TUTOR_MODEL_ID) { $env:TUTOR_MODEL_ID } else { [IO.Path]::GetFileNameWithoutExtension($modelPath) }
            throw "A llama-server is already healthy at $serverBase but serves model '$loadedModel', not the configured '$expected'. It was left untouched. Choose another TUTOR_SERVER_URL or configure TUTOR_MODEL_ID if this alias is intentional."
        }
        Write-Host "Reusing healthy llama-server at $serverBase (model: $loadedModel)."
    } elseif ($env:TUTOR_MANAGE_LLAMA -eq '0') {
        throw "No ready llama-server with a loaded model is available at $serverBase, and TUTOR_MANAGE_LLAMA=0. Start the local server or enable managed startup."
    } else {
        $portListener = Get-NetTCPConnection -State Listen -LocalPort $serverPort -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($portListener) { throw "Port $serverPort is occupied by PID $($portListener.OwningProcess), but it does not serve a compatible loaded model at $serverBase. The process was left untouched; choose another TUTOR_SERVER_URL or resolve the port owner." }
        $logDir = Join-Path $projectRoot 'logs'
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        $stdout = Join-Path $logDir 'llama-server.stdout.log'
        $stderr = Join-Path $logDir 'llama-server.stderr.log'
        $llamaArgs = @('--host', $serverHost, '--port', [string]$serverPort, '--model', $modelPath, '--ctx-size', $ctxSize, '--n-gpu-layers', $gpuLayers)
        if ($env:TUTOR_LLAMA_ARGS_JSON) {
            try { $llamaArgs += @(ConvertFrom-Json -InputObject $env:TUTOR_LLAMA_ARGS_JSON -ErrorAction Stop) }
            catch { throw "TUTOR_LLAMA_ARGS_JSON must be a JSON array of additional arguments, such as '[`"--threads`",`"8`"]'. Details: $($_.Exception.Message)" }
        }
        Write-Host "Starting managed llama-server with model $([IO.Path]::GetFileName($modelPath)) on $serverBase."
        $managedServer = Start-Process -FilePath $serverExe -ArgumentList $llamaArgs -WorkingDirectory (Split-Path -Parent $serverExe) -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $deadline = [DateTime]::UtcNow.AddSeconds($readyTimeout)
        $loadedModel = $null
        while ([DateTime]::UtcNow -lt $deadline) {
            if ($managedServer.HasExited) { throw "Managed llama-server exited early with code $($managedServer.ExitCode). See $stderr and $stdout. Check GPU memory, model access, and server arguments." }
            $loadedModel = Get-LoadedModel $serverBase
            if ($loadedModel) { break }
            Start-Sleep -Milliseconds 1000
        }
        if (-not $loadedModel) { throw "llama-server did not report a loaded model at $serverBase within $readyTimeout seconds. See $stderr and $stdout; check model loading and GPU memory." }
        if (-not (Test-ExpectedModel $loadedModel)) { throw "llama-server became ready with unexpected model '$loadedModel'. Expected a model matching '$([IO.Path]::GetFileNameWithoutExtension($modelPath))'. See $stderr." }
        Write-Host "Managed llama-server is ready (model: $loadedModel, PID $($managedServer.Id))."
    }

    $existingApp = Get-NetTCPConnection -State Listen -LocalPort $appPort -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existingApp) {
        try { $null = Invoke-RestMethod -Uri "http://${appHost}:${appPort}/api/v1/health" -TimeoutSec 2 }
        catch { throw "Tutor port $appPort is already in use by PID $($existingApp.OwningProcess), but its health route is unavailable. The process was left untouched. Set TUTOR_APP_PORT to a free port." }
        throw "The English Speaking Tutor already appears to be running at http://${appHost}:${appPort}. Its process was left untouched. Set TUTOR_APP_PORT to another port to start another copy."
    }

    $logDir = Join-Path $projectRoot 'logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $backendLog = Join-Path $logDir 'tutor-app.stdout.log'
    $backendError = Join-Path $logDir 'tutor-app.stderr.log'
    $env:TUTOR_SERVER_URL = $serverBase
    $shutdownToken = [Guid]::NewGuid().ToString('N')
    $env:TUTOR_SHUTDOWN_TOKEN = $shutdownToken
    $backendScript = Join-Path $projectRoot 'scripts\run_backend.py'
    $backend = Start-Process -FilePath $python -ArgumentList @($backendScript, '--host', $appHost, '--port', [string]$appPort) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $backendLog -RedirectStandardError $backendError
    $appUrl = "http://${appHost}:${appPort}"
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    $appReady = $false
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($backend.HasExited) { throw "Tutor backend exited early with code $($backend.ExitCode). See $backendError and $backendLog." }
        try {
            $health = Invoke-RestMethod -Uri "$appUrl/api/v1/health" -TimeoutSec 2
            if ($health.ok) { $appReady = $true; break }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    if (-not $appReady) { throw "Tutor backend did not become healthy at $appUrl within 45 seconds. See $backendError and $backendLog." }
    $diagnostics = Invoke-RestMethod -Uri "$appUrl/api/v1/diagnostics" -TimeoutSec 5
    if (-not $diagnostics.llama_server.available -or -not (Test-ExpectedModel ([string]$diagnostics.llama_server.model))) {
        throw "Tutor backend is healthy, but its LLM diagnostic does not match the configured Qwen model. Details: $($diagnostics.llama_server | ConvertTo-Json -Compress)"
    }
    Write-Host "English Speaking Tutor is ready at $appUrl (model: $($diagnostics.llama_server.model))."
    Write-Host "Logs: $logDir"
    if ($env:TUTOR_OPEN_BROWSER -ne '0') { Start-Process $appUrl }
    Write-Host 'Keep this window open. Press Ctrl+C to stop the tutor and its managed llama-server, if this script started it.'
    while (-not $stopRequested -and -not $backend.HasExited) { Start-Sleep -Seconds 1 }
    if ($backend.HasExited) { throw "Tutor backend stopped with exit code $($backend.ExitCode). See $backendError and $backendLog." }
} catch {
    Write-Error $_
    exit 1
} finally {
    Stop-ManagedBackend $backend $appUrl $shutdownToken
    Stop-OwnedProcess $managedServer 'llama-server'
    if ($null -eq $previousShutdownToken) { Remove-Item Env:TUTOR_SHUTDOWN_TOKEN -ErrorAction SilentlyContinue }
    else { $env:TUTOR_SHUTDOWN_TOKEN = $previousShutdownToken }
}
