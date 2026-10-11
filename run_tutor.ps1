$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$script = Join-Path $projectRoot "scripts\voice_tutor.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project environment not found at $python. Create it with 'uv sync' from the project root."
}

& $python -c "import whisper, sounddevice, soundfile, kokoro"
if ($LASTEXITCODE -ne 0) {
    throw "The project environment is missing tutor dependencies. From the project root, run 'uv sync' and then '.\run_tutor.ps1'."
}

& $python $script
exit $LASTEXITCODE
