@echo off
setlocal

REM Set proxy environment variables
@REM set http_proxy=http://127.0.0.1:21882
@REM set https_proxy=http://127.0.0.1:21882

REM Create necessary directories
if not exist "docs" mkdir docs

REM Check if backend directory exists
if not exist "backend" (
    echo Error: backend directory not found
    exit /b 1
)

echo Starting Course Materials RAG System...
echo Make sure you have set your DEEPSEEK_API_KEY in .env
@REM echo Proxy set to: %http_proxy%

REM Save current directory and change to backend directory
pushd backend

REM Start the server with trace logging
uv run uvicorn app:app --reload --port 8000 --log-level trace

REM Return to original directory after server stops (even if Ctrl+C)
popd

REM End local environment to restore original state
endlocal