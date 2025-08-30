@echo off
echo Running code quality checks...
echo.

echo Running Ruff linting...
uv run ruff check backend/ main.py
set RUFF_EXIT=%ERRORLEVEL%

echo.
echo Running Black check...
uv run black --check backend/ main.py
set BLACK_EXIT=%ERRORLEVEL%

echo.
echo Running isort check...
uv run isort --check-only backend/ main.py
set ISORT_EXIT=%ERRORLEVEL%

echo.
echo Running MyPy type checking...
uv run mypy backend/ main.py
set MYPY_EXIT=%ERRORLEVEL%

echo.
echo Running tests...
uv run pytest backend/tests/
set PYTEST_EXIT=%ERRORLEVEL%

echo.
echo ================================
echo QUALITY CHECK RESULTS:
echo ================================

if %RUFF_EXIT% NEQ 0 (
    echo ❌ Ruff: FAILED
) else (
    echo ✅ Ruff: PASSED
)

if %BLACK_EXIT% NEQ 0 (
    echo ❌ Black: FAILED
) else (
    echo ✅ Black: PASSED
)

if %ISORT_EXIT% NEQ 0 (
    echo ❌ isort: FAILED
) else (
    echo ✅ isort: PASSED
)

if %MYPY_EXIT% NEQ 0 (
    echo ❌ MyPy: FAILED
) else (
    echo ✅ MyPy: PASSED
)

if %PYTEST_EXIT% NEQ 0 (
    echo ❌ Tests: FAILED
) else (
    echo ✅ Tests: PASSED
)

echo.
set /a TOTAL_EXIT=%RUFF_EXIT% + %BLACK_EXIT% + %ISORT_EXIT% + %MYPY_EXIT% + %PYTEST_EXIT%

if %TOTAL_EXIT% NEQ 0 (
    echo Overall: FAILED
    exit /b 1
) else (
    echo Overall: PASSED
    exit /b 0
)