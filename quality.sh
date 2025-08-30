#!/bin/bash
echo "Running code quality checks..."
echo

echo "Running Ruff linting..."
uv run ruff check backend/ main.py
RUFF_EXIT=$?

echo
echo "Running Black check..."
uv run black --check backend/ main.py
BLACK_EXIT=$?

echo
echo "Running isort check..."
uv run isort --check-only backend/ main.py
ISORT_EXIT=$?

echo
echo "Running MyPy type checking..."
uv run mypy backend/ main.py
MYPY_EXIT=$?

echo
echo "Running tests..."
uv run pytest backend/tests/
PYTEST_EXIT=$?

echo
echo "================================"
echo "QUALITY CHECK RESULTS:"
echo "================================"

if [ $RUFF_EXIT -ne 0 ]; then
    echo "❌ Ruff: FAILED"
else
    echo "✅ Ruff: PASSED"
fi

if [ $BLACK_EXIT -ne 0 ]; then
    echo "❌ Black: FAILED"
else
    echo "✅ Black: PASSED"
fi

if [ $ISORT_EXIT -ne 0 ]; then
    echo "❌ isort: FAILED"
else
    echo "✅ isort: PASSED"
fi

if [ $MYPY_EXIT -ne 0 ]; then
    echo "❌ MyPy: FAILED"
else
    echo "✅ MyPy: PASSED"
fi

if [ $PYTEST_EXIT -ne 0 ]; then
    echo "❌ Tests: FAILED"
else
    echo "✅ Tests: PASSED"
fi

echo
TOTAL_EXIT=$((RUFF_EXIT + BLACK_EXIT + ISORT_EXIT + MYPY_EXIT + PYTEST_EXIT))

if [ $TOTAL_EXIT -ne 0 ]; then
    echo "Overall: FAILED"
    exit 1
else
    echo "Overall: PASSED"
    exit 0
fi