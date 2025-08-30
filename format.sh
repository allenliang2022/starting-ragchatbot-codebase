#!/bin/bash
echo "Running code formatting..."
echo

echo "Running Black..."
uv run black backend/ main.py

echo
echo "Running isort..."
uv run isort backend/ main.py

echo
echo "Code formatting complete!"