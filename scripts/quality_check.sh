#!/bin/bash

echo "Running code quality checks..."

echo "\nRunning black..."
black src/ || exit 1

echo "\nRunning isort..."
isort src/ || exit 1

echo "\nRunning mypy..."
mypy src/ || exit 1

echo "\nRunning flake8..."
flake8 src/ || exit 1

echo "\nRunning tests..."
pytest || exit 1

echo "\nAll checks passed!" 