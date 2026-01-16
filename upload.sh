#!/bin/bash

# in .env
# TWINE_USERNAME=__token__
# TWINE_PASSWORD=pypi-xxxxxxxx

# --- Load .env file ---
if [ -f .env ]; then
    echo "Loading variables from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "ERROR: .env file not found!"
    exit 1
fi

# delete stuff in dist
rm -rf dist

# Create a temporary Virtual Environment
poetry build

#  Metadata Check
echo "Checking metadata with twine..."
poetry run twine check dist/*

#  Publish

read -p "Upload to TestPyPI (t) or PyPI (p)? " -n 1 -r
echo
if [[ $REPLY =~ ^[Tt]$ ]]; then
    REPO="https://test.pypi.org/legacy/"
elif [[ $REPLY =~ ^[Pp]$ ]]; then
    REPO="https://upload.pypi.org/legacy/"
else
    echo "Cancelled."
    exit 1
fi

poetry run twine upload --repository-url $REPO dist/*