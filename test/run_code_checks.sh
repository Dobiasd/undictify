#!/usr/bin/env bash
set -e

find undictify -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint
find undictify -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 mypy --strict
find undictify -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 python3 -m unittest
