#!/usr/bin/env bash
set -e

./test/install_from_local.sh
./test/run_code_checks.sh
./test/run_examples.sh
