#!/usr/bin/env bash
set -e

./test/install_from_local.sh
./test/run_code_checks_3_7_and_up.sh
./test/run_examples.sh
