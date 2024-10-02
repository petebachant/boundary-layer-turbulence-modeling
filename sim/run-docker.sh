#!/usr/bin/env bash

COMMAND="$@"

docker run -it --rm -w /sim -v $(PWD):/sim blsim bash -c "$COMMAND"
