#!/usr/bin/env bash

docker run -it --rm -w /sim -v $(PWD):/sim blsim "$@"
