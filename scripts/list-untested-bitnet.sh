#!/bin/bash

find pkg/bitnet -iname '*.go'|grep -vF '_test.go'|sed -re 's/\.go$//'|while read FILE; do test -f "$FILE""_test.go" || echo "$FILE"".go"; done
