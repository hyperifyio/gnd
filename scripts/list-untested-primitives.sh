#!/bin/bash

ls pkg/primitives/*.go|grep -v _|while read FILE; do TEST=$(echo pkg/primitives/$(basename "$FILE" .go)_test.go); if test -f $TEST; then :; else echo $TEST; fi; done
