#!/bin/bash

# Run tests with a 30-second timeout
go test -v -timeout 30s ./pkg/bitnet/model/...

# Run benchmarks with a 30-second timeout
go test -v -timeout 30s -bench=. -benchmem ./pkg/bitnet/model/... 