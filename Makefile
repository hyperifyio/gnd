.PHONY: all test clean build-gnd build-gndc build-gndtest test test-verbose test-coverage

# Set default Go flags including test timeout
export GOFLAGS=-test.timeout=30s

all: build

build: build-gnd build-gndc build-gndtest

build-gnd:
	go build -o bin/gnd cmd/gnd/main.go

#build-gndc:
#	go build -o bin/gndc cmd/gndc/main.go

#build-gndtest:
#	go build -o bin/gndtest cmd/gndtest/main.go

# Default timeout for tests
TEST_TIMEOUT = 30s

# Run tests with default timeout
test:
	go test -timeout $(TEST_TIMEOUT) ./...

# Run tests with verbose output
test-verbose:
	go test -v -timeout $(TEST_TIMEOUT) ./...

# Run tests with coverage
test-coverage:
	go test -timeout $(TEST_TIMEOUT) -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out

# Run benchmarks
bench:
	go test -bench=. -benchmem -timeout $(TEST_TIMEOUT) ./...

clean:
	rm -f bin/gnd bin/gndc bin/gndtest
	go clean

# Development helpers
fmt:
	go fmt ./...

vet:
	go vet ./...

lint:
	golangci-lint run

# Run all checks
check: fmt vet lint test 
