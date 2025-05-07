.PHONY: all test clean build-gnd build-gndc build-gndtest

all: build

build: build-gnd build-gndc build-gndtest

build-gnd:
	go build -o bin/gnd cmd/gnd/main.go

build-gndc:
	go build -o bin/gndc cmd/gndc/main.go

build-gndtest:
	go build -o bin/gndtest cmd/gndtest/main.go

test:
	go test ./... -v

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