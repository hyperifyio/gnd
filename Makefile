.PHONY: all test clean

all: build

build:
	go build -o bin/gendo cmd/gendo/main.go

test:
	go test ./... -v

clean:
	rm -rf bin/
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