# Top-level entrypoints for building, testing, and packaging gomem.
# Run `make` (or `make all`) for the full local check.

GOOS := $(shell go env GOOS)
PYTHON_LIB := python/$(if $(filter darwin,$(GOOS)),libgomem.dylib,$(if $(filter windows,$(GOOS)),gomem.dll,libgomem.so))
PYTHON_HDR := python/libgomem.h

.DEFAULT_GOAL := all

.PHONY: all
all: fmt-check vet build test python-test ## check format, vet, build, test Go and Python

.PHONY: build
build: ## compile the library and all examples
	go build ./...

.PHONY: test
test: ## run the full test suite
	go test ./...

.PHONY: test-race
test-race: ## run the full test suite with the race detector
	go test -race -count=1 ./...

.PHONY: bench
bench: ## run the recall and reflect benchmarks
	go test -run '^$$' -bench 'BenchmarkRecall_10K|BenchmarkRecall_100K' -count 6 ./internal/recall
	go test -run '^$$' -bench 'BenchmarkReflect' -count 6 ./internal/reflect

.PHONY: bench-embed
bench-embed: ## 10s Remember+embed benchmark against the live API (reads .env; ARGS="-c 4")
	set -a; [ -f .env ] && . ./.env; set +a; go run ./examples/embed_bench $(ARGS)

.PHONY: test-e2e-live
test-e2e-live: ## TestE2E against the live embedding API (reads .env)
	set -a; [ -f .env ] && . ./.env; set +a; GOMEM_TEST_LIVE_EMBED=1 go test -count=1 -timeout 3m -run TestE2E -v

.PHONY: vet
vet: ## static analysis
	go vet ./...

.PHONY: fmt
fmt: ## format all Go sources
	gofmt -w .

.PHONY: fmt-check
fmt-check: ## fail if any Go source is unformatted
	@test -z "$$(gofmt -l .)" || (echo "unformatted:" && gofmt -l . && exit 1)

.PHONY: python-test
python-test: ## run Python unit tests (no shared library required)
	python3 -m unittest discover -s python -p 'test_*.py'

.PHONY: tidy
tidy: ## reconcile go.mod / go.sum
	go mod tidy

.PHONY: python
python: ## build the CGO shared library consumed by the Python client
	sh python/build.sh $(PYTHON_LIB)

.PHONY: python-smoke
python-smoke: python ## build the bridge and run the Python demo
	python3 examples/python_demo/demo.py

.PHONY: demo
demo: ## run the basic recall example
	go run ./examples/basic_recall

.PHONY: clean
clean: ## remove build artifacts
	rm -f python/libgomem.so python/libgomem.dylib python/gomem.dll $(PYTHON_HDR)
	find . -name '*.test' -type f -delete

.PHONY: help
help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'
