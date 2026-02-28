.PHONY: test coverage coverage-html install-tools tarpaulin

# Default target
test:
	cargo test

# Install coverage tools
install-tools:
	@echo "Installing LLVM tools and cargo-llvm-cov..."
	@rustup component add llvm-tools-preview || echo "Warning: rustup not found, please install llvm-tools manually."
	cargo install cargo-llvm-cov
	cargo install cargo-tarpaulin

# Run terminal coverage report (tarpaulin)
coverage:
	export PATH=$(PATH):$(HOME)/.cargo/bin && cargo tarpaulin --out Stdout --skip-clean --engine ptrace

# Generate HTML coverage report
coverage-html:
	export PATH=$(PATH):$(HOME)/.cargo/bin && cargo tarpaulin --out Html --skip-clean --engine ptrace
	@echo "Report generated in tarpaulin-report.html"

# Memory leak check using Valgrind
# Note: Using valgrind.supp to ignore a 48B "possibly lost" leak originating from
# the Rust test harness (std::thread::current::init_current).
# Confirmed as a false positive:
# 1. Traced to internal test harness thread-local allocation.
# 2. Confirmed via minimal reproduction (empty tests show the same trace).
leak-check-valgrind:
	@echo "Running memory leak check with valgrind..."
	VALGRIND_OPTS="--suppressions=valgrind.supp" cargo valgrind test

# Memory leak check using Miri
leak-check-miri:
	@echo "Running memory leak check with Miri..."
	cargo +nightly miri test

# Alternative leak check using ASAN (requires nightly)
leak-check-asan:
	ASAN_OPTIONS=detect_leaks=1 RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --lib --target x86_64-unknown-linux-gnu
	cargo test --doc

# Alternative: run coverage with tarpaulin (alias)
tarpaulin: coverage
