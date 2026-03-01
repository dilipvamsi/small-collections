# Documentation Rules for `small-collections`

1. **Public Item Documentation**: All public structs, enums, traits, associated types, methods, and functions MUST have `///` documentation comments. The crate must compile cleanly under `RUSTFLAGS="-W missing_docs" cargo check`.
2. **Compile-Time Size Assertions**: All collections that allocate heavily on the stack (e.g., using `heapless` or large arrays) MUST have a compile-time assertion in their `new()` constructor to ensure the struct size does not exceed 16 KB.
   - Example snippet:
     ```rust
     const {
         assert!(
             std::mem::size_of::<Self>() <= 16 * 1024,
             "Collection is too large! The struct size exceeds the 16KB limit. Reduce N."
         );
     }
     ```
3. **Architectural Pseudocode**: Custom `heapless_*` implementations should include architectural pseudocode in their top-level struct docstrings. This pseudocode should explain the memory layout, insert/remove algorithms, and performance characteristics (e.g., `O(N)` vs `O(1)`).
4. **Spill-to-Heap Clarity**: `Small*` wrappers that transparently spill to the heap (e.g., `SmallMap`, `SmallVec`, `SmallLruCache`) must clearly document this behavior, including exactly when it happens (e.g., when pushing beyond stack capacity `N`) and how it affects performance (e.g., allocating a heap backing store).
5. **Usage Examples**: The main module (`lib.rs`) should include comprehensive, runnable doctest usage examples demonstrating stack-to-heap transitions, dynamic trait usage (`Any*`), and explicit pure heapless collection usage.
6. **`Any*` Trait Abstractions**: Every structure should implement an associated `Any*` trait (`AnyMap`, `AnyVec`, `AnyLruCache`, etc.) that abstracts over both its stack-bound and heap-allocated storage mechanisms to allow generic polymorphism.
7. **Comprehensive Test Coverage**: All data structures must be accompanied by heavily descriptive unit tests covering zero-allocation edge cases, correct iteration logic, scale-up eviction, and capacity limits (`N`).
8. **Memory & Lifecycle Safety**: Implementations (especially those using `ManuallyDrop` or handling `Drop` traits across stack-to-heap swaps) must be explicitly checked for leaks (via Valgrind/external tools) when elements are ejected or cleared. Look out specifically for memory leaks within `IntoIter` instances.
9. **Consistent Naming**:
   - `Small*`: Core collections that automatically adapt from stack allocation to heap allocation (e.g., `SmallVec`).
   - `Heapless*`: Underlying explicit backends that reside completely within the stack without a heap counterpart and panic or return errors strictly upon capacity bound execution.

---

_Note: Keeping proper documentation not only ensures the safety of explicit stack limits across different environments but also serves as the public contract for when transparent collection spilling occurs in `small-collections`._
