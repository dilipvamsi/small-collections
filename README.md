# Small Collections

**High-performance `Map` and `Set` implementations with Small Object Optimization (SOO).**

`small_collections` provides `SmallMap` and `SmallSet`: collection types that live entirely on the stack for small capacities and automatically "spill" to the heap when they grow larger. This drastically reduces memory allocator pressure (malloc/free) and improves cache locality for small collections.

## 🚀 Features

*   **Stack Allocation:** Zero heap allocation while the collection size is ≤ `N`.
*   **Automatic Spilling:** Seamlessly transitions to a standard heap-allocated `HashMap` when capacity `N` is exceeded.
*   **Zero-Cost Move:** Spilling moves keys/values directly; no cloning required.
*   **Compile-Time Safety:** Enforces strict size limits during the build process to prevent accidental stack overflows.
*   **Fast Hashing:** Uses `FnvHasher` internally for extremely fast hashing of small keys.
*   **Interoperability:** `SmallSet` implements the `AnySet` trait, allowing efficient set operations (difference, subset, etc.) against standard `std::collections::HashSet` and `BTreeSet`.
*   **Standard API:** Implements `Entry` API, Iterators, and standard traits (`Debug`, `FromIterator`, `Extend`).

## 📦 Dependencies

This library relies on the following crates for its underlying storage and hashing mechanisms:

*   **`hashbrown`**: For the high-performance heap-allocated map (Raw Entry API support).
*   **`heapless`**: For the fixed-capacity stack-allocated map.
*   **`fnv`**: For fast, non-cryptographic hashing of small keys.

## 🛠 Usage

### SmallMap

A drop-in replacement for `HashMap`. You must specify the stack capacity `N` as a generic constant.

```rust
use small_collections::SmallMap;

fn main() {
    // Create a map with a stack capacity of 4.
    // This lives entirely on the stack.
    let mut map: SmallMap<String, i32, 4> = SmallMap::new();

    // These insertions happen on the Stack (No malloc)
    map.insert("Apple".to_string(), 10);
    map.insert("Banana".to_string(), 20);

    assert!(map.is_on_stack());
    assert_eq!(map.get("Apple"), Some(&10));

    // Insert more items to trigger a SPILL
    map.insert("Cherry".to_string(), 30);
    map.insert("Date".to_string(), 40);
    map.insert("Elderberry".to_string(), 50); // <--- Spills to Heap here

    // The API remains exactly the same
    assert!(!map.is_on_stack());
    assert_eq!(map.get("Elderberry"), Some(&50));
}
```

### SmallSet

A drop-in replacement for `HashSet`.

```rust
use small_collections::SmallSet;
use std::collections::HashSet;

fn main() {
    let mut small: SmallSet<i32, 4> = SmallSet::new();
    small.insert(1);
    small.insert(2);

    // Interoperate with standard collections
    let std_set: HashSet<i32> = vec![2, 3, 4].into_iter().collect();

    // Check intersection (Works efficiently!)
    for item in small.intersection(&std_set) {
        println!("Shared item: {}", item); // Prints 2
    }
}
```

## ⚠️ Safety & Constraints

To ensure high performance and prevent application crashes, `SmallMap` enforces constraints at **Compile Time**.

### 1. Capacity Rules (`heapless`)
The generic constant `N` (stack capacity) must adhere to the underlying storage rules:
*   `N` must be a **power of two** (e.g., 2, 4, 8, 16, 32...).
*   `N` must be **greater than 1**.

### 2. Stack Size Guard
Since `SmallMap` stores data inline, a large `N` or large `Value` type can easily exceed the thread stack size (causing a Segmentation Fault).

To prevent this, `SmallMap::new()` includes a **Compile-Time Assertion**.
*   **Limit:** The total size of the struct must not exceed **16 KB**.
*   **Behavior:** If your map is too large, **`cargo build` will fail**.

#### How to fix a build failure:
If you see an error like *"SmallMap is too large"*, you have two options:
1.  **Reduce `N`:** If you don't need that many items on the stack, reduce the capacity (e.g., from 128 to 32).
2.  **Box the Value:** If your value type is large, wrap it in a `Box`. This keeps the map structure small on the stack, while the bulky data lives on the heap.

```rust
// ❌ FAILS TO COMPILE (Too big for stack)
// struct Big([u8; 1000]);
// let map: SmallMap<u32, Big, 32> = SmallMap::new();

// ✅ FIXED (Use Box)
// let map: SmallMap<u32, Box<Big>, 32> = SmallMap::new();
```

#### Size Recommendation

| Environment | Recommended Limit (`MAX_STACK_SIZE`) |
| :--- | :--- |
| **General Purpose (Desktop/Server)** | **16 KB** (Current). Safe balance. |
| **High Performance / Games** | **4 KB**. Avoids heavy `memcpy`. |
| **Embedded / WASM** | **1 KB - 2 KB**. Stack is often very tight (e.g., 32KB total). |
| **Heavy Async/Web Servers** | **4 KB**. Prevents bloated Future states eating RAM. |

### 3. Trait Requirements (`Debug`)
Because `SmallMap` relies on `heapless::IndexMap` for stack storage, operations like `insert` and `Entry::or_insert` return a `Result` containing the Key/Value on failure (e.g., `Err((K, V))`).

To safely unwrap these results (asserting that capacity is available), **both `K` and `V` must implement `std::fmt::Debug`**.

*   **Required for:** `insert`, `entry`, `or_insert`, `and_modify`.
*   **Reason:** The compiler requires `Debug` to generate panic messages if an internal logic error (like a failed insertion into a non-full map) occurs.

```rust
// ❌ FAILS (MyStruct doesn't implement Debug)
// struct MyStruct;
// map.entry(1).or_insert(MyStruct);

// ✅ WORKS
// #[derive(Debug)]
// struct MyStruct;
// map.entry(1).or_insert(MyStruct);
```

## ⚡ Performance Architecture

This library is designed for scenarios with a **bimodal distribution of sizes**—where most collections are small, but some can grow large.

### 1. The Stack State
*   **Storage:** `heapless::index_map::FnvIndexMap`
*   **Allocator:** None. Uses inline stack memory.
*   **Hashing:** FNV (Fowler–Noll–Vo). Non-cryptographic but extremely fast for small keys, avoiding the startup overhead of SipHash.

### 2. The Heap State
*   **Storage:** `hashbrown::HashMap`
*   **Allocator:** Standard System Allocator.
*   **Hashing:** FNV (maintained for consistency).

### 3. The Spill Mechanism
When the map transitions from Stack to Heap, it performs a **bitwise copy** of the stack memory to "steal" ownership of the keys and values. It then inserts them into the heap map using the **Raw Entry API**. This avoids:
1.  Cloning keys/values (Standard moves).
2.  Double-hashing (Hashes are calculated once during migration).

## 🤝 Contributing

Contributions are welcome! Please ensure that any PRs include tests covering both the "Stack" state and the "Heap" state to ensure the spill logic is exercised correctly.

## 📄 License

MIT License
