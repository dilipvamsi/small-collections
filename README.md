# Small Collections

**High-performance `Map`, `Set`, and `String` implementations with Small Object Optimization (SOO).**

`small_collections` provides `SmallMap`, `SmallSet`, and `SmallString`: data structures that live entirely on the stack for small capacities and automatically "spill" to the heap when they grow larger. This drastically reduces memory allocator pressure (malloc/free) and improves cache locality.

## 🚀 Features

- **Stack Allocation:** Zero heap allocation while the collection size is ≤ `N`.
- **Automatic Spilling:** Seamlessly transitions to standard heap-allocated equivalents (`HashMap`, `HashSet`, `String`) when capacity `N` is exceeded.
- **Zero-Cost Move:** Spilling moves data directly; no cloning required.
- **Compile-Time Safety:** Enforces strict size limits during the build process to prevent accidental stack overflows.
- **Fast Hashing:** Maps and Sets use `FnvHasher` internally for extremely fast hashing of small keys.
- **Interoperability:** `SmallSet` implements the `AnySet` trait, allowing efficient set operations against standard collections.
- **Standard API:** Implements standard traits (`Debug`, `Display`, `FromIterator`, `Extend`, `Clone`, `Default`, `PartialEq`, `Hash`).

## 📦 Dependencies

This library relies on the following crates for its underlying storage and hashing mechanisms:

- **`hashbrown`**: For the high-performance heap-allocated map.
- **`heapless`**: For the fixed-capacity stack-allocated storage.
- **`fnv`**: For fast, non-cryptographic hashing of small keys (Maps/Sets).

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

### SmallString

A string type that stores short text inline and spills to a heap-allocated `String` when it exceeds `N` bytes.

```rust
use small_collections::SmallString;

fn main() {
    // Capacity 16 bytes. Fits "Hello World" (11 bytes).
    let mut s: SmallString<16> = SmallString::new();

    s.push_str("Hello");
    s.push_str(" World");

    assert!(s.is_on_stack());
    assert_eq!(s.as_str(), "Hello World");

    // Append more text to trigger a spill
    s.push_str(" - This part pushes it over the limit!");

    assert!(!s.is_on_stack());
    // Usage remains seamless
    println!("{}", s);
}
```

## ⚠️ Safety & Constraints

To ensure high performance and prevent application crashes, this library enforces constraints at **Compile Time**.

### 1. Capacity Rules (`heapless`)

The generic constant `N` (stack capacity) must adhere to the underlying storage rules:

- **For `SmallMap` / `SmallSet`:**
  - `N` must be a **power of two** (e.g., 2, 4, 8, 16...).
  - `N` must be **greater than 1**.
- **For `SmallString`:**
  - `N` can be any non-zero size (e.g., 20, 80, 100).
  - Powers of two are recommended for alignment, but **not required**.

### 2. Stack Size Guard

Since these collections store data inline, a large `N` (or large `Value` type in Maps) can easily exceed the thread stack size (causing a Segmentation Fault).

To prevent this, `new()` includes a **Compile-Time Assertion**.

- **Limit:** The total size of the struct must not exceed **16 KB**.
- **Behavior:** If your collection is too large, **`cargo build` will fail**.

#### How to fix a build failure:

If you see an error like _"SmallMap is too large"_, you have two options:

1.  **Reduce `N`:** If you don't need that many items on the stack, reduce the capacity.
2.  **Box the Value:** For `SmallMap`, wrap the value in a `Box<V>`. This keeps the map structure small on the stack.

#### Size Recommendation

| Environment                          | Recommended Limit (`MAX_STACK_SIZE`)                           |
| :----------------------------------- | :------------------------------------------------------------- |
| **General Purpose (Desktop/Server)** | **16 KB** (Current). Safe balance.                             |
| **High Performance / Games**         | **4 KB**. Avoids heavy `memcpy`.                               |
| **Embedded / WASM**                  | **1 KB - 2 KB**. Stack is often very tight (e.g., 32KB total). |
| **Heavy Async/Web Servers**          | **4 KB**. Prevents bloated Future states eating RAM.           |

## ⚡ Performance Architecture

This library is designed for scenarios with a **bimodal distribution of sizes**—where most collections are small, but some can grow large.

### 1. The Stack State

- **Storage:**
  - Map/Set: `heapless::IndexMap`
  - String: `heapless::String`
- **Allocator:** None. Uses inline stack memory.
- **Hashing:** FNV (Fowler–Noll–Vo). Non-cryptographic but extremely fast for small keys, avoiding the startup overhead of SipHash.

### 2. The Heap State

- **Storage:**
  - Map/Set: `hashbrown::HashMap`
  - String: `std::string::String`
- **Allocator:** Standard System Allocator.
- **Hashing:** FNV (maintained for consistency in Maps/Sets).

### 3. The Spill Mechanism

When a collection transitions from Stack to Heap, it performs a **bitwise copy** of the stack memory to "steal" ownership of the items. It then moves them into the heap structure. This avoids:

1.  Cloning keys/values (Standard moves).
2.  Double-hashing (Hashes are calculated once during migration).

## 🤝 Contributing

Contributions are welcome! Please ensure that any PRs include tests covering both the "Stack" state and the "Heap" state to ensure the spill logic is exercised correctly.

## 📄 License

MIT License
