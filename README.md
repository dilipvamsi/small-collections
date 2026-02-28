# Small Collections

**High-performance collections optimized with Small Object Optimization (SOO).**

`small_collections` provides a comprehensive suite of data structures—including `Map`, `Set`, `Vec`, `Deque`, `String`, `LRU Cache`, and more—that live entirely on the stack for small capacities and automatically "spill" to the heap when they grow larger. This drastically reduces memory allocator pressure (malloc/free) and improves cache locality.

## 🚀 Features

- **Wide Collection Support:** 12+ optimized collections including sequences, maps, sets, priority queues, and caches.
- **Stack Allocation:** Zero heap allocation while the collection size is ≤ `N`.
- **Automatic Spilling:** Seamlessly transitions to standard heap-allocated equivalents (`HashMap`, `Vec`, `String`, etc.) when capacity `N` is exceeded.
- **Zero-Cost Move:** Spilling moves data directly; no cloning required.
- **Compile-Time Safety:** Enforces strict size limits during the build process to prevent accidental stack overflows.
- **Standard API:** Implements standard traits (`Debug`, `Display`, `FromIterator`, `Extend`, `Clone`, `Default`, `PartialEq`, `Hash`) where applicable.

## 📦 Dependencies & Acknowledgments

`small_collections` is built on the shoulders of giants. We use best-in-class crates for our storage and hashing backends:

- **[`heapless`](https://crates.io/crates/heapless)**: Provides the foundational fixed-capacity stack storage.
- **[`hashbrown`](https://crates.io/crates/hashbrown)**: Our primary heap-allocated map backend, utilizing the `Raw Entry API` for efficient spills.
- **[`bitvec`](https://crates.io/crates/bitvec)**: Powers `SmallBitVec` with efficient bit-level manipulation.
- **[`lru`](https://crates.io/crates/lru)**: Provides the LRU eviction logic for `SmallLruCache`.
- **[`ordermap`](https://crates.io/crates/ordermap)**: Powers `SmallOrderedMap` for insertion-order preservation.
- **[`fnv`](https://crates.io/crates/fnv)**: A fast, non-cryptographic hasher used to ensure consistent state and performance between stack and heap transitions.

## 🛠 Usage

## 📦 Collections Catalog

`small_collections` covers almost all standard library collection types, optimized for stack-first storage.

### 1. Sequences

| Type             | Backend       | Use Case                                                              |
| :--------------- | :------------ | :-------------------------------------------------------------------- |
| **`SmallVec`**   | `Vec<T>`      | General purpose dynamic array. Use for most list-based workloads.     |
| **`SmallDeque`** | `VecDeque<T>` | Double-ended queue. Use when you need $O(1)$ push/pop from both ends. |

```rust
use small_collections::{SmallVec, SmallDeque};

// SmallVec: Efficient stack-based list
let mut v: SmallVec<i32, 4> = SmallVec::new();
v.push(10); // Stack

// SmallDeque: Efficient stack-based ring buffer
let mut d: SmallDeque<i32, 4> = SmallDeque::new();
d.push_front(1);
d.push_back(2);
```

---

### 2. Maps & Sets

We provide three varieties of associative collections depending on your ordering requirements:

| Type                                      | Underlying | Ordering  | Use Case                                                    |
| :---------------------------------------- | :--------- | :-------- | :---------------------------------------------------------- |
| **`SmallMap` / `SmallSet`**               | `HashMap`  | None      | Maximum performance for lookups. Uses FNV hashing on stack. |
| **`SmallBTreeMap` / `SmallBTreeSet`**     | `BTreeMap` | Sorted    | Use when you need keys to be kept in sorted order.          |
| **`SmallOrderedMap` / `SmallOrderedSet`** | `OrderMap` | Insertion | Use when you need to preserve the order items were added.   |

```rust
use small_collections::{SmallMap, SmallBTreeMap, SmallOrderedMap};

// Hash-based (Fastest)
let mut hm: SmallMap<String, i32, 4> = SmallMap::new();

// B-Tree based (Sorted)
let mut bm: SmallBTreeMap<i32, &str, 4> = SmallBTreeMap::new();
bm.insert(2, "World");
bm.insert(1, "Hello"); // Will be sorted as [1, 2]

// Ordered (Insertion Order)
let mut om: SmallOrderedMap<i32, &str, 4> = SmallOrderedMap::new();
om.insert(2, "World");
om.insert(1, "Hello"); // Preserves order as [2, 1]
```

---

### 3. Specialized Collections

| Type                  | Backend      | Use Case                                                           |
| :-------------------- | :----------- | :----------------------------------------------------------------- |
| **`SmallBinaryHeap`** | `BinaryHeap` | Priority queue. Efficiently find the maximum (or minimum) element. |
| **`SmallLruCache`**   | `LruCache`   | Fixed-size cache that evicts the "Least Recently Used" items.      |
| **`SmallBitVec`**     | `BitVec`     | Compact storage for booleans (1 bit per val).                      |

```rust
use small_collections::{SmallBinaryHeap, SmallLruCache, SmallBitVec};
use std::num::NonZeroUsize;

// Priority Queue
let mut heap: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
heap.push(10);
heap.push(20);
assert_eq!(heap.pop(), Some(20)); // Highest priority first

// LRU Cache
let mut cache: SmallLruCache<i32, i32, 8> = SmallLruCache::new(NonZeroUsize::new(2).unwrap());
cache.put(1, 10);
cache.put(2, 20);
cache.put(3, 30); // Evicts (1, 10) if capacity is reached on heap or stack limit N

// Compact Booleans
let mut bv: SmallBitVec<64> = SmallBitVec::new();
bv.push(true);
bv.push(false);
```

---

### 4. Utilities

| Type              | Backend  | Use Case                                                        |
| :---------------- | :------- | :-------------------------------------------------------------- |
| **`SmallString`** | `String` | Inline strings for IDs, short labels, and small formatted text. |

```rust
use small_collections::SmallString;

let mut s: SmallString<16> = SmallString::new();
s.push_str("Hello");
assert!(s.is_on_stack());
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

## ⚡ Performance Benchmarks

Measured using `Criterion` on small workloads (within stack capacity `N`).

| Collection            | Operation   | std/external Time | Small-Collections Time | Gain             |
| :-------------------- | :---------- | :---------------- | :--------------------- | :--------------- |
| **`SmallOrderedMap`** | Insert 8    | 197.35 ns         | 33.81 ns               | **5.84x faster** |
| **`SmallOrderedMap`** | Get 8       | 91.15 ns          | 24.15 ns               | **3.77x faster** |
| **`SmallMap`**        | Insert 8    | 182.66 ns         | 74.45 ns               | **2.45x faster** |
| **`SmallMap`**        | Get 8       | 90.28 ns          | 32.11 ns               | **2.81x faster** |
| **`SmallString`**     | Push 16     | 18.27 ns          | 7.27 ns                | **2.51x faster** |
| **`SmallLruCache`**   | Put 8       | 259.17 ns         | 102.48 ns              | **2.53x faster** |
| **`SmallBitVec`**     | Get 64      | 225.09 ns         | 108.49 ns              | **2.07x faster** |
| **`SmallBitVec`**     | Push 64     | 249.88 ns         | 141.74 ns              | **1.76x faster** |
| **`SmallBTreeMap`**   | Insert 8    | 85.65 ns          | 52.05 ns               | **1.65x faster** |
| **`SmallBTreeMap`**   | Get 8       | 34.89 ns          | 21.83 ns               | **1.60x faster** |
| **`SmallBinaryHeap`** | Push 8      | 36.59 ns          | 25.87 ns               | **1.41x faster** |
| **`SmallDeque`**      | Get 16      | 16.20 ns          | 14.01 ns               | **1.16x faster** |
| `SmallDeque`          | PushBack 16 | 40.81 ns          | 48.87 ns               | -19.7%           |
| `SmallVec`            | Push 16     | 24.36 ns          | 30.16 ns               | -23.8%           |
| `SmallVec`            | Access 16   | 7.04 ns           | 14.50 ns               | 2.06x slower     |
| `SmallLruCache`       | Get 8       | 44.40 ns          | 162.16 ns              | 3.6x slower      |
| `SmallBinaryHeap`     | Peek        | 259.18 ps         | 270.76 ps              | -4.4%            |

### Why use `small_collections`?

For bimodal workloads—where most collections are small but some grow large—the elimination of heap allocation and deallocation provides a significant speedup (up to **5.4x**). While `SmallVec` and `SmallDeque` incur a minor overhead for safety/state management, the associative collections deliver massive wins.

## 🏮 Design Philosophy

`small_collections` adheres to three core principles:

1. **Hybrid Storage**: We don't reinvent the wheel. We combine the safety of `heapless` stack arrays with the battle-tested performance of `hashbrown`, `std`, and `ordermap` for the heap path.
2. **Transparent Interoperability**: Through the `Any*` traits (e.g., `AnyMap`, `AnyString`), you can write generic code that handles both `Small*` and standard library types without performance penalties.
3. **Fail-Fast Safety**: We use compile-time constants and assertions to ensure that stack usage is explicit and guarded against overflows.

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
