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

## ⚙️ Optional Features

`small_collections` is modular. You can enable or disable groups of collections to minimize dependency overhead:

| Feature       | Collections Enabled                                                                    | Dependencies |
| :------------ | :------------------------------------------------------------------------------------- | :----------- |
| **`full`**    | All collections (Default)                                                              | All          |
| **`lru`**     | `SmallLruCache`, `HeaplessLruCache`, `HeaplessBTreeLruCache`, `HeaplessLinearLruCache` | `lru`        |
| **`ordered`** | `SmallOrderedMap`, `SmallOrderedSet`, `HeaplessOrderedMap`                             | `ordermap`   |
| **`bitvec`**  | `SmallBitVec`, `HeaplessBitVec`                                                        | `bitvec`     |

### Heapless LRU Variants

For `SmallLruCache`, there are three heapless backends optimized for different capacity ranges:

| Collection                   | Description                                                   | Access      |
| :--------------------------- | :------------------------------------------------------------ | :---------- |
| **`HeaplessLruCache`**       | Map-based LRU for high capacities.                            | $O(1)$      |
| **`HeaplessBTreeLruCache`**  | Binary-search LRU for medium capacities ($32 \le N \le 128$). | $O(\log N)$ |
| **`HeaplessLinearLruCache`** | Linear-scan LRU for ultra-small capacities ($N < 32$).        | $O(N)$      |

## Benchmarks: Which LRU to use?

| Capacity ($N$)       | Best for Writes    | Best for Reads  |
| :------------------- | :----------------- | :-------------- |
| **$N \le 16$**       | `Linear`           | `Linear`        |
| **$16 < N \le 64$**  | `Linear` / `BTree` | `BTree`         |
| **$64 < N \le 128$** | `BTree`            | `BTree` / `Map` |
| **$N > 128$**        | `Map`              | `Map`           |

Basic collections (`SmallVec`, `SmallDeque`, `SmallMap`, `SmallBTreeMap`, `SmallString`) are always available as they depend only on `heapless` and `fnv`.

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

| Type             | Backend       | Use Case                                                                  |
| :--------------- | :------------ | :------------------------------------------------------------------------ |
| **`SmallVec`**   | `Vec<T>`      | **Optimized** branchless architecture. Use for most list-based workloads. |
| **`SmallDeque`** | `VecDeque<T>` | Double-ended queue. Use when you need $O(1)$ push/pop from both ends.     |

```rust
use small_collections::{SmallVec, SmallDeque};

// SmallVec: Optimized branchless stack-based list
let mut v: SmallVec<i32, 4> = SmallVec::new();
v.push(10); // Stack access is neck-and-neck with std::vec::Vec

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

| Type                         | Backend      | Use Case                                                           |
| :--------------------------- | :----------- | :----------------------------------------------------------------- |
| **`SmallBinaryHeap`**        | `BinaryHeap` | Priority queue. Efficiently find the maximum (or minimum) element. |
| **`SmallLruCache`**          | `LruCache`   | Fixed-size cache that evicts the "Least Recently Used" items.      |
| **`HeaplessLinearLruCache`** | None (Stack) | **Ultra-fast** linear-scan LRU for tiny capacities ($N < 32$).     |
| **`SmallBitVec`**            | `BitVec`     | Compact storage for booleans (1 bit per val).                      |

```rust
use small_collections::{SmallBinaryHeap, SmallLruCache, SmallBitVec};
use std::num::NonZeroUsize;

// Priority Queue
let mut heap: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
heap.push(10);
heap.push(20);
assert_eq!(heap.pop(), Some(20)); // Highest priority first

// LRU Cache (Defaults to BTree backend)
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

`small_collections` is designed to be **neck-and-neck** with pure stack-allocated collections while providing the safety net of a heap spill.

### 1. Standard Comparison (Small N=8)

Measured using `Criterion` on very small workloads to show the base gain over heap-allocated defaults.

| Collection            | Operation   | std/external Time | Small-Collections Time | Gain             |
| :-------------------- | :---------- | :---------------- | :--------------------- | :--------------- |
| **`SmallOrderedMap`** | Insert 8    | 197.35 ns         | 33.81 ns               | **5.84x faster** |
| **`SmallOrderedMap`** | Get 8       | 91.15 ns          | 24.15 ns               | **3.77x faster** |
| **`SmallMap`**        | Insert 8    | 175.45 ns         | 73.08 ns               | **2.40x faster** |
| **`SmallMap`**        | Get 8       | 89.67 ns          | 41.51 ns               | **2.16x faster** |
| **`SmallString`**     | Push 16     | 17.58 ns          | 7.87 ns                | **2.23x faster** |
| **`SmallString`**     | Get (index) | 560.6 ps          | 523.7 ps               | **Competitive**  |
| **`SmallLruCache`**   | Put 8       | 242.88 ns         | 118.65 ns              | **2.05x faster** |
| **`SmallLruCache`**   | Get 8       | 47.53 ns          | 68.07 ns               | **O(log N)**     |
| **`SmallBitVec`**     | Get 64      | 225.09 ns         | 108.49 ns              | **2.07x faster** |
| **`SmallBitVec`**     | Push 64     | 249.88 ns         | 141.74 ns              | **1.76x faster** |
| **`SmallBTreeMap`**   | Insert 8    | 126.28 ns         | 61.85 ns               | **2.04x faster** |
| **`SmallBTreeMap`**   | Get 8       | 30.29 ns          | 23.29 ns               | **1.30x faster** |
| **`SmallBinaryHeap`** | Push 8      | 26.69 ns          | 27.60 ns               | **Competitive**  |
| **`SmallDeque`**      | PushBack 16 | 41.10 ns          | 29.15 ns               | **1.41x faster** |
| **`SmallDeque`**      | Get 16      | 15.86 ns          | 14.01 ns               | **1.13x faster** |
| **`SmallVec`**        | Access 16   | 12.39 ns          | 13.75 ns               | **Competitive**  |
| **`SmallVec`**        | Push 16     | 24.11 ns          | 30.06 ns               | **Competitive**  |

### 2. Heapless vs Small vs Std Comparison (N=16)

Benchmarked to measure the overhead of the "Small" tagged-union dispatch vs pure "Heapless" stack storage.

| Collection            | Operation | Std/External | **Small (Stack)** | **Heapless (Pure)** | Gain (Small vs Std) | Gain (Pure vs Std) |
| :-------------------- | :-------- | :----------- | :---------------- | :------------------ | :------------------ | :----------------- |
| **`SmallLruCache`**   | Put 16    | 462 ns       | **246 ns**        | 246 ns              | **1.88x faster**    | **1.88x faster**   |
| **`SmallLruCache`**   | Get 16    | 93 ns        | **88 ns**         | 82 ns               | **1.05x faster**    | **1.13x faster**   |
| **`SmallBTreeMap`**   | Insert 16 | 342 ns       | **160 ns**        | 159 ns              | **2.14x faster**    | **2.15x faster**   |
| **`SmallBTreeMap`**   | Get 16    | 110 ns       | **55 ns**         | 53 ns               | **2.00x faster**    | **2.08x faster**   |
| **`SmallBitVec`**     | Push 64   | 297 ns       | **132 ns**        | 90 ns               | **2.25x faster**    | **3.30x faster**   |
| **`SmallBitVec`**     | Get 64    | 233 ns       | **114 ns**        | 115 ns              | **2.04x faster**    | **2.03x faster**   |
| **`SmallOrderedMap`** | Insert 16 | 321 ns       | **101 ns**        | 89 ns               | **3.18x faster**    | **3.61x faster**   |
| **`SmallOrderedMap`** | Get 16    | 215 ns       | **76 ns**         | 78 ns               | **2.83x faster**    | **2.76x faster**   |

### 3. LRU Backend Comparison (Linear vs BTree vs Map)

For smaller capacities, the choice of backend significantly impacts performance.

| N   | Operation | Map (HeaplessLru) | **BTree (HeaplessBTreeLru)** | Linear (HeaplessLinearLru) | Best Case |
| :-- | :-------- | :---------------- | :--------------------------- | :------------------------- | :-------- |
| 8   | Put       | 587 ns            | 485 ns                       | **366 ns**                 | Linear    |
| 16  | Put       | 702 ns            | 572 ns                       | **522 ns**                 | Linear    |
| 64  | Put       | **1.14 µs**       | 1.62 µs                      | 3.96 µs                    | Map       |
| 16  | Get (Hit) | **95 ns**         | 151 ns                       | 362 ns                     | Map       |
| 64  | Get (Hit) | **409 ns**        | 927 ns                       | 5.54 µs                    | Map       |
| 128 | Get (Hit) | **846 ns**        | 1.93 µs                      | 23.4 µs                    | Map       |

_Benchmarks measured using Criterion. `Small` collections incur a negligible dispatch overhead but offer a seamless transition to the heap once capacity is reached._

## 🏗️ Design Rationale: Custom Stack Backends

While we leverage the `heapless` crate for foundational storage, `small_collections` includes several custom-built stack-allocated engines. This was necessary to fill gaps in the ecosystem and support our **spill-to-heap** protocol:

1.  **`HeaplessBTreeMap`**: Upstream `heapless` primarily provides `LinearMap` (O(N)) and `IndexMap`. We required a true B-Tree implementation to support sorted associative storage with $O(\log N)$ performance.
2.  **`HeaplessLruCache`**: Map-based LRU optimized for larger stack capacities. It uses a **Struct-of-Arrays (SoA)** layout for cache efficiency and a **singly-linked free-list** embedded within the next-pointer array to achieve O(1) allocation with zero extra memory overhead.
3.  **`HeaplessBitVec`**: Standard stack bit-arrays (like those in `bitvec::BitArray`) often have fixed lengths or lack the specific ownership-transfer APIs needed to "spill" bit-data into a heap-allocated `bitvec::BitVec` without cloning.
4.  **`HeaplessOrderedMap`**: Necessary to maintain strict insertion-order preservation while providing the "take ownership" hooks used by `SmallOrderedMap` during migration.
5.  **`HeaplessLinearLruCache`**: Optimized for tiny working sets ($N < 16$). Highly efficient linear scanning that eliminates hashing latency and minimizes stack metadata.
6.  **`HeaplessBTreeLruCache`**: The **default backend** for `SmallLruCache`. Bridges the gap with $O(\log N)$ binary search on a sorted index of physical slot IDs, providing stable performance without data shifting.
7.  **`SmallDeque`**: While `heapless` provides a `Deque`, ours uses a custom ring-buffer implementation to allow index management (head/len) to exist outside the storage union. This ensures backend independence and enables order-preserving, zero-copy spills to the heap.

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
- **Sequence**: Branchless `SmallVec`, `SmallDeque`
- **Map/Set**: `heapless::IndexMap`
- **String**: `heapless::String`
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
