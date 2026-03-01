#![doc = include_str!("../README.md")]
//! # Small Collections
//!
//! A collection of data structures optimized for small-buffer scenarios.
//!
//! ## Overview
//! This crate provides a variety of collections that are designed to reside on the stack
//! up to a specific capacity `N`. If the collection exceeds this capacity, it automatically
//! "spills" its contents to a corresponding heap-allocated collection from the standard library
//! or specialized crates (like `hashbrown` or `bitvec`).
//!
//! This approach balances the performance benefits of zero-allocation stack storage for small workloads
//! with the flexibility of heap storage for larger or unpredictable workloads.
//!
//! ## Key Features
//! - **Zero-Allocation Initial State:** All collections start on the stack.
//! - **Automatic Spill:** Seamless transition to heap storage when needed.
//! - **Efficient Spills:** Items are moved (bitwise copy/ownership transfer), never cloned during a spill.
//! - **Safety:** Extensively verified with Miri to ensure zero memory leaks and no Undefined Behavior (UB).
//!
//! ## Documentation Examples
//!
//! ### SmallVec
//! ```rust
//! use small_collections::{SmallVec, AnyVec};
//! let mut v: SmallVec<i32, 4> = SmallVec::new();
//! v.push(1);
//! v.push(2);
//! assert!(v.is_on_stack()); // Stays on stack
//!
//! v.extend(vec![3, 4, 5]);   // Exceeds capacity of 4
//! assert!(!v.is_on_stack()); // Spilled to heap
//! assert_eq!(v.pop(), Some(5));
//! ```
//!
//! ### SmallDeque
//! ```rust
//! use small_collections::{SmallDeque, AnyDeque};
//! let mut d: SmallDeque<i32, 4> = SmallDeque::new();
//! d.push_back(1);
//! d.push_front(2);
//! assert_eq!(d.pop_back(), Some(1));
//! assert!(d.is_on_stack());
//!
//! d.extend(vec![3, 4, 5, 6]);
//! assert!(!d.is_on_stack());
//! ```
//!
//! ### SmallString
//! ```rust
//! use small_collections::{SmallString, AnyString};
//! let mut s: SmallString<16> = SmallString::new();
//! s.push_str("Hello");
//! assert!(s.is_on_stack());
//!
//! s.push_str(", World! This is a long string that will spill.");
//! assert!(!s.is_on_stack());
//! assert_eq!(s.len(), 52);
//! ```
//!
//! ### SmallMap & SmallSet
//! ```rust
//! use small_collections::{SmallMap, SmallSet, AnyMap, AnySet};
//! let mut map: SmallMap<&str, i32, 4> = SmallMap::new();
//! map.insert("key", 10);
//! assert!(map.is_on_stack());
//!
//! let mut set: SmallSet<i32, 4> = SmallSet::new();
//! set.insert(1);
//! assert!(set.is_on_stack());
//! ```
//!
//! ### SmallBTreeMap & SmallBTreeSet
//! ```rust
//! use small_collections::{SmallBTreeMap, SmallBTreeSet};
//! let mut bmap: SmallBTreeMap<i32, i32, 4> = SmallBTreeMap::new();
//! bmap.insert(1, 10);
//!
//! let mut bset: SmallBTreeSet<i32, 4> = SmallBTreeSet::new();
//! bset.insert(1);
//! bset.insert(2);
//! bset.insert(3);
//! bset.insert(4);
//! bset.insert(5); // Spills to heap
//! assert!(!bset.is_on_stack());
//! ```
//!
//! ### SmallOrderedMap & SmallOrderedSet
//! ```rust
//! use small_collections::{SmallOrderedMap, SmallOrderedSet};
//! let mut omap: SmallOrderedMap<i32, i32, 4> = SmallOrderedMap::new();
//! omap.insert(1, 10); // Maintains insertion order
//!
//! let mut oset: SmallOrderedSet<i32, 4> = SmallOrderedSet::new();
//! oset.insert(1);
//! ```
//!
//! ### SmallBinaryHeap
//! ```rust
//! use small_collections::{SmallBinaryHeap, AnyHeap};
//! let mut heap: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
//! heap.push(10);
//! heap.push(20);
//! assert_eq!(heap.pop(), Some(20));
//! assert!(heap.is_on_stack());
//!
//! heap.extend(vec![30, 40, 50, 60]);
//! assert!(!heap.is_on_stack()); // Spills logic seamlessly
//! ```
//!
//! ### SmallLruCache
//! ```rust
//! use small_collections::{SmallLruCache, AnyLruCache};
//! use std::num::NonZeroUsize;
//! let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(NonZeroUsize::new(2).unwrap());
//! cache.put(1, 10);
//! cache.put(2, 20);
//! cache.put(3, 30); // Evicts key 1 based on LRU
//! assert_eq!(cache.get(&1), None);
//! assert_eq!(cache.get(&2), Some(&20));
//! ```
//!
//! ### SmallBitVec
//! ```rust
//! use small_collections::{SmallBitVec, AnyBitVec};
//! let mut bv: SmallBitVec<4> = SmallBitVec::new(); // 4 bytes = 32 bits
//! bv.push(true);
//! assert!(bv.get(0).unwrap());
//! assert!(bv.is_on_stack());
//!
//! for _ in 0..40 {
//!     bv.push(false);
//! }
//! assert!(!bv.is_on_stack()); // Automatically spilled to a heap allocated BitVec
//! ```
//!
//! ### Pure Heapless Collections
//! ```rust
//! use small_collections::{HeaplessBTreeMap, HeaplessBitVec};
//! // Use these if you want strict stack-only collections without heap-spill overhead.
//! let mut bmap: HeaplessBTreeMap<i32, i32, 4> = HeaplessBTreeMap::new();
//! assert_eq!(bmap.insert(1, 10), Ok(None));
//!
//! let mut bv: HeaplessBitVec<4> = HeaplessBitVec::new();
//! let _ = bv.push(true);
//! ```
//!
//! ## Capacity Constraints
//! Many collections using the `heapless` backend require `N` to be a power of two and greater than 1.
//! These constraints are enforced at compile time.

// --- Module Declarations ---

pub mod cache;
pub mod maps;
pub mod sets;
pub mod utils;
pub mod vecs;

pub mod heap;
pub mod string;

#[cfg(feature = "lru")]
pub use cache::heapless_btree_lru_cache::HeaplessBTreeLruCache;
#[cfg(feature = "lru")]
pub use cache::heapless_linear_lru_cache::HeaplessLinearLruCache;
#[cfg(feature = "lru")]
pub use cache::heapless_lru_cache::HeaplessLruCache;
#[cfg(feature = "lru")]
pub use cache::lru_cache::{AnyLruCache, SmallLruCache};
pub use heap::{AnyHeap, SmallBinaryHeap};
pub use maps::btree_map::{AnyBTreeMap, SmallBTreeMap};
pub use maps::heapless_btree_map::{Entry as BTreeEntry, HeaplessBTreeMap};
#[cfg(feature = "ordered")]
pub use maps::heapless_ordered_map::HeaplessOrderedMap;
pub use maps::map::{AnyMap, SmallMap, SmallMapIntoIter, SmallMapIter};
#[cfg(feature = "ordered")]
pub use maps::ordered_map::SmallOrderedMap;
pub use sets::btree_set::SmallBTreeSet;
#[cfg(feature = "ordered")]
pub use sets::ordered_set::SmallOrderedSet;
pub use sets::set::{AnySet, SetRefIter, SmallSet, SmallSetIntoIter};
pub use string::{AnyString, SmallString};
#[cfg(feature = "lru")]
pub use utils::index_type::IndexType;
#[cfg(feature = "bitvec")]
pub use vecs::bitvec::{AnyBitVec, SmallBitVec};
pub use vecs::deque::{AnyDeque, SmallDeque};
#[cfg(feature = "bitvec")]
pub use vecs::heapless_bitvec::HeaplessBitVec;
pub use vecs::vec::{AnyVec, SmallVec};
