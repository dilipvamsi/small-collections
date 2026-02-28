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
//! use small_collections::SmallVec;
//! let mut v: SmallVec<i32, 4> = SmallVec::new();
//! v.push(1);
//! v.push(2);
//! assert!(v.is_on_stack());
//! ```
//!
//! ### SmallDeque
//! ```rust
//! use small_collections::SmallDeque;
//! let mut d: SmallDeque<i32, 4> = SmallDeque::new();
//! d.push_back(1);
//! d.push_front(2);
//! assert_eq!(d.pop_back(), Some(1));
//! ```
//!
//! ### SmallString
//! ```rust
//! use small_collections::SmallString;
//! let mut s: SmallString<16> = SmallString::new();
//! s.push_str("Hello");
//! assert!(s.is_on_stack());
//! ```
//!
//! ### SmallMap & SmallSet
//! ```rust
//! use small_collections::{SmallMap, SmallSet};
//! let mut map: SmallMap<&str, i32, 4> = SmallMap::new();
//! map.insert("key", 10);
//!
//! let mut set: SmallSet<i32, 4> = SmallSet::new();
//! set.insert(1);
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
//! ```
//!
//! ### SmallOrderedMap & SmallOrderedSet
//! ```rust
//! use small_collections::{SmallOrderedMap, SmallOrderedSet};
//! let mut omap: SmallOrderedMap<i32, i32, 4> = SmallOrderedMap::new();
//! omap.insert(1, 10);
//!
//! let mut oset: SmallOrderedSet<i32, 4> = SmallOrderedSet::new();
//! oset.insert(1);
//! ```
//!
//! ### SmallBinaryHeap
//! ```rust
//! use small_collections::SmallBinaryHeap;
//! let mut heap: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
//! heap.push(10);
//! heap.push(20);
//! assert_eq!(heap.pop(), Some(20));
//! ```
//!
//! ### SmallLruCache
//! ```rust
//! use small_collections::SmallLruCache;
//! use std::num::NonZeroUsize;
//! let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(NonZeroUsize::new(2).unwrap());
//! cache.put(1, 10);
//! ```
//!
//! ### SmallBitVec
//! ```rust
//! use small_collections::SmallBitVec;
//! let mut bv: SmallBitVec<64> = SmallBitVec::new();
//! bv.push(true);
//! assert!(bv.get(0).unwrap());
//! ```
//!
//! ## Capacity Constraints
//! Many collections using the `heapless` backend require `N` to be a power of two and greater than 1.
//! These constraints are enforced at compile time.

// --- Module Declarations ---

#[cfg(feature = "bitvec")]
pub mod bitvec;
pub mod btree_map;
pub mod btree_set;
pub mod deque;
pub mod heap;
#[cfg(feature = "bitvec")]
pub mod heapless_bitvec;
pub mod heapless_btree_map;
#[cfg(feature = "lru")]
pub mod heapless_lru_cache;
#[cfg(feature = "ordered")]
pub mod heapless_ordered_map;
#[cfg(feature = "lru")]
pub mod lru_cache;
pub mod map;
#[cfg(feature = "ordered")]
pub mod ordered_map;
#[cfg(feature = "ordered")]
pub mod ordered_set;
pub mod set;
pub mod string;
pub mod vec;

// --- Re-exports ---

#[cfg(feature = "bitvec")]
pub use bitvec::{AnyBitVec, SmallBitVec};
pub use btree_map::{AnyBTreeMap, SmallBTreeMap};
pub use btree_set::SmallBTreeSet;
pub use deque::{AnyDeque, SmallDeque};
pub use heap::{AnyHeap, SmallBinaryHeap};
#[cfg(feature = "bitvec")]
pub use heapless_bitvec::HeaplessBitVec;
pub use heapless_btree_map::{Entry as BTreeEntry, HeaplessBTreeMap};
#[cfg(feature = "lru")]
pub use heapless_lru_cache::{HeaplessLruCache, IndexType};
#[cfg(feature = "ordered")]
pub use heapless_ordered_map::HeaplessOrderedMap;
#[cfg(feature = "lru")]
pub use lru_cache::{AnyLruCache, SmallLruCache};
pub use map::{AnyMap, SmallMap};
#[cfg(feature = "ordered")]
pub use ordered_map::SmallOrderedMap;
#[cfg(feature = "ordered")]
pub use ordered_set::SmallOrderedSet;
pub use set::{AnySet, SmallSet};
pub use string::{AnyString, SmallString};
pub use vec::{AnyVec, SmallVec};
