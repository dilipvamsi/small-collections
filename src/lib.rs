//! # Small Collections
//!
//! High-performance collection types that live on the stack for small sizes and automatically
//! spill to the heap when they grow larger.
//!
//! This crate provides `SmallMap`, `SmallSet`, and `SmallString`. These are drop-in replacements
//! for `HashMap`, `HashSet`, and `String` optimized for cases where collections often remain small.
//!
//! ## Key Features
//!
//! * **Stack Optimization:** Items are stored inline on the stack (no heap allocation) until the capacity `N` is exceeded.
//! * **Zero-Cost Spill:** When `N` is exceeded, items are moved to the heap without cloning (for maps) or with minimal overhead.
//! * **Performance:** Uses `FnvHasher` internally for extremely fast hashing on small keys.
//! * **Compile-Time Safety:** Enforces strict size limits (max 16KB) during the build process to prevent accidental stack overflows.
//! * **Interoperability:** `SmallSet` implements the `AnySet` trait, allowing efficient operations against standard `HashSet` and `BTreeSet`.
//!
//! ## Capacity Constraints (`N`)
//!
//! The capacity generic constant `N` (stack size) determines when the collection spills to the heap.
//!
//! * **For `SmallMap` and `SmallSet`:**
//!     * `N` must be a **power of two** (e.g., 2, 4, 8, 16, 32...).
//!     * `N` must be **greater than 1**.
//!     * *Constraint source:* Underlying `heapless::IndexMap`.
//!
//! * **For `SmallString`:**
//!     * `N` can be any non-zero size (e.g., 20, 80, 128).
//!     * Power-of-two is recommended for alignment but not enforced.
//!
//! ## Examples
//!
//! ### SmallMap
//!
//! ```rust
//! use small_collections::SmallMap;
//!
//! // Capacity 4. Lives on stack.
//! let mut map: SmallMap<String, i32, 4> = SmallMap::new();
//!
//! map.insert("A".to_string(), 10);
//! map.insert("B".to_string(), 20);
//!
//! assert!(map.is_on_stack());
//! assert_eq!(map.get("A"), Some(&10));
//!
//! // Insert 5th item -> Spills to Heap automatically
//! map.insert("C".to_string(), 30);
//! map.insert("D".to_string(), 40);
//! map.insert("E".to_string(), 50);
//!
//! assert!(!map.is_on_stack());
//! ```
//!
//! ### SmallSet
//!
//! ```rust
//! use small_collections::SmallSet;
//!
//! let mut set: SmallSet<i32, 2> = SmallSet::new();
//!
//! set.insert(1);
//! set.insert(2);
//! assert!(set.is_on_stack());
//!
//! // Spills on 3rd item
//! set.insert(3);
//! assert!(!set.is_on_stack());
//! ```
//!
//! ### SmallString
//!
//! ```rust
//! use small_collections::SmallString;
//!
//! // Capacity 16 bytes. Fits "Hello World" (11 bytes).
//! let mut s: SmallString<16> = SmallString::new();
//!
//! s.push_str("Hello");
//! s.push_str(" World");
//!
//! assert!(s.is_on_stack());
//! assert_eq!(s.as_str(), "Hello World");
//!
//! // Append more text to trigger a spill
//! s.push_str(" - This part pushes it over the limit!");
//!
//! assert!(!s.is_on_stack());
//! ```

// --- Module Declarations ---

pub mod map;
pub mod set;
pub mod string;

// --- Re-exports ---

pub use map::SmallMap;
pub use set::{AnySet, SmallSet};
pub use string::SmallString;
