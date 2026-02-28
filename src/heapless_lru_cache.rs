#![cfg(feature = "lru")]
//! Stack-allocated LRU cache — the stack half of [`SmallLruCache`](crate::SmallLruCache).
//!
//! # Design rationale
//! Implements a **Struct-of-Arrays (SoA)** doubly-linked list with a `FnvIndexMap` for O(1)
//! key lookup, O(1) LRU eviction, and O(1) MRU promotion — all without a single heap
//! allocation.

use core::mem::MaybeUninit;
use core::ptr;
use heapless::index_map::FnvIndexMap;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;

// ─── IndexType ────────────────────────────────────────────────────────────────

/// A sealed trait for integer types used as **compact doubly-linked-list node indices**.
///
/// Instead of pointer-based links, `HeaplessLruCache` stores indices into arrays.
/// This keeps the structure `#[no_std]`-friendly and saves 8 bytes per pointer on 64-bit
/// platforms.
///
/// # Sentinel value
/// `NONE` is the sentinel for "no node" (equivalent to a null pointer).  It is set to
/// the maximum value of the integer type (`255` for `u8`, `65535` for `u16`), so it can
/// never clash with a valid slot index as long as `N < NONE`.
///
/// # Implementations
/// | Type  | `NONE` | Maximum safe `N` |
/// |-------|--------|-----------------|
/// | `u8`  | 255    | 254             |
/// | `u16` | 65535  | 65534           |
pub trait IndexType: Copy + Eq + Hash + Debug + 'static {
    /// Sentinel value indicating "no node" (analogous to a null pointer).
    const NONE: Self;
    /// Converts this index to a `usize` for array access.
    fn as_usize(self) -> usize;
    /// Converts a `usize` slot index to this compact type.
    fn from_usize(i: usize) -> Self;
}

impl IndexType for u8 {
    const NONE: Self = 255;
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn from_usize(i: usize) -> Self {
        i as u8
    }
}

impl IndexType for u16 {
    const NONE: Self = 65535;
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn from_usize(i: usize) -> Self {
        i as u16
    }
}

// ─── HeaplessLruCache ─────────────────────────────────────────────────────────

/// A **stack-allocated LRU cache** using a Struct-of-Arrays doubly-linked list and a
/// `FnvIndexMap` for O(1) operations.
///
/// # Overview
/// | Operation | Complexity |
/// |-----------|------------|
/// | `put` (update existing key) | O(1) amortized |
/// | `put` (new key, under capacity) | O(1) amortized |
/// | `put` (LRU eviction) | O(1) amortized |
/// | `get` / `get_mut` | O(1) amortized |
/// | `peek` | O(1) |
///
/// # Overflow protocol
/// When `len() >= N` *and* `len() >= cap` simultaneously (i.e. at the true physical
/// capacity), [`put`](HeaplessLruCache::put) returns `(None, Err((key, value)))` and
/// the caller is expected to **spill** to a heap `LruCache`.
///
/// # Generic parameters
/// | Parameter | Meaning |
/// |-----------|---------|
/// | `K` | Key type; must implement `Eq + Hash + Clone` |
/// | `V` | Value type |
/// | `N` | Stack capacity (number of slots) |
/// | `I` | Index type; defaults to `u8` (max N = 254); use `u16` for N up to 65534 |
///
/// # Memory layout — Struct-of-Arrays (SoA)
/// Rather than storing `(key, value, prev, next)` structs per node (Array-of-Structs),
/// keys, values, `prev` pointers, and `next` pointers are stored in **separate arrays**.
/// This gives better cache behaviour when iterating over keys (e.g. to find the LRU tail)
/// because keys and index arrays fit in fewer cache lines.
///
/// # Design Considerations
/// - **`MaybeUninit` slots**: keys and values are stored as `[MaybeUninit<K>; N]` and
///   `[MaybeUninit<V>; N]`.  Only live slots (reachable from `head` via `nexts`) hold
///   initialized data.  `Drop` traverses the linked list to drop exactly those slots.
/// - **Unsafe access via `get_unchecked`**: all hot-path index dereferences bypass
///   bounds checking.  Correctness relies on the invariant that every `idx` stored in
///   `map`, `prevs`, or `nexts` is a valid slot index in `[0, N)`.
/// - **FNV hashing**: `FnvIndexMap` uses Fowler–Noll–Vo hashing which has lower
///   per-operation overhead than SipHash for small integer keys — important for the
///   latency-sensitive `get` path.
/// - **`K: Clone`**: the key must be cloned when writing to the keys array because the
///   same key is both stored there and inserted into the `FnvIndexMap`.
pub struct HeaplessLruCache<K, V, const N: usize, I: IndexType = u8> {
    /// O(1) key→slot-index lookup.
    pub map: FnvIndexMap<K, I, N>,
    /// Initialized keys for live nodes; uninitialized for free nodes.
    pub keys: [MaybeUninit<K>; N],
    /// Initialized values for live nodes; uninitialized for free nodes.
    pub values: [MaybeUninit<V>; N],
    /// `prevs[i]` is the slot index of the node *before* slot `i` in LRU order
    /// (towards the LRU tail), or `I::NONE` if `i == head`.
    pub prevs: [I; N],
    /// `nexts[i]` is the slot index of the node *after* slot `i` in LRU order
    /// (towards the MRU head), or `I::NONE` if `i == tail`.
    pub nexts: [I; N],
    /// Free-list stack: `free_nodes[free_top - 1]` is the next available slot.
    pub free_nodes: [I; N],
    /// Stack pointer into `free_nodes`. Decremented on allocation, incremented on free.
    pub free_top: usize,
    /// Slot index of the **most-recently used** node, or `I::NONE` when empty.
    pub head: I,
    /// Slot index of the **least-recently used** node, or `I::NONE` when empty.
    pub tail: I,
}

impl<K, V, const N: usize, I: IndexType> HeaplessLruCache<K, V, N, I>
where
    K: Eq + Hash + Clone,
{
    /// Creates an empty cache.  All slots are pushed onto the free-list.  No allocation occurs.
    pub fn new() -> Self {
        let mut free_nodes = [I::NONE; N];
        for i in 0..N {
            free_nodes[i] = I::from_usize(i);
        }
        Self {
            map: FnvIndexMap::new(),
            keys: unsafe { MaybeUninit::uninit().assume_init() },
            values: unsafe { MaybeUninit::uninit().assume_init() },
            prevs: [I::NONE; N],
            nexts: [I::NONE; N],
            free_nodes,
            free_top: N,
            head: I::NONE,
            tail: I::NONE,
        }
    }

    /// Returns the number of entries currently stored.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Unlinks node `idx` from the doubly-linked list.
    ///
    /// After this call the node is isolated: its `prev` and `next` still point to their
    /// old neighbours but those neighbours no longer reference `idx`.
    ///
    /// # Safety
    /// `idx` must be a live slot currently in the list.
    #[inline(always)]
    fn detach(&mut self, idx: I) {
        unsafe {
            let prev = *self.prevs.get_unchecked(idx.as_usize());
            let next = *self.nexts.get_unchecked(idx.as_usize());

            if prev != I::NONE {
                *self.nexts.get_unchecked_mut(prev.as_usize()) = next;
            } else {
                self.head = next;
            }

            if next != I::NONE {
                *self.prevs.get_unchecked_mut(next.as_usize()) = prev;
            } else {
                self.tail = prev;
            }
        }
    }

    /// Inserts node `idx` at the front (MRU end) of the list.
    ///
    /// # Safety
    /// `idx` must be a valid slot index and must *not* already be in the list.
    #[inline(always)]
    fn attach_front(&mut self, idx: I) {
        unsafe {
            *self.nexts.get_unchecked_mut(idx.as_usize()) = self.head;
            *self.prevs.get_unchecked_mut(idx.as_usize()) = I::NONE;

            if self.head != I::NONE {
                *self.prevs.get_unchecked_mut(self.head.as_usize()) = idx;
            } else {
                self.tail = idx;
            }
            self.head = idx;
        }
    }

    /// Moves node `idx` to the front (MRU end) of the list.
    ///
    /// This is the core "mark as recently used" operation.  If `idx` is already the
    /// head, this is a no-op.
    #[inline(always)]
    fn promote(&mut self, idx: I) {
        if idx == self.head {
            return;
        }
        self.detach(idx);
        self.attach_front(idx);
    }

    /// Inserts or updates a key-value pair subject to the logical capacity `cap`.
    ///
    /// The `cap` parameter is intentionally separate from `N` so that
    /// `SmallLruCache` can enforce a user-specified capacity that is smaller than `N`.
    ///
    /// # Returns
    /// | Variant | Meaning |
    /// |---------|---------|
    /// | `(Some(old), Ok(()))` | Key existed; old value replaced, node promoted to MRU. |
    /// | `(None, Ok(()))` | Key was new and a slot was available; entry added as MRU. |
    /// | `(None, Err((key, value)))` | All `N` physical slots are occupied (N < cap path); **caller must spill**. |
    ///
    /// # Eviction
    /// If `len() >= cap` the **LRU tail** is evicted before the new entry is inserted.
    /// If after eviction `len() >= N` (physical limit), the key and value are returned
    /// as `Err` for the caller to handle.
    pub fn put(&mut self, key: K, value: V, cap: usize) -> (Option<V>, Result<(), (K, V)>) {
        if let Some(idx_ref) = self.map.get(&key) {
            let idx = *idx_ref;
            unsafe {
                let old_v = ptr::replace(
                    self.values.get_unchecked_mut(idx.as_usize()).as_mut_ptr(),
                    value,
                );
                self.promote(idx);
                return (Some(old_v), Ok(()));
            }
        }

        if self.len() >= cap {
            let lru_idx = self.tail;
            unsafe {
                let key_ptr = self.keys.get_unchecked(lru_idx.as_usize()).as_ptr();
                self.map.remove(&*key_ptr);
                self.detach(lru_idx);
                ptr::drop_in_place(self.keys.get_unchecked_mut(lru_idx.as_usize()).as_mut_ptr());
                ptr::drop_in_place(
                    self.values
                        .get_unchecked_mut(lru_idx.as_usize())
                        .as_mut_ptr(),
                );
            }
            self.free_nodes[self.free_top] = lru_idx;
            self.free_top += 1;
        }

        if self.len() >= N {
            return (None, Err((key, value)));
        }

        self.free_top -= 1;
        let idx = self.free_nodes[self.free_top];
        unsafe {
            ptr::write(
                self.keys.get_unchecked_mut(idx.as_usize()).as_mut_ptr(),
                key.clone(),
            );
            ptr::write(
                self.values.get_unchecked_mut(idx.as_usize()).as_mut_ptr(),
                value,
            );
        }
        let _ = self.map.insert(key, idx);
        self.attach_front(idx);
        (None, Ok(()))
    }

    /// Returns a shared reference to the value for `key` and **promotes** it to MRU.
    ///
    /// Returns `None` if the key is not present.
    /// Accepts any type `Q` where `K: Borrow<Q>`.
    #[inline(always)]
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let idx = *self.map.get(key)?;
        self.promote(idx);
        unsafe { Some(&*self.values.get_unchecked(idx.as_usize()).as_ptr()) }
    }

    /// Returns an exclusive reference to the value for `key` and **promotes** it to MRU.
    ///
    /// Returns `None` if the key is not present.
    /// Accepts any type `Q` where `K: Borrow<Q>`.
    #[inline(always)]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let idx = *self.map.get(key)?;
        self.promote(idx);
        unsafe { Some(&mut *self.values.get_unchecked_mut(idx.as_usize()).as_mut_ptr()) }
    }

    /// Returns a shared reference to the value for `key` **without** changing LRU order.
    ///
    /// Returns `None` if the key is not present.
    /// Accepts any type `Q` where `K: Borrow<Q>`.
    #[inline(always)]
    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let idx = *self.map.get(key)?;
        unsafe { Some(&*self.values.get_unchecked(idx.as_usize()).as_ptr()) }
    }
}

impl<K, V, const N: usize, I: IndexType> Drop for HeaplessLruCache<K, V, N, I> {
    /// Walks the live linked list (head → tail) and drops each key and value in order.
    ///
    /// Free slots are never visited, which is correct because only live slots contain
    /// initialized `K` and `V` data.
    fn drop(&mut self) {
        let mut curr = self.head;
        while curr != I::NONE {
            let next = unsafe { *self.nexts.get_unchecked(curr.as_usize()) };
            unsafe {
                ptr::drop_in_place(self.keys.get_unchecked_mut(curr.as_usize()).as_mut_ptr());
                ptr::drop_in_place(self.values.get_unchecked_mut(curr.as_usize()).as_mut_ptr());
            }
            curr = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_stack_stack_ops_basic() {
        let mut stack: HeaplessLruCache<i32, i32, 4> = HeaplessLruCache::new();
        assert_eq!(stack.len(), 0);
        let _ = stack.put(1, 10, 4);
        let _ = stack.put(2, 20, 4);
        assert_eq!(stack.len(), 2);
        assert_eq!(stack.get(&1), Some(&10));
        assert_eq!(stack.get(&2), Some(&20));
    }

    #[test]
    fn test_lru_stack_stack_ops_eviction() {
        let mut stack: HeaplessLruCache<i32, i32, 2> = HeaplessLruCache::new();
        let _ = stack.put(1, 10, 2);
        let _ = stack.put(2, 20, 2);
        // Access 1 to make it MRU; 2 becomes LRU
        let _ = stack.get(&1);
        let _ = stack.put(3, 30, 2); // Should evict 2
        assert_eq!(stack.get(&2), None);
        assert_eq!(stack.get(&1), Some(&10));
        assert_eq!(stack.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_stack_stack_ops_update() {
        let mut stack: HeaplessLruCache<i32, i32, 4> = HeaplessLruCache::new();
        let (old, res) = stack.put(1, 10, 4);
        assert_eq!(old, None);
        assert!(res.is_ok());
        let (old2, res2) = stack.put(1, 99, 4);
        assert_eq!(old2, Some(10));
        assert!(res2.is_ok());
        assert_eq!(stack.get(&1), Some(&99));
    }

    #[test]
    fn test_lru_stack_stack_ops_full_returns_err() {
        let mut stack: HeaplessLruCache<i32, i32, 2> = HeaplessLruCache::new();
        let _ = stack.put(1, 10, 2);
        let _ = stack.put(2, 20, 2);
        // cap == N == 2, so no eviction possible, stack full
        let (old, res) = stack.put(3, 30, 2);
        // cap == N means evict LRU then insert -- should succeed
        assert!(res.is_ok());
        let _ = old;

        // exceed N
        let mut stack2: HeaplessLruCache<i32, i32, 2> = HeaplessLruCache::new();
        let _ = stack2.put(1, 10, 99);
        let _ = stack2.put(2, 20, 99);
        let (_, res2) = stack2.put(3, 30, 99);
        assert_eq!(res2, Err((3, 30)));
    }

    #[test]
    fn test_lru_stack_stack_ops_peek() {
        let mut stack: HeaplessLruCache<i32, i32, 4> = HeaplessLruCache::new();
        let _ = stack.put(1, 10, 4);
        let _ = stack.put(2, 20, 4);
        // peek at 1 (should not promote)
        assert_eq!(stack.peek(&1), Some(&10));
        // head should still be slot 2 (most recently inserted key=2 gets the lower free slot index)
        // free_top starts at N=4: first insert takes slot 3, second takes slot 2 -> head=2
        assert_eq!(stack.head, 2);
    }

    #[test]
    fn test_lru_stack_traits_drop() {
        // Ensure no double-free / leaks (best checked under Miri, but smoke-test here)
        let mut stack: HeaplessLruCache<String, String, 4> = HeaplessLruCache::new();
        let _ = stack.put("a".to_string(), "alpha".to_string(), 4);
        let _ = stack.put("b".to_string(), "beta".to_string(), 4);
        drop(stack); // Should not panic
    }
}
