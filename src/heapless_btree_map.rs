//! Stack-allocated sorted map and its key-value entry type.
//!
//! This module is the **stack half** of [`SmallBTreeMap`](crate::SmallBTreeMap).  It holds up to
//! `N` entries in a `heapless::Vec` sorted by key and exposes an `insert` that returns
//! `Err((key, value))` when the vec is full so the caller can transparently *spill* to a heap
//! `BTreeMap`.

use core::borrow::Borrow;
use core::cmp::Ordering;
use heapless::Vec as HeaplessVec;

// ─── Entry ────────────────────────────────────────────────────────────────────

/// A key-value pair whose ordering is determined **solely** by the key.
///
/// This newtype is stored inside [`HeaplessBTreeMap`]'s internal `heapless::Vec` so that
/// `binary_search_by` can use the standard `Ord` implementation without needing a
/// separate comparator every call site.
///
/// # Design Consideration
/// The value is intentionally excluded from all comparison traits.  This mirrors the
/// semantics of `BTreeMap`: two entries with the same key are considered *equal*
/// regardless of their values, making in-place replacement safe.
#[derive(Debug, Clone)]
pub struct Entry<K, V>(pub K, pub V);

impl<K: PartialEq, V> PartialEq for Entry<K, V> {
    /// Returns `true` iff the two entries share the same key.
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<K: Eq, V> Eq for Entry<K, V> {}

impl<K: PartialOrd, V> PartialOrd for Entry<K, V> {
    /// Delegates to the key's `partial_cmp`; the value is ignored.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<K: Ord, V> Ord for Entry<K, V> {
    /// Delegates to the key's `cmp`; the value is ignored.
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

// ─── HeaplessBTreeMap ─────────────────────────────────────────────────────────

/// A **stack-allocated**, sorted map backed by a `heapless::Vec<Entry<K, V>, N>`.
///
/// # Overview
/// Entries are kept in ascending key order at all times.  Insertions and lookups use
/// `binary_search_by` for **O(log N)** key comparisons.  Insertion still takes **O(N)**
/// time overall because items may need to be shifted to maintain sorted order — acceptable
/// for small `N` (e.g. ≤ 32) where the constant factor of a move is very cheap.
///
/// # Overflow protocol
/// When the map is full and a *new* key is inserted, [`insert`](HeaplessBTreeMap::insert)
/// returns `Err((key, value))` with the original key and value back.  The caller is
/// expected to **spill** to a heap-backed `BTreeMap` and retry the insert there.
///
/// # Generic parameters
/// | Parameter | Meaning |
/// |-----------|---------|
/// | `K`       | Key type; must implement `Ord` |
/// | `V`       | Value type |
/// | `N`       | Stack capacity (number of entries) |
///
/// # Design Considerations
/// - **No allocator dependency**: uses `heapless::Vec`, so the entire structure lives in
///   the caller's stack frame.
/// - **`Borrow<Q>` support**: `get`, `get_mut`, and `remove` accept any borrowed form of
///   the key (e.g. `&str` for `String` keys) via the standard `Borrow` trait.
/// - **Sorted-vec vs hash**: sorted-vec was chosen because `BTreeMap` semantics require
///   sorted iteration.  A hash-based approach would break `IntoIterator` order guarantees.
///
/// # Pseudo-code Implementation
/// `HeaplessBTreeMap` maintains a sorted array of `(Key, Value)` entries.
///
/// ```text
/// // 1. Lookup (get)
/// idx = binary_search(key) // O(log N)
/// if idx: return values[idx]
///
/// // 2. Insertion (insert)
/// idx = binary_search(key)
/// if idx:
///     update values[idx]; return old
/// else:
///     if len == N: return Err (Full)
///     shift_right(idx..len) // O(N)
///     insert (key, value) at idx
///
/// // 3. Removal (remove)
/// idx = binary_search(key)
/// if idx:
///     val = values[idx]
///     shift_left(idx+1..len) // O(N)
///     return val
/// ```
#[derive(Debug, Clone)]
pub struct HeaplessBTreeMap<K, V, const N: usize> {
    buf: HeaplessVec<Entry<K, V>, N>,
}

impl<K: Ord, V, const N: usize> HeaplessBTreeMap<K, V, N> {
    /// Creates an empty map.  No allocation occurs.
    pub fn new() -> Self {
        Self {
            buf: HeaplessVec::new(),
        }
    }

    /// Returns the number of entries currently stored.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Returns `true` if the map has reached its compile-time capacity `N`.
    ///
    /// When `is_full()` returns `true`, inserting a *new* key will return `Err`.
    pub fn is_full(&self) -> bool {
        self.buf.is_full()
    }

    /// Removes all entries, dropping the keys and values in place.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Inserts or updates a key-value pair.
    ///
    /// # Returns
    /// | Variant | Meaning |
    /// |---------|---------|
    /// | `Ok(Some(old))` | Key already existed; old value returned, new value stored. |
    /// | `Ok(None)` | Key was new; entry inserted in sort position. |
    /// | `Err((key, value))` | Map is full and key is new; **caller must spill to heap**. |
    ///
    /// # Complexity
    /// O(log N) for the binary search, O(N) for the potential shift on insertion.
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>, (K, V)> {
        match self.buf.binary_search_by(|e| e.0.cmp(&key)) {
            Ok(idx) => Ok(Some(core::mem::replace(&mut self.buf[idx].1, value))),
            Err(idx) => {
                if self.buf.is_full() {
                    Err((key, value))
                } else {
                    self.buf.insert(idx, Entry(key, value)).ok().unwrap();
                    Ok(None)
                }
            }
        }
    }

    /// Returns a shared reference to the value associated with `key`, or `None`.
    ///
    /// Accepts any type `Q` where `K: Borrow<Q>` and `Q: Ord`.
    /// Complexity: O(log N).
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.buf.binary_search_by(|e| e.0.borrow().cmp(key)) {
            Ok(idx) => Some(&self.buf[idx].1),
            Err(_) => None,
        }
    }

    /// Returns an exclusive reference to the value associated with `key`, or `None`.
    ///
    /// Accepts any type `Q` where `K: Borrow<Q>` and `Q: Ord`.
    /// Complexity: O(log N).
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.buf.binary_search_by(|e| e.0.borrow().cmp(key)) {
            Ok(idx) => Some(&mut self.buf[idx].1),
            Err(_) => None,
        }
    }

    /// Removes the entry associated with `key` and returns its value, or `None` if absent.
    ///
    /// Accepts any type `Q` where `K: Borrow<Q>` and `Q: Ord`.
    /// Complexity: O(log N) for the search, O(N) for the shift-left after removal.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.buf.binary_search_by(|e| e.0.borrow().cmp(key)) {
            Ok(idx) => Some(self.buf.remove(idx).1),
            Err(_) => None,
        }
    }

    /// Returns an iterator over `&Entry<K, V>` in ascending key order.
    ///
    /// The iterator yields shared references; entries cannot be modified through it.
    pub fn iter(&self) -> core::slice::Iter<'_, Entry<K, V>> {
        self.buf.iter()
    }

    /// Consumes `self` and returns the underlying `heapless::Vec`.
    ///
    /// Useful when the caller needs raw access to the sorted buffer, e.g. to drain entries
    /// during a spill-to-heap operation.
    pub fn into_vec(self) -> HeaplessVec<Entry<K, V>, N> {
        self.buf
    }
}

impl<K: Ord, V, const N: usize> Default for HeaplessBTreeMap<K, V, N> {
    /// Creates an empty map.  Equivalent to [`HeaplessBTreeMap::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V, const N: usize> IntoIterator for HeaplessBTreeMap<K, V, N> {
    type Item = Entry<K, V>;
    type IntoIter = heapless::vec::IntoIter<Entry<K, V>, N, usize>;

    /// Consumes `self` and returns an iterator over `Entry<K, V>` in ascending key order.
    fn into_iter(self) -> Self::IntoIter {
        self.buf.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heapless_btree_map_stack_ops_sorted_order() {
        let mut map: HeaplessBTreeMap<i32, i32, 4> = HeaplessBTreeMap::new();
        map.insert(3, 30).unwrap();
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();

        let keys: Vec<_> = map.iter().map(|e| e.0).collect();
        assert_eq!(keys, vec![1, 2, 3]);
    }

    #[test]
    fn test_heapless_btree_map_stack_ops_update() {
        let mut map: HeaplessBTreeMap<i32, i32, 4> = HeaplessBTreeMap::new();
        assert_eq!(map.insert(1, 10), Ok(None));
        assert_eq!(map.insert(1, 20), Ok(Some(10)));
        assert_eq!(map.get(&1), Some(&20));
    }

    #[test]
    fn test_heapless_btree_map_stack_ops_remove() {
        let mut map: HeaplessBTreeMap<i32, i32, 4> = HeaplessBTreeMap::new();
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), None);
    }

    #[test]
    fn test_heapless_btree_map_stack_ops_full_returns_err() {
        let mut map: HeaplessBTreeMap<i32, i32, 2> = HeaplessBTreeMap::new();
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        assert!(map.is_full());
        assert_eq!(map.insert(3, 30), Err((3, 30)));
    }

    #[test]
    fn test_heapless_btree_map_stack_ops_get_mut() {
        let mut map: HeaplessBTreeMap<i32, i32, 4> = HeaplessBTreeMap::new();
        map.insert(1, 10).unwrap();
        if let Some(v) = map.get_mut(&1) {
            *v = 99;
        }
        assert_eq!(map.get(&1), Some(&99));
    }

    #[test]
    fn test_heapless_btree_map_traits_clone_default() {
        let map1: HeaplessBTreeMap<i32, i32, 4> = HeaplessBTreeMap::default();
        let mut map2 = map1.clone();
        map2.insert(1, 1).unwrap();
        assert_eq!(map1.len(), 0);
        assert_eq!(map2.len(), 1);
    }

    #[test]
    fn test_heapless_btree_map_traits_entry_ord() {
        let e1 = Entry(1i32, 10i32);
        let e2 = Entry(1i32, 20i32);
        let e3 = Entry(2i32, 10i32);
        assert_eq!(e1, e2);
        assert!(e1 < e3);
        assert_eq!(e1.cmp(&e2), Ordering::Equal);
        assert_eq!(e1.partial_cmp(&e3), Some(Ordering::Less));
    }
}
