#![cfg(feature = "ordered")]
//! Stack-allocated insertion-order map.
//!
//! This module is the **stack half** of [`SmallOrderedMap`](crate::SmallOrderedMap).
//! It wraps `heapless::LinearMap` and adds full [`Borrow`]-based lookup and a `remove` that
//! preserves insertion order, returning `Err((key, value))` when the inner map is full so
//! the caller can transparently *spill* to a heap `OrderMap`.

use std::borrow::Borrow;
use std::hash::Hash;

use heapless::LinearMap;

// ─── HeaplessOrderedMap ───────────────────────────────────────────────────────

/// A **stack-allocated**, insertion-order-preserving map backed by `heapless::LinearMap`.
///
/// # Overview
/// Entries are stored in the order they were first inserted.  Lookup, insertion, and
/// removal all use a **linear scan** — O(N) — which is acceptable for small `N` (≤ 32)
/// where the overhead of hashing or tree-rotation would dominate.
///
/// # Overflow protocol
/// When the map is full and a *new* key is inserted, [`insert`](HeaplessOrderedMap::insert)
/// returns `Err((key, value))` with the original key and value back.  The caller is
/// expected to **spill** to a heap-backed `OrderMap` and retry the insert there.
///
/// # Generic parameters
/// | Parameter | Meaning |
/// |-----------|---------|
/// | `K`       | Key type; must implement `Eq + Hash` (required by `LinearMap`) |
/// | `V`       | Value type |
/// | `N`       | Stack capacity (number of entries) |
///
/// # Design Considerations
/// - **`Borrow<Q>` support**: `get`, `get_mut`, `remove`, and `contains_key` accept any
///   borrowed form of the key (e.g. `&str` for `String` keys).  `heapless::LinearMap`
///   only supports `K: PartialEq` lookups internally, so we implement linear scans
///   ourselves using `<K as Borrow<Q>>::borrow`.
/// - **`remove` rebuilds**: because `LinearMap` has no `remove_at` API, `remove` rebuilds
///   the map by iterating and skipping the matching entry.  This is O(N) but allocation-
///   free and preserves insertion order.
/// - **`K: Eq + Hash` struct bound**: `heapless::LinearMap` requires these bounds at the
///   struct level for its `Debug` and `Clone` derives, so they appear on the struct rather
///   than only on `impl` blocks.
#[derive(Debug, Clone)]
pub struct HeaplessOrderedMap<K: Eq + Hash, V, const N: usize> {
    map: LinearMap<K, V, N>,
}

impl<K: Eq + Hash, V, const N: usize> HeaplessOrderedMap<K, V, N> {
    /// Creates an empty map.  No allocation occurs.
    pub fn new() -> Self {
        Self {
            map: LinearMap::new(),
        }
    }

    /// Returns the number of entries currently stored.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns `true` if the map has reached its compile-time capacity `N`.
    ///
    /// When `is_full()` returns `true`, inserting a *new* key will return `Err`.
    pub fn is_full(&self) -> bool {
        self.map.len() == N
    }

    /// Removes all entries, dropping keys and values in place.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Inserts or updates a key-value pair.
    ///
    /// # Returns
    /// | Variant | Meaning |
    /// |---------|---------|
    /// | `Ok(Some(old))` | Key already existed; old value returned, new value stored in place. |
    /// | `Ok(None)` | Key was new; entry appended (insertion order preserved). |
    /// | `Err((key, value))` | Map is full and key is new; **caller must spill to heap**. |
    ///
    /// # Complexity
    /// O(N) — linear scan to check for the existing key before delegating to `LinearMap`.
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>, (K, V)> {
        if self.map.len() == N && !self.map.contains_key(&key) {
            return Err((key, value));
        }
        Ok(self.map.insert(key, value).ok().flatten())
    }

    /// Returns a shared reference to the value associated with `key`, or `None`.
    ///
    /// Accepts any type `Q` where `K: Borrow<Q>` and `Q: Hash + Eq`.
    /// Complexity: O(N) linear scan.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map
            .iter()
            .find(|&(k, _)| <K as Borrow<Q>>::borrow(k) == key)
            .map(|(_, v)| v)
    }

    /// Returns an exclusive reference to the value associated with `key`, or `None`.
    ///
    /// Accepts any type `Q` where `K: Borrow<Q>` and `Q: Hash + Eq`.
    /// Complexity: O(N) linear scan.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map
            .iter_mut()
            .find(|(k, _)| <K as Borrow<Q>>::borrow(k) == key)
            .map(|(_, v)| v)
    }

    /// Removes and returns the value for `key`, preserving insertion order for remaining entries.
    ///
    /// Accepts any type `Q` where `K: Borrow<Q>` and `Q: Hash + Eq`.
    ///
    /// # Implementation note
    /// `heapless::LinearMap::remove` requires `K: PartialEq<Q>`, which is more restrictive
    /// than `K: Borrow<Q>`.  To stay generic we drain the map into a temporary buffer,
    /// skipping the matching entry, then write the buffer back.  This preserves insertion
    /// order and is allocation-free, at the cost of O(N) time.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // LinearMap::remove requires K: PartialEq<Q>.
        // We rebuild to stay generic over Borrow<Q>.
        let mut temp: LinearMap<K, V, N> = LinearMap::new();
        let mut removed = None;

        let old = core::mem::replace(&mut self.map, LinearMap::new());
        for (k, v) in old.into_iter() {
            if k.borrow() == key && removed.is_none() {
                removed = Some(v);
            } else {
                let _ = temp.insert(k, v);
            }
        }
        self.map = temp;
        removed
    }

    /// Returns `true` if the map contains an entry for `key`.
    ///
    /// Accepts any type `Q` where `K: Borrow<Q>` and `Q: Hash + Eq`.
    /// Complexity: O(N).
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Returns an iterator over `(&K, &V)` pairs in insertion order.
    pub fn iter(&self) -> heapless::linear_map::Iter<'_, K, V> {
        self.map.iter()
    }

    /// Consumes `self` and returns the underlying `heapless::LinearMap`.
    ///
    /// Useful when the caller needs direct access to the inner map, e.g. during spill-to-heap.
    pub fn into_inner(self) -> LinearMap<K, V, N> {
        self.map
    }
}

impl<K: Eq + Hash, V, const N: usize> Default for HeaplessOrderedMap<K, V, N> {
    /// Creates an empty map.  Equivalent to [`HeaplessOrderedMap::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash, V, const N: usize> IntoIterator for HeaplessOrderedMap<K, V, N> {
    type Item = (K, V);
    type IntoIter = heapless::linear_map::IntoIter<K, V, N>;

    /// Consumes `self` and returns an iterator over `(K, V)` pairs in insertion order.
    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heapless_ordered_map_stack_ops_insertion_order() {
        let mut map: HeaplessOrderedMap<i32, i32, 4> = HeaplessOrderedMap::new();
        map.insert(3, 30).unwrap();
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();

        let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![3, 1, 2]);
    }

    #[test]
    fn test_heapless_ordered_map_stack_ops_update() {
        let mut map: HeaplessOrderedMap<i32, i32, 4> = HeaplessOrderedMap::new();
        assert_eq!(map.insert(1, 10), Ok(None));
        assert_eq!(map.insert(1, 20), Ok(Some(10)));
        assert_eq!(map.get(&1), Some(&20));
    }

    #[test]
    fn test_heapless_ordered_map_stack_ops_remove() {
        let mut map: HeaplessOrderedMap<String, i32, 4> = HeaplessOrderedMap::new();
        map.insert("a".into(), 1).unwrap();
        map.insert("b".into(), 2).unwrap();
        assert_eq!(map.remove("a"), Some(1));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("b"), Some(&2));
    }

    #[test]
    fn test_heapless_ordered_map_stack_ops_full_returns_err() {
        let mut map: HeaplessOrderedMap<i32, i32, 2> = HeaplessOrderedMap::new();
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        assert!(map.is_full());
        assert_eq!(map.insert(3, 30), Err((3, 30)));
    }

    #[test]
    fn test_heapless_ordered_map_stack_ops_get_mut() {
        let mut map: HeaplessOrderedMap<i32, i32, 4> = HeaplessOrderedMap::new();
        map.insert(1, 10).unwrap();
        if let Some(v) = map.get_mut(&1) {
            *v = 42;
        }
        assert_eq!(map.get(&1), Some(&42));
    }

    #[test]
    fn test_heapless_ordered_map_stack_ops_contains_key() {
        let mut map: HeaplessOrderedMap<i32, i32, 4> = HeaplessOrderedMap::new();
        map.insert(1, 10).unwrap();
        assert!(map.contains_key(&1));
        assert!(!map.contains_key(&2));
    }

    #[test]
    fn test_heapless_ordered_map_traits_clone_default() {
        let map1: HeaplessOrderedMap<i32, i32, 4> = HeaplessOrderedMap::default();
        let mut map2 = map1.clone();
        map2.insert(7, 70).unwrap();
        assert_eq!(map1.len(), 0);
        assert_eq!(map2.len(), 1);
    }

    #[test]
    fn test_heapless_ordered_map_stack_ops_borrow_lookup() {
        let mut map: HeaplessOrderedMap<String, i32, 4> = HeaplessOrderedMap::new();
        map.insert("Apple".to_string(), 100).unwrap();
        assert_eq!(map.get("Apple"), Some(&100));
        assert_eq!(map.get_mut("Apple"), Some(&mut 100));
    }
}
