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
/// # Architecture & Pseudocode
/// This map stores elements linearly in the exact sequence they were added. It relies
/// on a simple contiguous array layout under the hood, yielding O(N) operations.
///
/// - `map`: A `heapless::LinearMap<K, V, N>`.
///
/// ## Insert Algorithm
/// ```text
/// 1. If map is physically full (`len == N`) AND the key is not already inside:
///    a. Return `Err((key, value))` (triggers heap spill in `SmallOrderedMap`).
/// 2. Else (capacity available or updating existing key):
///    a. Let `old_val = map.insert(key, value)`.
///    b. Return `Ok(old_val)`.
/// ```
///
/// ## Remove Algorithm (Order-Preserving)
/// ```text
/// 1. Initialize a temporary empty `LinearMap`.
/// 2. Move the elements from the current map into `old_map` using `core::mem::replace`.
/// 3. Iterate through `old_map` elements:
///    a. If `element.key == target_key` (and we haven't removed it yet):
///       i. Save `element.value` as the return value.
///    b. Else:
///       i. Insert `element` into the temporary map.
/// 4. Replace `self.map` with the temporary map.
/// 5. Return the saved `old_val`.
/// ```
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

impl<K, V, const N: usize> PartialEq for HeaplessOrderedMap<K, V, N>
where
    K: Eq + Hash,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

impl<K, V, const N: usize> Eq for HeaplessOrderedMap<K, V, N>
where
    K: Eq + Hash,
    V: Eq,
{
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

#[cfg(test)]
mod heapless_ordered_map_coverage_tests {
    use super::*;

    #[test]
    fn test_is_empty_false() {
        let mut map: HeaplessOrderedMap<i32, i32, 2> = HeaplessOrderedMap::new();
        map.insert(1, 10).unwrap();
        assert!(!map.is_empty());
    }

    #[test]
    fn test_get_get_mut_missing() {
        let mut map: HeaplessOrderedMap<i32, i32, 2> = HeaplessOrderedMap::new();
        map.insert(1, 10).unwrap();

        assert_eq!(map.get(&2), None);
        assert_eq!(map.get_mut(&2), None);
    }

    #[test]
    fn test_into_inner() {
        let mut map: HeaplessOrderedMap<i32, i32, 2> = HeaplessOrderedMap::new();
        map.insert(1, 10).unwrap();
        let inner = map.into_inner();
        assert_eq!(inner.len(), 1);
        assert_eq!(inner.get(&1), Some(&10));
    }

    #[test]
    fn test_partial_eq_variants() {
        let mut m1: HeaplessOrderedMap<i32, i32, 4> = HeaplessOrderedMap::new();
        m1.insert(1, 10).unwrap();

        let mut m2: HeaplessOrderedMap<i32, i32, 4> = HeaplessOrderedMap::new();
        m2.insert(1, 10).unwrap();

        let mut m3: HeaplessOrderedMap<i32, i32, 4> = HeaplessOrderedMap::new();
        m3.insert(1, 10).unwrap();
        m3.insert(2, 20).unwrap();

        let mut m4: HeaplessOrderedMap<i32, i32, 4> = HeaplessOrderedMap::new();
        m4.insert(1, 99).unwrap();

        assert_eq!(m1, m2);
        assert_ne!(m1, m3); // len diff
        assert_ne!(m1, m4); // val diff
    }
}
