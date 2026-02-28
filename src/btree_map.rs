//! Sorted map that lives on the stack and spills to the heap.
//!
//! This module provides [`SmallBTreeMap`] — a map that stores up to `N` entries
//! in a stack-allocated sorted vector ([`HeaplessBTreeMap`]) and transparently
//! migrates to a `std::collections::BTreeMap` when the stack overflows.
//!
//! [`AnyBTreeMap`] is an object-safe trait that unifies both storage backends.

use core::borrow::Borrow;
use core::mem::ManuallyDrop;
use std::collections::BTreeMap;
use std::fmt::{self, Debug};
use std::iter::FromIterator;

use crate::heapless_btree_map::{Entry, HeaplessBTreeMap};

/// An object-safe abstraction over B-Tree map types.
///
/// Implemented by `BTreeMap<K, V>` (heap) and `SmallBTreeMap<K, V, N>` (small/stack) so
/// that callers can write backend-agnostic code.
pub trait AnyBTreeMap<K, V> {
    /// Returns the number of key-value pairs.
    fn len(&self) -> usize;
    /// Returns `true` if the map is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Inserts `(key, value)`, returning the previous value if the key existed.
    fn insert(&mut self, key: K, value: V) -> Option<V>;
    /// Returns a shared reference to the value for `key`, or `None`.
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized;
    /// Returns an exclusive reference to the value for `key`, or `None`.
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized;
    /// Removes and returns the value for `key`, or `None` if absent.
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized;
    /// Removes all entries.
    fn clear(&mut self);
}

impl<K: Ord, V> AnyBTreeMap<K, V> for BTreeMap<K, V> {
    fn len(&self) -> usize {
        self.len()
    }
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert(key, value)
    }
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get_mut(key)
    }
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.remove(key)
    }
    fn clear(&mut self) {
        self.clear();
    }
}

/// A sorted map that lives on the stack for up to `N` entries, then spills to the heap.
///
/// # Storage strategy
/// Uses a tagged union ([`MapData`]) with:
/// - **Stack side**: [`HeaplessBTreeMap<K, V, N>`] — a sorted `heapless::Vec`.
/// - **Heap side**: `std::collections::BTreeMap<K, V>`.
///
/// Once a spill occurs the struct permanently uses the heap side.
///
/// # Overflow protocol
/// When [`insert`](SmallBTreeMap::insert) is called on a full stack map with a new key,
/// `spill_to_heap` migrates all entries to a `BTreeMap` and then re-inserts the new pair.
///
/// # Generic parameters
/// | Parameter | Meaning |
/// |-----------|--------|
/// | `K` | Key type; must implement `Ord` |
/// | `V` | Value type |
/// | `N` | Stack capacity — max entries before spill |
///
/// # Design Considerations
/// - **`len` tracked separately**: instead of accessing the union variant to call `.len()`,
///   the length is cached in a plain `usize` field so `len()` never touches unsafe code.
/// - **Sorted iteration**: the stack-side sorted order is preserved during spill, so
///   `BTreeMap`'s sorted iteration invariant is maintained without re-sorting.
/// - **`Entry` API**: the `entry` method (in `map.rs` / `SmallMap`) can be layered on top;
///   this struct deliberately keeps its API minimal.
///
/// # Safety
/// `on_stack` determines which variant of `MapData` is active. Only that variant
/// may be accessed. All unsafe union accesses must first check `on_stack`.
pub struct SmallBTreeMap<K, V, const N: usize> {
    on_stack: bool,
    len: usize,
    data: MapData<K, V, N>,
}

impl<K: Ord, V, const N: usize> AnyBTreeMap<K, V> for SmallBTreeMap<K, V, N> {
    fn len(&self) -> usize {
        self.len()
    }
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert(key, value)
    }
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get_mut(key)
    }
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.remove(key)
    }
    fn clear(&mut self) {
        self.clear();
    }
}

/// The internal storage for `SmallBTreeMap`.
///
/// We use `ManuallyDrop` because the compiler cannot know which field is active
/// and therefore cannot automatically drop the correct one.
union MapData<K, V, const N: usize> {
    stack: ManuallyDrop<HeaplessBTreeMap<K, V, N>>,
    heap: ManuallyDrop<BTreeMap<K, V>>,
}

impl<K, V, const N: usize> SmallBTreeMap<K, V, N>
where
    K: Ord,
{
    pub const MAX_STACK_SIZE: usize = 16 * 1024;

    pub fn new() -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= SmallBTreeMap::<K, V, N>::MAX_STACK_SIZE,
                "SmallBTreeMap is too large! Reduce N."
            );
        }

        Self {
            on_stack: true,
            len: 0,
            data: MapData {
                stack: ManuallyDrop::new(HeaplessBTreeMap::new()),
            },
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        if cap <= N {
            Self::new()
        } else {
            Self {
                on_stack: false,
                len: 0,
                data: MapData::<K, V, N> {
                    heap: ManuallyDrop::new(BTreeMap::new()),
                },
            }
        }
    }

    #[inline]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        unsafe {
            if self.on_stack {
                (*self.data.stack).clear();
            } else {
                (*self.data.heap).clear();
            }
        }
        self.len = 0;
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        unsafe {
            if self.on_stack {
                let stack = &mut *self.data.stack;
                match stack.insert(key, value) {
                    Ok(old) => {
                        if old.is_none() {
                            self.len += 1;
                        }
                        return old;
                    }
                    Err((k, v)) => {
                        // Stack is full: spill, then insert into heap with the returned k/v.
                        self.spill_to_heap();
                        let old = (*self.data.heap).insert(k, v);
                        if old.is_none() {
                            self.len += 1;
                        }
                        return old;
                    }
                }
            }

            // Heap path
            let old = (*self.data.heap).insert(key, value);
            if old.is_none() {
                self.len += 1;
            }
            old
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        unsafe {
            if self.on_stack {
                self.data.stack.get(key)
            } else {
                (*self.data.heap).get(key)
            }
        }
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        unsafe {
            if self.on_stack {
                (*self.data.stack).get_mut(key)
            } else {
                (*self.data.heap).get_mut(key)
            }
        }
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        unsafe {
            if self.on_stack {
                let old = (*self.data.stack).remove(key);
                if old.is_some() {
                    self.len -= 1;
                }
                old
            } else {
                let old = (*self.data.heap).remove(key);
                if old.is_some() {
                    self.len -= 1;
                }
                old
            }
        }
    }

    #[inline(never)]
    unsafe fn spill_to_heap(&mut self) {
        unsafe {
            let mut heap = BTreeMap::new();
            // ManuallyDrop::take copies the bits out; safe because we immediately overwrite.
            let stack_map = ManuallyDrop::take(&mut self.data.stack);

            for entry in stack_map {
                heap.insert(entry.0, entry.1);
            }

            self.data.heap = ManuallyDrop::new(heap);
            self.on_stack = false;
        }
    }

    pub fn iter(&self) -> Iter<'_, K, V> {
        unsafe {
            if self.on_stack {
                Iter::Stack(self.data.stack.iter())
            } else {
                Iter::Heap((*self.data.heap).iter())
            }
        }
    }
}

pub enum Iter<'a, K: Ord, V> {
    Stack(core::slice::Iter<'a, Entry<K, V>>),
    Heap(std::collections::btree_map::Iter<'a, K, V>),
}

impl<'a, K: Ord, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Iter::Stack(i) => i.next().map(|e| (&e.0, &e.1)),
            Iter::Heap(i) => i.next(),
        }
    }
}

/// Convenience alias used by `SmallBTreeSet`.
pub type IntoIter<K, V, const N: usize> = SmallBTreeMapIntoIter<K, V, N>;

pub struct HeaplessBTreeMapIntoIter<K, V, const N: usize> {
    inner: heapless::vec::IntoIter<Entry<K, V>, N, usize>,
}

impl<K: Ord, V, const N: usize> Iterator for HeaplessBTreeMapIntoIter<K, V, N> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|e| (e.0, e.1))
    }
}

pub enum SmallBTreeMapIntoIter<K, V, const N: usize> {
    Stack(HeaplessBTreeMapIntoIter<K, V, N>),
    Heap(std::collections::btree_map::IntoIter<K, V>),
}

impl<K: Ord, V, const N: usize> Iterator for SmallBTreeMapIntoIter<K, V, N> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SmallBTreeMapIntoIter::Stack(i) => i.next(),
            SmallBTreeMapIntoIter::Heap(i) => i.next(),
        }
    }
}

impl<K, V, const N: usize> IntoIterator for SmallBTreeMap<K, V, N>
where
    K: Ord,
{
    type Item = (K, V);
    type IntoIter = SmallBTreeMapIntoIter<K, V, N>;

    fn into_iter(self) -> Self::IntoIter {
        let mut this = ManuallyDrop::new(self);
        unsafe {
            if this.on_stack {
                SmallBTreeMapIntoIter::Stack(HeaplessBTreeMapIntoIter {
                    inner: ManuallyDrop::<HeaplessBTreeMap<K, V, N>>::take(&mut this.data.stack)
                        .into_iter(),
                })
            } else {
                SmallBTreeMapIntoIter::Heap(ManuallyDrop::take(&mut this.data.heap).into_iter())
            }
        }
    }
}

impl<K, V, const N: usize> Drop for SmallBTreeMap<K, V, N> {
    fn drop(&mut self) {
        unsafe {
            if self.on_stack {
                ManuallyDrop::drop(&mut self.data.stack);
            } else {
                ManuallyDrop::drop(&mut self.data.heap);
            }
        }
    }
}

impl<K, V, const N: usize> Clone for SmallBTreeMap<K, V, N>
where
    K: Ord + Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        unsafe {
            if self.on_stack {
                Self {
                    on_stack: true,
                    len: self.len,
                    data: MapData {
                        stack: ManuallyDrop::new((*self.data.stack).clone()),
                    },
                }
            } else {
                Self {
                    on_stack: false,
                    len: self.len,
                    data: MapData {
                        heap: ManuallyDrop::new((*self.data.heap).clone()),
                    },
                }
            }
        }
    }
}

impl<K: Ord, V, const N: usize> Default for SmallBTreeMap<K, V, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Debug, V: Debug, const N: usize> Debug for SmallBTreeMap<K, V, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, const N: usize> FromIterator<(K, V)> for SmallBTreeMap<K, V, N>
where
    K: Ord,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = Self::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K, V, const N: usize> Extend<(K, V)> for SmallBTreeMap<K, V, N>
where
    K: Ord,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

// --- Index Traits ---
use std::ops::{Index, IndexMut};

impl<K, V, Q, const N: usize> Index<&Q> for SmallBTreeMap<K, V, N>
where
    K: Ord + Borrow<Q>,
    Q: Ord + ?Sized,
{
    type Output = V;

    fn index(&self, key: &Q) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, V, Q, const N: usize> IndexMut<&Q> for SmallBTreeMap<K, V, N>
where
    K: Ord + Borrow<Q>,
    Q: Ord + ?Sized,
{
    fn index_mut(&mut self, key: &Q) -> &mut Self::Output {
        self.get_mut(key).expect("no entry found for key")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Ordering;

    #[test]
    fn test_btree_stack_ops_sorted_order() {
        let mut map: SmallBTreeMap<i32, i32, 4> = SmallBTreeMap::new();
        map.insert(3, 30);
        map.insert(1, 10);
        map.insert(2, 20);

        let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 2, 3]);
    }

    #[test]
    fn test_btree_spill_trigger_on_insert() {
        let mut map: SmallBTreeMap<i32, i32, 2> = SmallBTreeMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        assert!(map.is_on_stack());

        map.insert(0, 0); // Spill
        assert!(!map.is_on_stack());

        let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![0, 1, 2]);
    }

    #[test]
    fn test_btree_any_storage_get_mut() {
        let mut map: SmallBTreeMap<i32, i32, 4> = SmallBTreeMap::new();
        map.insert(1, 10);
        if let Some(v) = map.get_mut(&1) {
            *v = 20;
        }
        assert_eq!(map.get(&1), Some(&20));

        map.insert(2, 200);
        map.insert(3, 300);
        map.insert(4, 400);
        map.insert(5, 500); // Spill
        assert!(!map.is_on_stack());

        if let Some(v) = map.get_mut(&5) {
            *v = 555;
        }
        assert_eq!(map.get(&5), Some(&555));
    }

    #[test]
    fn test_btree_any_storage_remove() {
        let mut map: SmallBTreeMap<i32, i32, 4> = SmallBTreeMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.len(), 1);
        assert!(map.get(&1).is_none());

        map.insert(3, 30);
        map.insert(4, 40);
        map.insert(5, 50); // Spill
        assert_eq!(map.remove(&5), Some(50));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_btree_any_storage_clear() {
        let mut map: SmallBTreeMap<i32, i32, 4> = SmallBTreeMap::new();
        map.insert(1, 10);
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        for i in 0..5 {
            map.insert(i, i * 10);
        }
        assert!(!map.is_on_stack());
        map.clear();
        assert!(map.is_empty());
        assert!(!map.is_on_stack()); // Stays allocated on heap (conceptually empty)
    }

    #[test]
    fn test_btree_traits_exhaustive() {
        let mut map: SmallBTreeMap<i32, i32, 4> =
            SmallBTreeMap::from_iter([(1, 10), (3, 30), (2, 20)]);
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&20));

        // Update
        map.insert(2, 22);
        assert_eq!(map.get(&2), Some(&22));

        // Remove
        assert_eq!(map.remove(&2), Some(22));
        assert_eq!(map.len(), 2);
        assert!(map.get(&2).is_none());

        // Clone
        let cloned = map.clone();
        assert_eq!(cloned.len(), 2);

        // Debug
        let debug = format!("{:?}", map);
        assert!(debug.contains("1: 10"));
        assert!(debug.contains("3: 30"));

        // FromIterator
        let collected: SmallBTreeMap<i32, i32, 4> = vec![(1, 10), (2, 20)].into_iter().collect();
        assert_eq!(collected.len(), 2);

        // Extend
        let mut map2 = SmallBTreeMap::<i32, i32, 4>::new();
        map2.extend(vec![(1, 10), (2, 20)]);
        assert_eq!(map2.len(), 2);

        // IntoIterator (Stack)
        let vec: Vec<_> = map2.into_iter().collect();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], (1, 10));
        assert_eq!(vec[1], (2, 20));

        // Spill and traits
        let mut map_spill: SmallBTreeMap<i32, i32, 2> = SmallBTreeMap::new();
        map_spill.insert(1, 10);
        map_spill.insert(2, 20);
        map_spill.insert(3, 30); // Spill
        assert!(!map_spill.is_on_stack());

        let cloned_heap = map_spill.clone();
        assert_eq!(cloned_heap.len(), 3);

        let debug_heap = format!("{:?}", map_spill);
        assert!(debug_heap.contains("1: 10"));

        // IntoIterator (Heap)
        let vec_heap: Vec<_> = map_spill.into_iter().collect();
        assert_eq!(vec_heap.len(), 3);
    }

    #[test]
    fn test_btree_entry_edge_cases() {
        // Entry comparisons
        let e1 = Entry(1, 10);
        let e2 = Entry(1, 20);
        let e3 = Entry(2, 10);
        assert_eq!(e1, e2);
        assert!(e1 < e3);
        assert_eq!(e1.cmp(&e2), Ordering::Equal);
        assert_eq!(e1.partial_cmp(&e3), Some(Ordering::Less));
    }

    #[test]
    fn test_btree_any_storage_remove_non_existent() {
        let mut map: SmallBTreeMap<i32, i32, 2> = SmallBTreeMap::new();
        assert_eq!(map.remove(&1), None);
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30); // Spill
        assert_eq!(map.remove(&4), None);
        assert_eq!(map.remove(&3), Some(30));
    }

    #[test]
    fn test_btree_traits_into_iter_empty() {
        let map_empty: SmallBTreeMap<i32, i32, 4> = SmallBTreeMap::new();
        let mut it = map_empty.into_iter();
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_btree_any_storage_heap_manipulation() {
        let mut map: SmallBTreeMap<i32, i32, 2> =
            vec![(1, 10), (2, 20), (3, 30)].into_iter().collect();
        assert!(!map.is_on_stack());

        // get_mut heap
        if let Some(v) = map.get_mut(&1) {
            *v = 11;
        }
        assert_eq!(map[&1], 11);

        // clear heap
        map.clear();
        assert!(map.is_empty());
        assert!(!map.is_on_stack());
    }

    #[test]
    fn test_btree_stack_binary_search_order() {
        let mut s: SmallBTreeMap<i32, i32, 4> = SmallBTreeMap::new();
        s.insert(2, 2);
        s.insert(1, 1);
        assert_eq!(s.get(&1), Some(&1));
    }
}
