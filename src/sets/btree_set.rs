//! Sorted set that lives on the stack and spills to the heap.
//!
//! [`SmallBTreeSet`] is a thin wrapper around `SmallBTreeMap<T, (), N>`,
//! inheriting all the sorted-order guarantees and the stack→heap spill protocol.

use crate::AnySet;
use crate::SmallBTreeMap;
use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::iter::FromIterator;

/// A sorted set that lives on the stack for up to `N` elements, then spills to the heap.
///
/// Implemented as a zero-overhead wrapper around `SmallBTreeMap<T, (), N>` so that `()`
/// zero-sized values add no memory cost.  All iteration is in ascending key order.
///
/// # Generic parameters
/// | Parameter | Meaning |
/// |-----------|--------|
/// | `T` | Element type; must implement `Ord` |
/// | `N` | Stack capacity — max elements before spill |
pub struct SmallBTreeSet<T, const N: usize> {
    inner: SmallBTreeMap<T, (), N>,
}

impl<T, const N: usize> SmallBTreeSet<T, N>
where
    T: Ord,
{
    /// Creates a new empty sorted set.
    pub fn new() -> Self {
        Self {
            inner: SmallBTreeMap::new(),
        }
    }

    /// Creates a new empty set starting on the heap with the given initial capacity.
    ///
    /// If `cap <= N` this is equivalent to [`new`](SmallBTreeSet::new).
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: SmallBTreeMap::with_capacity(cap),
        }
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the set contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clears the set, removing all elements.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Adds a value to the set.
    ///
    /// Returns whether the value was newly inserted.
    pub fn insert(&mut self, value: T) -> bool {
        self.inner.insert(value, ()).is_none()
    }

    /// Returns `true` if the set contains a value.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get(value).is_some()
    }

    /// Removes a value from the set. Returns whether the value was present in the set.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.remove(value).is_some()
    }

    /// Returns an iterator visiting all elements in ascending order.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.inner.iter(),
        }
    }

    /// Returns `true` if the set is stored on the stack.
    pub fn is_on_stack(&self) -> bool {
        self.inner.is_on_stack()
    }
}

impl<T, const N: usize> AnySet<T> for SmallBTreeSet<T, N>
where
    T: Ord,
{
    fn contains(&self, value: &T) -> bool {
        self.contains(value)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// A structure representing `Iter`.
pub struct Iter<'a, T: Ord> {
    inner: crate::maps::btree_map::Iter<'a, T, ()>,
}

impl<'a, T: Ord> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
}

impl<T, const N: usize> IntoIterator for SmallBTreeSet<T, N>
where
    T: Ord,
{
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.inner.into_iter(),
        }
    }
}

/// A structure representing `IntoIter`.
pub struct IntoIter<T: Ord, const N: usize> {
    inner: crate::maps::btree_map::IntoIter<T, (), N>,
}

impl<T: Ord, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
}

impl<T, const N: usize> Clone for SmallBTreeSet<T, N>
where
    T: Ord + Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T, const N: usize> Default for SmallBTreeSet<T, N>
where
    T: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Debug for SmallBTreeSet<T, N>
where
    T: Ord + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T, const N: usize> FromIterator<T> for SmallBTreeSet<T, N>
where
    T: Ord,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::new();
        for value in iter {
            set.insert(value);
        }
        set
    }
}

impl<T, const N: usize> Extend<T> for SmallBTreeSet<T, N>
where
    T: Ord,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }
}

use std::cmp::Ordering;

impl<T, const N: usize, S> PartialEq<S> for SmallBTreeSet<T, N>
where
    T: Ord,
    S: AnySet<T>,
{
    fn eq(&self, other: &S) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|v| other.contains(v))
    }
}

impl<T, const N: usize> Eq for SmallBTreeSet<T, N> where T: Ord + Eq {}

impl<T, const N: usize> PartialOrd for SmallBTreeSet<T, N>
where
    T: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T, const N: usize> Ord for SmallBTreeSet<T, N>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btree_set_stack_ops_sorted_order() {
        let mut set: SmallBTreeSet<i32, 4> = SmallBTreeSet::new();
        set.insert(3);
        set.insert(1);
        set.insert(2);

        assert_eq!(set.len(), 3);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
        assert!(!set.contains(&4));

        let values: Vec<_> = set.iter().cloned().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_btree_set_spill_trigger_on_insert() {
        let mut set: SmallBTreeSet<i32, 2> = SmallBTreeSet::new();
        set.insert(1);
        set.insert(2);
        assert!(set.is_on_stack());

        set.insert(0); // Spill
        assert!(!set.is_on_stack());

        let values: Vec<_> = set.iter().cloned().collect();
        assert_eq!(values, vec![0, 1, 2]);
    }

    #[test]
    fn test_btree_set_any_storage_remove() {
        let mut set: SmallBTreeSet<i32, 4> = SmallBTreeSet::new();
        set.insert(1);
        set.insert(2);
        assert!(set.remove(&1));
        assert_eq!(set.len(), 1);
        assert!(!set.contains(&1));

        set.insert(3);
        set.insert(4);
        set.insert(5); // Spill
        assert!(set.remove(&5));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_btree_set_any_storage_clear() {
        let mut set: SmallBTreeSet<i32, 4> = SmallBTreeSet::new();
        set.insert(1);
        set.clear();
        assert!(set.is_empty());

        for i in 0..5 {
            set.insert(i);
        }
        assert!(!set.is_on_stack());
        set.clear();
        assert!(set.is_empty());
        assert!(!set.is_on_stack());
    }

    #[test]
    fn test_btree_set_traits_exhaustive() {
        let mut set: SmallBTreeSet<i32, 4> = SmallBTreeSet::new();
        set.insert(1);
        set.insert(2);

        // Clone
        let cloned = set.clone();
        assert_eq!(cloned.len(), 2);
        assert!(cloned.contains(&1));

        // Debug
        let debug = format!("{:?}", set);
        assert!(debug.contains("1"));

        // FromIterator
        let collected: SmallBTreeSet<i32, 4> = vec![1, 2].into_iter().collect();
        assert_eq!(collected.len(), 2);

        // Extend
        let mut set2 = SmallBTreeSet::<i32, 4>::new();
        set2.extend(vec![1, 2]);
        assert_eq!(set2.len(), 2);

        // IntoIterator
        let vec: Vec<_> = set2.into_iter().collect();
        assert_eq!(vec.len(), 2);
        assert!(vec.contains(&1));
        assert!(vec.contains(&2));
    }

    #[test]
    fn test_btree_set_traits_any_set_impl() {
        let set: SmallBTreeSet<i32, 4> = vec![1, 2].into_iter().collect();
        let any: &dyn AnySet<i32> = &set;
        assert_eq!(any.len(), 2);
        assert!(any.contains(&1));
    }

    #[test]
    fn test_btree_set_traits_equality() {
        let set: SmallBTreeSet<i32, 4> = vec![1, 2].into_iter().collect();
        let s2 = set.clone();
        assert_eq!(set, s2);
    }

    #[test]
    fn test_btree_set_any_storage_clone_heap() {
        let set: SmallBTreeSet<i32, 2> = vec![1, 2, 3].into_iter().collect();
        assert!(!set.is_on_stack());

        // clone heap
        let cloned = set.clone();
        assert_eq!(cloned.len(), 3);
    }

    #[test]
    fn test_btree_set_any_storage_with_capacity() {
        let s_cap = SmallBTreeSet::<i32, 2>::with_capacity(10);
        assert!(!s_cap.is_on_stack());
    }

    #[test]
    fn test_btree_set_traits_comparison() {
        let set1: SmallBTreeSet<i32, 2> = vec![1, 2].into_iter().collect();
        let set2: SmallBTreeSet<i32, 2> = vec![1, 2].into_iter().collect();
        let set3: SmallBTreeSet<i32, 2> = vec![1, 3].into_iter().collect();

        // PartialEq (generic)
        assert_eq!(set1, set2);
        assert_ne!(set1, set3);

        // PartialOrd / Ord
        assert!(set1 < set3);
        assert!(set3 > set1);

        // Interop with std::collections::BTreeSet
        let std_set: std::collections::BTreeSet<i32> = vec![1, 2].into_iter().collect();
        assert_eq!(set1, std_set);
    }

    #[test]
    fn test_btree_set_coverage_gaps() {
        // Default
        let set: SmallBTreeSet<i32, 4> = Default::default();
        assert!(set.is_empty());

        // PartialEq differing lengths
        let set1: SmallBTreeSet<i32, 4> = vec![1, 2].into_iter().collect();
        let set2: SmallBTreeSet<i32, 4> = vec![1].into_iter().collect();
        assert_ne!(set1, set2); // hits `if self.len() != other.len()`

        // Ord cmp
        use std::cmp::Ordering;
        assert_eq!(set1.cmp(&set1), Ordering::Equal);
        assert_eq!(set1.cmp(&set2), Ordering::Greater);
    }
}
