#![cfg(feature = "ordered")]
//! Insertion-order-preserving set that lives on the stack and spills to the heap.
//!
//! [`SmallOrderedSet`] is a thin wrapper around `SmallOrderedMap<T, (), N>`, inheriting
//! the insertion-order-preserving semantics and the stack→heap spill protocol defined
//! in [`ordered_map`](crate::ordered_map).

use crate::AnySet;
use crate::SmallOrderedMap;
use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::iter::FromIterator;

/// An insertion-order-preserving set that lives on the stack for up to `N` elements,
/// then spills to a heap-backed `ordermap::OrderMap`.
///
/// Implemented as a zero-overhead wrapper around `SmallOrderedMap<T, (), N>` so `()`
/// zero-sized values add no memory cost.
///
/// # Generic parameters
/// | Parameter | Meaning |
/// |-----------|--------|
/// | `T` | Element type; must implement `Eq + Hash` |
/// | `N` | Stack capacity — max elements before spill |
///
/// # Design Consideration
/// - **Insertion order**: unlike `SmallSet` which uses an unordered `FnvIndexMap` on the
///   stack, this type also preserves insertion order after spill via `OrderMap`.
///   The trade-off is O(N) lookup on the stack (linear scan) vs. O(1) for `SmallSet`.
pub struct SmallOrderedSet<T: Eq + Hash, const N: usize> {
    map: SmallOrderedMap<T, (), N>,
}

impl<T, const N: usize> SmallOrderedSet<T, N>
where
    T: Eq + Hash,
{
    /// Creates a new empty ordered set.
    pub fn new() -> Self {
        Self {
            map: SmallOrderedMap::new(),
        }
    }

    /// Returns `true` if the set is currently storing data on the stack.
    #[inline]
    pub fn is_on_stack(&self) -> bool {
        self.map.is_on_stack()
    }

    /// Returns the number of elements in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the set contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clears the set, removing all values.
    #[inline]
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Adds a value to the set. Returns `true` if the value was newly inserted.
    pub fn insert(&mut self, value: T) -> bool {
        if self.map.contains_key(&value) {
            false
        } else {
            self.map.insert(value, ());
            true
        }
    }

    /// Returns `true` if the set contains a value.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.contains_key(value)
    }

    /// Removes a value from the set. Returns `true` if the value was present.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q> + PartialEq<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.remove(value).is_some()
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let old_map = std::mem::replace(&mut self.map, SmallOrderedMap::new());
        for (k, _) in old_map {
            if f(&k) {
                self.map.insert(k, ());
            }
        }
    }

    /// Returns an iterator visiting all elements in insertion order.
    pub fn iter(&self) -> SetRefIter<'_, T> {
        SetRefIter {
            iter: self.map.iter(),
        }
    }
}

// --- Traits ---

impl<T, const N: usize> AnySet<T> for SmallOrderedSet<T, N>
where
    T: Eq + Hash,
{
    fn contains(&self, value: &T) -> bool {
        self.contains(value)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T, const N: usize> Clone for SmallOrderedSet<T, N>
where
    T: Eq + Hash + Clone,
{
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl<T, const N: usize> Default for SmallOrderedSet<T, N>
where
    T: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Debug for SmallOrderedSet<T, N>
where
    T: Eq + Hash + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T, const N: usize> FromIterator<T> for SmallOrderedSet<T, N>
where
    T: Eq + Hash,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::new();
        for val in iter {
            set.insert(val);
        }
        set
    }
}

impl<T, const N: usize> IntoIterator for SmallOrderedSet<T, N>
where
    T: Eq + Hash,
{
    type Item = T;
    type IntoIter = SmallSetIntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        SmallSetIntoIter {
            iter: self.map.into_iter(),
        }
    }
}

pub struct SmallSetIntoIter<T: Eq, const N: usize> {
    iter: crate::maps::ordered_map::SmallMapIntoIter<T, (), N>,
}

impl<T, const N: usize> Iterator for SmallSetIntoIter<T, N>
where
    T: Eq + Hash,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }
}

pub struct SetRefIter<'a, T> {
    iter: crate::maps::ordered_map::SmallMapIter<'a, T, ()>,
}

impl<'a, T> Iterator for SetRefIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }
}

impl<T, const N: usize> Extend<T> for SmallOrderedSet<T, N>
where
    T: Eq + Hash,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<T, const N: usize, S> PartialEq<S> for SmallOrderedSet<T, N>
where
    T: Eq + Hash,
    S: AnySet<T>,
{
    fn eq(&self, other: &S) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|v| other.contains(v))
    }
}

impl<T, const N: usize> Eq for SmallOrderedSet<T, N> where T: Eq + Hash {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordered_set_stack_ops_insertion_order() {
        let mut set: SmallOrderedSet<i32, 4> = SmallOrderedSet::new();
        set.insert(3);
        set.insert(1);
        set.insert(2);

        let vals: Vec<_> = set.iter().cloned().collect();
        assert_eq!(vals, vec![3, 1, 2]);
    }

    #[test]
    fn test_ordered_set_spill_trigger_on_insert() {
        let mut set: SmallOrderedSet<i32, 2> = SmallOrderedSet::new();
        set.insert(1);
        set.insert(2);
        assert!(set.is_on_stack());

        set.insert(3);
        assert!(!set.is_on_stack());

        let vals: Vec<_> = set.iter().cloned().collect();
        assert_eq!(vals, vec![1, 2, 3]);
    }

    #[test]
    fn test_ordered_set_any_storage_lifecycle_duplicates() {
        let mut set: SmallOrderedSet<String, 2> = SmallOrderedSet::new();
        assert!(set.insert("a".into()));
        assert!(!set.insert("a".into())); // Duplicate
        assert!(set.insert("b".into()));
        assert_eq!(set.len(), 2);
        assert!(set.is_on_stack());

        assert!(set.insert("c".into())); // Spill
        assert!(!set.is_on_stack());
        assert_eq!(set.len(), 3);
        assert!(set.contains("a"));
        assert!(set.contains("b"));
        assert!(set.contains("c"));

        assert!(set.remove("a"));
        assert!(!set.contains("a"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_ordered_set_traits_interop() {
        use std::collections::HashSet;
        let set: SmallOrderedSet<i32, 4> = vec![1, 2, 3].into_iter().collect();
        let mut std_set = HashSet::new();
        std_set.insert(1);
        std_set.insert(2);
        std_set.insert(3);

        assert_eq!(set, std_set);
    }

    #[test]
    fn test_ordered_set_any_storage_clear() {
        let mut set: SmallOrderedSet<i32, 2> = SmallOrderedSet::new();
        assert!(set.is_empty());
        set.insert(1);
        set.clear();
        assert!(set.is_empty());

        set.insert(1);
        set.insert(2);
        set.insert(3); // Spill
        set.clear();
        assert!(set.is_empty());
        assert!(!set.is_on_stack());
    }

    #[test]
    fn test_ordered_set_traits_exhaustive() {
        let mut set: SmallOrderedSet<i32, 4> = SmallOrderedSet::new();
        set.insert(1);
        set.insert(2);

        // Clone
        let cloned = set.clone();
        assert_eq!(cloned.len(), 2);
        assert!(cloned.contains(&1));

        // Debug
        let debug = format!("{:?}", set);
        assert!(debug.contains("1"));

        // Default
        let def: SmallOrderedSet<i32, 4> = SmallOrderedSet::default();
        assert!(def.is_empty());

        // FromIterator / IntoIterator
        let set2: SmallOrderedSet<i32, 4> = vec![1, 2].into_iter().collect();
        let vec: Vec<_> = set2.into_iter().collect();
        assert_eq!(vec, vec![1, 2]);

        // Extend
        let mut set3 = SmallOrderedSet::<i32, 4>::new();
        set3.extend(vec![1, 2]);
        assert_eq!(set3.len(), 2);
    }

    #[test]
    fn test_ordered_set_traits_any_set_impl() {
        let set: SmallOrderedSet<i32, 2> = vec![1, 2, 3].into_iter().collect();
        assert!(!set.is_on_stack());

        // AnySet trait
        let any: &dyn AnySet<i32> = &set;
        assert_eq!(any.len(), 3);
        assert!(any.contains(&1));
    }

    #[test]
    fn test_ordered_set_traits_partial_eq_variants() {
        let set: SmallOrderedSet<i32, 2> = vec![1, 2, 3].into_iter().collect();
        let set2: SmallOrderedSet<i32, 2> = vec![1, 2].into_iter().collect();
        assert_ne!(set, set2);
    }

    #[test]
    fn test_ordered_set_coverage_gaps() {
        let mut set: SmallOrderedSet<i32, 4> = vec![1, 2, 3, 4].into_iter().collect();

        // Retain: keep evens
        set.retain(|x| x % 2 == 0);

        assert_eq!(set.len(), 2);
        assert!(set.contains(&2));
        assert!(set.contains(&4));
    }
}
