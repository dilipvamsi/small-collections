use crate::map::SmallMap;
use std::borrow::Borrow;
use std::collections::{BTreeSet, HashSet};
use std::fmt::{self, Debug};
use std::hash::{BuildHasher, Hash};
use std::iter::FromIterator;

// ==================================================================================
// 1. The Interoperability Trait
// ==================================================================================

/// A trait for any collection that supports efficient containment checks.
///
/// This allows `SmallSet` to perform set operations (like `difference` or `is_subset`)
/// against standard library sets (`HashSet`, `BTreeSet`) without converting them first.
pub trait AnySet<T> {
    /// Returns `true` if the collection contains the value.
    fn contains(&self, value: &T) -> bool;
}

// Support SmallSet
impl<T, const N: usize> AnySet<T> for SmallSet<T, N>
where
    T: Eq + Hash + Debug,
{
    fn contains(&self, value: &T) -> bool {
        SmallSet::contains(self, value)
    }
}

// Support standard HashSet
impl<T, S> AnySet<T> for HashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn contains(&self, value: &T) -> bool {
        self.contains(value)
    }
}

// Support standard BTreeSet
impl<T> AnySet<T> for BTreeSet<T>
where
    T: Ord,
{
    fn contains(&self, value: &T) -> bool {
        self.contains(value)
    }
}

// ==================================================================================
// 2. SmallSet Implementation
// ==================================================================================

/// A set that lives on the stack for `N` items, then automatically spills to the heap.
///
/// # Implementation Details
/// This is a wrapper around `SmallMap<T, (), N>`. Since `()` is a zero-sized type in Rust,
/// this wrapper has **zero memory overhead** compared to a raw set implementation.
///
/// * **Stack State:** Zero allocations. Extremely fast FNV hashing.
/// * **Heap State:** Standard `HashMap` performance.
/// * **Spill:** Zero-allocation move. Keys are moved, not cloned.
pub struct SmallSet<T, const N: usize> {
    map: SmallMap<T, (), N>,
}

impl<T, const N: usize> SmallSet<T, N>
where
    T: Eq + Hash + Debug,
{
    /// Creates a new empty set.
    ///
    /// The set initially lives on the stack.
    pub fn new() -> Self {
        Self {
            map: SmallMap::new(),
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
    ///
    /// This keeps the allocated memory (if on heap) for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Adds a value to the set.
    ///
    /// Returns `true` if the value was newly inserted.
    /// Returns `false` if the value was already present.
    ///
    /// If the stack capacity `N` is exceeded, this triggers a spill to the heap.
    pub fn insert(&mut self, value: T) -> bool {
        if self.map.contains_key(&value) {
            false
        } else {
            self.map.insert(value, ());
            true
        }
    }

    /// Returns `true` if the set contains a value.
    ///
    /// This method is generic over `Q` to allow looking up `String` keys with `&str`.
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
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.remove(value).is_some()
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns `false`.
    /// This method operates in O(n) time.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        // Safe implementation: We replace the map with a new empty one,
        // iterate the old one, and re-insert items that pass the filter.
        // This ensures proper Move semantics (items are moved back in, not cloned).
        let old_map = std::mem::replace(&mut self.map, SmallMap::new());
        for (k, _) in old_map {
            if f(&k) {
                self.map.insert(k, ());
            }
        }
    }

    // --- Set Operations ---

    /// Returns an iterator visiting all elements in arbitrary order.
    pub fn iter(&self) -> SetRefIter<'_, T> {
        SetRefIter {
            iter: self.map.iter(),
        }
    }

    /// Visits the values representing the difference, i.e., the values that are in `self` but not in `other`.
    ///
    /// `other` can be any collection implementing `AnySet` (`SmallSet`, `HashSet`, `BTreeSet`).
    pub fn difference<'a, S>(&'a self, other: &'a S) -> impl Iterator<Item = &'a T>
    where
        S: AnySet<T>,
    {
        self.iter().filter(move |v| !other.contains(v))
    }

    /// Visits the values representing the intersection, i.e., the values that are both in `self` and `other`.
    pub fn intersection<'a, S>(&'a self, other: &'a S) -> impl Iterator<Item = &'a T>
    where
        S: AnySet<T>,
    {
        self.iter().filter(move |v| other.contains(v))
    }

    /// Visits the values representing the union, i.e., all the values in `self` or `other`, without duplicates.
    ///
    /// `other` can be any iterator yielding `&T`.
    pub fn union<'a, I>(&'a self, other: I) -> impl Iterator<Item = &'a T>
    where
        I: IntoIterator<Item = &'a T>,
        I::IntoIter: 'a,
    {
        // Iterate self, then iterate 'other' ONLY if 'self' doesn't contain the item.
        self.iter()
            .chain(other.into_iter().filter(move |v| !self.contains(v)))
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    pub fn is_disjoint<S>(&self, other: &S) -> bool
    where
        S: AnySet<T>,
    {
        self.iter().all(|v| !other.contains(v))
    }

    /// Returns `true` if `self` is a subset of `other`.
    ///
    /// `other` must be a Set (implement `AnySet`) to ensure O(1) lookups.
    pub fn is_subset<S>(&self, other: &S) -> bool
    where
        S: AnySet<T>,
    {
        self.iter().all(|v| other.contains(v))
    }

    /// Returns `true` if `self` is a superset of `other`.
    ///
    /// `other` can be any iterator. We must ensure `T` lives as long as the iterator `'a`.
    pub fn is_superset<'a, I>(&self, other: I) -> bool
    where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
    {
        other.into_iter().all(|v| self.contains(v))
    }
}

// ==================================================================================
// 3. Trait Implementations
// ==================================================================================

impl<T: Eq + Hash + Debug, const N: usize> Default for SmallSet<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Debug + Eq + Hash, const N: usize> Debug for SmallSet<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

// Allows `set.iter().collect()` or `vec.into_iter().collect()`
impl<T: Eq + Hash + Debug, const N: usize> FromIterator<T> for SmallSet<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = SmallSet::new();
        for i in iter {
            set.insert(i);
        }
        set
    }
}

// Allows `for x in set` (Consuming Iterator)
impl<T: Eq + Hash + Debug, const N: usize> IntoIterator for SmallSet<T, N> {
    type Item = T;
    type IntoIter = SmallSetIntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        SmallSetIntoIter {
            iter: self.map.into_iter(),
        }
    }
}

/// A consuming iterator for `SmallSet`.
pub struct SmallSetIntoIter<T, const N: usize> {
    // We reuse the map's iterator logic. It yields (K, V).
    iter: crate::map::SmallMapIntoIter<T, (), N>,
}

impl<T, const N: usize> Iterator for SmallSetIntoIter<T, N> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        // Discard the empty tuple value, return key.
        self.iter.next().map(|(k, _)| k)
    }
}

// ==================================================================================
// 4. Reference Iterator Support
// ==================================================================================

/// An iterator over the references to values in a SmallSet.
pub struct SetRefIter<'a, T> {
    // We wrap the underlying map iterator
    iter: crate::map::SmallMapIter<'a, T, ()>,
}

impl<'a, T> Iterator for SetRefIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        // The map yields (&Key, &Value). We only want the Key.
        self.iter.next().map(|(k, _)| k)
    }
}

// Allows `for x in &set` or passing `&set` to methods like `union`
impl<'a, T, const N: usize> IntoIterator for &'a SmallSet<T, N>
where
    T: Eq + Hash + Debug,
{
    type Item = &'a T;
    type IntoIter = SetRefIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ==================================================================================
// 5. Tests
// ==================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    // --- 1. Basic CRUD & Stack Behavior ---
    #[test]
    fn test_basic_crud_stack() {
        let mut set: SmallSet<i32, 4> = SmallSet::new();

        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(set.is_on_stack());

        // Insert
        assert!(set.insert(10));
        assert!(set.insert(20));
        assert_eq!(set.len(), 2);

        // Contains
        assert!(set.contains(&10));
        assert!(!set.contains(&99));

        // Remove
        assert!(set.remove(&10));
        assert!(!set.contains(&10));
        assert_eq!(set.len(), 1);

        // Clear
        set.clear();
        assert!(set.is_empty());
        assert!(set.is_on_stack()); // Clearing shouldn't change storage mode
    }

    #[test]
    fn test_duplicates_handling() {
        let mut set: SmallSet<String, 4> = SmallSet::new();

        assert!(set.insert("A".to_string()));
        assert_eq!(set.len(), 1);

        // Duplicate insert returns false
        assert!(!set.insert("A".to_string()));
        assert_eq!(set.len(), 1); // Length should not increase
    }

    // --- 2. Spill Logic (The Critical Test) ---
    #[test]
    fn test_spill_threshold() {
        // N = 2. Should hold 2 items on stack. 3rd item triggers spill.
        let mut set: SmallSet<i32, 2> = SmallSet::new();

        set.insert(1);
        set.insert(2);
        assert!(set.is_on_stack());

        // Trigger Spill
        set.insert(3);

        assert!(!set.is_on_stack());
        assert_eq!(set.len(), 3);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
    }

    #[test]
    fn test_heap_growth() {
        let mut set: SmallSet<i32, 2> = SmallSet::new();

        // Push well past stack capacity
        for i in 0..100 {
            set.insert(i);
        }

        assert!(!set.is_on_stack());
        assert_eq!(set.len(), 100);
        assert!(set.contains(&50));
    }

    // --- 3. Iterators ---
    #[test]
    fn test_iter() {
        let set: SmallSet<i32, 4> = vec![1, 2, 3].into_iter().collect();
        let collected: Vec<_> = set.iter().cloned().collect(); // Arbitrary order

        assert_eq!(collected.len(), 3);
        assert!(collected.contains(&1));
        assert!(collected.contains(&2));
        assert!(collected.contains(&3));
    }

    #[test]
    fn test_into_iter_stack() {
        let mut set: SmallSet<i32, 4> = SmallSet::new();
        set.insert(1);
        set.insert(2);

        // Consuming iterator from Stack state
        let vec: Vec<i32> = set.into_iter().collect();
        assert_eq!(vec.len(), 2);
        assert!(vec.contains(&1));
        assert!(vec.contains(&2));
    }

    #[test]
    fn test_into_iter_heap() {
        let mut set: SmallSet<i32, 2> = SmallSet::new();
        set.insert(1);
        set.insert(2);
        set.insert(3); // Spilled

        // Consuming iterator from Heap state
        let vec: Vec<i32> = set.into_iter().collect();
        assert_eq!(vec.len(), 3);
        assert!(vec.contains(&1));
    }

    // --- 4. Set Algebra (Difference, Union, Intersection) ---
    #[test]
    fn test_difference() {
        let a: SmallSet<i32, 4> = vec![1, 2, 3].into_iter().collect();
        let b: SmallSet<i32, 4> = vec![3, 4, 5].into_iter().collect();

        // A - B = {1, 2}
        let diff: Vec<_> = a.difference(&b).cloned().collect();
        assert_eq!(diff.len(), 2);
        assert!(diff.contains(&1));
        assert!(diff.contains(&2));
        assert!(!diff.contains(&3));
    }

    #[test]
    fn test_intersection() {
        let a: SmallSet<i32, 4> = vec![1, 2, 3].into_iter().collect();
        let b: SmallSet<i32, 4> = vec![2, 3, 4].into_iter().collect();

        // A & B = {2, 3}
        let int: Vec<_> = a.intersection(&b).cloned().collect();
        assert_eq!(int.len(), 2);
        assert!(int.contains(&2));
        assert!(int.contains(&3));
        assert!(!int.contains(&1));
    }

    #[test]
    fn test_union() {
        let a: SmallSet<i32, 4> = vec![1, 2].into_iter().collect();
        let b: SmallSet<i32, 4> = vec![2, 3].into_iter().collect();

        // A | B = {1, 2, 3}
        // Passing &b works because we implemented IntoIterator for &SmallSet
        let u: Vec<_> = a.union(&b).cloned().collect();
        assert_eq!(u.len(), 3);
        assert!(u.contains(&1));
        assert!(u.contains(&2));
        assert!(u.contains(&3));
    }

    // --- 5. Boolean Logic (Subset, Disjoint) ---
    #[test]
    fn test_is_disjoint() {
        let a: SmallSet<i32, 4> = vec![1, 2].into_iter().collect();
        let b: SmallSet<i32, 4> = vec![3, 4].into_iter().collect();
        let c: SmallSet<i32, 4> = vec![2, 3].into_iter().collect();

        assert!(a.is_disjoint(&b)); // No shared elements
        assert!(!a.is_disjoint(&c)); // Shares '2'
    }

    #[test]
    fn test_is_subset() {
        let sub: SmallSet<i32, 4> = vec![1, 2].into_iter().collect();
        let sup: SmallSet<i32, 4> = vec![1, 2, 3].into_iter().collect();

        assert!(sub.is_subset(&sup));
        assert!(!sup.is_subset(&sub));

        // Empty set is subset of everything
        let empty: SmallSet<i32, 4> = SmallSet::new();
        assert!(empty.is_subset(&sub));
    }

    #[test]
    fn test_is_superset() {
        let sup: SmallSet<i32, 4> = vec![1, 2, 3].into_iter().collect();
        let sub_vec = vec![1, 2];

        assert!(sup.is_superset(&sub_vec)); // Works with Vec iterator
        assert!(!sup.is_superset(&vec![1, 99])); // Missing 99
    }

    // --- 6. Interoperability (Std Sets) ---
    #[test]
    fn test_interop_hashset() {
        let small: SmallSet<i32, 4> = vec![1, 2].into_iter().collect();
        let std_set: HashSet<i32> = vec![1, 2, 3].into_iter().collect();

        assert!(small.is_subset(&std_set));

        // Difference against std::HashSet
        let diff: Vec<_> = small.difference(&std_set).collect();
        assert!(diff.is_empty());
    }

    #[test]
    fn test_interop_btreeset() {
        let small: SmallSet<i32, 4> = vec![1, 2].into_iter().collect();
        let btree: BTreeSet<i32> = vec![2, 3].into_iter().collect();

        // Intersection with BTreeSet
        let int: Vec<_> = small.intersection(&btree).cloned().collect();
        assert_eq!(int, vec![2]);
    }

    // --- 7. Retain (Filter) ---
    #[test]
    fn test_retain() {
        let mut set: SmallSet<i32, 4> = vec![1, 2, 3, 4, 5].into_iter().collect();
        // Should spill (5 > 4)
        assert!(!set.is_on_stack());

        // Keep only even numbers
        set.retain(|x| x % 2 == 0);

        assert_eq!(set.len(), 2);
        assert!(set.contains(&2));
        assert!(set.contains(&4));
        assert!(!set.contains(&1));
    }
}
