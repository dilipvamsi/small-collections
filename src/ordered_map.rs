use core::mem::ManuallyDrop;
use core::ptr;
use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::iter::FromIterator;
use std::ops::{Index, IndexMut};

use heapless::LinearMap;
use ordermap::OrderMap;

/// A trait for abstraction over different map types (Stack, Heap, Small).
/// (Imported or redefined for convenience in this module)
pub use crate::map::AnyMap;

impl<K: Eq + Hash, V, S: std::hash::BuildHasher> AnyMap<K, V> for OrderMap<K, V, S> {
    fn len(&self) -> usize {
        self.len()
    }
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert(key, value)
    }
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get_mut(key)
    }
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove(key)
    }
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.contains_key(key)
    }
    fn clear(&mut self) {
        self.clear();
    }
}

/// A map that preserves insertion order and lives on the stack for `N` items, then spills to the heap.
///
/// # Overview
/// This collection uses `heapless::LinearMap` for stack storage and
/// `ordermap::OrderMap` for heap storage. Insertion order is maintained in both states.
///
/// # Safety
/// * `on_stack` tag determines which side of the `MapData` union is active.
/// * `ManuallyDrop` is used to manage union variant destruction.
pub struct SmallOrderedMap<K, V, const N: usize> {
    on_stack: bool,
    data: MapData<K, V, N>,
}

impl<K: Eq + Hash, V, const N: usize> AnyMap<K, V> for SmallOrderedMap<K, V, N> {
    fn len(&self) -> usize {
        self.len()
    }
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert(key, value)
    }
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get_mut(key)
    }
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // SmallOrderedMap::remove has K: Borrow<Q> + PartialEq<Q> bound in original impl
        // But for AnyMap we just need it to work.
        self.remove(key)
    }
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.contains_key(key)
    }
    fn clear(&mut self) {
        self.clear();
    }
}

/// Internal storage for `SmallOrderedMap`.
union MapData<K, V, const N: usize> {
    stack: ManuallyDrop<LinearMap<K, V, N>>,
    heap: ManuallyDrop<OrderMap<K, V>>,
}

impl<K, V, const N: usize> SmallOrderedMap<K, V, N>
where
    K: Eq + Hash,
{
    pub const MAX_STACK_SIZE: usize = 16 * 1024;

    /// Creates a new empty ordered map on the stack.
    pub fn new() -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= SmallOrderedMap::<K, V, N>::MAX_STACK_SIZE,
                "SmallOrderedMap is too large! Reduce N."
            );
        }

        Self {
            on_stack: true,
            data: MapData {
                stack: ManuallyDrop::new(LinearMap::new()),
            },
        }
    }

    /// Returns `true` if the map is currently storing data on the stack.
    #[inline]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        unsafe {
            if self.on_stack {
                self.data.stack.len()
            } else {
                self.data.heap.len()
            }
        }
    }

    /// Returns `true` if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the map, removing all key-value pairs.
    pub fn clear(&mut self) {
        unsafe {
            if self.on_stack {
                (*self.data.stack).clear();
            } else {
                (*self.data.heap).clear();
            }
        }
    }

    /// Inserts a key-value pair into the map.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        unsafe {
            if self.on_stack {
                let stack_map = &mut *self.data.stack;

                if stack_map.len() == N && !stack_map.contains_key(&key) {
                    self.spill_to_heap();
                } else {
                    return stack_map.insert(key, value).ok().flatten();
                }
            }

            (*self.data.heap).insert(key, value)
        }
    }

    /// Retrieves a reference to the value corresponding to the key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        unsafe {
            if self.on_stack {
                // LinearMap doesn't support Borrow lookups natively
                self.data
                    .stack
                    .iter()
                    .find(|&(k, _)| <K as Borrow<Q>>::borrow(k) == key)
                    .map(|(_, v)| v)
            } else {
                self.data.heap.get(key)
            }
        }
    }

    /// Retrieves a mutable reference to the value corresponding to the key.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        unsafe {
            if self.on_stack {
                (*self.data.stack)
                    .iter_mut()
                    .find(|&(ref k, _)| <K as Borrow<Q>>::borrow(k) == key)
                    .map(|(_, v)| v)
            } else {
                (*self.data.heap).get_mut(key)
            }
        }
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        unsafe {
            if self.on_stack {
                let stack = &mut *self.data.stack;
                // Linear scan using borrow() to avoid PartialEq<Q> bound on K
                stack
                    .iter()
                    .find(|(k, _)| <K as Borrow<Q>>::borrow(k) == key)?;

                // heapless::LinearMap doesn't have remove_at,
                // but we can manually rebuild or if it has a way to remove by key...
                // Actually, LinearMap::remove(key) requires K: PartialEq<Q>.
                // We'll use the fact that we have the index to manually swap_remove if it's the only way,
                // or just call into_iter and collect into a new stack map (since N is small).

                // Better: LinearMap::remove(key) is what we want, but we can't call it if K doesn't impl PartialEq<Q>.
                // Wait, if we know it's at 'idx', we can just remove it from the underlying buffer?
                // LinearMap is a private wrapper around a heapless::Vec of (K, V).

                // For now, let's use a slightly less efficient but correct implementation for the trait:
                let mut temp_stack: LinearMap<K, V, N> = LinearMap::new();
                let mut removed_val = None;

                // We move items out and back in
                let old_stack = core::ptr::read(stack);
                for (k, v) in old_stack.into_iter() {
                    if k.borrow() == key && removed_val.is_none() {
                        removed_val = Some(v);
                    } else {
                        let _ = temp_stack.insert(k, v);
                    }
                }
                core::ptr::write(stack, temp_stack);
                removed_val
            } else {
                (*self.data.heap).remove(key)
            }
        }
    }

    /// Returns `true` if the map contains a value for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    #[inline(never)]
    unsafe fn spill_to_heap(&mut self) {
        unsafe {
            let stack_map = ptr::read(&*self.data.stack);
            let mut new_heap = OrderMap::with_capacity(stack_map.len() * 2);

            for (key, value) in stack_map.into_iter() {
                new_heap.insert(key, value);
            }

            ptr::write(&mut self.data.heap, ManuallyDrop::new(new_heap));
            self.on_stack = false;
        }
    }
}

// --- Index Traits ---

impl<K, V, Q, const N: usize> Index<&Q> for SmallOrderedMap<K, V, N>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash + ?Sized,
{
    type Output = V;

    fn index(&self, key: &Q) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, V, Q, const N: usize> IndexMut<&Q> for SmallOrderedMap<K, V, N>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash + ?Sized,
{
    fn index_mut(&mut self, key: &Q) -> &mut Self::Output {
        self.get_mut(key).expect("no entry found for key")
    }
}

// --- Traits ---

impl<K, V, const N: usize> Drop for SmallOrderedMap<K, V, N> {
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

impl<K, V, const N: usize> Default for SmallOrderedMap<K, V, N>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, const N: usize> Clone for SmallOrderedMap<K, V, N>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        unsafe {
            if self.on_stack {
                Self {
                    on_stack: true,
                    data: MapData {
                        stack: ManuallyDrop::new((*self.data.stack).clone()),
                    },
                }
            } else {
                Self {
                    on_stack: false,
                    data: MapData {
                        heap: ManuallyDrop::new((*self.data.heap).clone()),
                    },
                }
            }
        }
    }
}

impl<K, V, const N: usize> Debug for SmallOrderedMap<K, V, N>
where
    K: Eq + Hash + Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, const N: usize> SmallOrderedMap<K, V, N>
where
    K: Eq + Hash,
{
    pub fn iter(&self) -> SmallMapIter<'_, K, V> {
        unsafe {
            if self.on_stack {
                SmallMapIter::Stack(self.data.stack.iter())
            } else {
                SmallMapIter::Heap(self.data.heap.iter())
            }
        }
    }
}

pub enum SmallMapIter<'a, K, V> {
    Stack(heapless::linear_map::Iter<'a, K, V>),
    Heap(ordermap::map::Iter<'a, K, V>),
}

impl<'a, K, V> Iterator for SmallMapIter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SmallMapIter::Stack(i) => i.next(),
            SmallMapIter::Heap(i) => i.next(),
        }
    }
}

impl<K, V, const N: usize> IntoIterator for SmallOrderedMap<K, V, N>
where
    K: Eq + Hash,
{
    type Item = (K, V);
    type IntoIter = SmallMapIntoIter<K, V, N>;

    fn into_iter(self) -> Self::IntoIter {
        let this = ManuallyDrop::new(self);
        unsafe {
            if this.on_stack {
                SmallMapIntoIter::Stack(ptr::read(&*this.data.stack).into_iter())
            } else {
                SmallMapIntoIter::Heap(ptr::read(&*this.data.heap).into_iter())
            }
        }
    }
}

pub enum SmallMapIntoIter<K: Eq, V, const N: usize> {
    Stack(heapless::linear_map::IntoIter<K, V, N>),
    Heap(ordermap::map::IntoIter<K, V>),
}

impl<K, V, const N: usize> Iterator for SmallMapIntoIter<K, V, N>
where
    K: Eq + Hash,
{
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SmallMapIntoIter::Stack(i) => i.next(),
            SmallMapIntoIter::Heap(i) => i.next(),
        }
    }
}

impl<K, V, const N: usize> FromIterator<(K, V)> for SmallOrderedMap<K, V, N>
where
    K: Eq + Hash,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = SmallOrderedMap::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K, V, const N: usize> Extend<(K, V)> for SmallOrderedMap<K, V, N>
where
    K: Eq + Hash,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordered_map_stack_ops_insertion_order() {
        let mut map: SmallOrderedMap<i32, i32, 4> = SmallOrderedMap::new();
        map.insert(3, 30);
        map.insert(1, 10);
        map.insert(2, 20);

        let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![3, 1, 2]);
    }

    #[test]
    fn test_ordered_map_spill_trigger_on_insert() {
        let mut map: SmallOrderedMap<i32, i32, 2> = SmallOrderedMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        assert!(map.is_on_stack());

        map.insert(3, 30); // Spill
        assert!(!map.is_on_stack());

        let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 2, 3]);
    }

    #[test]
    fn test_ordered_map_any_storage_lifecycle() {
        let mut map: SmallOrderedMap<String, i32, 2> = SmallOrderedMap::new();
        map.insert("a".into(), 1);
        map.insert("b".into(), 2);
        assert_eq!(map.get("a"), Some(&1));
        assert_eq!(map.remove("a"), Some(1));
        assert_eq!(map.len(), 1);
        assert!(map.is_on_stack());

        map.insert("c".into(), 3);
        map.insert("d".into(), 4); // Spill
        assert!(!map.is_on_stack());
        assert_eq!(map.len(), 3);
        assert_eq!(map.get("b"), Some(&2));
        assert_eq!(map.get("c"), Some(&3));
        assert_eq!(map.get("d"), Some(&4));
    }

    #[test]
    fn test_ordered_map_any_storage_get_mut() {
        let mut map: SmallOrderedMap<i32, i32, 4> = SmallOrderedMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        if let Some(v) = map.get_mut(&1) {
            *v = 11;
        }
        assert_eq!(map[&1], 11);
        map[&2] = 22;
        assert_eq!(map.get(&2), Some(&22));

        for i in 3..6 {
            map.insert(i, i * 10);
        }
        assert!(!map.is_on_stack());
        map[&5] = 55;
        assert_eq!(map.get(&5), Some(&55));
    }

    #[test]
    fn test_ordered_map_any_storage_clear() {
        let mut map: SmallOrderedMap<i32, i32, 2> = SmallOrderedMap::new();
        assert!(map.is_empty());
        map.insert(1, 10);
        map.clear();
        assert!(map.is_empty());

        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30); // Spill
        map.clear();
        assert!(map.is_empty());
        assert!(!map.is_on_stack());
    }

    #[test]
    fn test_ordered_map_traits_exhaustive() {
        let mut map: SmallOrderedMap<i32, i32, 4> = SmallOrderedMap::new();
        map.insert(1, 10);
        map.insert(2, 20);

        // Clone
        let cloned = map.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.get(&1), Some(&10));

        // Debug
        let debug = format!("{:?}", map);
        assert!(debug.contains("1: 10"));

        // Default
        let def: SmallOrderedMap<i32, i32, 4> = SmallOrderedMap::default();
        assert!(def.is_empty());

        // FromIterator
        let collected: SmallOrderedMap<i32, i32, 4> = vec![(1, 10), (2, 20)].into_iter().collect();
        assert_eq!(collected.len(), 2);
        let items: Vec<_> = collected.iter().collect();
        assert_eq!(items, vec![(&1, &10), (&2, &20)]);

        // Extend
        let mut map2 = SmallOrderedMap::<i32, i32, 4>::new();
        map2.extend(vec![(1, 10), (2, 20)]);
        assert_eq!(map2.len(), 2);

        // IntoIterator
        let vec: Vec<_> = map2.into_iter().collect();
        assert_eq!(vec.len(), 2);
        assert!(vec.contains(&(1, 10)));
        assert!(vec.contains(&(2, 20)));
    }

    #[test]
    fn test_ordered_map_any_storage_borrow_lookups() {
        // LinearMap borrow lookup
        let mut map: SmallOrderedMap<String, i32, 4> = SmallOrderedMap::new();
        map.insert("Apple".to_string(), 100);
        assert_eq!(map.get("Apple"), Some(&100));
        assert_eq!(map.get_mut("Apple"), Some(&mut 100));
    }

    #[test]
    fn test_ordered_map_any_storage_clone_heap() {
        let h: SmallOrderedMap<i32, i32, 2> = vec![(1, 1), (2, 2), (3, 3)].into_iter().collect();
        assert!(!h.is_on_stack());

        let h2 = h.clone();
        assert_eq!(h2.len(), 3);
        assert!(!h2.is_on_stack());
    }

    #[test]
    fn test_ordered_map_traits_into_iter_heap() {
        let h: SmallOrderedMap<i32, i32, 2> = vec![(1, 1), (2, 2), (3, 3)].into_iter().collect();
        let mut it = h.into_iter();
        assert_eq!(it.next(), Some((1, 1)));
    }
}
