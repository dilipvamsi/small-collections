#![cfg(feature = "lru")]
//! LRU cache that lives on the stack and spills to the heap.

use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::num::NonZeroUsize;
use core::ptr;
use lru::LruCache;
use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::hash::Hash;

use crate::HeaplessBTreeLruCache;
use crate::IndexType;

/// An object-safe abstraction over LRU cache types.
pub trait AnyLruCache<K, V> {
    /// Returns the number of key-value pairs that are currently on this backend.
    fn len(&self) -> usize;
    /// Returns true if the cache is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the logical maximum capacity of this cache backend.
    fn cap(&self) -> NonZeroUsize;
    /// Inserts a key-value pair, updating the value if the key already exists.
    /// Returns the old value if the key was present.
    fn put(&mut self, key: K, value: V) -> Option<V>;
    /// Inserts a key-value pair with a specific maximum capacity enforcement.
    /// Returns `(old_value, Result<(), (key, value)>)`. The result is an error if capacity is reached and the backend cannot grow.
    fn put_with_cap(
        &mut self,
        key: K,
        value: V,
        cap: NonZeroUsize,
    ) -> (Option<V>, Result<(), (K, V)>);
    /// Returns a reference to the value corresponding to the key, moving it to the MRU position.
    fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;
    /// Returns a mutable reference to the value corresponding to the key, moving it to the MRU position.
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;
    /// Returns a reference to the value corresponding to the key without updating the LRU state.
    fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;
    /// Clears the cache, removing all values.
    fn clear(&mut self);
    /// Removes and returns the explicitly Least Recently Used key-value pair.
    fn pop_lru(&mut self) -> Option<(K, V)>;
    /// Checks if the cache contains the given key.
    fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;
    /// Pushes a key-value pair into the cache. If the key already exists, updates the value.
    /// If pushing causes the capacity to be exceeded, returns the evicted LRU entry.
    fn push(&mut self, key: K, value: V) -> Option<(K, V)>;
    /// Removes the given key from the cache and returns its associated value.
    fn pop<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;
    /// Removes the given key from the cache and returns the (key, value) pair.
    fn pop_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;
    /// Explicitly marks the given key as the Most Recently Used.
    fn promote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;
    /// Explicitly marks the given key as the Least Recently Used.
    fn demote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;
    /// Returns a reference to the Least Recently Used pair without removing it.
    fn peek_lru(&self) -> Option<(&K, &V)>;
}

/// Helper trait for backends that support iteration.
pub trait LruIteratorSupport<'a, K: 'a, V: 'a> {
    type Iter: Iterator<Item = (&'a K, &'a V)>;
    type IterMut: Iterator<Item = (&'a K, &'a mut V)>;
    fn iter(&'a self) -> Self::Iter;
    fn iter_mut(&'a mut self) -> Self::IterMut;
}

impl<K: Hash + Eq + Ord, V> AnyLruCache<K, V> for LruCache<K, V> {
    fn len(&self) -> usize {
        self.len()
    }
    fn cap(&self) -> NonZeroUsize {
        self.cap()
    }
    fn put(&mut self, key: K, value: V) -> Option<V> {
        self.put(key, value)
    }
    fn put_with_cap(
        &mut self,
        key: K,
        value: V,
        _cap: NonZeroUsize,
    ) -> (Option<V>, Result<(), (K, V)>) {
        (self.put(key, value), Ok(()))
    }
    fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.get(key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.get_mut(key)
    }
    fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.peek(key)
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn pop_lru(&mut self) -> Option<(K, V)> {
        self.pop_lru()
    }
    fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.contains(key)
    }
    fn push(&mut self, key: K, value: V) -> Option<(K, V)> {
        self.push(key, value)
    }
    fn pop<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.pop(key)
    }
    fn pop_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.pop_entry(key)
    }
    fn promote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.promote(key)
    }
    fn demote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.demote(key)
    }
    fn peek_lru(&self) -> Option<(&K, &V)> {
        self.peek_lru()
    }
}

impl<'a, K: Hash + Eq + Ord + 'a, V: 'a> LruIteratorSupport<'a, K, V> for LruCache<K, V> {
    type Iter = lru::Iter<'a, K, V>;
    type IterMut = lru::IterMut<'a, K, V>;
    fn iter(&'a self) -> Self::Iter {
        self.iter()
    }
    fn iter_mut(&'a mut self) -> Self::IterMut {
        self.iter_mut()
    }
}

pub union LruData<K, V, S> {
    pub stack: ManuallyDrop<S>,
    pub heap: ManuallyDrop<LruCache<K, V>>,
}

pub struct SmallLruCache<
    K,
    V,
    const N: usize,
    I: IndexType = u8,
    S = HeaplessBTreeLruCache<K, V, N, I>,
> where
    S: AnyLruCache<K, V>,
{
    num_entries: usize,
    capacity: NonZeroUsize,
    on_stack: bool,
    data: LruData<K, V, S>,
    _phantom_i: PhantomData<I>,
}

impl<K, V, const N: usize, I, S> SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V>,
{
    /// Returns the number of key-value pairs currently in the cache.
    pub fn len(&self) -> usize {
        self.num_entries
    }
    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }
    /// Returns the maximum capacity of the cache.
    pub fn capacity(&self) -> NonZeroUsize {
        self.capacity
    }

    /// Returns a reference to the value corresponding to the key, moving it to the MRU position.
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if self.on_stack {
            unsafe { (*self.data.stack).get(key) }
        } else {
            unsafe { (*self.data.heap).get(key) }
        }
    }

    /// Returns a mutable reference to the value corresponding to the key, moving it to the MRU position.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if self.on_stack {
            unsafe { (*self.data.stack).get_mut(key) }
        } else {
            unsafe { (*self.data.heap).get_mut(key) }
        }
    }

    /// Returns a reference to the value corresponding to the key without updating the LRU state.
    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if self.on_stack {
            unsafe { (*self.data.stack).peek(key) }
        } else {
            unsafe { (*self.data.heap).peek(key) }
        }
    }

    /// Pushes a key-value pair into the cache. If the key already exists, updates the value and returns it.
    ///
    /// If the stack capacity `N` is exceeded during put, this will transparently allocate a `lru::LruCache`
    /// and spill all elements to the heap.
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        if self.on_stack {
            let (old_v, res) =
                unsafe { (*self.data.stack).put_with_cap(key, value, self.capacity) };
            if res.is_ok() {
                self.num_entries = unsafe { (*self.data.stack).len() };
                return old_v;
            } else {
                let (k, v) = res.err().unwrap();
                self.spill_to_heap();
                let old = unsafe { (*self.data.heap).put(k, v) };
                self.num_entries = unsafe { (*self.data.heap).len() };
                return old;
            }
        }
        let old = unsafe { (*self.data.heap).put(key, value) };
        self.num_entries = unsafe { (*self.data.heap).len() };
        old
    }

    /// Internal method to transition storage from stack to heap.
    fn spill_to_heap(&mut self) {
        if !self.on_stack {
            return;
        }
        let mut heap = LruCache::new(self.capacity);
        unsafe {
            let mut stack = ManuallyDrop::take(&mut self.data.stack);
            while let Some((k, v)) = stack.pop_lru() {
                heap.put(k, v);
            }
            ptr::write(&mut self.data.heap, ManuallyDrop::new(heap));
            self.on_stack = false;
        }
    }

    /// Clears the cache, removing all key-value pairs.
    pub fn clear(&mut self) {
        if self.on_stack {
            unsafe { (*self.data.stack).clear() }
        } else {
            unsafe { (*self.data.heap).clear() }
        }
        self.num_entries = 0;
    }

    /// Returns `true` if this cache is currently allocated entirely on the stack.
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }

    /// Returns `true` if the cache contains a value for the specified key.
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if self.on_stack {
            unsafe { (*self.data.stack).contains(key) }
        } else {
            unsafe { (*self.data.heap).contains(key) }
        }
    }

    /// Pushes a key-value pair into the cache. If the key already exists, updates the value.
    /// If pushing causes the capacity to be exceeded, returns the evicted LRU pair.
    pub fn push(&mut self, key: K, value: V) -> Option<(K, V)> {
        if self.on_stack {
            let res = unsafe { (*self.data.stack).push(key, value) };
            self.num_entries = unsafe { (*self.data.stack).len() };
            res
        } else {
            let res = unsafe { (*self.data.heap).push(key, value) };
            self.num_entries = unsafe { (*self.data.heap).len() };
            res
        }
    }

    /// Removes and returns the value corresponding to the key from the cache.
    pub fn pop<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        let res = if self.on_stack {
            unsafe { (*self.data.stack).pop(key) }
        } else {
            unsafe { (*self.data.heap).pop(key) }
        };
        if res.is_some() {
            self.num_entries -= 1;
        }
        res
    }

    /// Removes and returns the key-value pair corresponding to the key from the cache.
    pub fn pop_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        let res = if self.on_stack {
            unsafe { (*self.data.stack).pop_entry(key) }
        } else {
            unsafe { (*self.data.heap).pop_entry(key) }
        };
        if res.is_some() {
            self.num_entries -= 1;
        }
        res
    }

    /// Removes and returns the Least Recently Used (LRU) key-value pair.
    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        let res = if self.on_stack {
            unsafe { (*self.data.stack).pop_lru() }
        } else {
            unsafe { (*self.data.heap).pop_lru() }
        };
        if res.is_some() {
            self.num_entries -= 1;
        }
        res
    }

    /// Explictly promotes the corresponding key to the Most Recently Used (MRU) position.
    pub fn promote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if self.on_stack {
            unsafe { (*self.data.stack).promote(key) }
        } else {
            unsafe { (*self.data.heap).promote(key) }
        }
    }

    /// Explicitly demotes the corresponding key to the Least Recently Used (LRU) position.
    pub fn demote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if self.on_stack {
            unsafe { (*self.data.stack).demote(key) }
        } else {
            unsafe { (*self.data.heap).demote(key) }
        }
    }

    /// Returns references to the Least Recently Used (LRU) key-value pair without removing it.
    pub fn peek_lru(&self) -> Option<(&K, &V)> {
        if self.on_stack {
            unsafe { (*self.data.stack).peek_lru() }
        } else {
            unsafe { (*self.data.heap).peek_lru() }
        }
    }

    /// Resizes the cache to a new maximum capacity.
    /// If the new capacity exceeds the stack limit `N`, it transparently spills to the heap.
    pub fn resize(&mut self, cap: NonZeroUsize) {
        if self.on_stack {
            if cap.get() > N {
                self.spill_to_heap();
                unsafe { (*self.data.heap).resize(cap) };
            }
        } else {
            unsafe { (*self.data.heap).resize(cap) };
        }
        self.capacity = cap;
    }

    /// Returns a reference to the value for the key, or inserts a new one.
    pub fn get_or_insert<F>(&mut self, key: K, f: F) -> &V
    where
        F: FnOnce() -> V,
    {
        if self.contains(&key) {
            self.get(&key).unwrap()
        } else {
            self.put(key.clone(), f());
            self.get(&key).unwrap()
        }
    }

    /// Returns a mutable reference to the value for the key, or inserts a new one.
    pub fn get_or_insert_mut<F>(&mut self, key: K, f: F) -> &mut V
    where
        F: FnOnce() -> V,
    {
        if self.contains(&key) {
            self.get_mut(&key).unwrap()
        } else {
            self.put(key.clone(), f());
            self.get_mut(&key).unwrap()
        }
    }

    pub fn iter(&self) -> Iter<'_, K, V, N, I, S>
    where
        for<'a> S: LruIteratorSupport<'a, K, V>,
    {
        if self.on_stack {
            Iter::Stack(unsafe { (*self.data.stack).iter() })
        } else {
            Iter::Heap(unsafe { (*self.data.heap).iter() })
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, N, I, S>
    where
        for<'a> S: LruIteratorSupport<'a, K, V>,
    {
        if self.on_stack {
            IterMut::Stack(unsafe { (*self.data.stack).iter_mut() })
        } else {
            IterMut::Heap(unsafe { (*self.data.heap).iter_mut() })
        }
    }
}

impl<'a, K, V, const N: usize, I, S> LruIteratorSupport<'a, K, V> for SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone + 'a,
    V: 'a,
    I: IndexType,
    S: AnyLruCache<K, V> + for<'b> LruIteratorSupport<'b, K, V>,
{
    type Iter = Iter<'a, K, V, N, I, S>;
    type IterMut = IterMut<'a, K, V, N, I, S>;
    fn iter(&'a self) -> Self::Iter {
        self.iter()
    }
    fn iter_mut(&'a mut self) -> Self::IterMut {
        self.iter_mut()
    }
}

impl<K, V, const N: usize, I, S> SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + Default,
{
    pub fn new(cap: NonZeroUsize) -> Self {
        Self {
            num_entries: 0,
            capacity: cap,
            on_stack: true,
            data: LruData {
                stack: ManuallyDrop::new(S::default()),
            },
            _phantom_i: PhantomData,
        }
    }
}

impl<K: Hash + Eq + Ord + Clone, V, const N: usize, I: IndexType, S> Default
    for SmallLruCache<K, V, N, I, S>
where
    S: AnyLruCache<K, V> + Default,
{
    fn default() -> Self {
        Self::new(NonZeroUsize::new(N.max(1)).unwrap())
    }
}

impl<K, V, const N: usize, I, S> AnyLruCache<K, V> for SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V>,
{
    fn len(&self) -> usize {
        self.num_entries
    }
    fn cap(&self) -> NonZeroUsize {
        self.capacity
    }
    fn put(&mut self, key: K, value: V) -> Option<V> {
        self.put(key, value)
    }
    fn put_with_cap(
        &mut self,
        key: K,
        value: V,
        cap: NonZeroUsize,
    ) -> (Option<V>, Result<(), (K, V)>) {
        let (old, res) = if self.on_stack {
            unsafe { (*self.data.stack).put_with_cap(key, value, cap) }
        } else {
            (unsafe { (*self.data.heap).put(key, value) }, Ok(()))
        };
        if res.is_ok() && old.is_none() {
            self.num_entries += 1;
        }
        (old, res)
    }
    fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.get(key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.get_mut(key)
    }
    fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.peek(key)
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn pop_lru(&mut self) -> Option<(K, V)> {
        self.pop_lru()
    }
    fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.contains(key)
    }
    fn push(&mut self, key: K, value: V) -> Option<(K, V)> {
        self.push(key, value)
    }
    fn pop<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.pop(key)
    }
    fn pop_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.pop_entry(key)
    }
    fn promote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.promote(key)
    }
    fn demote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.demote(key)
    }
    fn peek_lru(&self) -> Option<(&K, &V)> {
        self.peek_lru()
    }
}

impl<K, V, const N: usize, I, S> Drop for SmallLruCache<K, V, N, I, S>
where
    I: IndexType,
    S: AnyLruCache<K, V>,
{
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

impl<K, V, const N: usize, I, S> Clone for SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    V: Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + Clone,
{
    fn clone(&self) -> Self {
        if self.on_stack {
            Self {
                num_entries: self.num_entries,
                capacity: self.capacity,
                on_stack: true,
                data: LruData {
                    stack: ManuallyDrop::new(unsafe { (*self.data.stack).clone() }),
                },
                _phantom_i: PhantomData,
            }
        } else {
            Self {
                num_entries: self.num_entries,
                capacity: self.capacity,
                on_stack: false,
                data: LruData {
                    heap: ManuallyDrop::new(unsafe { (*self.data.heap).clone() }),
                },
                _phantom_i: PhantomData,
            }
        }
    }
}

impl<K, V, const N: usize, I, S> Debug for SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone + Debug,
    V: Debug,
    I: IndexType,
    S: AnyLruCache<K, V>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmallLruCache")
            .field("on_stack", &self.on_stack)
            .field("capacity", &self.capacity)
            .field("num_entries", &self.num_entries)
            .finish()
    }
}

impl<K, V, const N: usize, I, S> FromIterator<(K, V)> for SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + Default,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut cache = Self::default();
        for (k, v) in iter {
            cache.put(k, v);
        }
        cache
    }
}

impl<K, V, const N: usize, I, S> Extend<(K, V)> for SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V>,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.put(k, v);
        }
    }
}

impl<'a, K: 'a, V: 'a, const N: usize, I: IndexType, S> Debug for Iter<'a, K, V, N, I, S>
where
    S: AnyLruCache<K, V> + LruIteratorSupport<'a, K, V>,
    S::Iter: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Iter::Stack(it) => f.debug_tuple("Iter::Stack").field(it).finish(),
            Iter::Heap(_) => f.write_str("Iter::Heap(lru::Iter)"),
            Iter::_Phantom(_) => unreachable!(),
        }
    }
}

pub enum Iter<'a, K: 'a, V: 'a, const N: usize, I: IndexType, S>
where
    S: AnyLruCache<K, V> + LruIteratorSupport<'a, K, V>,
{
    Stack(S::Iter),
    Heap(lru::Iter<'a, K, V>),
    _Phantom(PhantomData<I>),
}

impl<'a, K, V, const N: usize, I, S> Iterator for Iter<'a, K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + LruIteratorSupport<'a, K, V>,
{
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Iter::Stack(it) => it.next(),
            Iter::Heap(it) => it.next(),
            Iter::_Phantom(_) => unreachable!(),
        }
    }
}

impl<'a, K: 'a, V: 'a, const N: usize, I: IndexType, S> Debug for IterMut<'a, K, V, N, I, S>
where
    S: AnyLruCache<K, V> + LruIteratorSupport<'a, K, V>,
    S::IterMut: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IterMut::Stack(it) => f.debug_tuple("IterMut::Stack").field(it).finish(),
            IterMut::Heap(_) => f.write_str("IterMut::Heap(lru::IterMut)"),
            IterMut::_Phantom(_) => unreachable!(),
        }
    }
}

pub enum IterMut<'a, K: 'a, V: 'a, const N: usize, I: IndexType, S>
where
    S: AnyLruCache<K, V> + LruIteratorSupport<'a, K, V>,
{
    Stack(S::IterMut),
    Heap(lru::IterMut<'a, K, V>),
    _Phantom(PhantomData<I>),
}

impl<'a, K, V, const N: usize, I, S> Iterator for IterMut<'a, K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + LruIteratorSupport<'a, K, V>,
{
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IterMut::Stack(it) => it.next(),
            IterMut::Heap(it) => it.next(),
            IterMut::_Phantom(_) => unreachable!(),
        }
    }
}

pub enum IntoIter<K, V, const N: usize, I: IndexType, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + IntoIterator<Item = (K, V)>,
{
    Stack(S::IntoIter),
    Heap(lru::IntoIter<K, V>),
    _Phantom(PhantomData<I>),
}

impl<K, V, const N: usize, I, S> Iterator for IntoIter<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + IntoIterator<Item = (K, V)>,
{
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IntoIter::Stack(it) => it.next(),
            IntoIter::Heap(it) => it.next(),
            IntoIter::_Phantom(_) => unreachable!(),
        }
    }
}

impl<K, V, const N: usize, I, S> IntoIterator for SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + IntoIterator<Item = (K, V)>,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, N, I, S>;
    fn into_iter(self) -> Self::IntoIter {
        let on_stack = self.on_stack;
        // SAFETY: We use ptr::read to move the data out of the union and then forget(self)
        // so that the Drop trait of SmallLruCache (which would try to drop the union) doesn't run.
        let data = unsafe { ptr::read(&self.data) };
        core::mem::forget(self);

        if on_stack {
            IntoIter::Stack(unsafe { ManuallyDrop::into_inner(data.stack) }.into_iter())
        } else {
            IntoIter::Heap(unsafe { ManuallyDrop::into_inner(data.heap) }.into_iter())
        }
    }
}

impl<'a, K, V, const N: usize, I, S> IntoIterator for &'a SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + for<'b> LruIteratorSupport<'b, K, V>,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, N, I, S>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, const N: usize, I, S> IntoIterator for &'a mut SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + for<'b> LruIteratorSupport<'b, K, V>,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, N, I, S>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod lru_cache_basic_tests {
    use super::*;
    use crate::HeaplessBTreeLruCache;
    use crate::HeaplessLinearLruCache;
    use crate::HeaplessLruCache;

    fn test_basic_lru<S: AnyLruCache<i32, i32> + Default>() {
        let mut cache = S::default();
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get(&2), Some(&20));
        assert_eq!(cache.get(&3), Some(&30));

        // Test Promotion: After get(&1), 1 should be MRU.
        cache.get(&1);

        // Order should be (LRU) 2 -> 3 -> 1 (MRU)
        assert_eq!(cache.peek_lru(), Some((&2, &20)));
        assert_eq!(cache.pop_lru(), Some((2, 20)));
        assert_eq!(cache.pop_lru(), Some((3, 30)));
        assert_eq!(cache.pop_lru(), Some((1, 10)));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_heapless_lru_basic() {
        test_basic_lru::<HeaplessLruCache<i32, i32, 8>>();
    }

    #[test]
    fn test_heapless_btree_lru_basic() {
        test_basic_lru::<HeaplessBTreeLruCache<i32, i32, 10>>();
    }

    #[test]
    fn test_heapless_linear_lru_basic() {
        test_basic_lru::<HeaplessLinearLruCache<i32, i32, 10>>();
    }

    #[test]
    fn test_small_lru_spill() {
        let mut cache: SmallLruCache<i32, i32, 2> = SmallLruCache::default();
        cache.put(1, 10);
        cache.put(2, 20);
        assert!(cache.is_on_stack());

        cache.resize(NonZeroUsize::new(5).unwrap());
        cache.put(3, 30);
        assert!(!cache.is_on_stack());
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get(&2), Some(&20));
        assert_eq!(cache.get(&3), Some(&30));
    }

    #[test]
    fn test_small_lru_eviction() {
        let mut cache: SmallLruCache<i32, i32, 2> = SmallLruCache::default();
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);

        assert_eq!(cache.len(), 2);
        assert!(!cache.contains(&1));
    }

    #[test]
    fn test_iteration_order() {
        let mut cache: SmallLruCache<i32, i32, 5> = SmallLruCache::default();
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);

        let items: Vec<_> = cache.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(items, vec![(3, 30), (2, 20), (1, 10)]);

        let items_into: Vec<_> = cache.into_iter().collect();
        assert_eq!(items_into, vec![(3, 30), (2, 20), (1, 10)]);
    }

    #[test]
    fn test_promote_demote() {
        let mut cache: SmallLruCache<i32, i32, 3> = SmallLruCache::default();
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);

        cache.demote(&3);
        assert_eq!(cache.peek_lru(), Some((&3, &30)));

        cache.promote(&1);
        let items: Vec<_> = cache.iter().map(|(&k, _)| k).collect();
        assert_eq!(items, vec![1, 2, 3]);
    }
}

#[cfg(test)]
mod lru_cache_coverage_tests {
    use super::*;
    use crate::HeaplessBTreeLruCache;
    use crate::HeaplessLinearLruCache;
    use crate::HeaplessLruCache;
    use std::num::NonZeroUsize;

    #[test]
    fn test_any_lru_cache_trait_implementation() {
        let mut cache: SmallLruCache<i32, i32, 2> = SmallLruCache::default();

        // Use a generic function to test trait implementation
        fn check<S: AnyLruCache<i32, i32>>(any_cache: &mut S) {
            assert_eq!(any_cache.len(), 0);
            assert_eq!(any_cache.cap(), NonZeroUsize::new(2).unwrap());

            any_cache.put(1, 10);
            assert_eq!(any_cache.len(), 1);
            assert_eq!(any_cache.get(&1), Some(&10));
            assert_eq!(any_cache.peek(&1), Some(&10));
            assert!(any_cache.contains(&1));

            let _ = any_cache.put_with_cap(2, 20, NonZeroUsize::new(2).unwrap());
            assert_eq!(any_cache.len(), 2);

            any_cache.promote(&1);
            any_cache.demote(&1);
            assert_eq!(any_cache.peek_lru(), Some((&1, &10)));

            any_cache.push(3, 30); // Should evict 1
            assert!(!any_cache.contains(&1));

            any_cache.pop(&2);
            assert_eq!(any_cache.len(), 1);

            any_cache.clear();
            assert!(any_cache.is_empty());
        }

        check(&mut cache);
    }

    #[test]
    fn test_debug_implementations() {
        let cache_hash: HeaplessLruCache<i32, i32, 8> = HeaplessLruCache::new();
        let cache_btree: HeaplessBTreeLruCache<i32, i32, 8> = HeaplessBTreeLruCache::new();
        let cache_linear: HeaplessLinearLruCache<i32, i32, 8> = HeaplessLinearLruCache::new();
        let cache_small: SmallLruCache<i32, i32, 8> = SmallLruCache::default();

        println!("{:?}", cache_hash);
        println!("{:?}", cache_btree);
        println!("{:?}", cache_linear);
        println!("{:?}", cache_small);
    }

    #[test]
    fn test_clear_all_backends() {
        fn test_clear<S: AnyLruCache<i32, i32> + Default>() {
            let mut cache = S::default();
            cache.put(1, 10);
            cache.clear();
            assert!(cache.is_empty());
        }

        test_clear::<HeaplessLruCache<i32, i32, 8>>();
        test_clear::<HeaplessBTreeLruCache<i32, i32, 8>>();
        test_clear::<HeaplessLinearLruCache<i32, i32, 8>>();
        test_clear::<SmallLruCache<i32, i32, 8>>();
    }

    #[test]
    fn test_clone_all_backends() {
        fn test_clone<S: AnyLruCache<i32, i32> + Default + Clone>() {
            let mut cache = S::default();
            cache.put(1, 10);
            let mut cloned = cache.clone();
            assert_eq!(cloned.len(), 1);
            cloned.put(2, 20);
            assert_eq!(cache.len(), 1);
        }

        test_clone::<HeaplessLruCache<i32, i32, 8>>();
        test_clone::<HeaplessBTreeLruCache<i32, i32, 8>>();
        test_clone::<HeaplessLinearLruCache<i32, i32, 8>>();
        test_clone::<SmallLruCache<i32, i32, 8>>();
    }

    #[test]
    fn test_itermut_all_backends() {
        fn test_itermut<
            S: AnyLruCache<i32, i32> + Default + for<'a> LruIteratorSupport<'a, i32, i32>,
        >() {
            let mut cache = S::default();
            cache.put(1, 10);
            cache.put(2, 20);
            for (_, v) in cache.iter_mut() {
                *v += 1;
            }
            assert_eq!(cache.get(&1), Some(&11));
            assert_eq!(cache.get(&2), Some(&21));
        }

        test_itermut::<HeaplessLruCache<i32, i32, 8>>();
        test_itermut::<HeaplessBTreeLruCache<i32, i32, 8>>();
        test_itermut::<HeaplessLinearLruCache<i32, i32, 8>>();
        test_itermut::<SmallLruCache<i32, i32, 8>>();
    }

    #[test]
    fn test_small_lru_heap_mode_coverage() {
        // Force heap mode by resizing to something larger than N (8 in this case)
        let mut cache: SmallLruCache<i32, i32, 2> =
            SmallLruCache::new(NonZeroUsize::new(5).unwrap());
        // Since N=2 and cap=5, it might start on stack if it can, but let's just push 3 items.
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30); // Should trigger spill as N=2
        assert!(!cache.is_on_stack());

        // Use a generic function to test trait implementation in heap mode
        fn check_heap<S: AnyLruCache<i32, i32>>(any_cache: &mut S) {
            any_cache.get(&1);
        }
        check_heap(&mut cache);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get_mut(&1), Some(&mut 10));
        assert_eq!(cache.peek(&1), Some(&10));
        assert!(cache.contains(&1));

        cache.promote(&3);
        cache.demote(&3);
        assert_eq!(cache.peek_lru(), Some((&3, &30)));

        cache.push(4, 40);
        cache.pop_lru();
        cache.pop(&1);
        cache.pop_entry(&2);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_cache_backend_direct_coverage() {
        // Test the LruCache (from lru crate) delegation directly
        let mut cache = LruCache::new(NonZeroUsize::new(10).unwrap());

        fn check_lru<S: AnyLruCache<i32, i32>>(any: &mut S) {
            any.put(1, 10);
            let _ = any.put_with_cap(2, 20, NonZeroUsize::new(5).unwrap());
            any.get(&1);
            any.get_mut(&1);
            any.peek(&1);
            any.contains(&1);
            any.promote(&1);
            any.demote(&1);
            any.peek_lru();
            any.pop_lru();
            any.push(3, 30);
            any.pop(&3);
            any.pop_entry(&2);
            any.clear();
        }

        check_lru(&mut cache);
    }
    #[test]
    fn test_lru_cache_fuzz_operations() {
        // Simple LCG random number generator
        struct SimpleRng(u64);
        impl SimpleRng {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                self.0
            }
            fn gen_range(&mut self, min: i32, max: i32) -> i32 {
                let range = (max - min) as u64;
                if range == 0 {
                    return min;
                }
                min + (self.next() % range) as i32
            }
        }

        fn stress<
            S: AnyLruCache<i32, i32>
                + Default
                + Clone
                + for<'a> LruIteratorSupport<'a, i32, i32>
                + Debug
                + IntoIterator<Item = (i32, i32)>
                + Extend<(i32, i32)>,
        >(
            rng: &mut SimpleRng,
        ) {
            let mut cache = S::default();
            let cap = cache.cap().get();

            for _ in 0..500 {
                let op = rng.gen_range(0, 13);
                let key = rng.gen_range(0, cap as i32 * 2);
                let val = rng.gen_range(0, 100);

                match op {
                    0 => {
                        let _ = cache.put(key, val);
                    }
                    1 => {
                        let _ = cache.get(&key);
                    }
                    2 => {
                        let _ = cache.get_mut(&key);
                    }
                    3 => {
                        let _ = cache.peek(&key);
                    }
                    4 => {
                        let _ = cache.contains(&key);
                    }
                    5 => {
                        let _ = cache.pop(&key);
                    }
                    6 => {
                        cache.promote(&key);
                    }
                    7 => {
                        cache.demote(&key);
                    }
                    8 => {
                        let _ = cache.clone();
                    }
                    9 => {
                        let _ = cache.iter().count();
                        let _ = cache.iter_mut().count();
                        let _ = format!("{:?}", cache);
                    }
                    10 => {
                        let mut c = cache.clone();
                        let _ = c.pop_entry(&key);
                        let _ = c.pop_lru();
                        let _ = c.peek_lru();
                        c.clear();
                    }
                    11 => {
                        let c = cache.clone();
                        let _ = c.into_iter().count();
                    }
                    12 => {
                        let mut c = cache.clone();
                        c.extend(vec![(key, val), (key + 1, val + 1)]);
                    }
                    _ => {}
                }
            }
        }

        let mut rng = SimpleRng(42);
        stress::<HeaplessLruCache<i32, i32, 16>>(&mut rng);
        stress::<HeaplessBTreeLruCache<i32, i32, 16>>(&mut rng);
        stress::<HeaplessLinearLruCache<i32, i32, 16>>(&mut rng);
        stress::<SmallLruCache<i32, i32, 4>>(&mut rng);
    }

    #[test]
    fn test_lru_cache_get_or_insert_variants() {
        let mut cache: SmallLruCache<i32, i32, 2> = SmallLruCache::default();

        // Case 1: Insert new
        let val = cache.get_or_insert(1, || 10);
        assert_eq!(*val, 10);

        // Case 2: Get existing
        let val = cache.get_or_insert(1, || 20);
        assert_eq!(*val, 10);

        // Case 3: Mut insert new
        let val = cache.get_or_insert_mut(2, || 20);
        assert_eq!(*val, 20);

        // Case 4: Mut get existing
        let val = cache.get_or_insert_mut(2, || 30);
        assert_eq!(*val, 20);
    }

    #[test]
    fn test_lru_cache_debug_impls_and_raw_heap_ops() {
        let mut cache: SmallLruCache<i32, i32, 2> = SmallLruCache::default();
        let _ = cache.capacity();
        cache.put(1, 10);
        cache.put(1, 11); // old.is_some() path in put

        // LruCache direct iters
        let mut lru = lru::LruCache::new(NonZeroUsize::new(5).unwrap());
        lru.put(1, 10);
        let _ = lru.iter().count();
        let _ = lru.iter_mut().count();
        let _ = lru.cap();
    }
}
