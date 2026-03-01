#![cfg(feature = "lru")]
//! Stack-allocated LRU cache using linear search for $O(N)$ lookups.

use core::mem::MaybeUninit;
use core::num::NonZeroUsize;
use core::ptr;
use std::borrow::Borrow;
use std::hash::Hash;

use crate::AnyLruCache;
use crate::IndexType;

/// A **stack-allocated LRU cache** using linear search for $O(N)$ lookups.
///
/// # Architecture & Pseudocode
/// This cache is highly optimized for tiny workloads ($N < 32$) where the overhead
/// of hashing or binary searches outweighs a simple unstructured linear scan.
/// It uses a **Struct-of-Arrays (SoA)** layout combined with an intrusive
/// doubly-linked list and a free list.
///
/// - `keys`, `values`: Storage slots for the actual data (`MaybeUninit`).
/// - `prevs`, `nexts`: Parallel arrays representing the doubly-linked list of LRU order.
///   The `nexts` array also doubles as the singly-linked free list for available slots.
/// - `head`, `tail`: Pointers to the Most Recently Used (MRU) and Least Recently Used (LRU) slots.
/// - `free_head`: Pointer to the first available empty slot.
///
/// ## Put Algorithm
/// ```text
/// 1. Linearly scan the connected LRU list (following `nexts` from `head`) for the key.
/// 2. If found at `idx`:
///    a. Update `values[idx]`.
///    b. Promote `idx` to `head` (MRU).
/// 3. Else (new key):
///    a. If cache is logically full (`len >= cap`), call `pop_lru_internal()`:
///       i.  Read `idx = tail`.
///       ii. Detach `idx` from LRU list, push to `free_head`.
///    b. If `len == N` (absolute stack capacity), return Err (triggers heap spill in wrapper).
///    c. Pop `idx` from `free_head`.
///    d. Write `key` to `keys[idx]` and `value` to `values[idx]`.
///    e. Attach `idx` to `head` (MRU).
/// ```
pub struct HeaplessLinearLruCache<K, V, const N: usize, I: IndexType = u8> {
    /// Automatically generated documentation for this item.
    pub keys: [MaybeUninit<K>; N],
    /// Automatically generated documentation for this item.
    pub values: [MaybeUninit<V>; N],
    /// Automatically generated documentation for this item.
    pub prevs: [I; N],
    /// Automatically generated documentation for this item.
    pub nexts: [I; N],
    /// Automatically generated documentation for this item.
    pub head: I,
    /// Automatically generated documentation for this item.
    pub tail: I,
    /// Automatically generated documentation for this item.
    pub num_entries: I,
    /// Automatically generated documentation for this item.
    pub free_head: I,
}

impl<K, V, const N: usize, I: IndexType> HeaplessLinearLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
{
    /// Automatically generated documentation for this item.
    pub fn new() -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= 16 * 1024,
                "HeaplessLinearLruCache is too large! The struct size exceeds the 16KB limit. Reduce N."
            );
        }
        let prevs = [I::NONE; N];
        let mut nexts = [I::NONE; N];
        let mut idx = I::ZERO;
        for slot in nexts.iter_mut() {
            idx = idx.inc();
            *slot = idx;
        }
        if N > 0 {
            nexts[N - 1] = I::NONE;
        }

        Self {
            keys: unsafe { MaybeUninit::uninit().assume_init() },
            values: unsafe { MaybeUninit::uninit().assume_init() },
            prevs,
            nexts,
            head: I::NONE,
            tail: I::NONE,
            num_entries: I::ZERO,
            free_head: if N > 0 { I::ZERO } else { I::NONE },
        }
    }

    /// Returns the number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.num_entries.as_usize()
    }

    /// Returns `true` if the collection is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.num_entries.is_zero()
    }

    /// Returns a reference to the value corresponding to the key.
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(idx) = self.find_index(key) {
            self.promote_idx(idx);
            Some(unsafe { &*self.values[idx.as_usize()].as_ptr() })
        } else {
            None
        }
    }

    /// Returns a reference to the value corresponding to the key.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(idx) = self.find_index(key) {
            self.promote_idx(idx);
            Some(unsafe { &mut *self.values[idx.as_usize()].as_mut_ptr() })
        } else {
            None
        }
    }

    /// Returns a reference to the next item.
    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(idx) = self.find_index(key) {
            Some(unsafe { &*self.values[idx.as_usize()].as_ptr() })
        } else {
            None
        }
    }

    /// Returns a reference to the next item.
    pub fn peek_lru(&self) -> Option<(&K, &V)> {
        if self.tail != I::NONE {
            let idx = self.tail.as_usize();
            unsafe { Some((&*self.keys[idx].as_ptr(), &*self.values[idx].as_ptr())) }
        } else {
            None
        }
    }

    /// Returns an iterator over the elements.
    pub fn iter(&self) -> Iter<'_, K, V, N, I> {
        Iter {
            cache: self,
            curr: self.head,
            remaining: self.num_entries.as_usize(),
        }
    }

    /// Returns an iterator over the elements.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, N, I> {
        IterMut {
            curr: self.head,
            remaining: self.num_entries.as_usize(),
            keys: &self.keys as *const [MaybeUninit<K>; N],
            values: &mut self.values as *mut [MaybeUninit<V>; N],
            nexts: &self.nexts as *const [I; N],
            _marker: core::marker::PhantomData,
        }
    }

    /// Pushes a key-value pair, updating the LRU list.
    pub fn put(&mut self, key: K, value: V, cap: usize) -> (Option<V>, Result<(), (K, V)>) {
        if let Some(idx) = self.find_index(key.borrow()) {
            let old = unsafe { ptr::replace(self.values[idx.as_usize()].as_mut_ptr(), value) };
            self.promote_idx(idx);
            (Some(old), Ok(()))
        } else {
            if self.len() >= cap {
                self.pop_lru_internal();
            }

            if self.len() >= N {
                return (None, Err((key, value)));
            }

            let idx = self.free_head;
            self.free_head = self.nexts[idx.as_usize()];
            unsafe {
                ptr::write(self.keys[idx.as_usize()].as_mut_ptr(), key);
                ptr::write(self.values[idx.as_usize()].as_mut_ptr(), value);
            }
            self.attach_front(idx);
            self.num_entries = self.num_entries.inc();
            (None, Ok(()))
        }
    }

    fn find_index<Q>(&self, key: &Q) -> Option<I>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut curr = self.head;
        while curr != I::NONE {
            let idx = curr.as_usize();
            let k = unsafe { &*self.keys[idx].as_ptr() };
            if k.borrow() == key {
                return Some(curr);
            }
            curr = self.nexts[idx];
        }
        None
    }

    fn promote_idx(&mut self, idx: I) {
        if idx != self.head {
            self.detach(idx);
            self.attach_front(idx);
        }
    }

    fn detach(&mut self, idx: I) {
        let (p, n) = (self.prevs[idx.as_usize()], self.nexts[idx.as_usize()]);
        if p != I::NONE {
            self.nexts[p.as_usize()] = n;
        } else {
            self.head = n;
        }
        if n != I::NONE {
            self.prevs[n.as_usize()] = p;
        } else {
            self.tail = p;
        }
    }

    fn attach_front(&mut self, idx: I) {
        self.prevs[idx.as_usize()] = I::NONE;
        self.nexts[idx.as_usize()] = self.head;
        if self.head != I::NONE {
            self.prevs[self.head.as_usize()] = idx;
        } else {
            self.tail = idx;
        }
        self.head = idx;
    }

    fn attach_back(&mut self, idx: I) {
        self.nexts[idx.as_usize()] = I::NONE;
        self.prevs[idx.as_usize()] = self.tail;
        if self.tail != I::NONE {
            self.nexts[self.tail.as_usize()] = idx;
        } else {
            self.head = idx;
        }
        self.tail = idx;
    }

    fn pop_lru_internal(&mut self) -> Option<(K, V)> {
        if self.tail != I::NONE {
            let idx = self.tail;
            let key = unsafe { ptr::read(self.keys[idx.as_usize()].as_ptr()) };
            let val = unsafe { ptr::read(self.values[idx.as_usize()].as_ptr()) };
            self.detach(idx);
            self.nexts[idx.as_usize()] = self.free_head;
            self.free_head = idx;
            self.num_entries = self.num_entries.dec();
            Some((key, val))
        } else {
            None
        }
    }

    fn pop_mru_internal(&mut self) -> Option<(K, V)> {
        if self.head != I::NONE {
            let idx = self.head;
            let key = unsafe { ptr::read(self.keys[idx.as_usize()].as_ptr()) };
            let val = unsafe { ptr::read(self.values[idx.as_usize()].as_ptr()) };
            self.detach(idx);
            self.nexts[idx.as_usize()] = self.free_head;
            self.free_head = idx;
            self.num_entries = self.num_entries.dec();
            Some((key, val))
        } else {
            None
        }
    }
}

impl<K: Hash + Eq + Clone, V, const N: usize, I: IndexType> Default
    for HeaplessLinearLruCache<K, V, N, I>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, const N: usize, I: IndexType> AnyLruCache<K, V> for HeaplessLinearLruCache<K, V, N, I>
where
    K: Hash + Eq + Ord + Clone,
{
    fn len(&self) -> usize {
        self.num_entries.as_usize()
    }
    fn cap(&self) -> NonZeroUsize {
        NonZeroUsize::new(N.max(1)).unwrap()
    }
    fn put(&mut self, key: K, value: V) -> Option<V> {
        self.put(key, value, N).0
    }
    fn put_with_cap(
        &mut self,
        key: K,
        value: V,
        cap: NonZeroUsize,
    ) -> (Option<V>, Result<(), (K, V)>) {
        self.put(key, value, cap.get())
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
        let mut curr = self.head;
        while curr != I::NONE {
            let idx = curr.as_usize();
            let next = self.nexts[idx];
            unsafe {
                ptr::drop_in_place(self.keys[idx].as_mut_ptr());
                ptr::drop_in_place(self.values[idx].as_mut_ptr());
            }
            curr = next;
        }
        self.head = I::NONE;
        self.tail = I::NONE;
        self.num_entries = I::ZERO;
        self.free_head = if N > 0 { I::ZERO } else { I::NONE };
        if N > 0 {
            let mut idx = I::ZERO;
            for slot in self.nexts.iter_mut() {
                idx = idx.inc();
                *slot = idx;
            }
            self.nexts[N - 1] = I::NONE;
        }
    }
    fn pop_lru(&mut self) -> Option<(K, V)> {
        self.pop_lru_internal()
    }
    fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.find_index(key).is_some()
    }
    fn push(&mut self, key: K, value: V) -> Option<(K, V)> {
        match self.put(key, value, N) {
            (None, Err(item)) => Some(item),
            _ => None,
        }
    }
    fn pop<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if let Some(idx) = self.find_index(key) {
            unsafe {
                ptr::drop_in_place(self.keys[idx.as_usize()].as_mut_ptr());
                let val = ptr::read(self.values[idx.as_usize()].as_ptr());
                self.detach(idx);
                self.nexts[idx.as_usize()] = self.free_head;
                self.free_head = idx;
                self.num_entries = self.num_entries.dec();
                Some(val)
            }
        } else {
            None
        }
    }
    fn pop_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if let Some(idx) = self.find_index(key) {
            unsafe {
                let key = ptr::read(self.keys[idx.as_usize()].as_ptr());
                let val = ptr::read(self.values[idx.as_usize()].as_ptr());
                self.detach(idx);
                self.nexts[idx.as_usize()] = self.free_head;
                self.free_head = idx;
                self.num_entries = self.num_entries.dec();
                Some((key, val))
            }
        } else {
            None
        }
    }
    fn promote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if let Some(idx) = self.find_index(key) {
            self.promote_idx(idx);
        }
    }
    fn demote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if let Some(idx) = self.find_index(key) {
            if idx != self.tail {
                self.detach(idx);
                self.attach_back(idx);
            }
        }
    }
    fn peek_lru(&self) -> Option<(&K, &V)> {
        self.peek_lru()
    }
}

impl<'a, K, V, const N: usize, I: IndexType> crate::cache::lru_cache::LruIteratorSupport<'a, K, V>
    for HeaplessLinearLruCache<K, V, N, I>
where
    K: Hash + Eq + Ord + Clone + 'a,
    V: 'a,
{
    type Iter = Iter<'a, K, V, N, I>;
    type IterMut = IterMut<'a, K, V, N, I>;
    fn iter(&'a self) -> Self::Iter {
        self.iter()
    }
    fn iter_mut(&'a mut self) -> Self::IterMut {
        self.iter_mut()
    }
}

/// A structure representing `IntoIter`.
pub struct IntoIter<K, V, const N: usize, I: IndexType> {
    cache: HeaplessLinearLruCache<K, V, N, I>,
}

/// A structure representing `Iter`.
#[derive(Debug)]
pub struct Iter<'a, K: 'a, V: 'a, const N: usize, I: IndexType> {
    cache: &'a HeaplessLinearLruCache<K, V, N, I>,
    curr: I,
    remaining: usize,
}

impl<'a, K, V, const N: usize, I: IndexType> Iterator for Iter<'a, K, V, N, I> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == I::NONE {
            return None;
        }
        let idx = self.curr.as_usize();
        let (key, val) = unsafe {
            (
                &*self.cache.keys[idx].as_ptr(),
                &*self.cache.values[idx].as_ptr(),
            )
        };
        self.curr = self.cache.nexts[idx];
        self.remaining -= 1;
        Some((key, val))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K, V, const N: usize, I: IndexType> ExactSizeIterator for Iter<'a, K, V, N, I> {}

/// A structure representing `IterMut`.
#[derive(Debug)]
pub struct IterMut<'a, K: 'a, V: 'a, const N: usize, I: IndexType> {
    curr: I,
    remaining: usize,
    keys: *const [MaybeUninit<K>; N],
    values: *mut [MaybeUninit<V>; N],
    nexts: *const [I; N],
    _marker: core::marker::PhantomData<&'a mut V>,
}

impl<'a, K, V, const N: usize, I: IndexType> Iterator for IterMut<'a, K, V, N, I> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == I::NONE {
            return None;
        }
        let idx = self.curr.as_usize();
        unsafe {
            let key = &*(*self.keys)[idx].as_ptr();
            let val = &mut *(*self.values)[idx].as_mut_ptr();
            self.curr = (*self.nexts)[idx];
            self.remaining -= 1;
            Some((key, val))
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K, V, const N: usize, I: IndexType> ExactSizeIterator for IterMut<'a, K, V, N, I> {}

impl<K, V, const N: usize, I: IndexType> Iterator for IntoIter<K, V, N, I>
where
    K: Hash + Eq + Ord + Clone,
{
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        self.cache.pop_mru_internal()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.cache.len();
        (len, Some(len))
    }
}

impl<K, V, const N: usize, I: IndexType> ExactSizeIterator for IntoIter<K, V, N, I> where
    K: Hash + Eq + Ord + Clone
{
}

impl<K, V, const N: usize, I: IndexType> IntoIterator for HeaplessLinearLruCache<K, V, N, I>
where
    K: Hash + Eq + Ord + Clone,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, N, I>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { cache: self }
    }
}

impl<'a, K, V, const N: usize, I: IndexType> IntoIterator for &'a HeaplessLinearLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, N, I>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, const N: usize, I: IndexType> IntoIterator
    for &'a mut HeaplessLinearLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, N, I>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, const N: usize, I> FromIterator<(K, V)> for HeaplessLinearLruCache<K, V, N, I>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut cache = Self::default();
        for (k, v) in iter {
            let _ = cache.put(k, v, N);
        }
        cache
    }
}

impl<K, V, const N: usize, I> Extend<(K, V)> for HeaplessLinearLruCache<K, V, N, I>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            let _ = self.put(k, v, N);
        }
    }
}

impl<K, V, const N: usize, I: IndexType> Clone for HeaplessLinearLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut new_keys: [MaybeUninit<K>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        let mut new_vals: [MaybeUninit<V>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        let mut curr = self.head;
        while curr != I::NONE {
            let idx = curr.as_usize();
            let k = unsafe { &*self.keys[idx].as_ptr() };
            let v = unsafe { &*self.values[idx].as_ptr() };
            unsafe {
                ptr::write(new_keys[idx].as_mut_ptr(), k.clone());
                ptr::write(new_vals[idx].as_mut_ptr(), v.clone());
            }
            curr = self.nexts[idx];
        }
        Self {
            keys: new_keys,
            values: new_vals,
            prevs: self.prevs,
            nexts: self.nexts,
            head: self.head,
            tail: self.tail,
            num_entries: self.num_entries,
            free_head: self.free_head,
        }
    }
}

impl<K, V, const N: usize, I: IndexType> Drop for HeaplessLinearLruCache<K, V, N, I> {
    fn drop(&mut self) {
        let mut curr = self.head;
        while curr != I::NONE {
            let idx = curr.as_usize();
            let next = self.nexts[idx];
            unsafe {
                ptr::drop_in_place(self.keys[idx].as_mut_ptr());
                ptr::drop_in_place(self.values[idx].as_mut_ptr());
            }
            curr = next;
        }
    }
}

impl<K, V, const N: usize, I: IndexType> std::fmt::Debug for HeaplessLinearLruCache<K, V, N, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeaplessLinearLruCache")
            .field("num_entries", &self.num_entries)
            .field("head", &self.head)
            .field("tail", &self.tail)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_lookup_and_eviction() {
        let mut cache: HeaplessLinearLruCache<i32, i32, 2> = HeaplessLinearLruCache::new();
        let _ = cache.put(1, 10, 2);
        let _ = cache.put(2, 20, 2);

        assert_eq!(cache.get(&1), Some(&10)); // 1 is MRU
        let _ = cache.put(3, 30, 2); // Should evict 2

        assert!(cache.contains(&1));
        assert!(cache.contains(&3));
        assert!(!cache.contains(&2));
    }

    #[test]
    fn test_empty_ops() {
        let mut cache: HeaplessLinearLruCache<i32, i32, 2> = HeaplessLinearLruCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.pop_lru(), None);
    }
}
