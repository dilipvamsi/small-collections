#![cfg(feature = "lru")]
//! Stack-allocated LRU cache using an IndexMap for $O(1)$ lookups.

use core::mem::MaybeUninit;
use core::num::NonZeroUsize;
use core::ptr;
use std::borrow::Borrow;
use std::hash::Hash;

use crate::AnyLruCache;
use crate::IndexType;
use heapless::IndexMap;
use heapless::index_map::FnvIndexMap;

/// A **stack-allocated LRU cache** with $O(1)$ performance.
///
/// # Architecture & Pseudocode
/// This cache uses a **Struct-of-Arrays (SoA)** layout combined with an intrusive
/// doubly-linked list and a free list, all residing entirely on the stack.
///
/// - `map`: A `heapless::IndexMap<K, I, N>` mapping keys to their physical slot indices.
/// - `keys`, `values`: Storage slots for the actual data (`MaybeUninit` to avoid generic `Default` bounds).
/// - `prevs`, `nexts`: Parallel arrays representing the doubly-linked list of LRU order.
///   The `nexts` array also doubles as the singly-linked free list for available slots.
/// - `head`, `tail`: Pointers to the Most Recently Used (MRU) and Least Recently Used (LRU) slots.
/// - `free_head`: Pointer to the first available empty slot.
///
/// ## Put Algorithm
/// ```text
/// 1. If key exists in `map` at `idx`:
///    a. Update `values[idx]`.
///    b. Promote `idx` to `head` (MRU).
/// 2. Else (new key):
///    a. If cache is logically full (`len >= cap`), call `pop_lru_internal()`:
///       i.  Read `idx = tail`.
///       ii. Remove `keys[idx]` from `map`.
///       iii. Detach `idx` from LRU list, push to `free_head`.
///    b. If `len == N` (absolute stack capacity), return Err (triggers heap spill in wrapper).
///    c. Pop `idx` from `free_head`.
///    d. Write `key` to `keys[idx]` and `value` to `values[idx]`.
///    e. Insert `key -> idx` into `map`.
///    f. Attach `idx` to `head` (MRU).
/// ```
pub struct HeaplessLruCache<K, V, const N: usize, I: IndexType = u8> {
    pub map: FnvIndexMap<K, I, N>,
    pub keys: [MaybeUninit<K>; N],
    pub values: [MaybeUninit<V>; N],
    pub prevs: [I; N],
    pub nexts: [I; N],
    pub free_head: I,
    pub head: I,
    pub tail: I,
    pub num_entries: I,
}

impl<K, V, const N: usize, I: IndexType> HeaplessLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
{
    pub fn new() -> Self {
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
            map: IndexMap::new(),
            keys: unsafe { MaybeUninit::uninit().assume_init() },
            values: unsafe { MaybeUninit::uninit().assume_init() },
            prevs,
            nexts,
            free_head: if N > 0 { I::ZERO } else { I::NONE },
            head: I::NONE,
            tail: I::NONE,
            num_entries: I::ZERO,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.num_entries.as_usize()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.num_entries.is_zero()
    }

    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(&idx) = self.map.get(key) {
            self.promote_idx(idx);
            Some(unsafe { &*self.values[idx.as_usize()].as_ptr() })
        } else {
            None
        }
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(&idx) = self.map.get(key) {
            self.promote_idx(idx);
            Some(unsafe { &mut *self.values[idx.as_usize()].as_mut_ptr() })
        } else {
            None
        }
    }

    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(&idx) = self.map.get(key) {
            Some(unsafe { &*self.values[idx.as_usize()].as_ptr() })
        } else {
            None
        }
    }

    pub fn peek_lru(&self) -> Option<(&K, &V)> {
        if self.tail != I::NONE {
            let idx = self.tail.as_usize();
            unsafe { Some((&*self.keys[idx].as_ptr(), &*self.values[idx].as_ptr())) }
        } else {
            None
        }
    }

    pub fn iter(&self) -> Iter<'_, K, V, N, I> {
        Iter {
            cache: self,
            curr: self.head,
            remaining: self.num_entries.as_usize(),
        }
    }

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

    pub fn put(&mut self, key: K, value: V, cap: usize) -> (Option<V>, Result<(), (K, V)>) {
        if let Some(&idx) = self.map.get(key.borrow()) {
            let old = unsafe { ptr::replace(self.values[idx.as_usize()].as_mut_ptr(), value) };
            self.promote_idx(idx);
            return (Some(old), Ok(()));
        }

        if self.len() >= cap {
            self.pop_lru_internal();
        }

        if self.len() >= N {
            return (None, Err((key, value)));
        }

        let idx = self.free_head;
        self.free_head = self.nexts[idx.as_usize()];
        self.map.insert(key.clone(), idx).ok();
        unsafe {
            ptr::write(self.keys[idx.as_usize()].as_mut_ptr(), key);
            ptr::write(self.values[idx.as_usize()].as_mut_ptr(), value);
        }
        self.attach_front(idx);
        self.num_entries = self.num_entries.inc();
        (None, Ok(()))
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
            self.map.remove(key.borrow());
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
            self.map.remove(key.borrow());
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
    for HeaplessLruCache<K, V, N, I>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, const N: usize, I: IndexType> AnyLruCache<K, V> for HeaplessLruCache<K, V, N, I>
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
        self.map.clear();
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
        self.map.contains_key(key)
    }
    fn push(&mut self, key: K, value: V) -> Option<(K, V)> {
        match self.put(key, value, N) {
            (Some(_v), Ok(())) => None, // Not really useful for push which wants (K, V)
            (None, Ok(())) => None,
            (None, Err(item)) => Some(item),
            _ => None,
        }
    }
    fn pop<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if let Some(idx) = self.map.remove(key) {
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
        if let Some(idx) = self.map.remove(key) {
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
        if let Some(&idx) = self.map.get(key) {
            self.promote_idx(idx);
        }
    }
    fn demote<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        if let Some(&idx) = self.map.get(key) {
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
    for HeaplessLruCache<K, V, N, I>
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

#[derive(Debug)]
pub struct Iter<'a, K: 'a, V: 'a, const N: usize, I: IndexType> {
    cache: &'a HeaplessLruCache<K, V, N, I>,
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

pub struct IntoIter<K, V, const N: usize, I: IndexType> {
    cache: HeaplessLruCache<K, V, N, I>,
}

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

impl<K, V, const N: usize, I: IndexType> IntoIterator for HeaplessLruCache<K, V, N, I>
where
    K: Hash + Eq + Ord + Clone,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, N, I>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { cache: self }
    }
}

impl<'a, K, V, const N: usize, I: IndexType> IntoIterator for &'a HeaplessLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, N, I>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, const N: usize, I: IndexType> IntoIterator for &'a mut HeaplessLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, N, I>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, const N: usize, I> FromIterator<(K, V)> for HeaplessLruCache<K, V, N, I>
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

impl<K, V, const N: usize, I> Extend<(K, V)> for HeaplessLruCache<K, V, N, I>
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

impl<K, V, const N: usize, I: IndexType> Clone for HeaplessLruCache<K, V, N, I>
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
            map: self.map.clone(),
            keys: new_keys,
            values: new_vals,
            prevs: self.prevs,
            nexts: self.nexts,
            free_head: self.free_head,
            head: self.head,
            tail: self.tail,
            num_entries: self.num_entries,
        }
    }
}

impl<K, V, const N: usize, I: IndexType> Drop for HeaplessLruCache<K, V, N, I> {
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

impl<K, V, const N: usize, I: IndexType> std::fmt::Debug for HeaplessLruCache<K, V, N, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeaplessLruCache")
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
    fn test_capacity_eviction() {
        let mut cache: HeaplessLruCache<i32, i32, 4> = HeaplessLruCache::new();
        let _ = cache.put(1, 10, 4);
        let _ = cache.put(2, 20, 4);
        let _ = cache.put(3, 30, 4);
        let _ = cache.put(4, 40, 4);
        assert_eq!(cache.len(), 4);

        // This should evict 1
        let _ = cache.put(5, 50, 4);
        assert_eq!(cache.len(), 4);
        assert!(!cache.map.contains_key(&1));
        assert_eq!(cache.get(&2), Some(&20));

        // 2 is now MRU. Evict next one (3).
        let _ = cache.put(6, 60, 4);
        assert!(!cache.map.contains_key(&3));
    }

    #[test]
    fn test_promotion_demote() {
        let mut cache: HeaplessLruCache<i32, i32, 4> = HeaplessLruCache::new();
        let _ = cache.put(1, 10, 4);
        let _ = cache.put(2, 20, 4);
        let _ = cache.put(3, 30, 4);

        // 3(MRU), 2, 1(LRU)
        cache.demote(&3);
        // 2(MRU), 1, 3(LRU)
        assert_eq!(cache.peek_lru(), Some((&3, &30)));

        cache.promote(&1);
        // 1(MRU), 2, 3(LRU)
        let keys: Vec<_> = cache.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 2, 3]);
    }

    #[test]
    fn test_clone_independence() {
        let mut cache: HeaplessLruCache<i32, i32, 4> = HeaplessLruCache::new();
        let _ = cache.put(1, 10, 4);
        let _ = cache.put(2, 20, 4);

        let mut cloned = cache.clone();
        let _ = cloned.put(3, 30, 4);

        assert_eq!(cloned.len(), 3);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut cache: HeaplessLruCache<i32, i32, 4> = HeaplessLruCache::new();
        let _ = cache.put(1, 10, 4);
        let _ = cache.put(2, 20, 4);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.free_head.as_usize(), 0);

        // Can still use it
        let _ = cache.put(1, 10, 4);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_from_iter_and_extend() {
        let mut cache = HeaplessLruCache::<i32, i32, 4>::from_iter(vec![(1, 10), (2, 20)]);
        cache.extend(vec![(3, 30), (4, 40)]);
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.get(&1), Some(&10));
    }
}
