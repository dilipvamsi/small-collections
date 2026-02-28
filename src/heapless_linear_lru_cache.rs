#![cfg(feature = "lru")]
//! Stack-allocated LRU cache using linear scanning for $O(N)$ lookups.
//!
//! # Implementation details
//! - **Linear Scanning**: simple and efficient for very small capacities ($N < 16$).
//! - **Doubly-linked list**: `prevs` and `nexts` maintain the LRU order.
//! - **Struct-of-Arrays (SoA)**: separate arrays for `keys` and `values`.

use core::mem::MaybeUninit;
use core::num::NonZeroUsize;
use core::ptr;
use std::borrow::Borrow;
use std::hash::Hash;

use crate::index_type::IndexType;
use crate::lru_cache::AnyLruCache;

/// A **stack-allocated LRU cache** using linear scanning.
pub struct HeaplessLinearLruCache<K, V, const N: usize, I: IndexType = u8> {
    pub keys: [MaybeUninit<K>; N],
    pub values: [MaybeUninit<V>; N],
    pub prevs: [I; N],
    pub nexts: [I; N],
    pub head: I,
    pub tail: I,
    pub num_entries: I,
    pub free_head: I,
}

impl<K, V, const N: usize, I: IndexType> HeaplessLinearLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
{
    /// Creates a new, empty `HeaplessLinearLruCache`.
    ///
    /// # Pseudo Code:
    /// ```text
    /// initialize prevs with NONE
    /// initialize nexts as free-list (0->1->...->N-1->NONE)
    /// return Self { ..., free_head: 0, head: NONE, tail: NONE, num_entries: 0 }
    /// ```
    pub fn new() -> Self {
        let prevs = [I::NONE; N];
        let mut nexts = [I::NONE; N];
        // Initialize free-list: 0 -> 1 -> ... -> N-1 -> NONE
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

    /// Returns the number of elements in the cache.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.num_entries.as_usize()
    }

    /// Returns `true` if the cache contains no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.num_entries.is_zero()
    }

    /// Returns a reference to the value associated with the key and promotes it to MRU.
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if let Some(idx) = self.find_index(key) {
            self.promote(idx);
            Some(unsafe { &*self.values[idx.as_usize()].as_ptr() })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the value associated with the key and promotes it to MRU.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if let Some(idx) = self.find_index(key) {
            self.promote(idx);
            Some(unsafe { &mut *self.values[idx.as_usize()].as_mut_ptr() })
        } else {
            None
        }
    }

    /// Returns a reference to the value associated with the key without promoting it.
    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if let Some(idx) = self.find_index(key) {
            Some(unsafe { &*self.values[idx.as_usize()].as_ptr() })
        } else {
            None
        }
    }

    /// Linearly scans the keys to find the index of a key.
    ///
    /// # Pseudo Code:
    /// ```text
    /// curr = head
    /// while curr != NONE:
    ///     if keys[curr] == key:
    ///         return Some(curr)
    ///     curr = nexts[curr]
    /// return None
    /// ```
    fn find_index<Q>(&self, key: &Q) -> Option<I>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
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

    /// Inserts a key-value pair.
    ///
    /// # Pseudo Code:
    /// ```text
    /// idx = find_index(key)
    /// if Found(idx):
    ///     replace value
    ///     promote(idx)
    ///     return (old_value, Ok)
    ///
    /// if num_entries >= cap:
    ///     idx_to_evict = tail
    ///     evict_key = keys[idx_to_evict]
    ///     detach(idx_to_evict)
    ///     return idx_to_evict to free list
    ///     num_entries -= 1
    ///
    /// if full (num_entries >= N):
    ///     return (None, Error)
    ///
    /// idx = free_head
    /// free_head = nexts[idx]
    /// write key and value to arrays[idx]
    /// attach_front(idx)
    /// num_entries += 1
    /// return (None, Ok)
    /// ```
    pub fn put(&mut self, key: K, value: V, cap: usize) -> (Option<V>, Result<(), (K, V)>) {
        if let Some(idx) = self.find_index(key.borrow()) {
            let old = unsafe { ptr::replace(self.values[idx.as_usize()].as_mut_ptr(), value) };
            self.promote(idx);
            return (Some(old), Ok(()));
        }

        if self.len() >= cap {
            let lru_idx = self.tail;
            if lru_idx != I::NONE {
                unsafe {
                    let old_key = ptr::read(self.keys[lru_idx.as_usize()].as_ptr());
                    let old_val = ptr::read(self.values[lru_idx.as_usize()].as_ptr());
                    self.detach(lru_idx);
                    self.nexts[lru_idx.as_usize()] = self.free_head;
                    self.free_head = lru_idx;
                    self.num_entries = self.num_entries.dec();
                    let _ = (old_key, old_val);
                }
            }
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

    fn promote(&mut self, idx: I) {
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
    /// Removes and returns the least recently used (LRU) item from the cache.
    fn pop_lru(&mut self) -> Option<(K, V)> {
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
