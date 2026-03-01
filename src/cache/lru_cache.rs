#![cfg(feature = "lru")]
//! LRU cache that lives on the stack and spills to the heap.
//!
//! Provides [`SmallLruCache`] — an LRU cache that uses a stack-allocated
//! backend for small workloads and spills to the [`lru`](https://docs.rs/lru)
//! crate's `LruCache` once the stack capacity is exceeded.
//!
//! [`AnyLruCache`] is an object-safe trait abstracting over both backends.

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
///
/// Implemented by `lru::LruCache<K, V>` (heap) and `SmallLruCache<K, V, N, I, S>` (small/stack)
/// so that callers can be backend-agnostic.
pub trait AnyLruCache<K, V> {
    /// Returns the number of entries currently cached.
    fn len(&self) -> usize;

    /// Returns `true` if the cache is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the logical capacity of the cache.
    fn cap(&self) -> NonZeroUsize;

    /// Inserts or updates `(key, value)`, evicting the LRU entry if at capacity.
    /// Returns the previous value if the key already existed.
    fn put(&mut self, key: K, value: V) -> Option<V>;

    /// Special put for heapless backends that might fail if physically full.
    ///
    /// Returns:
    /// - `Option<V>`: The old value if the key existed.
    /// - `Result<(), (K, V)>`: `Ok(())` if insertion succeeded, or `Err((K, V))` if the backend is physically full.
    fn put_with_cap(
        &mut self,
        key: K,
        value: V,
        cap: NonZeroUsize,
    ) -> (Option<V>, Result<(), (K, V)>);

    /// Returns a shared reference to the value for `key` and promotes it to MRU.
    fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;

    /// Returns an exclusive reference to the value for `key` and promotes it to MRU.
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;

    /// Returns a shared reference to the value for `key` **without** changing LRU order.
    fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;

    /// Removes all entries.
    fn clear(&mut self);

    /// Removes and returns the LRU entry.
    /// Used primarily for spilling data between backends.
    fn pop_lru(&mut self) -> Option<(K, V)>;
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
}

pub type DefaultHasher = std::collections::hash_map::RandomState;

/// Tagged union holding either the stack backend `S` or the heap `LruCache`.
///
/// # Safety
/// `SmallLruCache::on_stack` is the discriminant.
pub union LruData<K, V, S> {
    pub stack: ManuallyDrop<S>,
    pub heap: ManuallyDrop<LruCache<K, V>>,
}

/// An LRU cache that lives on the stack for up to `N` entries, then spills to
/// a heap-allocated `lru::LruCache`.
///
/// # Storage strategy
/// Uses a tagged union ([`LruData`]) with:
/// - **Stack side**: A generic storage type `S` (e.g., [`HeaplessBTreeLruCache`]).
/// - **Heap side**: `lru::LruCache<K, V>` — standard linked-hash-map-based LRU.
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
    /// Returns the number of entries currently in the cache.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.num_entries
    }

    /// Returns `true` if the cache is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }

    /// Returns the logical capacity of the cache.
    #[inline(always)]
    pub fn capacity(&self) -> NonZeroUsize {
        self.capacity
    }

    /// Returns a reference to the value associated with the key and promotes it to MRU.
    #[inline(always)]
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

    /// Returns a mutable reference to the value associated with the key and promotes it to MRU.
    #[inline(always)]
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

    /// Returns a reference to the value associated with the key without promoting it.
    #[inline(always)]
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

    /// Inserts a key-value pair into the cache.
    ///
    /// # Pseudo Code:
    /// ```text
    /// if on_stack:
    ///     try put into stack backend (with capacity check)
    ///     if success:
    ///         update num_entries and return
    ///     else (stack full/collision):
    ///         spill_to_heap()
    ///         fallthrough to heap logic
    ///
    /// put into heap backend
    /// update num_entries
    /// ```
    #[inline(always)]
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        if self.on_stack {
            let (old_v, res) =
                unsafe { (*self.data.stack).put_with_cap(key, value, self.capacity) };
            match res {
                Ok(()) => {
                    self.num_entries = unsafe { (*self.data.stack).len() };
                    return old_v;
                }
                Err((k, v)) => {
                    self.spill_to_heap();
                    let (res, _) = unsafe { (*self.data.heap).put_with_cap(k, v, self.capacity) };
                    self.num_entries = unsafe { (*self.data.heap).len() };
                    return res;
                }
            }
        }
        let (res, _) = unsafe { (*self.data.heap).put_with_cap(key, value, self.capacity) };
        self.num_entries = unsafe { (*self.data.heap).len() };
        res
    }

    /// Moves all data from the stack backend to the heap backend.
    ///
    /// # Pseudo Code:
    /// ```text
    /// create new heap LruCache
    /// take stack backend out of union
    /// while stack has entries (pop_lru):
    ///     insert entry into heap cache
    /// write heap cache back into union
    /// set on_stack = false
    /// ```
    #[inline(never)]
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

    /// Removes all entries from the cache.
    pub fn clear(&mut self) {
        if self.on_stack {
            unsafe { (*self.data.stack).clear() };
        } else {
            unsafe { (*self.data.heap).clear() };
        }
        self.num_entries = 0;
    }

    /// Returns `true` if the cache is currently using the stack-allocated backend.
    #[inline(always)]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }
}

/// Separate impl for methods requiring Default backend.
impl<K, V, const N: usize, I, S> SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + Default,
{
    /// Creates a new `SmallLruCache` with a specified capacity.
    ///
    /// It starts using the stack-allocated backend `S` and will
    /// automatically spill to the heap once the capacity is exceeded.
    pub fn new(cap: NonZeroUsize) -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= 16 * 1024,
                "SmallLruCache is too large! Reduce N."
            );
        }

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
        Self::put(self, key, value)
    }
    fn put_with_cap(
        &mut self,
        key: K,
        value: V,
        cap: NonZeroUsize,
    ) -> (Option<V>, Result<(), (K, V)>) {
        if self.on_stack {
            unsafe { (*self.data.stack).put_with_cap(key, value, cap) }
        } else {
            (
                unsafe { (*self.data.heap).put_with_cap(key, value, cap).0 },
                Ok(()),
            )
        }
    }
    fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        Self::get(self, key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        Self::get_mut(self, key)
    }
    fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        Self::peek(self, key)
    }
    fn clear(&mut self) {
        Self::clear(self);
    }
    fn pop_lru(&mut self) -> Option<(K, V)> {
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
    K: Hash + Eq + Debug + Ord + Clone,
    V: Debug,
    I: IndexType,
    S: AnyLruCache<K, V>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("SmallLruCache");
        d.field("on_stack", &self.on_stack);
        d.field("capacity", &self.capacity);
        d.field("num_entries", &self.num_entries);
        d.finish()
    }
}

impl<K, V, const N: usize, I, S> FromIterator<(K, V)> for SmallLruCache<K, V, N, I, S>
where
    K: Hash + Eq + Ord + Clone,
    I: IndexType,
    S: AnyLruCache<K, V> + Default,
{
    fn from_iter<II: IntoIterator<Item = (K, V)>>(iter: II) -> Self {
        let mut cache = Self::new(NonZeroUsize::new(N.max(1)).unwrap());
        for (k, v) in iter {
            cache.put(k, v);
        }
        cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HeaplessBTreeLruCache;
    use crate::HeaplessLinearLruCache;

    #[test]
    fn test_lru_sizes() {
        println!(
            "Size of SmallLruCache<i32, i32, 16>: {}",
            std::mem::size_of::<SmallLruCache<i32, i32, 16>>()
        );
    }

    #[test]
    fn test_lru_cache_stack_ops_basic() {
        let cap = NonZeroUsize::new(4).unwrap();
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        assert!(cache.is_on_stack());
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        cache.put(1, 10);
        cache.put(2, 20);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get(&2), Some(&20));
    }

    #[test]
    fn test_lru_cache_stack_ops_eviction_lru() {
        let cap = NonZeroUsize::new(2).unwrap();
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        cache.put(1, 10);
        cache.put(2, 20);
        cache.get(&1);
        cache.put(3, 30);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_cache_spill_trigger() {
        let cap = NonZeroUsize::new(8).unwrap();
        let mut cache: SmallLruCache<i32, i32, 2> = SmallLruCache::new(cap);
        cache.put(1, 10);
        cache.put(2, 20);
        assert!(cache.is_on_stack());
        cache.put(3, 30);
        assert!(!cache.is_on_stack());
        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get(&2), Some(&20));
        assert_eq!(cache.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_cache_different_backends() {
        let cap = NonZeroUsize::new(4).unwrap();

        let mut linear: SmallLruCache<i32, i32, 4, u8, HeaplessLinearLruCache<i32, i32, 4>> =
            SmallLruCache::new(cap);
        linear.put(1, 10);
        assert_eq!(linear.get(&1), Some(&10));

        let mut btree: SmallLruCache<i32, i32, 4, u8, HeaplessBTreeLruCache<i32, i32, 4>> =
            SmallLruCache::new(cap);
        btree.put(1, 10);
        assert_eq!(btree.get(&1), Some(&10));
    }
}
