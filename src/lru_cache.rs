#![cfg(feature = "lru")]
//! LRU cache that lives on the stack and spills to the heap.
//!
//! Provides [`SmallLruCache`] — an LRU cache that uses a stack-allocated
//! [`HeaplessLruCache`] for small workloads and spills to the [`lru`](https://docs.rs/lru)
//! crate's `LruCache` once the stack capacity is exceeded.
//!
//! [`AnyLruCache`] is an object-safe trait abstracting over both backends.

use core::mem::ManuallyDrop;
use core::num::NonZeroUsize;
use core::ptr;
use lru::LruCache;
use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::hash::Hash;

use crate::heapless_lru_cache::{HeaplessLruCache, IndexType};

/// An object-safe abstraction over LRU cache types.
///
/// Implemented by `lru::LruCache<K, V>` (heap) and `SmallLruCache<K, V, N, I>` (small/stack)
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
    /// Returns a shared reference to the value for `key` and promotes it to MRU.
    fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;
    /// Returns an exclusive reference to the value for `key` and promotes it to MRU.
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;
    /// Returns a shared reference to the value for `key` **without** changing LRU order.
    fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;
    /// Removes all entries.
    fn clear(&mut self);
}

impl<K: Hash + Eq, V> AnyLruCache<K, V> for LruCache<K, V> {
    fn len(&self) -> usize {
        self.len()
    }
    fn cap(&self) -> NonZeroUsize {
        self.cap()
    }
    fn put(&mut self, key: K, value: V) -> Option<V> {
        self.put(key, value)
    }
    fn get<Q>(&mut self, key: &Q) -> Option<&V>
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
    fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.peek(key)
    }
    fn clear(&mut self) {
        self.clear();
    }
}

pub type DefaultHasher = std::collections::hash_map::RandomState;

/// Tagged union holding either the stack [`HeaplessLruCache`] or the heap `LruCache`.
///
/// # Safety
/// `SmallLruCache::on_stack` is the discriminant. Only the active variant must be
/// read or written. The stack variant's `Drop` is managed manually via
/// `ManuallyDrop`; the heap variant's `LruCache` drop is invoked explicitly in
/// `SmallLruCache::drop`.
pub union LruData<K, V, const N: usize, I: IndexType> {
    pub stack: ManuallyDrop<HeaplessLruCache<K, V, N, I>>,
    pub heap: ManuallyDrop<LruCache<K, V>>,
}

/// An LRU cache that lives on the stack for up to `N` entries, then spills to
/// a heap-allocated `lru::LruCache`.
///
/// # Storage strategy
/// Uses a tagged union ([`LruData`]) with:
/// - **Stack side**: [`HeaplessLruCache<K, V, N, I>`] — SoA doubly-linked list + FNV map.
/// - **Heap side**: `lru::LruCache<K, V>` — standard linked-hash-map-based LRU.
///
/// # Generic parameters
/// | Parameter | Meaning |
/// |-----------|--------|
/// | `K` | Key type; must implement `Eq + Hash + Clone` |
/// | `V` | Value type |
/// | `N` | Stack capacity — max entries before spill |
/// | `I` | Index type; defaults to `u8` (max N=254); use `u16` for larger caches |
///
/// # Design Considerations
/// - **Capacity is user-defined**: the `cap` field stores the logical capacity
///   (as `NonZeroUsize`), which may be smaller than `N`.  Eviction happens when
///   `len >= cap`; a spill happens when the stack is also physically full (`len >= N`).
/// - **`I: IndexType`**: choosing a smaller index type (`u8`) saves 3 bytes per node
///   on the stack, which matters when `N` is large.
/// - **Stack→Heap spill is permanent**: after spill, all operations go to `LruCache`.
///   The cache never moves back to the stack.
pub struct SmallLruCache<K, V, const N: usize, I: IndexType = u8> {
    len: usize,
    capacity: NonZeroUsize,
    on_stack: bool,
    data: LruData<K, V, N, I>,
}

impl<K, V, const N: usize, I: IndexType> AnyLruCache<K, V> for SmallLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
{
    fn len(&self) -> usize {
        self.len
    }
    fn cap(&self) -> NonZeroUsize {
        self.capacity
    }
    fn put(&mut self, key: K, value: V) -> Option<V> {
        self.put(key, value)
    }
    fn get<Q>(&mut self, key: &Q) -> Option<&V>
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
    fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.peek(key)
    }
    fn clear(&mut self) {
        self.clear();
    }
}

impl<K, V, const N: usize, I: IndexType> SmallLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
{
    pub const MAX_STACK_SIZE: usize = 16 * 1024;

    pub fn new(cap: NonZeroUsize) -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= 16 * 1024,
                "SmallLruCache is too large! Reduce N."
            );
        }
        Self {
            len: 0,
            capacity: cap,
            on_stack: true,
            data: LruData {
                stack: ManuallyDrop::new(HeaplessLruCache::new()),
            },
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn capacity(&self) -> NonZeroUsize {
        self.capacity
    }

    #[inline(always)]
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.on_stack {
            unsafe { (*self.data.stack).get(key) }
        } else {
            unsafe { (*self.data.heap).get(key) }
        }
    }

    #[inline(always)]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.on_stack {
            unsafe { (*self.data.stack).get_mut(key) }
        } else {
            unsafe { (*self.data.heap).get_mut(key) }
        }
    }

    #[inline(always)]
    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        unsafe {
            if self.on_stack {
                let idx = *(*self.data.stack).map.get(key)?;
                Some(
                    &*(*self.data.stack)
                        .values
                        .get_unchecked(idx.as_usize())
                        .as_ptr(),
                )
            } else {
                (*self.data.heap).peek(key)
            }
        }
    }

    #[inline(always)]
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        if self.on_stack {
            unsafe {
                let stack = &mut *self.data.stack;
                let (old_v, res) = stack.put(key, value, self.capacity.get());
                match res {
                    Ok(()) => {
                        self.len = stack.len();
                        return old_v;
                    }
                    Err((k, v)) => {
                        self.spill_to_heap();
                        let res = (*self.data.heap).put(k, v);
                        self.len = (*self.data.heap).len();
                        return res;
                    }
                }
            }
        }
        unsafe {
            let res = (*self.data.heap).put(key, value);
            self.len = (*self.data.heap).len();
            res
        }
    }

    #[inline(never)]
    unsafe fn spill_to_heap(&mut self) {
        let mut heap = LruCache::new(self.capacity);
        unsafe {
            let stack = ManuallyDrop::take(&mut self.data.stack);
            let mut curr = stack.tail;
            while curr != I::NONE {
                let key = ptr::read(stack.keys.get_unchecked(curr.as_usize()).as_ptr());
                let val = ptr::read(stack.values.get_unchecked(curr.as_usize()).as_ptr());
                heap.put(key, val);
                curr = *stack.prevs.get_unchecked(curr.as_usize());
            }
            core::mem::forget(stack);

            ptr::write(&mut self.data.heap, ManuallyDrop::new(heap));
            self.on_stack = false;
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        unsafe {
            if self.on_stack {
                let cap = self.capacity;
                ManuallyDrop::drop(&mut self.data.stack);
                self.data.stack = ManuallyDrop::new(HeaplessLruCache::new());
                self.capacity = cap;
            } else {
                (*self.data.heap).clear();
            }
        }
        self.len = 0;
    }

    #[inline(always)]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }
}

impl<K, V, const N: usize, I: IndexType> Drop for SmallLruCache<K, V, N, I> {
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

impl<K, V, const N: usize, I: IndexType> Clone for SmallLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        unsafe {
            if self.on_stack {
                let mut new_stack = HeaplessLruCache::new();
                let mut curr = (*self.data.stack).tail;
                let mut nodes_to_add = std::vec::Vec::new();
                while curr != I::NONE {
                    let key = (*self.data.stack)
                        .keys
                        .get_unchecked(curr.as_usize())
                        .assume_init_ref();
                    let val = (*self.data.stack)
                        .values
                        .get_unchecked(curr.as_usize())
                        .assume_init_ref();
                    nodes_to_add.push((key.clone(), val.clone()));
                    curr = *(*self.data.stack).prevs.get_unchecked(curr.as_usize());
                }
                for (k, v) in nodes_to_add {
                    let _ = new_stack.put(k, v, self.capacity.get());
                }

                Self {
                    len: self.len,
                    on_stack: true,
                    capacity: self.capacity,
                    data: LruData {
                        stack: ManuallyDrop::new(new_stack),
                    },
                }
            } else {
                let heap_cloned = (*self.data.heap).clone();
                Self {
                    len: self.len,
                    on_stack: false,
                    capacity: self.capacity,
                    data: LruData {
                        heap: ManuallyDrop::new(heap_cloned),
                    },
                }
            }
        }
    }
}

impl<K, V, const N: usize, I: IndexType> Debug for SmallLruCache<K, V, N, I>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("SmallLruCache");
        d.field("on_stack", &self.on_stack);
        d.field("capacity", &self.capacity);
        d.field("len", &self.len);
        d.finish()
    }
}

impl<K, V, const N: usize, I: IndexType> FromIterator<(K, V)> for SmallLruCache<K, V, N, I>
where
    K: Hash + Eq + Clone,
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

    // ─── size diagnostic ──────────────────────────────────────────────────────
    #[test]
    fn test_lru_sizes() {
        println!(
            "Size of SmallLruCache<i32, i32, 16>: {}",
            std::mem::size_of::<SmallLruCache<i32, i32, 16>>()
        );
        println!(
            "Size of HeaplessLruCache<i32, i32, 16>: {}",
            std::mem::size_of::<HeaplessLruCache<i32, i32, 16>>()
        );
    }

    // ─── stack operations ─────────────────────────────────────────────────────
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
    fn test_lru_cache_stack_ops_update() {
        let cap = NonZeroUsize::new(4).unwrap();
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        let old = cache.put(1, 10);
        assert_eq!(old, None);
        let old2 = cache.put(1, 99);
        assert_eq!(old2, Some(10));
        assert_eq!(cache.get(&1), Some(&99));
    }

    #[test]
    fn test_lru_cache_stack_ops_eviction_lru() {
        let cap = NonZeroUsize::new(2).unwrap();
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        cache.put(1, 10);
        cache.put(2, 20);
        // Access 1 → promotes 1, makes 2 LRU
        cache.get(&1);
        cache.put(3, 30); // evicts 2
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_cache_stack_ops_peek_no_promote() {
        let cap = NonZeroUsize::new(2).unwrap();
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        cache.put(1, 10);
        cache.put(2, 20);
        // peek 1 without promoting
        assert_eq!(cache.peek(&1), Some(&10));
        // Now put 3 — should evict LRU which is still 1 (peek didn't promote)
        cache.put(3, 30);
        assert_eq!(
            cache.get(&1),
            None,
            "1 should have been evicted (peek didn't promote)"
        );
    }

    #[test]
    fn test_lru_cache_stack_ops_get_mut() {
        let cap = NonZeroUsize::new(4).unwrap();
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        cache.put(1, 10);
        if let Some(v) = cache.get_mut(&1) {
            *v = 42;
        }
        assert_eq!(cache.peek(&1), Some(&42));
    }

    #[test]
    fn test_lru_cache_stack_ops_get_nonexistent() {
        let cap = NonZeroUsize::new(4).unwrap();
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        assert_eq!(cache.get(&99), None);
        assert_eq!(cache.peek(&99), None);
        assert_eq!(cache.get_mut(&99), None);
    }

    #[test]
    fn test_lru_cache_stack_ops_clear() {
        let cap = NonZeroUsize::new(4).unwrap();
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        cache.put(1, 10);
        cache.put(2, 20);
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.is_on_stack());
        // Can still insert after clear
        cache.put(3, 30);
        assert_eq!(cache.get(&3), Some(&30));
    }

    // ─── spill ────────────────────────────────────────────────────────────────
    // NOTE: Spill to heap happens when the physical stack (N slots) is full and
    // HeaplessLruCache::put cannot fit the new entry even after LRU eviction.
    // This occurs when cap > N: after evicting LRU, put tries to fill slot 0..N
    // but the key is in a different slot chain, so `Err` is returned → spill.
    // Easiest trigger: N=2 with cap=8; after 2 entries the stack is full but cap
    // allows more, so the 3rd insert triggers spill.
    #[test]
    fn test_lru_cache_spill_trigger() {
        // N=2: only 2 physical stack slots. cap=8: logical capacity allows 8.
        // After 2 inserts the stack is physically full; the 3rd insert spills.
        let cap = NonZeroUsize::new(8).unwrap();
        let mut cache: SmallLruCache<i32, i32, 2> = SmallLruCache::new(cap);
        cache.put(1, 10);
        cache.put(2, 20);
        assert!(cache.is_on_stack());
        cache.put(3, 30); // 3rd key can't fit in 2-slot stack → spill
        assert!(!cache.is_on_stack());
        // All 3 entries still reachable on heap
        assert_eq!(cache.get(&2), Some(&20));
        assert_eq!(cache.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_cache_spill_heap_ops() {
        let cap = NonZeroUsize::new(8).unwrap();
        let mut cache: SmallLruCache<i32, i32, 2> = SmallLruCache::new(cap);
        for i in 0..10i32 {
            cache.put(i, i * 10);
        }
        assert!(!cache.is_on_stack()); // spilled after 3rd unique key
        // cap=8 so last 8 keys survive LRU eviction, first 2 evicted
        assert_eq!(cache.get(&9), Some(&90));
        assert_eq!(cache.get(&0), None); // evicted by LRU
        assert_eq!(cache.get(&1), None); // evicted by LRU
    }

    // ─── clone ────────────────────────────────────────────────────────────────
    #[test]
    fn test_lru_cache_traits_clone_stack() {
        let cap = NonZeroUsize::new(4).unwrap();
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        cache.put(1, 10);
        cache.put(2, 20);
        let mut cloned = cache.clone();
        cloned.put(3, 30);
        assert_eq!(cache.len(), 2);
        assert_eq!(cloned.len(), 3);
    }

    #[test]
    fn test_lru_cache_traits_clone_heap() {
        let cap = NonZeroUsize::new(8).unwrap();
        let mut cache: SmallLruCache<i32, i32, 2> = SmallLruCache::new(cap);
        for i in 0..10i32 {
            cache.put(i, i);
        }
        assert!(!cache.is_on_stack()); // spilled after 3rd key
        let cloned = cache.clone();
        assert!(!cloned.is_on_stack());
        assert_eq!(cloned.len(), 8); // cap=8, 10 inserts → last 8 alive
    }

    // ─── FromIterator ─────────────────────────────────────────────────────────
    #[test]
    fn test_lru_cache_traits_from_iter() {
        let cache: SmallLruCache<i32, i32, 4> = vec![(1, 10), (2, 20)].into_iter().collect();
        assert_eq!(cache.len(), 2);
        assert!(cache.is_on_stack());
    }

    // ─── cap / AnyLruCache trait ───────────────────────────────────────────────
    // NOTE: AnyLruCache has generic methods (get<Q>, peek<Q>, get_mut<Q>) so it
    // is NOT object-safe. We exercise the trait methods through a concrete type.
    #[test]
    fn test_lru_cache_any_lru_cache_trait() {
        use std::num::NonZeroUsize;
        let cap = NonZeroUsize::new(3).unwrap();
        // Call through the trait by explicitly typing the impl
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        // AnyLruCache methods:
        assert_eq!(
            <SmallLruCache<i32, i32, 4> as AnyLruCache<i32, i32>>::cap(&cache),
            cap
        );
        assert_eq!(
            <SmallLruCache<i32, i32, 4> as AnyLruCache<i32, i32>>::put(&mut cache, 1, 10),
            None
        );
        assert_eq!(
            <SmallLruCache<i32, i32, 4> as AnyLruCache<i32, i32>>::get(&mut cache, &1),
            Some(&10)
        );
        assert_eq!(
            <SmallLruCache<i32, i32, 4> as AnyLruCache<i32, i32>>::peek(&cache, &1),
            Some(&10)
        );
        assert!(
            <SmallLruCache<i32, i32, 4> as AnyLruCache<i32, i32>>::get_mut(&mut cache, &1)
                .is_some()
        );
        <SmallLruCache<i32, i32, 4> as AnyLruCache<i32, i32>>::clear(&mut cache);
        assert!(<SmallLruCache<i32, i32, 4> as AnyLruCache<i32, i32>>::is_empty(&cache));
    }

    // ─── Debug ────────────────────────────────────────────────────────────────
    #[test]
    fn test_lru_cache_traits_debug() {
        let cap = NonZeroUsize::new(4).unwrap();
        let mut cache: SmallLruCache<i32, i32, 4> = SmallLruCache::new(cap);
        cache.put(1, 10);
        let s = format!("{:?}", cache);
        assert!(s.contains("on_stack"));
        assert!(s.contains("capacity"));
    }

    // ─── heap clear ───────────────────────────────────────────────────────────
    #[test]
    fn test_lru_cache_any_storage_heap_clear() {
        let cap = NonZeroUsize::new(8).unwrap();
        let mut cache: SmallLruCache<i32, i32, 2> = SmallLruCache::new(cap);
        for i in 0..10i32 {
            cache.put(i, i);
        }
        assert!(!cache.is_on_stack()); // spilled after 3rd key
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(!cache.is_on_stack()); // stays on heap after clear
    }
}
