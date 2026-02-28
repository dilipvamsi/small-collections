use core::mem::ManuallyDrop;
use core::num::NonZeroUsize;
use core::ptr;
use heapless::index_map::FnvIndexMap;
use lru::LruCache;
use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::hash::Hash;

/// A trait for abstraction over different LRU cache types (Stack, Heap, Small).
pub trait AnyLruCache<K, V> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn cap(&self) -> NonZeroUsize;
    fn put(&mut self, key: K, value: V) -> Option<V>;
    fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;
    fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;
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

/// A stack-optimized Least Recently Used (LRU) cache.
///
/// # Overview
/// It lives on the stack for up to `N` items, then spills to the heap using the `lru` crate.
/// On the stack, it uses `heapless::FnvIndexMap`. Promotion (MRU) is implemented by
/// removing and re-inserting the item, which is O(N) on the stack but O(1) effectively
/// for small N. On the heap, it's O(1).
///
/// # Safety
/// * `on_stack` tag determines which side of the `LruData` union is active.
/// * `ManuallyDrop` manages destruction of the active variant.
pub struct SmallLruCache<K, V, const N: usize> {
    on_stack: bool,
    capacity: NonZeroUsize,
    data: LruData<K, V, N>,
}

impl<K, V, const N: usize> AnyLruCache<K, V> for SmallLruCache<K, V, N>
where
    K: Hash + Eq + Clone,
{
    fn len(&self) -> usize {
        self.len()
    }
    fn cap(&self) -> NonZeroUsize {
        self.capacity()
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

/// Internal storage for `SmallLruCache`.
union LruData<K, V, const N: usize> {
    stack: ManuallyDrop<FnvIndexMap<K, V, N>>,
    heap: ManuallyDrop<LruCache<K, V>>,
}

impl<K, V, const N: usize> SmallLruCache<K, V, N>
where
    K: Hash + Eq + Clone,
{
    pub const MAX_STACK_SIZE: usize = 16 * 1024;

    /// Creates a new LRU cache with the given total capacity.
    pub fn new(cap: NonZeroUsize) -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= SmallLruCache::<K, V, N>::MAX_STACK_SIZE,
                "SmallLruCache is too large! Reduce N."
            );
        }

        Self {
            on_stack: true,
            capacity: cap,
            data: LruData {
                stack: ManuallyDrop::new(FnvIndexMap::new()),
            },
        }
    }

    /// Returns the number of elements in the cache.
    pub fn len(&self) -> usize {
        unsafe {
            if self.on_stack {
                self.data.stack.len()
            } else {
                self.data.heap.len()
            }
        }
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the total capacity of the cache.
    pub fn capacity(&self) -> NonZeroUsize {
        self.capacity
    }

    /// Returns a reference to the value of the key in the cache or `None` if it is not present.
    ///
    /// If the key is present, it is promoted to the Most Recently Used (MRU) position.
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        unsafe {
            if self.on_stack {
                let stack = &mut *self.data.stack;

                let mut found_key = None;
                for k in stack.keys() {
                    if k.borrow() == key {
                        found_key = Some(k.clone());
                        break;
                    }
                }

                if let Some(k) = found_key {
                    let v = stack.remove(key).unwrap();
                    let _ = stack.insert(k, v);
                    return stack.get(key);
                }
                None
            } else {
                (*self.data.heap).get(key)
            }
        }
    }

    /// Returns a mutable reference to the value of the key in the cache or `None` if it is not present.
    ///
    /// If the key is present, it is promoted to the MRU position.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        unsafe {
            if self.on_stack {
                let stack = &mut *self.data.stack;
                let mut found_key = None;
                for k in stack.keys() {
                    if k.borrow() == key {
                        found_key = Some(k.clone());
                        break;
                    }
                }

                if let Some(k) = found_key {
                    let v = stack.remove(key).unwrap();
                    let _ = stack.insert(k, v);
                    return stack.get_mut(key);
                }
                None
            } else {
                (*self.data.heap).get_mut(key)
            }
        }
    }

    /// Returns a reference to the value of the key in the cache without promoting it.
    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        unsafe {
            if self.on_stack {
                (*self.data.stack).get(key)
            } else {
                (*self.data.heap).peek(key)
            }
        }
    }

    /// Puts a key-value pair into the cache.
    ///
    /// If the key already exists, it is updated and promoted to MRU.
    /// If the cache is full, the least recently used item is evicted.
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        unsafe {
            if self.on_stack {
                if (*self.data.stack).contains_key(&key) {
                    let old = (*self.data.stack).remove(&key).unwrap();
                    let _ = (*self.data.stack).insert(key, value);
                    return Some(old);
                }

                if (*self.data.stack).len() == N || (*self.data.stack).len() >= self.capacity.get()
                {
                    if (*self.data.stack).len() >= self.capacity.get() {
                        // Evict LRU (first item in IndexMap)
                        let first_key = (*self.data.stack).keys().next().cloned();
                        if let Some(k) = first_key {
                            (*self.data.stack).remove(&k);
                        }
                    } else {
                        self.spill_to_heap();
                    }
                }

                if self.on_stack {
                    let _ = (*self.data.stack).insert(key, value);
                    return None;
                }
            }

            (*self.data.heap).put(key, value)
        }
    }

    #[inline(never)]
    unsafe fn spill_to_heap(&mut self) {
        let mut heap = LruCache::new(self.capacity);
        unsafe {
            let stack = ptr::read(&*self.data.stack);
            for (k, v) in stack.into_iter() {
                heap.put(k, v);
            }
        }
        self.data.heap = ManuallyDrop::new(heap);
        self.on_stack = false;
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        unsafe {
            if self.on_stack {
                (*self.data.stack).clear();
            } else {
                (*self.data.heap).clear();
            }
        }
    }

    /// Returns `true` if the cache is currently on the stack.
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }
}

impl<K, V, const N: usize> Drop for SmallLruCache<K, V, N> {
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

impl<K, V, const N: usize> Clone for SmallLruCache<K, V, N>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        unsafe {
            if self.on_stack {
                Self {
                    on_stack: true,
                    capacity: self.capacity,
                    data: LruData {
                        stack: ManuallyDrop::new((*self.data.stack).clone()),
                    },
                }
            } else {
                Self {
                    on_stack: false,
                    capacity: self.capacity,
                    data: LruData {
                        heap: ManuallyDrop::new((*self.data.heap).clone()),
                    },
                }
            }
        }
    }
}

impl<K, V, const N: usize> Debug for SmallLruCache<K, V, N>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("SmallLruCache");
        d.field("on_stack", &self.on_stack);
        d.field("capacity", &self.capacity);
        unsafe {
            if self.on_stack {
                d.field("len", &self.data.stack.len());
            } else {
                d.field("len", &self.data.heap.len());
            }
        }
        d.finish()
    }
}

impl<K, V, const N: usize> FromIterator<(K, V)> for SmallLruCache<K, V, N>
where
    K: Hash + Eq + Clone,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        // We use a default capacity if not specified, or we could require it.
        // For FromIterator, we'll use N as the default capacity.
        let mut cache = Self::new(NonZeroUsize::new(N).unwrap());
        for (k, v) in iter {
            cache.put(k, v);
        }
        cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_stack_eviction() {
        let mut cache: SmallLruCache<&str, i32, 4> =
            SmallLruCache::new(NonZeroUsize::new(2).unwrap());
        cache.put("a", 1);
        cache.put("b", 2);
        assert_eq!(cache.len(), 2);

        cache.put("c", 3); // Should evict "a" if capacity is 2
        assert_eq!(cache.len(), 2);
        assert!(cache.peek(&"a").is_none());
        assert!(cache.peek(&"b").is_some());
    }

    #[test]
    fn test_lru_cache_stack_promotion() {
        let mut cache: SmallLruCache<&str, i32, 4> =
            SmallLruCache::new(NonZeroUsize::new(2).unwrap());
        cache.put("a", 1);
        cache.put("b", 2);
        assert_eq!(cache.get(&"a"), Some(&1)); // Promote "a"
        cache.put("c", 3); // Should evict "b"
        assert!(cache.peek(&"b").is_none());
        assert!(cache.peek(&"a").is_some());
    }

    #[test]
    fn test_lru_cache_spill_trigger_on_put() {
        let mut cache: SmallLruCache<i32, i32, 2> =
            SmallLruCache::new(NonZeroUsize::new(10).unwrap());
        cache.put(1, 10);
        cache.put(2, 20);
        assert!(cache.is_on_stack());

        cache.put(3, 30); // Spill
        assert!(!cache.is_on_stack());
        assert_eq!(cache.len(), 3);
        assert_eq!(*cache.get(&1).unwrap(), 10);
    }

    #[test]
    fn test_lru_cache_any_storage_get_mut() {
        let mut cache: SmallLruCache<i32, i32, 4> =
            SmallLruCache::new(NonZeroUsize::new(10).unwrap());
        cache.put(1, 10);
        if let Some(v) = cache.get_mut(&1) {
            *v = 11;
        }
        assert_eq!(cache.peek(&1), Some(&11));

        for i in 2..6 {
            cache.put(i, i * 10);
        }
        assert!(!cache.is_on_stack());
        if let Some(v) = cache.get_mut(&5) {
            *v = 55;
        }
        assert_eq!(cache.peek(&5), Some(&55));
    }

    #[test]
    fn test_lru_cache_traits_capacity() {
        let cap = NonZeroUsize::new(5).unwrap();
        let cache: SmallLruCache<i32, i32, 2> = SmallLruCache::new(cap);
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), cap);
    }

    #[test]
    fn test_lru_cache_traits_clone() {
        let mut cache: SmallLruCache<i32, i32, 4> =
            SmallLruCache::new(NonZeroUsize::new(10).unwrap());
        cache.put(1, 10);
        let mut cloned = cache.clone();
        cloned.put(2, 20);
        assert!(cache.peek(&2).is_none());
        assert!(cloned.peek(&2).is_some());

        // Clone heap
        for i in 3..8 {
            cache.put(i, i * 10);
        }
        assert!(!cache.is_on_stack());
        let cloned_heap = cache.clone();
        assert_eq!(cloned_heap.len(), cache.len());
        assert_eq!(cloned_heap.peek(&1), cache.peek(&1));
    }

    #[test]
    fn test_lru_cache_traits_debug() {
        let mut cache: SmallLruCache<i32, i32, 4> =
            SmallLruCache::new(NonZeroUsize::new(10).unwrap());
        cache.put(1, 10);
        let debug_stack = format!("{:?}", cache);
        assert!(debug_stack.contains("on_stack: true"));
        assert!(debug_stack.contains("len: 1"));

        for i in 2..7 {
            cache.put(i, i * 10);
        }
        assert!(!cache.is_on_stack());
        let debug_heap = format!("{:?}", cache);
        assert!(debug_heap.contains("on_stack: false"));
        assert!(debug_heap.contains("len: 6"));
    }

    #[test]
    fn test_lru_cache_any_storage_get_none() {
        let mut cache: SmallLruCache<i32, i32, 4> =
            SmallLruCache::new(NonZeroUsize::new(10).unwrap());
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get_mut(&1), None);

        for i in 0..5 {
            cache.put(i, i);
        }
        assert_eq!(cache.get(&10), None);
        assert_eq!(cache.get_mut(&10), None);
    }

    #[test]
    fn test_lru_cache_any_storage_clear() {
        let mut cache: SmallLruCache<i32, i32, 2> =
            SmallLruCache::new(NonZeroUsize::new(10).unwrap());
        cache.put(1, 10);
        cache.clear();
        assert!(cache.is_empty());

        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30); // Spill
        assert!(!cache.is_on_stack());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_cache_any_storage_get_after_spill() {
        let mut lru: SmallLruCache<i32, i32, 2> =
            SmallLruCache::new(std::num::NonZeroUsize::new(10).unwrap());
        lru.put(1, 1);
        lru.put(2, 2);
        lru.put(3, 3); // Spills to heap

        assert_eq!(lru.get(&1), Some(&1));
        assert_eq!(lru.get_mut(&2), Some(&mut 2));
    }

    #[test]
    fn test_lru_cache_stack_ops_overwrite() {
        let mut s: SmallLruCache<i32, i32, 4> =
            SmallLruCache::new(std::num::NonZeroUsize::new(10).unwrap());
        s.put(1, 10);
        s.put(1, 11); // overwrite
        assert_eq!(s.get(&1), Some(&11));
    }
}
