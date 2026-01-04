use core::mem::ManuallyDrop;
use core::ptr;
use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::FromIterator;
use std::ops::{Index, IndexMut};

// Use 'hashbrown' directly for the Raw Entry API (allows preventing double-hashing during spill)
use hashbrown::HashMap;
// Use 'fnv' to match heapless's internal hasher for consistent performance
use fnv::FnvBuildHasher;
// Use 'heapless' for the stack storage
use heapless::index_map::FnvIndexMap;

/// A map that lives on the stack for `N` items, then automatically spills to the heap.
///
/// # Overview
/// * **Stack State:** Zero allocations. Extremely fast FNV hashing. Data is stored inline.
/// * **Heap State:** Standard `HashMap` performance. Data is stored on the heap.
/// * **Spill:** Occurs automatically when the stack capacity `N` is exceeded. This is a "Zero-Allocation Move"—keys/values are moved, not cloned.
///
/// # Capacity Constraints (`N`)
/// Due to the underlying `heapless` implementation constraints:
/// * `N` must be a **power of two** (e.g., 2, 4, 8, 16, 32).
/// * `N` must be **greater than 1**.
///
/// **Compilation will fail if these constraints are not met.**
pub struct SmallMap<K, V, const N: usize> {
    /// Tracks whether data is currently in `data.stack` or `data.heap`.
    /// This acts as the "tag" for our manual tagged union.
    on_stack: bool,

    /// The storage union. Only one field is active at a time.
    data: MapData<K, V, N>,
}

/// Internal storage union.
///
/// We use `ManuallyDrop` because the compiler cannot know which field is active based on `on_stack`
/// and therefore cannot automatically drop the correct one. We must handle this manually in `impl Drop`.
union MapData<K, V, const N: usize> {
    stack: ManuallyDrop<FnvIndexMap<K, V, N>>,
    heap: ManuallyDrop<HashMap<K, V, FnvBuildHasher>>,
}

// --- 1. Core Implementation ---

impl<K, V, const N: usize> SmallMap<K, V, N>
where
    K: Eq + Hash,
{
    /// The maximum allowed stack size in bytes (16 KB).
    ///
    /// Because `SmallMap` stores data inline, a large `N` or large `Key`/`Value` types
    /// can easily exceed the thread stack size. This limit prevents that.
    pub const MAX_STACK_SIZE: usize = 16 * 1024;

    /// Creates a new empty map on the stack.
    ///
    /// # Compile-Time Safety Check
    /// This function enforces a strict size limit of **16 KB** (`MAX_STACK_SIZE`).
    ///
    /// Because the map size is known at compile time, the compiler will **fail to build**
    /// if the total size of `SmallMap<K, V, N>` exceeds this limit. This prevents
    /// accidental Stack Overflows (Segfaults).
    ///
    /// # How to fix the build error
    /// If your code fails to compile pointing to this assertion, you have two options:
    /// 1. **Reduce `N`:** If you don't need that many items on the stack.
    /// 2. **Box the Value:** Change `SmallMap<K, V, N>` to `SmallMap<K, Box<V>, N>`.
    ///    This moves the bulk of the data to the heap immediately, keeping the stack footprint small.
    pub fn new() -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= SmallMap::<K, V, N>::MAX_STACK_SIZE,
                "SmallMap is too large! The total struct size exceeds the 16KB safety limit. \
                 This will cause a Stack Overflow. \
                 Solution: Reduce N, or wrap your Value in Box<V>."
            );
        }

        Self {
            on_stack: true,
            data: MapData {
                stack: ManuallyDrop::new(FnvIndexMap::new()),
            },
        }
    }

    /// Returns `true` if the map is currently storing data on the stack.
    /// Returns `false` if it has spilled to the heap.
    #[inline]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        unsafe {
            // Safety: We check the `on_stack` tag to access the active union field.
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
    ///
    /// * **Stack:** Resets the index to 0.
    /// * **Heap:** Clears the map but keeps the allocated memory for reuse.
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
    ///
    /// If the map is on the stack and full, this triggers a **Spill to Heap**.
    /// This implementation explicitly checks capacity constraints before insertion,
    /// ensuring a clean separation between the "Spill Decision" and the "Insert Action".
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        unsafe {
            if self.on_stack {
                let stack_map = &mut *self.data.stack;

                // Check 1: Is the map full?
                if stack_map.len() == N {
                    // Check 2: Is this a NEW key? (If it's an update, we fit!)
                    // Note: We only pay the hashing cost of 'contains_key' if we are at the limit.
                    if !stack_map.contains_key(&key) {
                        self.spill_to_heap();
                        // Fall through to Heap Logic below...
                    } else {
                        // Map is full, but we are updating an existing key. Safe to proceed on stack.
                        return match stack_map.insert(key, value) {
                            Ok(old_val) => old_val,
                            Err(_) => unreachable!("Logic Error: Key exists, update must succeed"),
                        };
                    }
                } else {
                    // Map is not full. Safe to insert.
                    return match stack_map.insert(key, value) {
                        Ok(old_val) => old_val,
                        Err(_) => {
                            unreachable!("Logic Error: Capacity available, insert must succeed")
                        }
                    };
                }
            }

            // Heap Logic (Standard Insert)
            (*self.data.heap).insert(key, value)
        }
    }

    /// Retrieves a reference to the value corresponding to the key.
    ///
    /// This method is generic over the key type `Q`. This allows you to lookup
    /// values using a reference (like `&str`) without allocating a new owned
    /// key (like `String`).
    ///
    /// # How it works (The `Borrow` Trait)
    /// If the map stores keys of type `K` (e.g., `String`), you can pass a
    /// query key of type `Q` (e.g., `str`) as long as:
    /// 1. `K` implements `Borrow<Q>` (meaning `String` can be viewed as `str`).
    /// 2. `Q` implements `Hash` and `Eq`.
    /// 3. The hash of the borrowed `K` is identical to the hash of `Q`.
    ///
    /// # Example
    /// ```rust
    /// // Assuming generic import for doc test
    /// use small_collections::SmallMap;
    ///
    /// let mut map = SmallMap::<String, i32, 4>::new();
    /// map.insert("Apple".to_string(), 10);
    ///
    /// // Works efficiently without allocating a new String for the lookup:
    /// assert_eq!(map.get("Apple"), Some(&10));
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,          // The Key (String) can be borrowed as Q (str)
        Q: Hash + Eq + ?Sized, // Q is hashable and comparable (Unsized allows str)
    {
        unsafe {
            if self.on_stack {
                self.data.stack.get(key)
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
                (*self.data.stack).get_mut(key)
            } else {
                (*self.data.heap).get_mut(key)
            }
        }
    }

    /// Removes a key from the map, returning the value at the key if the key was previously in the map.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        unsafe {
            if self.on_stack {
                (*self.data.stack).remove(key)
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
        unsafe {
            if self.on_stack {
                self.data.stack.contains_key(key)
            } else {
                self.data.heap.contains_key(key)
            }
        }
    }

    /// **The Logic Core: Stack -> Heap Migration**
    ///
    /// This method performs a "Zero-Cost Move" of items from Stack to Heap.
    /// It avoids cloning keys/values and avoids re-hashing where possible.
    ///
    /// # Safety
    /// This function is marked `unsafe` because it manually manages raw pointers
    /// and union state. The caller must ensure the map is currently on the stack.
    #[inline(never)] // Optimization: Keep this cold path out of the hot 'insert' loop
    unsafe fn spill_to_heap(&mut self) {
        // We explicitly wrap the body in `unsafe` to satisfy `unsafe_op_in_unsafe_fn` lint
        unsafe {
            // 1. "Steal" the Stack Map
            // `ptr::read` does a bitwise copy of the map struct.
            // We effectively own the items now. The original memory in `self.data.stack`
            // is now considered "moved from" and should not be dropped.
            let stack_map = ptr::read(&*self.data.stack);

            // 2. Allocate Heap Map
            // We use capacity * 2 to give breathing room after the spill.
            // We use FnvBuildHasher to maintain consistent hashing behavior with the stack map.
            let mut new_heap =
                HashMap::with_capacity_and_hasher(stack_map.len() * 2, FnvBuildHasher::default());

            // Cache the hasher builder to avoid cloning it in the loop
            let hasher_builder = new_heap.hasher().clone();

            // 3. Migrate Items (The Efficient Way)
            // `into_iter()` consumes `stack_map`. Since we own it (via ptr::read),
            // this moves the Keys and Values directly. NO CLONES occur here.
            for (key, value) in stack_map.into_iter() {
                // A. Re-calculate hash using the Heap Map's hasher
                let mut hasher = hasher_builder.build_hasher();
                key.hash(&mut hasher);
                let hash = hasher.finish();

                // B. Insert using Raw Entry API
                // We skip the collision check and probe sequence because we know
                // we are inserting into a fresh, empty map.
                new_heap
                    .raw_entry_mut()
                    .from_key_hashed_nocheck(hash, &key)
                    .insert(key, value);
            }

            // 4. Overwrite Union Memory
            // CRITICAL: We use `ptr::write` to overwrite the union field.
            // If we used simple assignment (`self.data.heap = ...`), Rust would try to
            // Drop the *old* value at that memory address.
            // But the old value is the `stack` map bits! Treating stack bits as a
            // heap map pointer would cause a segfault (freeing invalid memory).
            // `ptr::write` overwrites blindly without dropping the old garbage.
            ptr::write(&mut self.data.heap, ManuallyDrop::new(new_heap));

            // 5. Flip the Switch
            self.on_stack = false;
        }
    }
}

/// Allows read access using `map[&key]`.
///
/// # Panics
/// Panics if the key is not present in the map.
impl<K, V, Q, const N: usize> Index<&Q> for SmallMap<K, V, N>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash + ?Sized,
{
    type Output = V;

    fn index(&self, key: &Q) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

/// Allows mutable access using `map[&key] = new_value`.
///
/// # Panics
/// Panics if the key is not present in the map.
impl<K, V, Q, const N: usize> IndexMut<&Q> for SmallMap<K, V, N>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash + ?Sized,
{
    fn index_mut(&mut self, key: &Q) -> &mut Self::Output {
        self.get_mut(key).expect("no entry found for key")
    }
}

// Add this to src/map.rs

// Manual implementation of Clone.
// We must check `on_stack` to know which field to clone.
impl<K, V, const N: usize> Clone for SmallMap<K, V, N>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        unsafe {
            if self.on_stack {
                // Clone the Stack Map
                let stack_clone = (*self.data.stack).clone();
                SmallMap {
                    on_stack: true,
                    data: MapData {
                        stack: ManuallyDrop::new(stack_clone),
                    },
                }
            } else {
                // Clone the Heap Map
                let heap_clone = (*self.data.heap).clone();
                SmallMap {
                    on_stack: false,
                    data: MapData {
                        heap: ManuallyDrop::new(heap_clone),
                    },
                }
            }
        }
    }
}

// --- 2. Entry API Support ---

impl<K, V, const N: usize> SmallMap<K, V, N>
where
    K: Eq + Hash,
{
    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    pub fn entry(&mut self, key: K) -> SmallMapEntry<'_, K, V, N> {
        unsafe {
            if self.on_stack {
                // Safety Pre-Check:
                // If we return a Stack Entry, user might call `.or_insert()`.
                // If the map is full, `heapless` panics on `or_insert`.
                // We must predict this: If full AND key is new, we spill NOW.
                let needs_spill = {
                    let stack_map = &mut self.data.stack;
                    stack_map.len() == N && !stack_map.contains_key(&key)
                };

                if needs_spill {
                    self.spill_to_heap();
                    // Fall through to Heap logic below
                } else {
                    // Safe to return Stack Entry
                    return SmallMapEntry::Stack((*self.data.stack).entry(key));
                }
            }
            // Heap Logic
            SmallMapEntry::Heap((*self.data.heap).entry(key))
        }
    }
}

/// A wrapper enum that unifies Stack and Heap entries.
pub enum SmallMapEntry<'a, K, V, const N: usize> {
    Stack(heapless::index_map::Entry<'a, K, V, N>),
    Heap(hashbrown::hash_map::Entry<'a, K, V, FnvBuildHasher>),
}

impl<'a, K, V, const N: usize> SmallMapEntry<'a, K, V, N>
where
    K: Eq + Hash,
{
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            // We use expect() because SmallMap::entry() guarantees capacity
            // exists if it returns a Stack variant.
            SmallMapEntry::Stack(e) => {
                // We handle the Result manually to avoid `unwrap()`/`expect()`.
                // This removes the need for V: Debug.
                match e.or_insert(default) {
                    Ok(v) => v,
                    // We ignore the error payload (_) so we don't need to print it.
                    Err(_) => {
                        unreachable!("Logic Error: Stack map capacity check failed in entry()")
                    }
                }
            }
            SmallMapEntry::Heap(e) => e.or_insert(default),
        }
    }

    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            SmallMapEntry::Stack(e) => match e.or_insert_with(default) {
                Ok(v) => v,
                Err(_) => unreachable!("Logic Error: Stack map capacity check failed in entry()"),
            },
            SmallMapEntry::Heap(e) => e.or_insert_with(default),
        }
    }

    pub fn and_modify<F: FnOnce(&mut V)>(self, f: F) -> Self {
        match self {
            SmallMapEntry::Stack(e) => SmallMapEntry::Stack(e.and_modify(f)),
            SmallMapEntry::Heap(e) => SmallMapEntry::Heap(e.and_modify(f)),
        }
    }

    pub fn key(&self) -> &K {
        match self {
            SmallMapEntry::Stack(e) => e.key(),
            SmallMapEntry::Heap(e) => e.key(),
        }
    }
}

// --- 3. Iterator Support ---

impl<K, V, const N: usize> SmallMap<K, V, N>
where
    K: Eq + Hash,
{
    /// Returns an iterator over the map.
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

/// Wrapper for iterators to hide the underlying type difference.
pub enum SmallMapIter<'a, K, V> {
    Stack(heapless::index_map::Iter<'a, K, V>),
    Heap(hashbrown::hash_map::Iter<'a, K, V>),
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

// --- 4. Trait Implementations ---

// Safety: ManuallyDrop fields inside Union are NOT dropped automatically.
// We must check `on_stack` and drop the correct field manually.
impl<K, V, const N: usize> Drop for SmallMap<K, V, N> {
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

// Default (Allows SmallMap::default())
impl<K: Eq + Hash, V, const N: usize> Default for SmallMap<K, V, N> {
    fn default() -> Self {
        Self::new()
    }
}

// Debug (Allows println!("{:?}", map))
// Note: We require V: Debug here to print values
impl<K: Debug + Eq + Hash, V: Debug, const N: usize> Debug for SmallMap<K, V, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

// FromIterator (Allows .collect())
impl<K, V, const N: usize> FromIterator<(K, V)> for SmallMap<K, V, N>
where
    K: Eq + Hash,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = SmallMap::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

// IntoIterator (Allows 'for (k,v) in map')
impl<K: Eq + Hash, V, const N: usize> IntoIterator for SmallMap<K, V, N> {
    type Item = (K, V);
    type IntoIter = SmallMapIntoIter<K, V, N>;

    fn into_iter(self) -> Self::IntoIter {
        // We need to move out of self.
        // We read 'self' bitwise, then forget the original so Drop doesn't run.
        let this = ManuallyDrop::new(self);
        unsafe {
            if this.on_stack {
                // ptr::read copies the stack map out, then we iterate it
                SmallMapIntoIter::Stack(ptr::read(&*this.data.stack).into_iter())
            } else {
                // ptr::read copies the heap map out
                SmallMapIntoIter::Heap(ptr::read(&*this.data.heap).into_iter())
            }
        }
    }
}

/// Wrapper for owning iterators
pub enum SmallMapIntoIter<K, V, const N: usize> {
    Stack(heapless::index_map::IntoIter<K, V, N>),
    Heap(hashbrown::hash_map::IntoIter<K, V>),
}

impl<K, V, const N: usize> Iterator for SmallMapIntoIter<K, V, N> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SmallMapIntoIter::Stack(i) => i.next(),
            SmallMapIntoIter::Heap(i) => i.next(),
        }
    }
}

// --- 5. Test Suite ---

#[cfg(test)]
mod tests {
    use super::*;

    // --- Basic Stack Operations ---
    #[test]
    fn test_stack_basic_operations() {
        let mut map: SmallMap<i32, i32, 4> = SmallMap::new();

        assert!(map.is_empty());
        assert!(map.is_on_stack());

        map.insert(1, 10);
        map.insert(2, 20);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&1), Some(&10));
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.get(&99), None);

        // Ensure we haven't spilled yet
        assert!(map.is_on_stack());
    }

    // --- The Critical Spill Test ---
    #[test]
    fn test_spill_trigger_and_persistence() {
        // Capacity is strictly 2
        let mut map: SmallMap<String, String, 2> = SmallMap::new();

        map.insert("Key1".into(), "Val1".into());
        map.insert("Key2".into(), "Val2".into());

        // Still on stack (2/2)
        assert!(map.is_on_stack());

        // TRIGGER SPILL: Insert 3rd item
        // This should trigger: ptr::read(stack) -> alloc heap -> move items -> ptr::write(union)
        map.insert("Key3".into(), "Val3".into());

        // 1. Check State Change
        assert!(!map.is_on_stack(), "Map should have spilled to heap");

        // 2. Check Data Integrity (Did previous items survive?)
        assert_eq!(map.get("Key1"), Some(&"Val1".to_string()));
        assert_eq!(map.get("Key2"), Some(&"Val2".to_string()));
        assert_eq!(map.get("Key3"), Some(&"Val3".to_string()));
        assert_eq!(map.len(), 3);
    }

    // --- Heap Operations (Post-Spill) ---
    #[test]
    fn test_heap_operations() {
        let mut map: SmallMap<i32, i32, 2> = SmallMap::new();

        // Fill and Spill
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30); // Spilled

        // Continue working on Heap
        map.insert(4, 40);
        map.remove(&1); // Remove from Heap

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&1), None);
        assert_eq!(map.get(&4), Some(&40));
        assert!(!map.is_on_stack());
    }

    // --- Overwriting Values ---
    #[test]
    fn test_overwrite() {
        let mut map: SmallMap<i32, i32, 2> = SmallMap::new();

        // Stack Overwrite
        map.insert(1, 10);
        map.insert(1, 99);
        assert_eq!(map.get(&1), Some(&99));

        // Spill
        map.insert(2, 20);
        map.insert(3, 30);

        // Heap Overwrite
        map.insert(1, 1000);
        assert_eq!(map.get(&1), Some(&1000));
    }

    // --- Entry API: Basic & Modify ---
    #[test]
    fn test_entry_api_basic() {
        let mut map: SmallMap<&str, i32, 4> = SmallMap::new();

        // or_insert (New Key)
        map.entry("A").or_insert(1);
        assert_eq!(map.get("A"), Some(&1));

        // or_insert (Existing Key)
        map.entry("A").or_insert(999);
        assert_eq!(map.get("A"), Some(&1)); // Should not change

        // and_modify
        map.entry("A").and_modify(|v| *v += 10);
        assert_eq!(map.get("A"), Some(&11));
    }

    // --- Entry API: Spill Edge Case ---
    #[test]
    fn test_entry_spill() {
        let mut map: SmallMap<&str, i32, 2> = SmallMap::new();
        map.insert("A", 1);
        map.insert("B", 2);

        // Map is full (2/2).
        // Calling .entry("C") checks capacity -> finds full -> spills -> returns Heap Entry
        // If we didn't handle this, heapless would panic inside or_insert.
        let val = map.entry("C").or_insert(3);
        *val += 10;

        assert!(!map.is_on_stack());
        assert_eq!(map.get("C"), Some(&13));
    }

    // --- Iterators & Traits ---
    #[test]
    fn test_iterators_and_traits() {
        let mut map: SmallMap<i32, i32, 2> = SmallMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30); // Spill

        // Test Reference Iterator
        let mut sum = 0;
        for (k, v) in map.iter() {
            sum += k + v;
        }
        assert_eq!(sum, (1 + 10) + (2 + 20) + (3 + 30));

        // Test FromIterator (Collect)
        let collected: SmallMap<i32, i32, 2> = vec![(1, 1), (2, 2), (3, 3)].into_iter().collect();
        assert_eq!(collected.len(), 3);
        assert!(!collected.is_on_stack());

        // Test Debug (ensure V: Debug logic works)
        let debug_str = format!("{:?}", collected);
        assert!(debug_str.contains("1: 1"));

        // Test IntoIterator (Consuming)
        let vec: Vec<(i32, i32)> = collected.into_iter().collect();
        assert_eq!(vec.len(), 3);
    }

    // --- Minimum valid size is 2 Edge Case ---
    #[test]
    fn test_minimum_capacity() {
        // heapless requires N >= 2 and Power of Two
        // This test ensures the library handles the smallest possible stack size correctly
        let mut map: SmallMap<i32, i32, 2> = SmallMap::new();

        map.insert(1, 1);
        map.insert(2, 2);
        assert!(map.is_on_stack()); // Holds 2 items

        map.insert(3, 3);
        assert!(!map.is_on_stack()); // Spills on 3rd
    }

    // --- Clear Operation ---
    #[test]
    fn test_clear() {
        let mut map: SmallMap<i32, i32, 2> = SmallMap::new();

        // Clear Stack
        map.insert(1, 1);
        map.clear();
        assert!(map.is_empty());
        assert!(map.is_on_stack());

        // Clear Heap
        map.insert(1, 1);
        map.insert(2, 2);
        map.insert(3, 3); // Spill
        map.clear();
        assert!(map.is_empty());
        assert!(!map.is_on_stack()); // Remains on heap, just empty
    }

    #[test]
    fn test_size_guard_allows_valid_sizes() {
        // 1. Small Map (Standard use case)
        let _small: SmallMap<i32, i32, 4> = SmallMap::new();

        // 2. Medium Map (Pushing the limit but safe)
        // Struct = 100 bytes. N = 64. Total ~6.4 KB.
        // This is < 16 KB, so it must compile and run.
        #[allow(dead_code)]
        struct MediumStruct([u8; 100]);

        // We verify that this instantiation does NOT panic/fail to build.
        let _medium: SmallMap<i32, MediumStruct, 32> = SmallMap::new();
    }

    #[test]
    fn test_index_read() {
        let mut map: SmallMap<i32, i32, 4> = SmallMap::new();
        map.insert(1, 10);
        map.insert(2, 20);

        // Read using Index syntax
        assert_eq!(map[&1], 10);
        assert_eq!(map[&2], 20);
    }

    #[test]
    fn test_index_assign() {
        let mut map: SmallMap<&str, i32, 4> = SmallMap::new();
        map.insert("A", 10);

        // Modify existing value using IndexMut syntax
        map[&"A"] = 999;

        assert_eq!(map.get("A"), Some(&999));
        assert_eq!(map[&"A"], 999);
    }

    #[test]
    fn test_index_borrowing() {
        // Demonstrate using &str to index a Map<String, _>
        let mut map: SmallMap<String, i32, 4> = SmallMap::new();
        map.insert("Apple".to_string(), 100);

        // We can use string literal "Apple" directly
        assert_eq!(map["Apple"], 100);

        // Mutate
        map["Apple"] = 200;
        assert_eq!(map["Apple"], 200);
    }

    #[test]
    #[should_panic(expected = "no entry found for key")]
    fn test_index_panic_on_missing() {
        let map: SmallMap<i32, i32, 4> = SmallMap::new();
        // This should panic
        let _val = map[&999];
    }

    #[test]
    fn test_map_clone() {
        let mut map: SmallMap<String, i32, 4> = SmallMap::new();
        map.insert("A".to_string(), 1);

        // This requires the impl Clone above
        let mut clone = map.clone();
        clone.insert("B".to_string(), 2);

        // Verify independence
        assert_eq!(map.len(), 1);
        assert_eq!(clone.len(), 2);
        assert_eq!(clone.get("A"), Some(&1));
    }
}
