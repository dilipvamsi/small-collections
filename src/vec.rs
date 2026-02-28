use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use core::ptr;
use core::slice;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A trait generalizing any vector-like contiguous collection.
///
/// This trait allows for high-level interoperability between different
/// sequence types. By implementing `AnyVec`, a type can be used seamlessly
/// with the `SmallVec` interop methods (like `extend_from_any`, `eq_any`, etc.)
/// without requiring manual conversion to a slice.
pub trait AnyVec<T> {
    /// The foundational method: returns the data as a slice.
    fn as_slice(&self) -> &[T];

    // --- Inspection ---

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    // --- Searching ---

    fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().contains(x)
    }

    // --- Iteration ---

    fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }
}

// Implement for Standard Library Vec
impl<T> AnyVec<T> for std::vec::Vec<T> {
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }
}

// Implement for Raw Slices (allows passing &[1, 2, 3])
impl<T> AnyVec<T> for [T] {
    fn as_slice(&self) -> &[T] {
        self
    }
}

// Implement for Arrays (allows passing [1, 2, 3])
impl<T, const N: usize> AnyVec<T> for [T; N] {
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }
}

// Implement for our SmallVec
impl<T, const N: usize> AnyVec<T> for SmallVec<T, N> {
    fn as_slice(&self) -> &[T] {
        self.as_slice() // Uses the method we defined in SmallVec
    }
}

/// A vector that lives on the stack for `N` items, then spills to the heap.
///
/// # Behavior
/// * **Stack Storage:** Uses `heapless::Vec` for the first `N` items.
/// * **Heap Spill:** Automatically moves to `std::vec::Vec` when capacity `N` is exceeded.
/// * **Interface:** Implements `Deref<Target=[T]>`, so all slice methods (iter, split, chunks) work automatically.
pub struct SmallVec<T, const N: usize> {
    on_stack: bool,
    data: VecData<T, N>,
}

/// The internal storage for `SmallVec`.
///
/// This is a manual tagged union that holds either a `heapless::Vec` (stack)
/// or a `std::vec::Vec` (heap). We use `ManuallyDrop` to prevent the compiler
/// from trying to drop both variants automatically, as it cannot know which
/// one is active.
union VecData<T, const N: usize> {
    stack: ManuallyDrop<heapless::Vec<T, N>>,
    heap: ManuallyDrop<std::vec::Vec<T>>,
}

impl<T, const N: usize> SmallVec<T, N> {
    pub const MAX_STACK_SIZE: usize = 16 * 1024;
    /// Creates a new empty SmallVec.
    pub fn new() -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= SmallVec::<T, N>::MAX_STACK_SIZE,
                "SmallVec is too large! Reduce N."
            );
        }
        Self {
            on_stack: true,
            data: VecData {
                stack: ManuallyDrop::new(heapless::Vec::new()),
            },
        }
    }

    /// Creates a SmallVec with a specific capacity on the heap immediately if required.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= N {
            Self::new()
        } else {
            Self {
                on_stack: false,
                data: VecData {
                    heap: ManuallyDrop::new(Vec::with_capacity(capacity)),
                },
            }
        }
    }

    // --- Inspection ---

    #[inline]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        unsafe {
            if self.on_stack {
                self.data.stack.len()
            } else {
                self.data.heap.len()
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        unsafe {
            if self.on_stack {
                N
            } else {
                self.data.heap.capacity()
            }
        }
    }

    // --- Modification ---

    /// Reserves capacity for at least `additional` more elements.
    /// Spills to heap if stack capacity is insufficient.
    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            if self.on_stack {
                if self.data.stack.len() + additional > N {
                    self.spill_to_heap();
                    (*self.data.heap).reserve(additional);
                }
            } else {
                (*self.data.heap).reserve(additional);
            }
        }
    }

    /// Appends an element to the back of a collection.
    pub fn push(&mut self, item: T) {
        unsafe {
            if self.on_stack {
                // Check if Stack is full
                if self.data.stack.len() == N {
                    self.spill_to_heap();
                    // Fallthrough: execution continues to the heap push at the bottom
                } else {
                    // Stack has space. Attempt push.
                    match (*self.data.stack).push(item) {
                        Ok(()) => return, // Success: exit early
                        Err(_) => unreachable!("Stack capacity check failed in push"),
                    }
                }
            }

            // We reach here if:
            // 1. We were already on the heap.
            // 2. We were on the stack, it was full, and we just spilled to heap.
            (*self.data.heap).push(item);
        }
    }

    /// Removes the last element from a vector and returns it, or None if it is empty.
    pub fn pop(&mut self) -> Option<T> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).pop()
            } else {
                (*self.data.heap).pop()
            }
        }
    }

    /// Inserts an element at position `index`.
    /// Panics if `index > len`.
    pub fn insert(&mut self, index: usize, element: T) {
        let len = self.len();
        assert!(
            index <= len,
            "insertion index (is {}) should be <= len (is {})",
            index,
            len
        );

        unsafe {
            if self.on_stack {
                // Check if we need to spill before inserting
                if len == N {
                    self.spill_to_heap();
                    (*self.data.heap).insert(index, element);
                } else {
                    // heapless::Vec::insert returns Result, we unwrap because we checked capacity
                    (*self.data.stack).insert(index, element).ok().unwrap();
                }
            } else {
                (*self.data.heap).insert(index, element);
            }
        }
    }

    /// Removes and returns the element at position `index`.
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(
            index < len,
            "removal index (is {}) should be < len (is {})",
            index,
            len
        );
        unsafe {
            if self.on_stack {
                (*self.data.stack).remove(index)
            } else {
                (*self.data.heap).remove(index)
            }
        }
    }

    /// Removes an element from the vector and returns it.
    /// The removed element is replaced by the last element of the vector.
    /// This is O(1).
    pub fn swap_remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(
            index < len,
            "swap_remove index (is {}) should be < len (is {})",
            index,
            len
        );
        unsafe {
            if self.on_stack {
                (*self.data.stack).swap_remove(index)
            } else {
                (*self.data.heap).swap_remove(index)
            }
        }
    }

    pub fn truncate(&mut self, len: usize) {
        unsafe {
            if self.on_stack {
                (*self.data.stack).truncate(len);
            } else {
                (*self.data.heap).truncate(len);
            }
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        // heapless::Vec does not have retain(), so we implement it manually for stack
        unsafe {
            if self.on_stack {
                let vec = &mut *self.data.stack;
                let mut del = 0;
                let len = vec.len();
                for i in 0..len {
                    if !f(&vec[i]) {
                        del += 1;
                    } else if del > 0 {
                        // Swap efficient move
                        let val = ptr::read(&vec[i]);
                        ptr::write(&mut vec[i - del], val);
                    }
                }
                if del > 0 {
                    vec.truncate(len - del);
                }
            } else {
                (*self.data.heap).retain(f);
            }
        }
    }

    pub fn shrink_to_fit(&mut self) {
        if !self.on_stack {
            unsafe { (*self.data.heap).shrink_to_fit() }
        }
    }

    /// Consumes the SmallVec and returns a standard Vec.
    pub fn into_vec(self) -> Vec<T> {
        let this = ManuallyDrop::new(self);
        unsafe {
            if this.on_stack {
                ptr::read(&*this.data.stack).into_iter().collect()
            } else {
                ptr::read(&*this.data.heap)
            }
        }
    }

    // --- Internal Helpers ---

    #[inline(never)]
    unsafe fn spill_to_heap(&mut self) {
        unsafe {
            let stack_vec = ptr::read(&*self.data.stack);
            // Default spill strategy: double the capacity to amortize costs
            let mut heap_vec = Vec::with_capacity(N * 2);
            heap_vec.extend(stack_vec.into_iter());
            ptr::write(&mut self.data.heap, ManuallyDrop::new(heap_vec));
            self.on_stack = false;
        }
    }
}

// --- Extended Functionality (Clone, Resize) ---

impl<T: Clone, const N: usize> SmallVec<T, N> {
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();
        if new_len > len {
            self.reserve(new_len - len);
            unsafe {
                if self.on_stack {
                    // heapless::Vec::resize returns Result<(), T>
                    match (*self.data.stack).resize(new_len, value) {
                        Ok(()) => {}
                        Err(_) => unreachable!("Stack capacity check failed in resize"),
                    }
                } else {
                    (*self.data.heap).resize(new_len, value);
                }
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Appends all elements in a slice to the `SmallVec`.
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.reserve(other.len());
        unsafe {
            if self.on_stack {
                // heapless::Vec::extend_from_slice returns Result<(), ()>
                match (*self.data.stack).extend_from_slice(other) {
                    Ok(()) => {}
                    Err(_) => unreachable!("Stack capacity check failed in extend_from_slice"),
                }
            } else {
                (*self.data.heap).extend_from_slice(other);
            }
        }
    }
}

impl<T, const N: usize> SmallVec<T, N> {
    // ... (keep existing new, push, pop, etc.) ...

    /// Extracts a slice containing the entire vector.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            if self.on_stack {
                // heapless::Vec has an inherent as_slice method
                self.data.stack.as_slice()
            } else {
                // std::vec::Vec has an inherent as_slice method
                self.data.heap.as_slice()
            }
        }
    }

    /// Extracts a mutable slice containing the entire vector.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            if self.on_stack {
                (*self.data.stack).as_mut_slice()
            } else {
                (*self.data.heap).as_mut_slice()
            }
        }
    }
}

// --- Trait Implementations ---

// 1. Deref / DerefMut (Slice access)
impl<T, const N: usize> Deref for SmallVec<T, N> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe {
            if self.on_stack {
                &self.data.stack
            } else {
                &self.data.heap
            }
        }
    }
}

impl<T, const N: usize> DerefMut for SmallVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            if self.on_stack {
                &mut *self.data.stack
            } else {
                &mut *self.data.heap
            }
        }
    }
}

// 2. Drop
impl<T, const N: usize> Drop for SmallVec<T, N> {
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

// 3. Clone
impl<T: Clone, const N: usize> Clone for SmallVec<T, N> {
    fn clone(&self) -> Self {
        unsafe {
            if self.on_stack {
                Self {
                    on_stack: true,
                    data: VecData {
                        stack: ManuallyDrop::new((*self.data.stack).clone()),
                    },
                }
            } else {
                Self {
                    on_stack: false,
                    data: VecData {
                        heap: ManuallyDrop::new((*self.data.heap).clone()),
                    },
                }
            }
        }
    }
}

// 4. Debug
impl<T: fmt::Debug, const N: usize> fmt::Debug for SmallVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

// 5. Default
impl<T, const N: usize> Default for SmallVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// 6. PartialEq / Eq
impl<T: PartialEq, const N: usize> PartialEq for SmallVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
}
impl<T: Eq, const N: usize> Eq for SmallVec<T, N> {}

// 7. Extend / FromIterator
impl<T, const N: usize> Extend<T> for SmallVec<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        self.reserve(lower);
        for i in iter {
            self.push(i);
        }
    }
}

impl<T, const N: usize> FromIterator<T> for SmallVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = SmallVec::new();
        vec.extend(iter);
        vec
    }
}

// --- Iterators ---

pub enum IntoIter<T, const N: usize> {
    Stack(heapless::Vec<T, N>), // heapless::Vec is an iterator in newer versions, or we use IntoIter
    Heap(std::vec::IntoIter<T>),
}

// Note: heapless::Vec itself is not an iterator, but it has an into_iter() that returns one.
// However, heapless::Vec::IntoIter struct is private/complex to name in some versions.
// We will wrap the heapless IntoIter.
pub struct SmallVecIntoIter<T, const N: usize> {
    iter: SmallVecIterEnum<T, N>,
}

enum SmallVecIterEnum<T, const N: usize> {
    Stack(heapless::vec::IntoIter<T, N, usize>),
    Heap(std::vec::IntoIter<T>),
}

impl<T, const N: usize> IntoIterator for SmallVec<T, N> {
    type Item = T;
    type IntoIter = SmallVecIntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let this = ManuallyDrop::new(self);
        unsafe {
            if this.on_stack {
                let stack_vec = ptr::read(&*this.data.stack);
                SmallVecIntoIter {
                    iter: SmallVecIterEnum::Stack(stack_vec.into_iter()),
                }
            } else {
                let heap_vec = ptr::read(&*this.data.heap);
                SmallVecIntoIter {
                    iter: SmallVecIterEnum::Heap(heap_vec.into_iter()),
                }
            }
        }
    }
}

impl<T, const N: usize> Iterator for SmallVecIntoIter<T, N> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.iter {
            SmallVecIterEnum::Stack(iter) => iter.next(),
            SmallVecIterEnum::Heap(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.iter {
            SmallVecIterEnum::Stack(iter) => iter.size_hint(),
            SmallVecIterEnum::Heap(iter) => iter.size_hint(),
        }
    }
}

impl<T, const N: usize> ExactSizeIterator for SmallVecIntoIter<T, N> {}

// --- Hash ---
impl<T: Hash, const N: usize> Hash for SmallVec<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the slice, same as std::Vec
        self.as_slice().hash(state);
    }
}

// --- PartialOrd (Ordering) ---
impl<T: PartialOrd, const N: usize> PartialOrd for SmallVec<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

// --- Ord (Strict Ordering) ---
impl<T: Ord, const N: usize> Ord for SmallVec<T, N> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

// --- AsRef ---
impl<T, const N: usize> AsRef<[T]> for SmallVec<T, N> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> AsMut<[T]> for SmallVec<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

// Allows generic usage like: std::io::Cursor::new(my_small_vec)
impl<T, const N: usize> AsRef<SmallVec<T, N>> for SmallVec<T, N> {
    fn as_ref(&self) -> &SmallVec<T, N> {
        self
    }
}

impl<T, const N: usize> SmallVec<T, N> {
    // --- Modification Interop ---

    /// Appends elements from any vector-like collection (Vec, Array, Slice).
    pub fn extend_from_any<V: AnyVec<T> + ?Sized>(&mut self, other: &V)
    where
        T: Clone,
    {
        self.extend_from_slice(other.as_slice());
    }

    // --- Comparison Interop ---

    /// Returns true if this SmallVec has the same elements as the other collection.
    pub fn eq_any<V: AnyVec<T> + ?Sized>(&self, other: &V) -> bool
    where
        T: PartialEq,
    {
        self.as_slice() == other.as_slice()
    }

    /// Compares this SmallVec with any other collection (Lexicographical order).
    pub fn cmp_any<V: AnyVec<T> + ?Sized>(&self, other: &V) -> Ordering
    where
        T: Ord,
    {
        self.as_slice().cmp(other.as_slice())
    }

    /// Returns true if this SmallVec starts with the contents of the other collection.
    pub fn starts_with_any<V: AnyVec<T> + ?Sized>(&self, other: &V) -> bool
    where
        T: PartialEq,
    {
        self.starts_with(other.as_slice())
    }

    /// Returns true if this SmallVec ends with the contents of the other collection.
    pub fn ends_with_any<V: AnyVec<T> + ?Sized>(&self, other: &V) -> bool
    where
        T: PartialEq,
    {
        self.ends_with(other.as_slice())
    }

    /// Returns true if `other` is a subsequence of this SmallVec.
    /// (Equivalent to std string `contains` but for vectors)
    pub fn contains_subsequence<V: AnyVec<T> + ?Sized>(&self, other: &V) -> bool
    where
        T: PartialEq,
    {
        let other_slice = other.as_slice();
        if other_slice.is_empty() {
            return true;
        }
        if other_slice.len() > self.len() {
            return false;
        }

        self.windows(other_slice.len())
            .any(|window| window == other_slice)
    }
}

// --- Test Suite ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_stack_push_pop_basic() {
        let mut vec: SmallVec<i32, 4> = SmallVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert!(vec.is_on_stack());
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1);
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_vec_spill_trigger_on_push() {
        let mut vec: SmallVec<i32, 2> = SmallVec::new();
        vec.push(1);
        vec.push(2);
        assert!(vec.is_on_stack());

        // This should trigger spill
        vec.push(3);
        assert!(!vec.is_on_stack());
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[2], 3);

        // Capacity should double (N * 2 = 4)
        assert!(vec.capacity() >= 4);
    }

    #[test]
    fn test_vec_spill_trigger_on_insert() {
        let mut vec: SmallVec<i32, 2> = SmallVec::new();
        vec.push(1);
        vec.push(3);

        // Insert at 1, should move to heap because capacity 2 is full
        vec.insert(1, 2);

        assert!(!vec.is_on_stack());
        assert_eq!(vec.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_vec_stack_insert_remove_swap() {
        let mut vec: SmallVec<i32, 4> = SmallVec::from_iter([10, 20, 30]);

        // Insert (Stack)
        vec.insert(1, 15); // [10, 15, 20, 30]
        assert_eq!(vec[1], 15);

        // Remove
        let removed = vec.remove(2); // Remove 20
        assert_eq!(removed, 20);
        assert_eq!(vec.as_slice(), &[10, 15, 30]);

        // Swap Remove
        vec.push(40); // [10, 15, 30, 40]
        let swapped = vec.swap_remove(0); // Replace 10 with 40
        assert_eq!(swapped, 10);
        assert_eq!(vec.as_slice(), &[40, 15, 30]);
    }

    #[test]
    fn test_vec_any_storage_retain() {
        let mut vec: SmallVec<i32, 8> = SmallVec::from_iter(0..10); // 10 items > 8, so Heap
        assert!(!vec.is_on_stack());

        vec.retain(|&x| x % 2 == 0);
        assert_eq!(vec.as_slice(), &[0, 2, 4, 6, 8]);

        // Test retain on stack
        let mut vec_stack: SmallVec<i32, 8> = SmallVec::from_iter(0..6);
        assert!(vec_stack.is_on_stack());
        vec_stack.retain(|&x| x % 2 != 0);
        assert_eq!(vec_stack.as_slice(), &[1, 3, 5]);
    }

    #[test]
    fn test_vec_any_storage_resize_clone() {
        let mut vec: SmallVec<i32, 4> = SmallVec::new();
        vec.resize(2, 0); // [0, 0]
        assert!(vec.is_on_stack());

        let vec2 = vec.clone();
        assert_eq!(vec, vec2);

        vec.resize(10, 5); // Spills, fills with 5
        assert!(!vec.is_on_stack());
        assert_eq!(vec.len(), 10);
        assert_eq!(vec[9], 5);
    }

    #[test]
    fn test_vec_traits_into_iter_basic() {
        let vec: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3]);
        let collected: Vec<i32> = vec.into_iter().map(|x| x * 2).collect();
        assert_eq!(collected, vec![2, 4, 6]);
    }

    #[test]
    fn test_vec_any_storage_drop_behavior() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let counter = Rc::new(RefCell::new(0));

        struct Dropper(Rc<RefCell<i32>>);
        impl Drop for Dropper {
            fn drop(&mut self) {
                *self.0.borrow_mut() += 1;
            }
        }

        {
            let mut vec: SmallVec<Dropper, 2> = SmallVec::new();
            vec.push(Dropper(counter.clone()));
            vec.push(Dropper(counter.clone())); // On Stack
        }
        assert_eq!(*counter.borrow(), 2);

        *counter.borrow_mut() = 0;

        {
            let mut vec: SmallVec<Dropper, 2> = SmallVec::new();
            vec.push(Dropper(counter.clone()));
            vec.push(Dropper(counter.clone()));
            vec.push(Dropper(counter.clone())); // Spill to Heap
        }
        assert_eq!(*counter.borrow(), 3);
    }

    #[test]
    fn test_vec_traits_anyvec_inspection() {
        let sv: SmallVec<i32, 4> = SmallVec::from_iter([10, 20, 30]);

        // Using AnyVec trait methods directly
        assert_eq!(sv.len(), 3);
        assert!(!sv.is_empty());
        assert_eq!(sv.as_slice().first(), Some(&10));
        assert_eq!(sv.as_slice().last(), Some(&30));
        assert_eq!(sv.get(1), Some(&20));

        // Iteration via AnyVec
        let sum: i32 = sv.iter().sum();
        assert_eq!(sum, 60);
    }

    #[test]
    fn test_vec_traits_interop_comparison() {
        let sv: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3]);

        // Compare against std::Vec
        let std_vec = vec![1, 2, 3];
        assert!(sv.eq_any(&std_vec));
        assert_eq!(sv.cmp_any(&std_vec), std::cmp::Ordering::Equal);

        // Compare against Array
        let arr = [1, 2, 4]; // Greater
        assert!(!sv.eq_any(&arr));
        assert_eq!(sv.cmp_any(&arr), std::cmp::Ordering::Less);

        // Compare against Slice
        let slice = &[1, 2][..];
        assert_eq!(sv.cmp_any(slice), std::cmp::Ordering::Greater);
    }

    #[test]
    fn test_vec_traits_interop_searching() {
        let sv: SmallVec<i32, 8> = SmallVec::from_iter([1, 2, 3, 4, 5]);

        // starts_with_any
        let prefix_vec = vec![1, 2];
        assert!(sv.starts_with_any(&prefix_vec));

        // ends_with_any
        let suffix_arr = [4, 5];
        assert!(sv.ends_with_any(&suffix_arr));

        // contains_subsequence (find [3, 4] inside [1, 2, 3, 4, 5])
        let sub_vec = vec![3, 4];
        assert!(sv.contains_subsequence(&sub_vec));

        let not_sub = [3, 5];
        assert!(!sv.contains_subsequence(&not_sub));
    }

    #[test]
    fn test_vec_traits_generic_function_usage() {
        // This function works on ANY vector type
        fn sum_elements<V: AnyVec<i32> + ?Sized>(v: &V) -> i32 {
            v.iter().sum()
        }

        let sv: SmallVec<i32, 4> = SmallVec::from_iter([1, 2]);
        let v = vec![3, 4];
        let arr = [5, 6];

        assert_eq!(sum_elements(&sv), 3);
        assert_eq!(sum_elements(&v), 7);
        assert_eq!(sum_elements(&arr), 11);
    }

    #[test]
    fn test_vec_any_storage_with_capacity() {
        let v: SmallVec<i32, 4> = SmallVec::with_capacity(2);
        assert!(v.is_on_stack());

        let v2: SmallVec<i32, 4> = SmallVec::with_capacity(10);
        assert!(!v2.is_on_stack());
        assert!(v2.capacity() >= 10);
    }

    #[test]
    fn test_vec_any_storage_shrink_to_fit() {
        let mut v: SmallVec<i32, 4> = SmallVec::with_capacity(10);
        v.push(1);
        v.shrink_to_fit();
        assert!(v.capacity() >= 1);
        // On stack, shrink_to_fit is a no-op as capacity is fixed N
        let mut v_stack: SmallVec<i32, 4> = SmallVec::new();
        v_stack.push(1);
        v_stack.shrink_to_fit();
        assert_eq!(v_stack.capacity(), 4);
    }

    #[test]
    fn test_vec_any_storage_extend_from_slice() {
        let mut v: SmallVec<i32, 4> = SmallVec::new();
        v.extend_from_slice(&[1, 2]);
        assert!(v.is_on_stack());
        assert_eq!(v.as_slice(), &[1, 2]);

        v.extend_from_slice(&[3, 4, 5]); // Spill
        assert!(!v.is_on_stack());
        assert_eq!(v.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_vec_any_storage_clear_and_empty() {
        let mut v: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3, 4, 5]);
        assert!(!v.is_on_stack());
        v.clear();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);

        let mut v_stack: SmallVec<i32, 4> = SmallVec::from_iter([1, 2]);
        v_stack.clear();
        assert!(v_stack.is_empty());
    }

    #[test]
    fn test_vec_traits_partial_eq_cross_storage() {
        let v1: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3]);
        let v2: SmallVec<i32, 2> = SmallVec::from_iter([1, 2, 3]); // v2 is on heap
        assert!(v1.is_on_stack());
        assert!(!v2.is_on_stack());
        // Compare as slices since PartialEq requires same N
        assert_eq!(&v1[..], &v2[..]);
    }

    #[test]
    fn test_vec_any_storage_as_mut_and_deref_mut() {
        let mut v: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3]);
        v[0] = 10;
        assert_eq!(v[0], 10);

        let slice = v.as_mut_slice();
        slice[1] = 20;
        assert_eq!(v[1], 20);

        let mut v2: SmallVec<i32, 2> = SmallVec::from_iter([1, 2, 3]); // Heap
        v2[2] = 30;
        assert_eq!(v2[2], 30);
    }

    #[test]
    #[should_panic(expected = "insertion index")]
    fn test_vec_any_storage_insert_panic() {
        let mut v: SmallVec<i32, 4> = SmallVec::new();
        v.insert(1, 10);
    }

    #[test]
    #[should_panic(expected = "removal index")]
    fn test_vec_any_storage_remove_panic() {
        let mut v: SmallVec<i32, 4> = SmallVec::from_iter([1]);
        v.remove(1);
    }

    #[test]
    #[should_panic(expected = "swap_remove index")]
    fn test_vec_any_storage_swap_remove_panic() {
        let mut v: SmallVec<i32, 4> = SmallVec::from_iter([1]);
        v.swap_remove(1);
    }

    #[test]
    fn test_vec_traits_exhaustive() {
        let mut v: SmallVec<i32, 4> = SmallVec::from_iter([3, 1, 2]);

        // Hash
        let mut s = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut s);
        let _ = s.finish();

        // PartialOrd / Ord
        let v2 = SmallVec::<i32, 4>::from_iter([1, 2, 3]);
        assert!(v > v2);
        assert_eq!(v.cmp(&v2), std::cmp::Ordering::Greater);

        // AsRef / AsMut
        let _: &[i32] = v.as_ref();
        let _: &mut [i32] = v.as_mut();

        // Default
        let def: SmallVec<i32, 4> = SmallVec::default();
        assert!(def.is_empty());

        // AnyVec methods explicitly
        let any_v: &dyn AnyVec<i32> = &v;
        assert_eq!(any_v.len(), 3);
        assert!(!any_v.is_empty());
        assert_eq!(any_v.get(0), Some(&3));
        assert!(any_v.contains(&2));
        assert_eq!(any_v.iter().count(), 3);
    }

    #[test]
    fn test_vec_any_storage_into_vec_move() {
        let v: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3]);
        let std_v = v.into_vec();
        assert_eq!(std_v, vec![1, 2, 3]);

        let mut v_heap: SmallVec<i32, 2> = SmallVec::new();
        v_heap.push(1);
        v_heap.push(2);
        v_heap.push(3);
        assert!(!v_heap.is_on_stack());
        let std_v2 = v_heap.into_vec();
        assert_eq!(std_v2, vec![1, 2, 3]);
    }

    #[test]
    fn test_vec_any_storage_gap_coverage() {
        let mut v: SmallVec<i32, 2> = vec![1, 2, 3].into_iter().collect();
        assert!(!v.is_on_stack());

        // heap accessors
        let _ = v.as_ptr();
        let _ = v.as_mut_ptr();
        assert_eq!(v.get_mut(0), Some(&mut 1));

        // resize
        v.resize(5, 0);
        assert_eq!(v.len(), 5);

        let mut v_stack: SmallVec<i32, 2> = SmallVec::new();
        v_stack.resize(4, 9); // Spill via resize
        assert!(!v_stack.is_on_stack());
        assert_eq!(v_stack[0], 9);

        // truncate stack
        let mut s: SmallVec<i32, 4> = vec![1, 2, 3].into_iter().collect();
        s.truncate(1);
        assert_eq!(s.len(), 1);

        // clear heap
        v.clear();
        assert!(v.is_empty());

        // reserve heap
        v.reserve(10);
        assert!(v.capacity() >= 10);

        // iter traits
        let mut it = v_stack.iter();
        let _ = it.size_hint();
        assert_eq!(it.next(), Some(&9));

        // Clone/Debug heap
        let v_heap: SmallVec<i32, 1> = vec![1, 2].into_iter().collect();
        let cloned = v_heap.clone();
        assert_eq!(cloned.len(), 2);
        let debug = format!("{:?}", v_heap);
        assert!(debug.contains("1"));

        // Default
        let def: SmallVec<i32, 4> = SmallVec::default();
        assert!(def.is_empty());
    }

    #[test]
    fn test_vec_any_storage_gap_coverage_v2() {
        // AnyVec for slice
        let s_slice: &[i32] = &[1, 2];
        assert_eq!(s_slice.len(), 2);

        // heap pop
        let mut v: SmallVec<i32, 1> = vec![1, 2].into_iter().collect();
        assert_eq!(v.pop(), Some(2));

        // heap insert/remove/swap_remove
        v.insert(1, 10);
        assert_eq!(v.remove(1), 10);
        v.push(20);
        assert_eq!(v.swap_remove(0), 1);
        assert_eq!(v[0], 20);

        // heap resize/extend_from_slice
        let mut v2: SmallVec<i32, 1> = SmallVec::new();
        v2.push(1);
        v2.resize(0, 0); // truncate branch
        v2.extend_from_slice(&[1, 2, 3]); // heap branch
        assert_eq!(v2.len(), 3);

        // as_mut_slice heap
        v2.as_mut_slice()[0] = 100;

        // cmp_any / starts_with_any etc
        assert!(v2.eq_any(&v2));
        assert!(v2.starts_with_any(&[100, 2]));
        assert!(v2.ends_with_any(&[2, 3]));
        assert!(v2.contains_subsequence(&[2]));
        assert_eq!(v2.cmp_any(&[100, 2, 3]), Ordering::Equal);
    }
}
