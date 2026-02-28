use core::mem::ManuallyDrop;
use core::ptr;
use std::collections::VecDeque;
use std::fmt;

/// A trait for abstraction over different double-ended queue types (Stack, Heap, Small).
pub trait AnyDeque<T> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn push_back(&mut self, item: T);
    fn push_front(&mut self, item: T);
    fn pop_back(&mut self) -> Option<T>;
    fn pop_front(&mut self) -> Option<T>;
    fn clear(&mut self);
    fn front(&self) -> Option<&T>;
    fn back(&self) -> Option<&T>;
    fn front_mut(&mut self) -> Option<&mut T>;
    fn back_mut(&mut self) -> Option<&mut T>;
}

impl<T> AnyDeque<T> for VecDeque<T> {
    fn len(&self) -> usize {
        self.len()
    }
    fn push_back(&mut self, item: T) {
        self.push_back(item);
    }
    fn push_front(&mut self, item: T) {
        self.push_front(item);
    }
    fn pop_back(&mut self) -> Option<T> {
        self.pop_back()
    }
    fn pop_front(&mut self) -> Option<T> {
        self.pop_front()
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn front(&self) -> Option<&T> {
        self.front()
    }
    fn back(&self) -> Option<&T> {
        self.back()
    }
    fn front_mut(&mut self) -> Option<&mut T> {
        self.front_mut()
    }
    fn back_mut(&mut self) -> Option<&mut T> {
        self.back_mut()
    }
}

/// A double-ended queue that lives on the stack for `N` items, then spills to the heap.
///
/// # Overview
/// This collection uses a `heapless::Deque` (ring-buffer) for stack storage and a
/// `std::collections::VecDeque` (ring-buffer) for heap storage.
///
/// # Invariants
/// * `on_stack` tag determines which side of the `DequeData` union is active.
/// * `N` must be a power of two due to `heapless` implementation details.
pub struct SmallDeque<T, const N: usize> {
    on_stack: bool,
    data: DequeData<T, N>,
}

impl<T, const N: usize> AnyDeque<T> for SmallDeque<T, N> {
    fn len(&self) -> usize {
        self.len()
    }
    fn push_back(&mut self, item: T) {
        self.push_back(item);
    }
    fn push_front(&mut self, item: T) {
        self.push_front(item);
    }
    fn pop_back(&mut self) -> Option<T> {
        self.pop_back()
    }
    fn pop_front(&mut self) -> Option<T> {
        self.pop_front()
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn front(&self) -> Option<&T> {
        self.front()
    }
    fn back(&self) -> Option<&T> {
        self.back()
    }
    fn front_mut(&mut self) -> Option<&mut T> {
        self.front_mut()
    }
    fn back_mut(&mut self) -> Option<&mut T> {
        self.back_mut()
    }
}

/// The internal storage for `SmallDeque`.
///
/// We use `ManuallyDrop` because the compiler cannot know which field is active
/// and therefore cannot automatically drop the correct one.
union DequeData<T, const N: usize> {
    stack: ManuallyDrop<heapless::Deque<T, N>>,
    heap: ManuallyDrop<VecDeque<T>>,
}

impl<T, const N: usize> SmallDeque<T, N> {
    /// Creates a new empty SmallDeque.
    pub fn new() -> Self {
        const {
            assert!(N.is_power_of_two(), "SmallDeque N must be a power of two");
        }

        Self {
            on_stack: true,
            data: DequeData {
                stack: ManuallyDrop::new(heapless::Deque::new()),
            },
        }
    }

    /// Creates a SmallDeque with a specific capacity on the heap immediately if required.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= N {
            Self::new()
        } else {
            Self {
                on_stack: false,
                data: DequeData {
                    heap: ManuallyDrop::new(VecDeque::with_capacity(capacity)),
                },
            }
        }
    }

    // --- Inspection ---

    #[inline]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }

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

    // --- Access ---

    pub fn front(&self) -> Option<&T> {
        unsafe {
            if self.on_stack {
                self.data.stack.front()
            } else {
                self.data.heap.front()
            }
        }
    }

    pub fn front_mut(&mut self) -> Option<&mut T> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).front_mut()
            } else {
                (*self.data.heap).front_mut()
            }
        }
    }

    pub fn back(&self) -> Option<&T> {
        unsafe {
            if self.on_stack {
                self.data.stack.back()
            } else {
                self.data.heap.back()
            }
        }
    }

    pub fn back_mut(&mut self) -> Option<&mut T> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).back_mut()
            } else {
                (*self.data.heap).back_mut()
            }
        }
    }

    /// Returns a reference to the element at the given index.
    pub fn get(&self, index: usize) -> Option<&T> {
        let (s1, s2) = self.as_slices();
        if index < s1.len() {
            Some(&s1[index])
        } else if index < s1.len() + s2.len() {
            Some(&s2[index - s1.len()])
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let (s1, s2) = self.as_mut_slices();
        let s1_len = s1.len();
        let s2_len = s2.len();
        if index < s1_len {
            Some(&mut s1[index])
        } else if index < s1_len + s2_len {
            Some(&mut s2[index - s1_len])
        } else {
            None
        }
    }

    // --- Modification ---

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

    pub fn push_back(&mut self, item: T) {
        unsafe {
            if self.on_stack {
                if self.data.stack.len() == N {
                    self.spill_to_heap();
                    // Fallthrough to heap
                } else {
                    match (*self.data.stack).push_back(item) {
                        Ok(()) => return,
                        Err(_) => unreachable!("Stack capacity check failed in push_back"),
                    }
                }
            }
            (*self.data.heap).push_back(item);
        }
    }

    pub fn push_front(&mut self, item: T) {
        unsafe {
            if self.on_stack {
                if self.data.stack.len() == N {
                    self.spill_to_heap();
                    // Fallthrough to heap
                } else {
                    match (*self.data.stack).push_front(item) {
                        Ok(()) => return,
                        Err(_) => unreachable!("Stack capacity check failed in push_front"),
                    }
                }
            }
            (*self.data.heap).push_front(item);
        }
    }

    pub fn pop_front(&mut self) -> Option<T> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).pop_front()
            } else {
                (*self.data.heap).pop_front()
            }
        }
    }

    pub fn pop_back(&mut self) -> Option<T> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).pop_back()
            } else {
                (*self.data.heap).pop_back()
            }
        }
    }

    pub fn swap_remove_front(&mut self, index: usize) -> Option<T> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).swap_remove_front(index)
            } else {
                (*self.data.heap).swap_remove_front(index)
            }
        }
    }

    pub fn swap_remove_back(&mut self, index: usize) -> Option<T> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).swap_remove_back(index)
            } else {
                (*self.data.heap).swap_remove_back(index)
            }
        }
    }

    pub fn insert(&mut self, index: usize, item: T) {
        let len = self.len();
        assert!(index <= len, "index out of bounds");

        unsafe {
            if self.on_stack {
                // heapless::Deque does not support arbitrary insertion easily.
                // We spill to heap to support this API safely and completely.
                self.spill_to_heap();
                (*self.data.heap).insert(index, item);
            } else {
                (*self.data.heap).insert(index, item);
            }
        }
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        unsafe {
            if self.on_stack {
                // heapless::Deque does not support arbitrary removal easily.
                self.spill_to_heap();
                (*self.data.heap).remove(index)
            } else {
                (*self.data.heap).remove(index)
            }
        }
    }

    pub fn clear(&mut self) {
        unsafe {
            if self.on_stack {
                (*self.data.stack).clear();
            } else {
                (*self.data.heap).clear();
            }
        }
    }

    pub fn truncate(&mut self, len: usize) {
        unsafe {
            if self.on_stack {
                // heapless doesn't have truncate, implement via pop_back loop
                let current_len = (*self.data.stack).len();
                if len < current_len {
                    for _ in 0..(current_len - len) {
                        (*self.data.stack).pop_back();
                    }
                }
            } else {
                (*self.data.heap).truncate(len);
            }
        }
    }

    // --- Slices & Iteration ---

    /// Returns a pair of slices which contain the contents of the deque.
    pub fn as_slices(&self) -> (&[T], &[T]) {
        unsafe {
            if self.on_stack {
                self.data.stack.as_slices()
            } else {
                self.data.heap.as_slices()
            }
        }
    }

    /// Returns a pair of mutable slices which contain the contents of the deque.
    pub fn as_mut_slices(&mut self) -> (&mut [T], &mut [T]) {
        unsafe {
            if self.on_stack {
                (*self.data.stack).as_mut_slices()
            } else {
                (*self.data.heap).as_mut_slices()
            }
        }
    }

    /// Rearranges the internal storage so the deque is one contiguous slice.
    pub fn make_contiguous(&mut self) -> &mut [T] {
        unsafe {
            if self.on_stack {
                // heapless doesn't support make_contiguous directly.
                // Spill to heap to satisfy the API.
                self.spill_to_heap();
                (*self.data.heap).make_contiguous()
            } else {
                (*self.data.heap).make_contiguous()
            }
        }
    }

    // --- Rotations ---

    pub fn rotate_left(&mut self, mid: usize) {
        unsafe {
            if self.on_stack {
                self.spill_to_heap();
                (*self.data.heap).rotate_left(mid);
            } else {
                (*self.data.heap).rotate_left(mid);
            }
        }
    }

    pub fn rotate_right(&mut self, k: usize) {
        unsafe {
            if self.on_stack {
                self.spill_to_heap();
                (*self.data.heap).rotate_right(k);
            } else {
                (*self.data.heap).rotate_right(k);
            }
        }
    }

    // --- Internals ---

    #[inline(never)]
    unsafe fn spill_to_heap(&mut self) {
        unsafe {
            let stack_deque = ptr::read(&*self.data.stack);
            let mut heap_deque = VecDeque::with_capacity(N * 2);
            // heapless::Deque implements IntoIterator, handling ring-buffer unwrapping automatically
            heap_deque.extend(stack_deque.into_iter());
            ptr::write(&mut self.data.heap, ManuallyDrop::new(heap_deque));
            self.on_stack = false;
        }
    }
}

// --- Iterators ---

pub struct Iter<'a, T> {
    inner: IterEnum<'a, T>,
}

enum IterEnum<'a, T> {
    Stack(heapless::deque::Iter<'a, T>),
    Heap(std::collections::vec_deque::Iter<'a, T>),
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            IterEnum::Stack(i) => i.next(),
            IterEnum::Heap(i) => i.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            IterEnum::Stack(i) => i.size_hint(),
            IterEnum::Heap(i) => i.size_hint(),
        }
    }
}

impl<T, const N: usize> SmallDeque<T, N> {
    pub fn iter(&self) -> Iter<'_, T> {
        unsafe {
            if self.on_stack {
                Iter {
                    inner: IterEnum::Stack(self.data.stack.iter()),
                }
            } else {
                Iter {
                    inner: IterEnum::Heap(self.data.heap.iter()),
                }
            }
        }
    }
}

pub struct IntoIter<T, const N: usize> {
    deque: SmallDeque<T, N>,
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.deque.pop_front()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.deque.len();
        (len, Some(len))
    }
}

impl<T, const N: usize> IntoIterator for SmallDeque<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { deque: self }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a SmallDeque<T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// --- Traits ---

impl<T, const N: usize> Drop for SmallDeque<T, N> {
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

impl<T: Clone, const N: usize> Clone for SmallDeque<T, N> {
    fn clone(&self) -> Self {
        unsafe {
            if self.on_stack {
                Self {
                    on_stack: true,
                    data: DequeData {
                        stack: ManuallyDrop::new((*self.data.stack).clone()),
                    },
                }
            } else {
                Self {
                    on_stack: false,
                    data: DequeData {
                        heap: ManuallyDrop::new((*self.data.heap).clone()),
                    },
                }
            }
        }
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for SmallDeque<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, const N: usize> Default for SmallDeque<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialEq, const N: usize> PartialEq for SmallDeque<T, N> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq, const N: usize> Eq for SmallDeque<T, N> {}

impl<T, const N: usize> Extend<T> for SmallDeque<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        self.reserve(lower);
        for item in iter {
            self.push_back(item);
        }
    }
}

impl<T, const N: usize> FromIterator<T> for SmallDeque<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut deque = Self::new();
        deque.extend(iter);
        deque
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deque_stack_lifecycle_basic() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        d.push_back(1);
        d.push_back(2);
        d.push_front(0); // [0, 1, 2]

        assert!(d.is_on_stack());
        assert_eq!(d.len(), 3);
        assert_eq!(d.front(), Some(&0));
        assert_eq!(d.back(), Some(&2));

        assert_eq!(d.pop_front(), Some(0));
        assert_eq!(d.pop_back(), Some(2));
        assert_eq!(d.pop_back(), Some(1));
        assert!(d.is_empty());
    }

    #[test]
    fn test_deque_spill_trigger_on_push_back() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        // Fill stack
        d.push_back(1);
        d.push_back(2);
        d.push_back(3);
        d.push_back(4);
        assert!(d.is_on_stack());

        // Push 5th -> Spill
        d.push_back(5);
        assert!(!d.is_on_stack());

        // Check integrity after spill
        assert_eq!(d.len(), 5);
        let vec: Vec<_> = d.iter().cloned().collect();
        assert_eq!(vec, vec![1, 2, 3, 4, 5]);

        // Continue heap ops
        d.push_front(0);
        assert_eq!(d.front(), Some(&0));
    }

    #[test]
    fn test_deque_spill_trigger_on_push_front() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        for i in 0..4 {
            d.push_back(i);
        } // [0,1,2,3]

        // Spill via push_front
        d.push_front(99); // [99, 0, 1, 2, 3]
        assert!(!d.is_on_stack());
        assert_eq!(d.front(), Some(&99));
        assert_eq!(d.back(), Some(&3));
    }

    #[test]
    fn test_deque_spill_trigger_on_insert() {
        // Test that `insert` triggers spill because stack doesn't support random insert efficiently
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        d.push_back(10);
        d.push_back(20);
        assert!(d.is_on_stack());

        d.insert(1, 15); // Should spill for safety/simplicity in this impl
        assert!(!d.is_on_stack());
        assert_eq!(d.iter().cloned().collect::<Vec<_>>(), vec![10, 15, 20]);
    }

    #[test]
    fn test_deque_any_storage_as_slices() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        d.push_back(1);
        d.push_back(2);
        // Force wrap-around in ring buffer
        d.pop_front(); // remove 1
        d.push_back(3);
        d.push_back(4);
        d.push_back(5); // [2, 3, 4, 5] (internally may wrap)

        // Since we pushed 5th, it spilled to heap.
        // Let's test on stack strictly first:
        let mut stack_d: SmallDeque<i32, 8> = SmallDeque::new();
        stack_d.push_back(1);
        stack_d.push_back(2);
        stack_d.pop_front();
        stack_d.push_back(3); // [2, 3]

        let (s1, s2) = stack_d.as_slices();
        assert_eq!(s1.len() + s2.len(), 2);
    }

    #[test]
    fn test_deque_any_storage_drop_behavior() {
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
            let mut d: SmallDeque<Dropper, 2> = SmallDeque::new();
            d.push_back(Dropper(counter.clone())); // Stack
        }
        assert_eq!(*counter.borrow(), 1);

        *counter.borrow_mut() = 0;
        {
            let mut d: SmallDeque<Dropper, 2> = SmallDeque::new();
            d.push_back(Dropper(counter.clone()));
            d.push_back(Dropper(counter.clone()));
            d.push_back(Dropper(counter.clone())); // Spill
        }
        assert_eq!(*counter.borrow(), 3);
    }

    #[test]
    fn test_deque_any_storage_with_capacity() {
        let d: SmallDeque<i32, 4> = SmallDeque::with_capacity(2);
        assert!(d.is_on_stack());

        let d2: SmallDeque<i32, 4> = SmallDeque::with_capacity(10);
        assert!(!d2.is_on_stack());
        assert!(d2.capacity() >= 10);
    }

    #[test]
    fn test_deque_any_storage_clear_truncate() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::from_iter([1, 2, 3]);
        assert!(d.is_on_stack());
        d.truncate(1);
        assert_eq!(d.len(), 1);
        assert_eq!(d.front(), Some(&1));

        d.clear();
        assert!(d.is_empty());

        // On heap
        let mut d: SmallDeque<i32, 4> = SmallDeque::from_iter([1, 2, 3, 4, 5]);
        assert!(!d.is_on_stack());
        d.truncate(2);
        assert_eq!(d.len(), 2);
        d.clear();
        assert!(d.is_empty());
    }

    #[test]
    fn test_deque_any_storage_rotations() {
        let mut d: SmallDeque<i32, 8> = SmallDeque::from_iter([1, 2, 3, 4]);
        assert!(d.is_on_stack());
        d.rotate_left(1); // [2, 3, 4, 1]
        assert!(!d.is_on_stack()); // Rotation triggers spill
        assert_eq!(d.iter().copied().collect::<Vec<_>>(), vec![2, 3, 4, 1]);

        d.rotate_right(1); // [1, 2, 3, 4]
        assert_eq!(d.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_deque_any_storage_mut_accessors() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::from_iter([1, 2]);
        if let Some(front) = d.front_mut() {
            *front = 10;
        }
        if let Some(back) = d.back_mut() {
            *back = 20;
        }
        assert_eq!(d.as_slices().0, &[10, 20]);

        d.push_back(30);
        d.push_back(40);
        d.push_back(50); // Spill

        let slices = d.as_mut_slices();
        if !slices.0.is_empty() {
            slices.0[0] = 100;
        }
        assert_eq!(d.front(), Some(&100));
    }

    #[test]
    fn test_deque_traits_iterators() {
        let d: SmallDeque<i32, 4> = SmallDeque::from_iter([1, 2, 3]);
        let mut iter = d.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);

        let vec: Vec<_> = d.into_iter().collect();
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_deque_traits_interop() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        d.extend(vec![1, 2]);

        // Clone
        let cloned = d.clone();
        assert_eq!(cloned, d);

        // Debug
        let debug = format!("{:?}", d);
        assert!(debug.contains("1"));

        // Default
        let def: SmallDeque<i32, 4> = SmallDeque::default();
        assert!(def.is_empty());

        // PartialEq cross capacity
        let d2: SmallDeque<i32, 8> = SmallDeque::from_iter([1, 2]);
        assert_eq!(d.len(), d2.len());
        assert!(d.iter().zip(d2.iter()).all(|(a, b)| a == b));
    }

    #[test]
    fn test_deque_any_storage_swap_remove() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::from_iter([1, 2, 3]);
        assert_eq!(d.swap_remove_front(1), Some(2));
        assert_eq!(d.iter().copied().collect::<Vec<_>>(), vec![1, 3]);

        let mut d_heap: SmallDeque<i32, 2> = SmallDeque::from_iter([1, 2, 3]);
        assert_eq!(d_heap.swap_remove_back(1), Some(2));
        assert_eq!(d_heap.iter().copied().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn test_deque_any_storage_gap_coverage() {
        let mut d: SmallDeque<i32, 2> = SmallDeque::new();
        // pop empty
        assert_eq!(d.pop_front(), None);
        assert_eq!(d.pop_back(), None);

        // heap branches
        d.push_back(1);
        d.push_back(2);
        d.push_front(0); // Spill
        assert!(!d.is_on_stack());

        d.push_back(3); // push_back on heap
        d.push_front(-1); // push_front on heap
        assert_eq!(d.len(), 5);

        d.reserve(10); // reserve on heap
        d.truncate(3); // truncate on heap
        assert_eq!(d.len(), 3);

        d.rotate_left(1); // rotate on heap
        d.rotate_right(1);
        d.make_contiguous(); // make_contiguous on heap

        assert_eq!(d.remove(1), Some(0)); // remove on heap
        assert_eq!(d.len(), 2);

        // Iter size hints
        let (low, high) = d.iter().size_hint();
        assert_eq!(low, 2);
        assert_eq!(high, Some(2));

        let mut stack_d: SmallDeque<i32, 4> = SmallDeque::new();
        stack_d.push_back(1);
        let (s_low, s_high) = stack_d.iter().size_hint();
        assert_eq!(s_low, 1);
        assert_eq!(s_high, Some(1));

        // swap_remove_front/back error cases (bounds already checked by inner but let's hit them)
        assert_eq!(stack_d.swap_remove_front(10), None);
        assert_eq!(stack_d.swap_remove_back(10), None);
    }
}
