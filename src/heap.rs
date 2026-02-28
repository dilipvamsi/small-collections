use core::cmp::Reverse;
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use core::ptr;
use std::collections::BinaryHeap;
use std::fmt::{Debug, Formatter, Result};

/// A trait for abstraction over different priority queue types (Stack, Heap, Small).
pub trait AnyHeap<T> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn push(&mut self, item: T);
    fn pop(&mut self) -> Option<T>;
    fn peek(&self) -> Option<&T>;
    fn clear(&mut self);
}

impl<T: Ord> AnyHeap<T> for BinaryHeap<T> {
    fn len(&self) -> usize {
        self.len()
    }
    fn push(&mut self, item: T) {
        self.push(item);
    }
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn peek(&self) -> Option<&T> {
        self.peek()
    }
    fn clear(&mut self) {
        self.clear();
    }
}

use heapless::binary_heap::BinaryHeap as HeaplessHeap;
pub use heapless::binary_heap::{Kind, Max, Min};

/// Internal trait to bridge heapless::Kind and std::collections::BinaryHeap.
pub trait HeapKind<T: Ord>: Kind {
    type Element: Ord;
    fn wrap(item: T) -> Self::Element;
    fn unwrap(item: Self::Element) -> T;
    fn wrap_ref(item: &T) -> &Self::Element;
    fn unwrap_ref(item: &Self::Element) -> &T;
}

impl<T: Ord> HeapKind<T> for Max {
    type Element = T;
    #[inline]
    fn wrap(item: T) -> T {
        item
    }
    #[inline]
    fn unwrap(item: T) -> T {
        item
    }
    #[inline]
    fn wrap_ref(item: &T) -> &T {
        item
    }
    #[inline]
    fn unwrap_ref(item: &T) -> &T {
        item
    }
}

impl<T: Ord> HeapKind<T> for Min {
    type Element = Reverse<T>;
    #[inline]
    fn wrap(item: T) -> Reverse<T> {
        Reverse(item)
    }
    #[inline]
    fn unwrap(item: Reverse<T>) -> T {
        item.0
    }
    #[inline]
    fn wrap_ref(item: &T) -> &Reverse<T> {
        // Reverse is #[repr(transparent)]
        unsafe { &*(item as *const T as *const Reverse<T>) }
    }
    #[inline]
    fn unwrap_ref(item: &Reverse<T>) -> &T {
        &item.0
    }
}

/// A priority queue that lives on the stack for `N` items, then spills to the heap.
///
/// # Behavior
/// * **Heap Kind:** Supports both `Max` (default) and `Min` heap behavior.
/// * **Spill:** Occurs automatically when pushing the (N+1)th item.
/// * **Complexity:** Push and Pop are O(log n). Spill is O(n).
///
/// # Safety Invariants
/// * `on_stack` tag determines which side of the `HeapData` union is active.
/// * Elements are stored in a binary heap order (array representation).
pub struct SmallBinaryHeap<T: Ord, const N: usize, K: Kind + HeapKind<T> = Max> {
    on_stack: bool,
    data: HeapData<T, N, K>,
}

impl<T: Ord, const N: usize, K: Kind + HeapKind<T>> AnyHeap<T> for SmallBinaryHeap<T, N, K> {
    fn len(&self) -> usize {
        self.len()
    }
    fn push(&mut self, item: T) {
        self.push(item);
    }
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn peek(&self) -> Option<&T> {
        self.peek()
    }
    fn clear(&mut self) {
        self.clear();
    }
}

/// Internal storage for `SmallBinaryHeap`.
///
/// We use `ManuallyDrop` because the compiler cannot know which field is active
/// and therefore cannot automatically drop the correct one.
union HeapData<T: Ord, const N: usize, K: Kind + HeapKind<T>> {
    stack: ManuallyDrop<HeaplessHeap<T, K, N>>,
    heap: ManuallyDrop<BinaryHeap<K::Element>>,
}

impl<T, const N: usize, K> SmallBinaryHeap<T, N, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    pub const MAX_STACK_SIZE: usize = 16 * 1024;

    /// Creates a new empty binary heap.
    pub fn new() -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= SmallBinaryHeap::<T, N, K>::MAX_STACK_SIZE,
                "SmallBinaryHeap is too large! Reduce N."
            );
        }

        Self {
            on_stack: true,
            data: HeapData {
                stack: ManuallyDrop::new(HeaplessHeap::new()),
            },
        }
    }

    /// Creates a new empty binary heap with a specific initial capacity on the heap.
    /// This immediately forces the storage to be on the heap, bypassing the stack optimization.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= N {
            Self::new()
        } else {
            Self {
                on_stack: false,
                data: HeapData {
                    heap: ManuallyDrop::new(BinaryHeap::with_capacity(capacity)),
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

    /// Returns the total capacity of the heap.
    /// If on stack, returns N. If on heap, returns the vector capacity.
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

    /// Reserves capacity for at least `additional` more elements to be inserted.
    /// May trigger a spill to heap if current stack capacity is insufficient.
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

    /// Discards as much additional capacity as possible.
    pub fn shrink_to_fit(&mut self) {
        if !self.on_stack {
            unsafe { (*self.data.heap).shrink_to_fit() }
        }
    }

    /// Returns a reference to the greatest item.
    pub fn peek(&self) -> Option<&T> {
        unsafe {
            if self.on_stack {
                self.data.stack.peek()
            } else {
                self.data.heap.peek().map(K::unwrap_ref)
            }
        }
    }

    /// Returns a mutable reference to the greatest item.
    pub fn peek_mut(&mut self) -> Option<SmallPeekMut<'_, T, N, K>> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).peek_mut().map(SmallPeekMut::Stack)
            } else {
                (*self.data.heap).peek_mut().map(SmallPeekMut::Heap)
            }
        }
    }

    pub fn push(&mut self, item: T) {
        unsafe {
            if self.on_stack {
                let stack_heap = &mut *self.data.stack;
                if stack_heap.len() == N {
                    self.spill_to_heap();
                    // Fallthrough to heap push
                } else {
                    stack_heap.push(item).ok().unwrap();
                    return;
                }
            }
            (*self.data.heap).push(K::wrap(item));
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).pop()
            } else {
                (*self.data.heap).pop().map(K::unwrap)
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

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut Self) {
        if other.is_empty() {
            return;
        }

        self.reserve(other.len());

        match (self.on_stack, other.on_stack) {
            (false, false) => unsafe {
                // Both on heap, but we can't easily merge two BinaryHeaps of different types
                // unless we drain.
                (*self.data.heap).extend((*other.data.heap).drain());
            },
            (false, true) => {
                while let Some(item) = other.pop() {
                    unsafe {
                        (*self.data.heap).push(K::wrap(item));
                    }
                }
            }
            (true, _) => {
                // If we are on stack, we just reserved, so we know we fit or spilled.
                while let Some(item) = other.pop() {
                    self.push(item);
                }
            }
        }
    }

    /// Retains only the elements specified by the predicate.
    /// Rebuilds the heap, complexity O(n).
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let items = self.drain().collect::<Vec<_>>();
        self.clear(); // Reset state
        for item in items {
            if f(&item) {
                self.push(item);
            }
        }
    }

    // --- Consumption ---

    /// Consumes the heap and returns the underlying vector in arbitrary (heap) order.
    pub fn into_vec(self) -> Vec<T> {
        let this = ManuallyDrop::new(self);
        unsafe {
            if this.on_stack {
                ptr::read(&*this.data.stack)
                    .into_vec()
                    .into_iter()
                    .collect()
            } else {
                ptr::read(&*this.data.heap)
                    .into_vec()
                    .into_iter()
                    .map(K::unwrap)
                    .collect()
            }
        }
    }

    /// Consumes the heap and returns a vector sorted from greatest to lowest.
    pub fn into_sorted_vec(mut self) -> Vec<T> {
        let mut vec = Vec::with_capacity(self.len());
        while let Some(item) = self.pop() {
            vec.push(item);
        }
        vec
    }

    // --- Internals ---

    #[inline(never)]
    unsafe fn spill_to_heap(&mut self) {
        unsafe {
            let stack_heap = ptr::read(&*self.data.stack);
            let std_heap = BinaryHeap::from_iter(stack_heap.into_vec().into_iter().map(K::wrap));
            ptr::write(&mut self.data.heap, ManuallyDrop::new(std_heap));
            self.on_stack = false;
        }
    }

    /// Helper for append/retain: drains all elements.
    fn drain(&mut self) -> IntoIter<T, N, K> {
        let old_self = std::mem::replace(self, Self::new());
        old_self.into_iter()
    }
}

// --- Wrapper Types ---

pub enum SmallPeekMut<'a, T, const N: usize, K: Kind + HeapKind<T>>
where
    T: Ord,
{
    Stack(heapless::binary_heap::PeekMut<'a, T, K, N>),
    Heap(std::collections::binary_heap::PeekMut<'a, K::Element>),
}

impl<'a, T, const N: usize, K> Deref for SmallPeekMut<'a, T, N, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        match self {
            SmallPeekMut::Stack(p) => p,
            SmallPeekMut::Heap(p) => K::unwrap_ref(p),
        }
    }
}

impl<'a, T, const N: usize, K> DerefMut for SmallPeekMut<'a, T, N, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            SmallPeekMut::Stack(p) => p,
            SmallPeekMut::Heap(p) => {
                // This is slightly incorrect for Min-Heap because mutating might violate the heap property
                // of std::collections::BinaryHeap<Reverse<T>> if we don't handle it carefully.
                // However, PeekMut's Drop handle's the re-heapification.
                // We need to return a mutable reference to the inner T.
                // BinaryHeap<Reverse<T>>::peek_mut() returns a PeekMut<Reverse<T>>.
                // DerefMut on that returns &mut Reverse<T>.
                // We need &mut T.
                let rev_mut: &mut K::Element = &mut **p;
                let t_mut: &mut T = unsafe { &mut *(rev_mut as *mut K::Element as *mut T) };
                t_mut
            }
        }
    }
}

impl<'a, T, const N: usize, K> SmallPeekMut<'a, T, N, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    pub fn pop(self) -> T {
        match self {
            SmallPeekMut::Stack(p) => heapless::binary_heap::PeekMut::pop(p),
            SmallPeekMut::Heap(p) => K::unwrap(std::collections::binary_heap::PeekMut::pop(p)),
        }
    }
}

// --- Iterators ---

pub struct IntoIter<T, const N: usize, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    heap: SmallBinaryHeap<T, N, K>,
}

impl<T, const N: usize, K> Iterator for IntoIter<T, N, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.heap.pop()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.heap.len();
        (len, Some(len))
    }
}

impl<T, const N: usize, K> IntoIterator for SmallBinaryHeap<T, N, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    type Item = T;
    type IntoIter = IntoIter<T, N, K>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { heap: self }
    }
}

impl<'a, T, const N: usize, K> IntoIterator for &'a SmallBinaryHeap<T, N, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    type Item = &'a T;
    type IntoIter = SmallHeapIter<'a, T, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub enum SmallHeapIter<'a, T, K: Kind + HeapKind<T>>
where
    T: Ord,
{
    Stack(core::slice::Iter<'a, T>),
    Heap(std::collections::binary_heap::Iter<'a, K::Element>),
}

impl<'a, T, K> Iterator for SmallHeapIter<'a, T, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SmallHeapIter::Stack(iter) => iter.next(),
            SmallHeapIter::Heap(iter) => iter.next().map(K::unwrap_ref),
        }
    }
}

impl<T, const N: usize, K> SmallBinaryHeap<T, N, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    pub fn iter(&self) -> SmallHeapIter<'_, T, K> {
        unsafe {
            if self.on_stack {
                SmallHeapIter::Stack(self.data.stack.iter())
            } else {
                SmallHeapIter::Heap(self.data.heap.iter())
            }
        }
    }
}

// --- Trait Implementations ---

impl<T: Ord, const N: usize, K: Kind + HeapKind<T>> Drop for SmallBinaryHeap<T, N, K> {
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

impl<T: Ord, const N: usize, K: Kind + HeapKind<T>> Default for SmallBinaryHeap<T, N, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize, K> Clone for SmallBinaryHeap<T, N, K>
where
    T: Ord + Clone,
    K: Kind + HeapKind<T>,
    K::Element: Clone,
{
    fn clone(&self) -> Self {
        unsafe {
            if self.on_stack {
                Self {
                    on_stack: true,
                    data: HeapData {
                        stack: ManuallyDrop::new((*self.data.stack).clone()),
                    },
                }
            } else {
                Self {
                    on_stack: false,
                    data: HeapData {
                        heap: ManuallyDrop::new((*self.data.heap).clone()),
                    },
                }
            }
        }
    }
}

impl<T, const N: usize, K> Debug for SmallBinaryHeap<T, N, K>
where
    T: Ord + Debug,
    K: Kind + HeapKind<T>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, const N: usize, K> Extend<T> for SmallBinaryHeap<T, N, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        self.reserve(lower);
        for elem in iter {
            self.push(elem);
        }
    }
}

impl<T, const N: usize, K> FromIterator<T> for SmallBinaryHeap<T, N, K>
where
    T: Ord,
    K: Kind + HeapKind<T>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut heap = Self::new();
        heap.extend(iter);
        heap
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_stack_ops_basic() {
        let mut heap: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
        assert!(heap.is_empty());
        assert!(heap.is_on_stack());

        heap.push(1);
        heap.push(5);
        heap.push(2);

        assert_eq!(heap.len(), 3);
        assert_eq!(heap.peek(), Some(&5));

        // Pop Order
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_heap_stack_min_heap_ops() {
        let mut heap: SmallBinaryHeap<i32, 4, Min> = SmallBinaryHeap::new();
        heap.push(10);
        heap.push(5);
        heap.push(15);

        assert_eq!(heap.peek(), Some(&5));
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(10));
        assert_eq!(heap.pop(), Some(15));
    }

    #[test]
    fn test_heap_spill_trigger_max() {
        let mut heap: SmallBinaryHeap<i32, 2, Max> = SmallBinaryHeap::new();
        heap.push(10);
        heap.push(20);
        assert!(heap.is_on_stack());

        heap.push(30);
        assert!(!heap.is_on_stack());
        assert_eq!(heap.pop(), Some(30));
    }

    #[test]
    fn test_heap_spill_trigger_min() {
        let mut heap: SmallBinaryHeap<i32, 2, Min> = SmallBinaryHeap::new();
        heap.push(30);
        heap.push(20);
        assert!(heap.is_on_stack());

        heap.push(10);
        assert!(!heap.is_on_stack());
        assert_eq!(heap.peek(), Some(&10));
        assert_eq!(heap.pop(), Some(10));
        assert_eq!(heap.pop(), Some(20));
        assert_eq!(heap.pop(), Some(30));
    }

    #[test]
    fn test_heap_any_storage_peek_mut_reorder() {
        let mut heap: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
        heap.push(10);
        heap.push(50);
        heap.push(20);

        if let Some(mut top) = heap.peek_mut() {
            assert_eq!(*top, 50);
            *top = 5;
        }

        assert_eq!(heap.peek(), Some(&20));
        assert_eq!(heap.pop(), Some(20));
        assert_eq!(heap.pop(), Some(10));
        assert_eq!(heap.pop(), Some(5));
    }

    #[test]
    fn test_heap_any_storage_peek_mut_min_heap() {
        let mut heap: SmallBinaryHeap<i32, 4, Min> = SmallBinaryHeap::new();
        heap.push(10);
        heap.push(5);
        heap.push(20);

        assert_eq!(heap.peek(), Some(&5));

        if let Some(mut top) = heap.peek_mut() {
            assert_eq!(*top, 5);
            *top = 100;
        }

        assert_eq!(heap.peek(), Some(&10));
        assert_eq!(heap.pop(), Some(10));
        assert_eq!(heap.pop(), Some(20));
        assert_eq!(heap.pop(), Some(100));
    }

    #[test]
    fn test_heap_any_storage_append() {
        let mut h1: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
        h1.push(1);
        h1.push(10);

        let mut h2: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
        h2.push(5);
        h2.push(15);

        h1.append(&mut h2);

        assert!(h2.is_empty());
        assert_eq!(h1.len(), 4);
        assert!(h1.is_on_stack());

        assert_eq!(h1.pop(), Some(15));
        assert_eq!(h1.pop(), Some(10));
    }

    #[test]
    fn test_heap_any_storage_retain() {
        let mut heap: SmallBinaryHeap<i32, 8> = SmallBinaryHeap::from_iter(0..10);
        assert!(!heap.is_on_stack());

        heap.retain(|&x| x % 2 == 0);

        let vec = heap.into_sorted_vec();
        assert_eq!(vec, vec![8, 6, 4, 2, 0]);
    }

    #[test]
    fn test_heap_traits_clone() {
        let mut h1: SmallBinaryHeap<i32, 2> = SmallBinaryHeap::new();
        h1.push(1);
        h1.push(2);

        let h2 = h1.clone();
        h1.push(3);

        assert!(h1.len() == 3 && !h1.is_on_stack());
        assert!(h2.len() == 2 && h2.is_on_stack());

        let mut h3 = h1.clone();
        assert!(!h3.is_on_stack());
        assert_eq!(h3.pop(), Some(3));
    }

    #[test]
    fn test_heap_any_storage_with_capacity() {
        let heap: SmallBinaryHeap<i32, 2> = SmallBinaryHeap::with_capacity(10);
        assert!(!heap.is_on_stack());
        assert!(heap.capacity() >= 10);

        let heap2: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::with_capacity(2);
        assert!(heap2.is_on_stack());
    }

    #[test]
    fn test_heap_any_storage_reserve_shrink() {
        let mut heap: SmallBinaryHeap<i32, 2> = SmallBinaryHeap::new();
        heap.push(1);
        heap.reserve(10);
        assert!(!heap.is_on_stack());
        assert!(heap.capacity() >= 11);

        heap.shrink_to_fit();
        assert!(heap.capacity() >= 1);
    }

    #[test]
    fn test_heap_any_storage_clear() {
        let mut heap: SmallBinaryHeap<i32, 2> = SmallBinaryHeap::new();
        heap.push(1);
        heap.push(2);
        heap.clear();
        assert!(heap.is_empty());
        assert!(heap.is_on_stack());

        heap.push(1);
        heap.push(2);
        heap.push(3); // Spill
        heap.clear();
        assert!(heap.is_empty());
        assert!(!heap.is_on_stack());
    }

    #[test]
    fn test_heap_traits_into_vec() {
        let mut heap: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
        heap.push(1);
        heap.push(3);
        heap.push(2);

        let v = heap.clone().into_vec();
        assert_eq!(v.len(), 3);
        assert!(v.contains(&1));
        assert!(v.contains(&2));
        assert!(v.contains(&3));

        let sv = heap.into_sorted_vec();
        assert_eq!(sv, vec![3, 2, 1]);
    }

    #[test]
    fn test_heap_traits_debug_default() {
        let heap: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::default();
        assert!(heap.is_empty());

        let mut heap = SmallBinaryHeap::<i32, 4>::new();
        heap.push(1);
        let debug = format!("{:?}", heap);
        assert!(debug.contains("1"));
    }

    #[test]
    fn test_heap_any_storage_drop_check() {
        use std::cell::RefCell;
        use std::rc::Rc;

        struct Tracker(Rc<RefCell<i32>>);
        impl PartialEq for Tracker {
            fn eq(&self, _: &Self) -> bool {
                true
            }
        }
        impl Eq for Tracker {}
        impl PartialOrd for Tracker {
            fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
                Some(std::cmp::Ordering::Equal)
            }
        }
        impl Ord for Tracker {
            fn cmp(&self, _: &Self) -> std::cmp::Ordering {
                std::cmp::Ordering::Equal
            }
        }
        impl Drop for Tracker {
            fn drop(&mut self) {
                *self.0.borrow_mut() += 1;
            }
        }

        let counter = Rc::new(RefCell::new(0));
        {
            let mut heap: SmallBinaryHeap<Tracker, 2> = SmallBinaryHeap::new();
            heap.push(Tracker(counter.clone()));
            heap.push(Tracker(counter.clone()));
        }
        assert_eq!(*counter.borrow(), 2);

        *counter.borrow_mut() = 0;
        {
            let mut heap: SmallBinaryHeap<Tracker, 2> = SmallBinaryHeap::new();
            heap.push(Tracker(counter.clone()));
            heap.push(Tracker(counter.clone()));
            heap.push(Tracker(counter.clone()));
        }
        assert_eq!(*counter.borrow(), 3);
    }

    #[test]
    fn test_heap_traits_min_heap_kind() {
        // Heap kind traits
        let _: Reverse<i32> = <Min as HeapKind<i32>>::wrap(1);
        assert_eq!(<Min as HeapKind<i32>>::unwrap(Reverse(1)), 1);
        let val = 1;
        let _ = <Min as HeapKind<i32>>::wrap_ref(&val);
        let rev = Reverse(val);
        assert_eq!(<Min as HeapKind<i32>>::unwrap_ref(&rev), &1);
    }

    #[test]
    fn test_heap_any_storage_pop_empty() {
        let mut h: SmallBinaryHeap<i32, 2> = SmallBinaryHeap::new();
        assert_eq!(h.pop(), None);
    }

    #[test]
    fn test_heap_any_storage_reserve_heap_side() {
        let mut h: SmallBinaryHeap<i32, 2> = SmallBinaryHeap::new();
        h.push(1);
        h.push(2);
        h.push(3); // Spill
        h.reserve(10);
        assert!(!h.is_on_stack());
        assert!(h.capacity() >= 13);
    }

    #[test]
    fn test_heap_any_storage_peek_mut_heap() {
        let mut h: SmallBinaryHeap<i32, 2> = vec![1, 2, 3].into_iter().collect();
        if let Some(mut top) = h.peek_mut() {
            assert_eq!(*top, 3);
            *top = 0;
        }
        assert_eq!(h.peek(), Some(&2));
    }

    #[test]
    fn test_heap_traits_into_iter_heap() {
        let h_into: SmallBinaryHeap<i32, 2> = vec![1, 2, 3].into_iter().collect();
        let mut it = h_into.into_iter();
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(it.next(), Some(3));
    }

    #[test]
    fn test_heap_traits_iter_any_storage() {
        let h_iter: SmallBinaryHeap<i32, 2> = vec![1, 2, 3].into_iter().collect();
        let mut it_ref = h_iter.iter();
        assert_eq!(it_ref.next(), Some(&3));

        let stack_h: SmallBinaryHeap<i32, 4> = vec![1, 2].into_iter().collect();
        let mut it_stack = stack_h.iter();
        assert_eq!(it_stack.next(), Some(&2));
    }

    #[test]
    fn test_heap_any_storage_append_complex() {
        let mut h_heap1: SmallBinaryHeap<i32, 2> = vec![1, 2, 3].into_iter().collect();
        let mut h_heap2: SmallBinaryHeap<i32, 2> = vec![4, 5, 6].into_iter().collect();
        h_heap1.append(&mut h_heap2);
        assert_eq!(h_heap1.len(), 6);

        let mut h_heap3: SmallBinaryHeap<i32, 2> = vec![1, 2, 3].into_iter().collect();
        let mut h_stack1: SmallBinaryHeap<i32, 2> = vec![10].into_iter().collect();
        h_heap3.append(&mut h_stack1);
        assert_eq!(h_heap3.len(), 4);
    }

    #[test]
    fn test_heap_any_storage_peek_mut_pop() {
        let mut h_pop: SmallBinaryHeap<i32, 4> = vec![1, 2].into_iter().collect();
        if let Some(p) = h_pop.peek_mut() {
            assert_eq!(p.pop(), 2);
        }
        assert_eq!(h_pop.len(), 1);

        let mut h_pop_heap: SmallBinaryHeap<i32, 2> = vec![1, 2, 3].into_iter().collect();
        if let Some(p) = h_pop_heap.peek_mut() {
            assert_eq!(p.pop(), 3);
        }
        assert_eq!(h_pop_heap.len(), 2);
    }

    #[test]
    fn test_heap_any_storage_append_edge_cases() {
        let mut h: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
        h.push(1);
        assert_eq!(h.capacity(), 4);

        let mut empty: SmallBinaryHeap<i32, 4> = SmallBinaryHeap::new();
        h.append(&mut empty); // append empty
        assert_eq!(h.len(), 1);
    }
}
