//! Contiguous vector that lives on the stack and spills to the heap.
//!
//! Provides [`SmallVec`] — stores up to `N` elements in a `[MaybeUninit<T>; N]` stack
//! array and transparently migrates to a `std::vec::Vec` when full.  Because it `Deref`s
//! to `[T]`, all standard slice methods are available without conversion.
//!
//! [`AnyVec`] is a slice-view trait implemented by `SmallVec`, `Vec`, slices (`[T]`), and
//! arrays (`[T; N]`) to enable generic comparison and extension helpers.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::{Deref, DerefMut};
use core::ptr;
use core::slice;

/// A trait generalizing any vector-like contiguous collection.
pub trait AnyVec<T> {
    fn as_slice(&self) -> &[T];

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().contains(x)
    }

    fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }
}

impl<T> AnyVec<T> for std::vec::Vec<T> {
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AnyVec<T> for [T] {
    fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T, const N: usize> AnyVec<T> for [T; N] {
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> AnyVec<T> for SmallVec<T, N> {
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }
}

pub union VecData<T, const N: usize> {
    pub stack: ManuallyDrop<[MaybeUninit<T>; N]>,
    pub heap: ManuallyDrop<std::vec::Vec<T>>,
}

pub struct SmallVec<T, const N: usize> {
    len: usize,
    capacity: usize,
    on_stack: bool,
    data: VecData<T, N>,
}

impl<T, const N: usize> SmallVec<T, N> {
    pub const MAX_STACK_SIZE: usize = 16 * 1024;

    pub fn new() -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= SmallVec::<T, N>::MAX_STACK_SIZE,
                "SmallVec is too large! Reduce N."
            );
        }
        Self {
            len: 0,
            capacity: N,
            on_stack: true,
            data: VecData {
                stack: ManuallyDrop::new(unsafe { MaybeUninit::uninit().assume_init() }),
            },
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= N {
            Self::new()
        } else {
            let heap_vec = std::vec::Vec::with_capacity(capacity);
            Self {
                len: 0,
                capacity: heap_vec.capacity(),
                on_stack: false,
                data: VecData {
                    heap: ManuallyDrop::new(heap_vec),
                },
            }
        }
    }

    #[inline(always)]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
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
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe {
                let ptr = if self.on_stack {
                    (*self.data.stack).as_ptr() as *const T
                } else {
                    (*self.data.heap).as_ptr()
                };
                Some(&*ptr.add(index))
            }
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            unsafe {
                let ptr = if self.on_stack {
                    (*self.data.stack).as_mut_ptr() as *mut T
                } else {
                    (*self.data.heap).as_mut_ptr()
                };
                Some(&mut *ptr.add(index))
            }
        } else {
            None
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        if self.len + additional > self.capacity {
            unsafe {
                if self.on_stack {
                    self.spill_to_heap();
                }
                (*self.data.heap).reserve(additional);
                self.capacity = (*self.data.heap).capacity();
            }
        }
    }

    #[inline(always)]
    pub fn push(&mut self, item: T) {
        if self.len < self.capacity {
            unsafe {
                let ptr = if self.on_stack {
                    (*self.data.stack).as_mut_ptr() as *mut T
                } else {
                    (*self.data.heap).as_mut_ptr()
                };
                ptr::write(ptr.add(self.len), item);
                self.len += 1;
                if !self.on_stack {
                    (*self.data.heap).set_len(self.len);
                }
            }
        } else {
            self.grow_and_push(item);
        }
    }

    #[inline(never)]
    fn grow_and_push(&mut self, item: T) {
        unsafe {
            if self.on_stack {
                self.spill_to_heap();
            }
            (*self.data.heap).push(item);
            self.len = (*self.data.heap).len();
            self.capacity = (*self.data.heap).capacity();
        }
    }

    #[inline(always)]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe {
                let ptr = if self.on_stack {
                    (*self.data.stack).as_ptr() as *mut T
                } else {
                    (*self.data.heap).as_mut_ptr()
                };
                let val = ptr::read(ptr.add(self.len));
                if !self.on_stack {
                    (*self.data.heap).set_len(self.len);
                }
                Some(val)
            }
        }
    }

    pub fn insert(&mut self, index: usize, element: T) {
        assert!(index <= self.len);
        if self.len == self.capacity {
            self.grow_for_insert(index, element);
        } else {
            unsafe {
                let ptr = if self.on_stack {
                    (*self.data.stack).as_mut_ptr() as *mut T
                } else {
                    (*self.data.heap).as_mut_ptr()
                };
                ptr::copy(ptr.add(index), ptr.add(index + 1), self.len - index);
                ptr::write(ptr.add(index), element);
                self.len += 1;
                if !self.on_stack {
                    (*self.data.heap).set_len(self.len);
                }
            }
        }
    }

    #[inline(never)]
    fn grow_for_insert(&mut self, index: usize, element: T) {
        unsafe {
            if self.on_stack {
                self.spill_to_heap();
            }
            (*self.data.heap).insert(index, element);
            self.len = (*self.data.heap).len();
            self.capacity = (*self.data.heap).capacity();
        }
    }

    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len);
        unsafe {
            let ptr = if self.on_stack {
                (*self.data.stack).as_mut_ptr() as *mut T
            } else {
                (*self.data.heap).as_mut_ptr()
            };
            let val = ptr::read(ptr.add(index));
            ptr::copy(ptr.add(index + 1), ptr.add(index), self.len - index - 1);
            self.len -= 1;
            if !self.on_stack {
                (*self.data.heap).set_len(self.len);
            }
            val
        }
    }

    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(index < self.len);
        unsafe {
            let ptr = if self.on_stack {
                (*self.data.stack).as_mut_ptr() as *mut T
            } else {
                (*self.data.heap).as_mut_ptr()
            };
            let val = ptr::read(ptr.add(index));
            let last = ptr::read(ptr.add(self.len - 1));
            if index < self.len - 1 {
                ptr::write(ptr.add(index), last);
            }
            self.len -= 1;
            if !self.on_stack {
                (*self.data.heap).set_len(self.len);
            }
            val
        }
    }

    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            unsafe {
                let ptr = if self.on_stack {
                    (*self.data.stack).as_mut_ptr() as *mut T
                } else {
                    (*self.data.heap).as_mut_ptr()
                };
                for i in len..self.len {
                    ptr::drop_in_place(ptr.add(i));
                }
            }
            self.len = len;
            if !self.on_stack {
                unsafe { (*self.data.heap).set_len(len) };
            }
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let len = self.len;
        let mut del = 0;
        unsafe {
            let ptr = if self.on_stack {
                (*self.data.stack).as_mut_ptr() as *mut T
            } else {
                (*self.data.heap).as_mut_ptr()
            };

            for i in 0..len {
                if !f(&*ptr.add(i)) {
                    ptr::drop_in_place(ptr.add(i));
                    del += 1;
                } else if del > 0 {
                    let val = ptr::read(ptr.add(i));
                    ptr::write(ptr.add(i - del), val);
                }
            }
            self.len -= del;
            if !self.on_stack {
                (*self.data.heap).set_len(self.len);
            }
        }
    }

    pub fn shrink_to_fit(&mut self) {
        if !self.on_stack {
            unsafe {
                (*self.data.heap).shrink_to_fit();
                self.capacity = (*self.data.heap).capacity();
            }
        }
    }

    pub fn into_vec(self) -> std::vec::Vec<T>
    where
        T: Clone,
    {
        let this = ManuallyDrop::new(self);
        unsafe {
            if this.on_stack {
                let ptr = (*this.data.stack).as_ptr() as *const T;
                slice::from_raw_parts(ptr, this.len).to_vec()
            } else {
                ptr::read(&*this.data.heap)
            }
        }
    }

    #[inline(never)]
    unsafe fn spill_to_heap(&mut self) {
        unsafe {
            let mut heap_vec = std::vec::Vec::with_capacity(self.capacity * 2);
            let ptr = (*self.data.stack).as_ptr() as *const T;
            for i in 0..self.len {
                heap_vec.push(ptr::read(ptr.add(i)));
            }
            ptr::write(&mut self.data.heap, ManuallyDrop::new(heap_vec));
            self.on_stack = false;
            self.capacity = (*self.data.heap).capacity();
        }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            let ptr = if self.on_stack {
                (*self.data.stack).as_ptr() as *const T
            } else {
                (*self.data.heap).as_ptr()
            };
            slice::from_raw_parts(ptr, self.len)
        }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            let ptr = if self.on_stack {
                (*self.data.stack).as_mut_ptr() as *mut T
            } else {
                (*self.data.heap).as_mut_ptr()
            };
            slice::from_raw_parts_mut(ptr, self.len)
        }
    }
}

impl<T: Clone, const N: usize> SmallVec<T, N> {
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len;
        if new_len > len {
            self.reserve(new_len - len);
            for _ in len..new_len {
                self.push(value.clone());
            }
        } else {
            self.truncate(new_len);
        }
    }

    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.reserve(other.len());
        for item in other {
            self.push(item.clone());
        }
    }
}

impl<T, const N: usize> Deref for SmallVec<T, N> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const N: usize> DerefMut for SmallVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> Drop for SmallVec<T, N> {
    fn drop(&mut self) {
        if self.on_stack {
            unsafe {
                let ptr = (*self.data.stack).as_mut_ptr() as *mut T;
                for i in 0..self.len {
                    ptr::drop_in_place(ptr.add(i));
                }
            }
        } else {
            unsafe {
                ManuallyDrop::drop(&mut self.data.heap);
            }
        }
    }
}

impl<T: Clone, const N: usize> Clone for SmallVec<T, N> {
    fn clone(&self) -> Self {
        if self.on_stack {
            let mut stack_arr: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
            let src = self.as_slice();
            for i in 0..self.len {
                stack_arr[i] = MaybeUninit::new(src[i].clone());
            }
            Self {
                len: self.len,
                capacity: N,
                on_stack: true,
                data: VecData {
                    stack: ManuallyDrop::new(stack_arr),
                },
            }
        } else {
            let heap_vec = unsafe { (*self.data.heap).clone() };
            Self {
                len: self.len,
                capacity: heap_vec.capacity(),
                on_stack: false,
                data: VecData {
                    heap: ManuallyDrop::new(heap_vec),
                },
            }
        }
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for SmallVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T, const N: usize> Default for SmallVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialEq, const N: usize> PartialEq for SmallVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<T: Eq, const N: usize> Eq for SmallVec<T, N> {}

impl<T, const N: usize> Extend<T> for SmallVec<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
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

pub struct SmallVecIntoIter<T, const N: usize> {
    iter: SmallVecIterEnum<T, N>,
}

enum SmallVecIterEnum<T, const N: usize> {
    Stack {
        data: [MaybeUninit<T>; N],
        pos: usize,
        len: usize,
    },
    Heap(std::vec::IntoIter<T>),
}

impl<T, const N: usize> IntoIterator for SmallVec<T, N> {
    type Item = T;
    type IntoIter = SmallVecIntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let this = ManuallyDrop::new(self);
        unsafe {
            if this.on_stack {
                SmallVecIntoIter {
                    iter: SmallVecIterEnum::Stack {
                        data: ptr::read(&*this.data.stack),
                        pos: 0,
                        len: this.len,
                    },
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
            SmallVecIterEnum::Stack { data, pos, len } => {
                if *pos < *len {
                    let val = unsafe { ptr::read(data[*pos].as_ptr()) };
                    *pos += 1;
                    Some(val)
                } else {
                    None
                }
            }
            SmallVecIterEnum::Heap(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.iter {
            SmallVecIterEnum::Stack { pos, len, .. } => {
                let remaining = len - pos;
                (remaining, Some(remaining))
            }
            SmallVecIterEnum::Heap(iter) => iter.size_hint(),
        }
    }
}

impl<T, const N: usize> ExactSizeIterator for SmallVecIntoIter<T, N> {}

impl<T, const N: usize> Drop for SmallVecIntoIter<T, N> {
    fn drop(&mut self) {
        if let SmallVecIterEnum::Stack { data, pos, len } = &mut self.iter {
            for i in *pos..*len {
                unsafe {
                    ptr::drop_in_place(data[i].as_mut_ptr());
                }
            }
        }
    }
}

impl<T: Hash, const N: usize> Hash for SmallVec<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T: PartialOrd, const N: usize> PartialOrd for SmallVec<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Ord, const N: usize> Ord for SmallVec<T, N> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T, const N: usize> AsRef<[T]> for SmallVec<T, N> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> std::borrow::Borrow<[T]> for SmallVec<T, N> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> std::borrow::BorrowMut<[T]> for SmallVec<T, N> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> AsMut<[T]> for SmallVec<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> AsRef<SmallVec<T, N>> for SmallVec<T, N> {
    fn as_ref(&self) -> &SmallVec<T, N> {
        self
    }
}

impl<T, const N: usize> core::ops::Index<usize> for SmallVec<T, N> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<T, const N: usize> core::ops::IndexMut<usize> for SmallVec<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("index out of bounds")
    }
}

impl<T, const N: usize> core::ops::Index<core::ops::RangeFull> for SmallVec<T, N> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, _: core::ops::RangeFull) -> &Self::Output {
        self.as_slice()
    }
}

impl<T, const N: usize> core::ops::IndexMut<core::ops::RangeFull> for SmallVec<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, _: core::ops::RangeFull) -> &mut Self::Output {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> SmallVec<T, N> {
    pub fn extend_from_any<V: AnyVec<T> + ?Sized>(&mut self, other: &V)
    where
        T: Clone,
    {
        self.extend_from_slice(other.as_slice());
    }

    pub fn eq_any<V: AnyVec<T> + ?Sized>(&self, other: &V) -> bool
    where
        T: PartialEq,
    {
        self.as_slice() == other.as_slice()
    }

    pub fn cmp_any<V: AnyVec<T> + ?Sized>(&self, other: &V) -> Ordering
    where
        T: Ord,
    {
        self.as_slice().cmp(other.as_slice())
    }

    pub fn starts_with_any<V: AnyVec<T> + ?Sized>(&self, other: &V) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().starts_with(other.as_slice())
    }

    pub fn ends_with_any<V: AnyVec<T> + ?Sized>(&self, other: &V) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().ends_with(other.as_slice())
    }

    pub fn contains_subsequence<V: AnyVec<T> + ?Sized>(&self, other: &V) -> bool
    where
        T: PartialEq,
    {
        let other_slice = other.as_slice();
        if other_slice.is_empty() {
            return true;
        }
        if other_slice.len() > self.len {
            return false;
        }
        self.as_slice()
            .windows(other_slice.len())
            .any(|w| w == other_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_traits_borrow() {
        use std::borrow::{Borrow, BorrowMut};
        let mut v: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3]);

        // Test Borrow<[i32]>
        let b: &[i32] = v.borrow();
        assert_eq!(b, &[1, 2, 3]);

        // Test BorrowMut<[i32]>
        let b_mut: &mut [i32] = v.borrow_mut();
        b_mut[0] = 10;
        assert_eq!(v.as_slice(), &[10, 2, 3]);
    }

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
        vec.push(3);
        assert!(!vec.is_on_stack());
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[2], 3);
        assert!(vec.capacity() >= 4);
    }

    #[test]
    fn test_vec_spill_trigger_on_insert() {
        let mut vec: SmallVec<i32, 2> = SmallVec::new();
        vec.push(1);
        vec.push(3);
        vec.insert(1, 2);
        assert!(!vec.is_on_stack());
        assert_eq!(vec.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_vec_stack_insert_remove_swap() {
        let mut vec: SmallVec<i32, 4> = SmallVec::from_iter([10, 20, 30]);
        vec.insert(1, 15);
        assert_eq!(vec[1], 15);
        let removed = vec.remove(2);
        assert_eq!(removed, 20);
        assert_eq!(vec.as_slice(), &[10, 15, 30]);
        vec.push(40);
        let swapped = vec.swap_remove(0);
        assert_eq!(swapped, 10);
        assert_eq!(vec.as_slice(), &[40, 15, 30]);
    }

    #[test]
    fn test_vec_any_storage_retain() {
        let mut vec: SmallVec<i32, 8> = SmallVec::from_iter(0..10);
        assert!(!vec.is_on_stack());
        vec.retain(|&x| x % 2 == 0);
        assert_eq!(vec.as_slice(), &[0, 2, 4, 6, 8]);
        let mut vec_stack: SmallVec<i32, 8> = SmallVec::from_iter(0..6);
        assert!(vec_stack.is_on_stack());
        vec_stack.retain(|&x| x % 2 != 0);
        assert_eq!(vec_stack.as_slice(), &[1, 3, 5]);
    }

    #[test]
    fn test_vec_any_storage_resize_clone() {
        let mut vec: SmallVec<i32, 4> = SmallVec::new();
        vec.resize(2, 0);
        assert!(vec.is_on_stack());
        let vec2 = vec.clone();
        assert_eq!(vec, vec2);
        vec.resize(10, 5);
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
        impl Clone for Dropper {
            fn clone(&self) -> Self {
                Dropper(self.0.clone())
            }
        }
        {
            let mut vec: SmallVec<Dropper, 2> = SmallVec::new();
            vec.push(Dropper(counter.clone()));
            vec.push(Dropper(counter.clone()));
        }
        assert_eq!(*counter.borrow(), 2);
        *counter.borrow_mut() = 0;
        {
            let mut vec: SmallVec<Dropper, 2> = SmallVec::new();
            vec.push(Dropper(counter.clone()));
            vec.push(Dropper(counter.clone()));
            vec.push(Dropper(counter.clone()));
        }
        assert_eq!(*counter.borrow(), 3);
    }

    #[test]
    fn test_vec_traits_anyvec_inspection() {
        let sv: SmallVec<i32, 4> = SmallVec::from_iter([10, 20, 30]);
        assert_eq!(sv.len(), 3);
        assert!(!sv.is_empty());
        assert_eq!(sv.as_slice().first(), Some(&10));
        assert_eq!(sv.get(1), Some(&20));
        let sum: i32 = sv.iter().sum();
        assert_eq!(sum, 60);
    }

    #[test]
    fn test_vec_traits_interop_comparison() {
        let sv: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3]);
        let std_vec = vec![1, 2, 3];
        assert!(sv.eq_any(&std_vec));
        assert_eq!(sv.cmp_any(&std_vec), std::cmp::Ordering::Equal);
        let arr = [1, 2, 4];
        assert!(!sv.eq_any(&arr));
        assert_eq!(sv.cmp_any(&arr), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_vec_traits_interop_searching() {
        let sv: SmallVec<i32, 8> = SmallVec::from_iter([1, 2, 3, 4, 5]);
        assert!(sv.starts_with_any(&vec![1, 2]));
        assert!(sv.ends_with_any(&[4, 5]));
        assert!(sv.contains_subsequence(&vec![3, 4]));
        assert!(!sv.contains_subsequence(&[3, 5]));
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
        v.extend_from_slice(&[3, 4, 5]);
        assert!(!v.is_on_stack());
        assert_eq!(v.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_vec_any_storage_clear_and_empty() {
        let mut v: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3, 4, 5]);
        assert!(!v.is_on_stack());
        v.clear();
        assert!(v.is_empty());
        let mut v_stack: SmallVec<i32, 4> = SmallVec::from_iter([1, 2]);
        v_stack.clear();
        assert!(v_stack.is_empty());
    }

    #[test]
    fn test_vec_traits_partial_eq_cross_storage() {
        let v1: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3]);
        let v2: SmallVec<i32, 2> = SmallVec::from_iter([1, 2, 3]);
        assert!(v1.is_on_stack());
        assert!(!v2.is_on_stack());
        assert_eq!(&v1[..], &v2[..]);
    }

    #[test]
    fn test_vec_any_storage_as_mut_and_deref_mut() {
        let mut v: SmallVec<i32, 4> = SmallVec::from_iter([1, 2, 3]);
        v[0] = 10;
        assert_eq!(v[0], 10);
        v.as_mut_slice()[1] = 20;
        assert_eq!(v[1], 20);
    }
}
