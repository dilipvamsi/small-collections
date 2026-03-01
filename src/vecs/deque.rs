//! Stack-allocated double-ended queue that spills to the heap when full.
//!
//! # Why not `heapless::Deque`?
//! `heapless::Deque` exists but was intentionally **not** used as the stack-side storage
//! for [`SmallDeque`].  The key reasons are:
//!
//! 1. **Forced power-of-two capacity**: `heapless::Deque<T, N>` internally requires
//!    `N` to be a power of two to support its bitmask-based wrap-around arithmetic.
//!    `SmallDeque` imposes the same constraint, yet needs to manage the ring-buffer
//!    indices (head/len) *outside* the stored data so they can be preserved independent
//!    of the storage backend.  Embedding them inside a `heapless::Deque` value stored
//!    in a union field would require reading from the (possibly heap-active) union branch,
//!    which is unsound without unsafe indirection that yields no benefit over the raw
//!    `[MaybeUninit<T>; N]` approach used here.
//!
//! 2. **No first-class drain/into-iter that preserves ring order**: when spilling to the
//!    heap we need to iterate in logical order (head … tail) to fill `VecDeque`.  The
//!    raw-array approach gives us direct index arithmetic (`wrap_add`) to do this;
//!    `heapless::Deque` would require an additional copy through its iterator which
//!    offers the same cost but less control.
//!
//! 3. **Union soundness**: the `DequeData` union holds either a raw array or a
//!    heap-allocated `VecDeque`.  `heapless::Deque` contains internal state (`read`,
//!    `write` cursors) that would be overwritten if treated as an uninitialised `VecDeque`
//!    and vice versa.  Using `[MaybeUninit<T>; N]` makes the union semantics trivial:
//!    the stack side is just memory, no drop glue.

use core::fmt;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr;
use std::collections::VecDeque;

// ─── AnyDeque ─────────────────────────────────────────────────────────────────

/// An object-safe abstraction over double-ended queue types.
///
/// Implemented by both `VecDeque<T>` (heap) and `SmallDeque<T, N>` (small/stack)
/// so that code can operate on a deque without knowing which backend is active.
pub trait AnyDeque<T> {
    /// Returns the number of elements in the deque.
    fn len(&self) -> usize;
    /// Returns `true` if the deque contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Appends an element to the back.
    fn push_back(&mut self, item: T);
    /// Prepends an element to the front.
    fn push_front(&mut self, item: T);
    /// Removes and returns the element from the back, or `None` if empty.
    fn pop_back(&mut self) -> Option<T>;
    /// Removes and returns the element from the front, or `None` if empty.
    fn pop_front(&mut self) -> Option<T>;
    /// Removes and returns the element at `index`, or `None` if out of bounds.
    fn remove(&mut self, index: usize) -> Option<T>;
    /// Removes all elements.
    fn clear(&mut self);
    /// Returns a shared reference to the front element, or `None` if empty.
    fn front(&self) -> Option<&T>;
    /// Returns a shared reference to the back element, or `None` if empty.
    fn back(&self) -> Option<&T>;
    /// Returns an exclusive reference to the front element, or `None` if empty.
    fn front_mut(&mut self) -> Option<&mut T>;
    /// Returns an exclusive reference to the back element, or `None` if empty.
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
    fn remove(&mut self, index: usize) -> Option<T> {
        self.remove(index)
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

/// Tagged union holding either the stack ring-buffer or the heap `VecDeque`.
///
/// # Safety
/// The active variant is tracked by `SmallDeque::on_stack`.  Only the active
/// variant must be accessed at any time.  The stack variant is
/// `[MaybeUninit<T>; N]`, so it has no drop glue — `ManuallyDrop` around the
/// stack side is therefore redundant but kept for symmetry with the heap side.
pub union DequeData<T, const N: usize> {
    pub stack: ManuallyDrop<[MaybeUninit<T>; N]>,
    pub heap: ManuallyDrop<VecDeque<T>>,
}

/// A double-ended queue that lives on the stack for up to `N` items, then spills to
/// a heap-allocated `VecDeque` transparently.
///
/// # Stack representation
/// Items are stored in a ring buffer of `[MaybeUninit<T>; N]` with a `head` cursor and
/// a `len` counter.  Two helpers, `wrap_add` and `wrap_sub`, implement the modular
/// arithmetic using a bitmask (requires `N` to be a power of two — enforced by a
/// `const` assertion in [`new`](SmallDeque::new)).
///
/// # Spill behaviour
/// When `push_back` or `push_front` causes `len > capacity` on the stack, the ring
/// buffer is drained in logical order into a freshly heap-allocated `VecDeque` via
/// `spill_to_heap`.  After a spill, all subsequent mutations go directly to the
/// `VecDeque`; the struct never returns to stack storage.
///
/// # Generic parameters
/// | Parameter | Meaning |
/// |-----------|--------|
/// | `T` | Element type |
/// | `N` | Stack capacity; **must be a power of two** |
///
/// # Compile-time assertions
/// `new()` uses `const { assert!(...) }` to verify:
/// - `size_of::<Self>() <= 16 KiB` — prevents accidental blowing of the stack frame.
/// - `N.is_power_of_two()` — required for bitmask ring-buffer arithmetic.
pub struct SmallDeque<T, const N: usize> {
    len: usize,
    capacity: usize,
    head: usize,
    on_stack: bool,
    data: DequeData<T, N>,
}

impl<T, const N: usize> AnyDeque<T> for SmallDeque<T, N> {
    fn len(&self) -> usize {
        self.len
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
    fn remove(&mut self, index: usize) -> Option<T> {
        self.remove(index)
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

impl<T, const N: usize> SmallDeque<T, N> {
    /// Maximum allowed struct size in bytes.  Prevents accidentally allocating huge
    /// arrays on the call stack.
    const MAX_STACK_SIZE: usize = 16 * 1024;

    /// Creates a new empty deque backed by stack storage.
    ///
    /// # Panics (compile-time)
    /// Asserts that `size_of::<Self>() <= 16 KiB` and that `N` is a power of two.
    pub fn new() -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= SmallDeque::<T, N>::MAX_STACK_SIZE,
                "SmallDeque is too large! Reduce N."
            );
            assert!(N.is_power_of_two(), "SmallDeque N must be a power of two");
        }
        Self {
            len: 0,
            capacity: N,
            head: 0,
            on_stack: true,
            data: DequeData {
                stack: ManuallyDrop::new(unsafe { MaybeUninit::uninit().assume_init() }),
            },
        }
    }

    /// Creates a deque that starts on the heap if `capacity > N`, otherwise on the stack.
    ///
    /// Useful when the caller already knows the required size will exceed `N`.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= N {
            Self::new()
        } else {
            let heap_deque = VecDeque::with_capacity(capacity);
            Self {
                len: 0,
                capacity: heap_deque.capacity(),
                head: 0,
                on_stack: false,
                data: DequeData {
                    heap: ManuallyDrop::new(heap_deque),
                },
            }
        }
    }

    /// Returns `true` if the deque is currently using stack storage.
    #[inline(always)]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }

    /// Returns the number of elements currently in the deque.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the deque contains no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the current capacity (not necessarily `N` after a spill).
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Maps a logical index into the ring buffer to a physical slot index.
    /// Uses bitmask `(capacity - 1)` — valid because `N` is a power of two.
    #[inline(always)]
    fn wrap_add(&self, idx: usize, add: usize) -> usize {
        (idx + add) & (self.capacity - 1)
    }

    /// Maps a logical index into the ring buffer, wrapping backwards.
    #[inline(always)]
    fn wrap_sub(&self, idx: usize, sub: usize) -> usize {
        (idx.wrapping_sub(sub)) & (self.capacity - 1)
    }

    /// Returns a shared reference to the element at logical `index`, or `None`.
    ///
    /// Logical index 0 is the front (oldest element).
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe {
                if self.on_stack {
                    let real_idx = self.wrap_add(self.head, index);
                    let ptr = (*self.data.stack).as_ptr() as *const T;
                    Some(&*ptr.add(real_idx))
                } else {
                    (*self.data.heap).get(index)
                }
            }
        } else {
            None
        }
    }

    /// Returns an exclusive reference to the element at logical `index`, or `None`.
    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            unsafe {
                if self.on_stack {
                    let real_idx = self.wrap_add(self.head, index);
                    let ptr = (*self.data.stack).as_mut_ptr() as *mut T;
                    Some(&mut *ptr.add(real_idx))
                } else {
                    (*self.data.heap).get_mut(index)
                }
            }
        } else {
            None
        }
    }

    /// Reserves capacity for at least `additional` more elements, spilling to the heap
    /// if necessary.
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

    /// Appends `item` to the back of the deque.  Spills to heap if the stack is full.
    #[inline(always)]
    pub fn push_back(&mut self, item: T) {
        if self.len < self.capacity && self.on_stack {
            unsafe {
                let tail = self.wrap_add(self.head, self.len);
                let ptr = (*self.data.stack).as_mut_ptr() as *mut T;
                ptr::write(ptr.add(tail), item);
                self.len += 1;
            }
        } else {
            self.grow_and_push_back(item);
        }
    }

    /// Cold path: spills to heap then delegates `push_back`.
    #[inline(never)]
    fn grow_and_push_back(&mut self, item: T) {
        unsafe {
            if self.on_stack {
                self.spill_to_heap();
            }
            (*self.data.heap).push_back(item);
            self.len = (*self.data.heap).len();
            self.capacity = (*self.data.heap).capacity();
        }
    }

    /// Prepends `item` to the front of the deque.  Spills to heap if the stack is full.
    #[inline(always)]
    pub fn push_front(&mut self, item: T) {
        if self.len < self.capacity && self.on_stack {
            unsafe {
                self.head = self.wrap_sub(self.head, 1);
                let ptr = (*self.data.stack).as_mut_ptr() as *mut T;
                ptr::write(ptr.add(self.head), item);
                self.len += 1;
            }
        } else {
            self.grow_and_push_front(item);
        }
    }

    /// Cold path: spills to heap then delegates `push_front`.
    #[inline(never)]
    fn grow_and_push_front(&mut self, item: T) {
        unsafe {
            if self.on_stack {
                self.spill_to_heap();
            }
            (*self.data.heap).push_front(item);
            self.len = (*self.data.heap).len();
            self.capacity = (*self.data.heap).capacity();
        }
    }

    /// Removes and returns the last element, or `None` if empty.
    #[inline(always)]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe {
                if self.on_stack {
                    let tail = self.wrap_add(self.head, self.len);
                    let ptr = (*self.data.stack).as_ptr() as *const T;
                    Some(ptr::read(ptr.add(tail)))
                } else {
                    (*self.data.heap).pop_back()
                }
            }
        }
    }

    /// Removes and returns the first element, or `None` if empty.
    #[inline(always)]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                let val = if self.on_stack {
                    let ptr = (*self.data.stack).as_ptr() as *const T;
                    let v = ptr::read(ptr.add(self.head));
                    self.head = self.wrap_add(self.head, 1);
                    v
                } else {
                    (*self.data.heap).pop_front().unwrap()
                };
                self.len -= 1;
                Some(val)
            }
        }
    }

    /// Removes the element at logical `index` and returns it, or `None` if out of bounds.
    ///
    /// Chooses whether to shift from the front or back depending on which is cheaper
    /// (shift the shorter half).
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }

        if index == 0 {
            return self.pop_front();
        }
        if index == self.len - 1 {
            return self.pop_back();
        }

        unsafe {
            if self.on_stack {
                let real_idx = self.wrap_add(self.head, index);
                let ptr = (*self.data.stack).as_mut_ptr() as *mut T;
                let val = ptr::read(ptr.add(real_idx));

                // Shift elements
                if index < self.len / 2 {
                    // Shift head forward
                    for i in (0..index).rev() {
                        let from = self.wrap_add(self.head, i);
                        let to = self.wrap_add(from, 1);
                        ptr::copy_nonoverlapping(ptr.add(from), ptr.add(to), 1);
                    }
                    self.head = self.wrap_add(self.head, 1);
                } else {
                    // Shift tail backward
                    for i in (index + 1)..self.len {
                        let from = self.wrap_add(self.head, i);
                        let to = self.wrap_sub(from, 1);
                        ptr::copy_nonoverlapping(ptr.add(from), ptr.add(to), 1);
                    }
                }
                self.len -= 1;
                Some(val)
            } else {
                let val = (*self.data.heap).remove(index);
                self.len = (*self.data.heap).len();
                val
            }
        }
    }

    /// Shortens the deque to `len`, dropping all elements beyond that point.
    ///
    /// If `len >= self.len()`, this is a no-op.
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Shortens the deque to at most `len` elements, dropping those beyond.
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            unsafe {
                if self.on_stack {
                    let ptr = (*self.data.stack).as_mut_ptr() as *mut T;
                    for i in len..self.len {
                        let real_idx = self.wrap_add(self.head, i);
                        ptr::drop_in_place(ptr.add(real_idx));
                    }
                } else {
                    (*self.data.heap).truncate(len);
                }
            }
            self.len = len;
        }
    }

    /// Returns a shared reference to the front element, or `None` if empty.
    #[inline(always)]
    pub fn front(&self) -> Option<&T> {
        self.get(0)
    }

    /// Returns a shared reference to the back element, or `None` if empty.
    #[inline(always)]
    pub fn back(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            self.get(self.len - 1)
        }
    }

    /// Returns an exclusive reference to the front element, or `None` if empty.
    #[inline(always)]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    /// Returns an exclusive reference to the back element, or `None` if empty.
    #[inline(always)]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            None
        } else {
            self.get_mut(self.len - 1)
        }
    }

    /// Returns up to two contiguous slices covering the logical range `[0, len)`.
    ///
    /// Returns `(head_slice, &[])` when the ring buffer hasn't wrapped, or
    /// `(head_slice, tail_slice)` when it has.  On the heap path delegates to
    /// `VecDeque::as_slices`.
    #[inline(always)]
    pub fn as_slices(&self) -> (&[T], &[T]) {
        unsafe {
            if self.on_stack {
                let ptr = (*self.data.stack).as_ptr() as *const T;
                if self.head + self.len <= self.capacity {
                    (
                        core::slice::from_raw_parts(ptr.add(self.head), self.len),
                        &[],
                    )
                } else {
                    let head_len = self.capacity - self.head;
                    let tail_len = self.len - head_len;
                    (
                        core::slice::from_raw_parts(ptr.add(self.head), head_len),
                        core::slice::from_raw_parts(ptr, tail_len),
                    )
                }
            } else {
                (*self.data.heap).as_slices()
            }
        }
    }

    /// Mutable counterpart of [`as_slices`](SmallDeque::as_slices).
    #[inline(always)]
    pub fn as_mut_slices(&mut self) -> (&mut [T], &mut [T]) {
        unsafe {
            if self.on_stack {
                let ptr = (*self.data.stack).as_mut_ptr() as *mut T;
                if self.head + self.len <= self.capacity {
                    (
                        core::slice::from_raw_parts_mut(ptr.add(self.head), self.len),
                        &mut [],
                    )
                } else {
                    let head_len = self.capacity - self.head;
                    let tail_len = self.len - head_len;
                    let (s1, s2) = (
                        core::slice::from_raw_parts_mut(ptr.add(self.head), head_len),
                        core::slice::from_raw_parts_mut(ptr, tail_len),
                    );
                    (s1, s2)
                }
            } else {
                (*self.data.heap).as_mut_slices()
            }
        }
    }

    /// Migrates all elements from the stack ring buffer to a heap `VecDeque`.
    ///
    /// Elements are written in logical order (front → back) so `VecDeque` indices
    /// match the caller's expectations.  After this call, `on_stack == false`.
    ///
    /// # Safety
    /// Must only be called when `on_stack == true`.  The stack variant of `data` must
    /// contain `len` initialized elements starting at `head`.
    #[inline(never)]
    unsafe fn spill_to_heap(&mut self) {
        unsafe {
            let mut heap_deque = VecDeque::with_capacity(self.capacity * 2);
            let ptr = (*self.data.stack).as_ptr() as *const T;
            for i in 0..self.len {
                let real_idx = self.wrap_add(self.head, i);
                heap_deque.push_back(ptr::read(ptr.add(real_idx)));
            }
            ptr::write(&mut self.data.heap, ManuallyDrop::new(heap_deque));
            self.on_stack = false;
            self.capacity = (*self.data.heap).capacity();
        }
    }
}

impl<T, const N: usize> Drop for SmallDeque<T, N> {
    fn drop(&mut self) {
        if self.on_stack {
            unsafe {
                let ptr = (*self.data.stack).as_mut_ptr() as *mut T;
                for i in 0..self.len {
                    let real_idx = self.wrap_add(self.head, i);
                    ptr::drop_in_place(ptr.add(real_idx));
                }
            }
        } else {
            unsafe {
                ManuallyDrop::drop(&mut self.data.heap);
            }
        }
    }
}

impl<T: Clone, const N: usize> Clone for SmallDeque<T, N> {
    fn clone(&self) -> Self {
        if self.on_stack {
            let mut stack_arr: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
            let (s1, s2) = self.as_slices();
            let mut idx = 0;
            for item in s1 {
                stack_arr[idx] = MaybeUninit::new(item.clone());
                idx += 1;
            }
            for item in s2 {
                stack_arr[idx] = MaybeUninit::new(item.clone());
                idx += 1;
            }
            Self {
                len: self.len,
                capacity: N,
                head: 0,
                on_stack: true,
                data: DequeData {
                    stack: ManuallyDrop::new(stack_arr),
                },
            }
        } else {
            let heap_deque = unsafe { (*self.data.heap).clone() };
            Self {
                len: self.len,
                capacity: heap_deque.capacity(),
                head: 0,
                on_stack: false,
                data: DequeData {
                    heap: ManuallyDrop::new(heap_deque),
                },
            }
        }
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for SmallDeque<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (s1, s2) = self.as_slices();
        f.debug_list().entries(s1.iter().chain(s2.iter())).finish()
    }
}

impl<T, const N: usize> Default for SmallDeque<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialEq, const N: usize> PartialEq for SmallDeque<T, N> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        let (s1_a, s2_a) = self.as_slices();
        let (s1_b, s2_b) = other.as_slices();
        s1_a.iter()
            .chain(s2_a.iter())
            .zip(s1_b.iter().chain(s2_b.iter()))
            .all(|(a, b)| a == b)
    }
}
impl<T: Eq, const N: usize> Eq for SmallDeque<T, N> {}

impl<T: PartialOrd, const N: usize> PartialOrd for SmallDeque<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let (s1_a, s2_a) = self.as_slices();
        let (s1_b, s2_b) = other.as_slices();
        s1_a.iter()
            .chain(s2_a.iter())
            .partial_cmp(s1_b.iter().chain(s2_b.iter()))
    }
}

impl<T: Ord, const N: usize> Ord for SmallDeque<T, N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let (s1_a, s2_a) = self.as_slices();
        let (s1_b, s2_b) = other.as_slices();
        s1_a.iter()
            .chain(s2_a.iter())
            .cmp(s1_b.iter().chain(s2_b.iter()))
    }
}

impl<T, const N: usize> Extend<T> for SmallDeque<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for i in iter {
            self.push_back(i);
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

#[cfg(test)]
mod tests {
    use super::*;

    // ─── basic stack ops ──────────────────────────────────────────────────────
    #[test]
    fn test_deque_stack_ops_basic() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        assert!(d.is_empty());
        assert!(d.is_on_stack());
        d.push_back(1);
        d.push_back(2);
        d.push_front(0);
        assert_eq!(d.len(), 3);
        assert_eq!(d.front(), Some(&0));
        assert_eq!(d.back(), Some(&2));
        assert_eq!(d.pop_front(), Some(0));
        assert_eq!(d.pop_back(), Some(2));
        assert_eq!(d.len(), 1);
        assert!(d.is_on_stack());
    }

    #[test]
    fn test_deque_stack_ops_pop_empty() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        assert_eq!(d.pop_front(), None);
        assert_eq!(d.pop_back(), None);
        assert_eq!(d.front(), None);
        assert_eq!(d.back(), None);
    }

    // ─── wrap-around (ring buffer) ────────────────────────────────────────────
    #[test]
    fn test_deque_stack_wrap_ring_buffer() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        d.push_back(1);
        d.push_back(2);
        d.pop_front(); // head advances
        d.pop_front();
        // head is now at index 2
        d.push_back(3);
        d.push_back(4);
        d.push_back(5);
        d.push_back(6); // wrapped: [3,4,5,6]
        assert!(d.is_on_stack());
        assert_eq!(d.pop_front(), Some(3));
        assert_eq!(d.pop_front(), Some(4));
        assert_eq!(d.pop_front(), Some(5));
        assert_eq!(d.pop_front(), Some(6));
    }

    // ─── spill ────────────────────────────────────────────────────────────────
    #[test]
    fn test_deque_spill_trigger() {
        let mut d: SmallDeque<i32, 2> = SmallDeque::new();
        d.push_back(1);
        d.push_back(2);
        assert!(d.is_on_stack());
        d.push_back(3); // spill
        assert!(!d.is_on_stack());
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn test_deque_spill_data_integrity() {
        let mut d: SmallDeque<i32, 2> = SmallDeque::new();
        d.push_back(10);
        d.push_back(20);
        d.push_back(30); // spill
        assert_eq!(d.pop_front(), Some(10));
        assert_eq!(d.pop_front(), Some(20));
        assert_eq!(d.pop_front(), Some(30));
        assert_eq!(d.pop_front(), None);
    }

    #[test]
    fn test_deque_spill_front_back_after_spill() {
        let mut d: SmallDeque<i32, 2> = SmallDeque::new();
        for i in 0..10 {
            d.push_back(i);
        }
        assert!(!d.is_on_stack());
        assert_eq!(d.front(), Some(&0));
        assert_eq!(d.back(), Some(&9));
        assert_eq!(d.len(), 10);
    }

    // ─── clear ────────────────────────────────────────────────────────────────
    #[test]
    fn test_deque_stack_ops_clear() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        d.push_back(1);
        d.push_back(2);
        d.clear();
        assert!(d.is_empty());
        assert!(d.is_on_stack());
        // Reusable after clear
        d.push_back(3);
        assert_eq!(d.pop_front(), Some(3));
    }

    // ─── get / as_slices ─────────────────────────────────────────────────────
    #[test]
    fn test_deque_stack_ops_get() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        d.push_back(10);
        d.push_back(20);
        d.push_back(30);
        assert_eq!(d.get(0), Some(&10));
        assert_eq!(d.get(2), Some(&30));
        assert_eq!(d.get(99), None);
    }

    #[test]
    fn test_deque_stack_ops_as_slices_contiguous() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        d.push_back(1);
        d.push_back(2);
        let (s1, s2) = d.as_slices();
        assert_eq!(s1, &[1, 2]);
        assert!(s2.is_empty());
    }

    #[test]
    fn test_deque_traits_comparison() {
        let d1: SmallDeque<i32, 4> = vec![1, 2, 3].into_iter().collect();
        let d2: SmallDeque<i32, 4> = vec![1, 2, 3].into_iter().collect();
        let d3: SmallDeque<i32, 4> = vec![1, 2, 4].into_iter().collect();
        let d4: SmallDeque<i32, 4> = vec![1, 2].into_iter().collect();

        assert_eq!(d1, d2);
        assert!(d1 < d3);
        assert!(d1 > d4);
        assert!(d3 > d1);
    }

    // ─── iter ────────────────────────────────────────────────────────────────
    #[test]
    fn test_deque_traits_iter_stack() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        d.push_back(1);
        d.push_back(2);
        d.push_back(3);
        // SmallDeque has no .iter(); use as_slices to verify logical order
        let (s1, s2) = d.as_slices();
        let v: Vec<_> = s1.iter().chain(s2.iter()).cloned().collect();
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn test_deque_traits_into_iter_stack() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        d.push_back(1);
        d.push_back(2);
        d.push_back(3);
        // SmallDeque has no IntoIterator; drain via pop_front
        let mut v = Vec::new();
        while let Some(x) = d.pop_front() {
            v.push(x);
        }
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn test_deque_traits_into_iter_heap() {
        let mut d: SmallDeque<i32, 2> = SmallDeque::new();
        d.extend([1, 2, 3, 4]);
        assert!(!d.is_on_stack());
        let mut v = Vec::new();
        while let Some(x) = d.pop_front() {
            v.push(x);
        }
        assert_eq!(v, vec![1, 2, 3, 4]);
    }

    // ─── FromIterator / Extend ────────────────────────────────────────────────
    #[test]
    fn test_deque_traits_from_iter_and_extend() {
        let d: SmallDeque<i32, 4> = vec![1, 2].into_iter().collect();
        assert_eq!(d.len(), 2);

        let mut d2: SmallDeque<i32, 4> = SmallDeque::new();
        d2.extend(vec![10, 20, 30]);
        assert_eq!(d2.len(), 3);
    }

    // ─── clone ────────────────────────────────────────────────────────────────
    #[test]
    fn test_deque_traits_clone_stack() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::from_iter(vec![1, 2, 3]);
        let mut cloned = d.clone();
        d.push_back(4);
        assert_eq!(d.len(), 4);
        assert_eq!(cloned.len(), 3);
        assert_eq!(cloned.pop_front(), Some(1));
    }

    #[test]
    fn test_deque_traits_clone_heap() {
        let d: SmallDeque<i32, 2> = vec![1, 2, 3, 4].into_iter().collect();
        let cloned = d.clone();
        assert!(!cloned.is_on_stack());
        assert_eq!(cloned.len(), 4);
    }

    // ─── Debug / PartialEq ────────────────────────────────────────────────────
    #[test]
    fn test_deque_traits_debug_and_eq() {
        let d: SmallDeque<i32, 4> = vec![1, 2, 3].into_iter().collect();
        let d2: SmallDeque<i32, 4> = vec![1, 2, 3].into_iter().collect();
        assert_eq!(d, d2);
        let debug = format!("{:?}", d);
        assert!(debug.contains('1'));
    }

    // ─── AnyDeque trait dispatch ──────────────────────────────────────────────
    #[test]
    fn test_deque_any_deque_trait() {
        let mut d: SmallDeque<i32, 4> = SmallDeque::new();
        let any: &mut dyn AnyDeque<i32> = &mut d;
        any.push_back(10);
        any.push_front(5);
        assert_eq!(any.len(), 2);
        assert!(!any.is_empty());
        assert_eq!(any.front(), Some(&5));
        assert_eq!(any.back(), Some(&10));
        assert_eq!(any.pop_front(), Some(5));
        any.clear();
        assert!(any.is_empty());
    }
}
