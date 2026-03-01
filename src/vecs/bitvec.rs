#![cfg(feature = "bitvec")]
//! Bit vector that lives on the stack and spills to the heap.
//!
//! Provides [`SmallBitVec`] — backed by [`HeaplessBitVec`] on the stack and
//! `bitvec::prelude::BitVec` on the heap.  [`AnyBitVec`] is the object-safe
//! inspection trait implemented by both.

use crate::HeaplessBitVec;
use bitvec::prelude::{BitOrder, BitSlice, BitVec, Lsb0};
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use core::ptr;
use std::borrow::{Borrow, BorrowMut};

/// A trait for uniform inspection of bit vectors.
pub trait AnyBitVec {
    /// Returns the number of bits in the bit vector.
    fn len(&self) -> usize;

    /// Returns `true` if the bit vector contains no bits.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the bit value at the given index, or `None` if out of bounds.
    fn get(&self, index: usize) -> Option<bool>;
}

impl<O: BitOrder> AnyBitVec for BitVec<u8, O> {
    fn len(&self) -> usize {
        self.as_bitslice().len()
    }
    fn get(&self, index: usize) -> Option<bool> {
        self.as_bitslice().get(index).map(|b| *b)
    }
}

/// A bit vector that lives on the stack for `N * 8` bits, then spills to a heap
/// `bitvec::BitVec`.
///
/// # Capacity
/// `N` is measured in **bytes**; total bit capacity is `N * 8`.
///
/// # Generic parameters
/// | Parameter | Meaning |
/// |-----------|--------|
/// | `N` | Stack capacity in bytes |
/// | `O` | Bit ordering within each byte; defaults to `Lsb0` |
///
/// # Design Considerations
/// - **Byte-granular storage**: bits are packed into `u8` bytes by `HeaplessBitVec`.
///   Spill copies them directly into a `BitVec<u8, O>`, keeping the same byte layout.
/// - **`on_stack` tag**: the `BitData` union is discriminated by `on_stack`, just like
///   all other `Small*` types in this crate.
///
/// # Safety
/// `on_stack` determines which variant of `BitData` is active.  Only the active
/// variant may be accessed.
pub struct SmallBitVec<const N: usize, O: BitOrder = Lsb0> {
    on_stack: bool,
    data: BitData<N, O>,
}

/// Internal storage for `SmallBitVec`.
union BitData<const N: usize, O: BitOrder> {
    stack: ManuallyDrop<HeaplessBitVec<N, O>>,
    // We use u8 as the storage primitive to match HeaplessBitVec.
    heap: ManuallyDrop<BitVec<u8, O>>,
}

impl<const N: usize, O: BitOrder> SmallBitVec<N, O> {
    /// Creates a new empty SmallBitVec.
    ///
    /// # Capacity
    /// `N` represents **BYTES**.
    /// Stack bit capacity = `N * 8`.
    pub fn new() -> Self {
        const {
            assert!(
                std::mem::size_of::<Self>() <= 16 * 1024,
                "SmallBitVec is too large! Reduce N."
            );
        }
        Self {
            on_stack: true,
            data: BitData {
                stack: ManuallyDrop::new(HeaplessBitVec::new()),
            },
        }
    }

    /// Returns the number of bits in the vector.
    #[inline(always)]
    pub fn len(&self) -> usize {
        unsafe {
            if self.on_stack {
                self.data.stack.len()
            } else {
                self.data.heap.len()
            }
        }
    }

    /// Returns `true` if the vector contains no bits.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the vector is currently storing data on the stack.
    #[inline(always)]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }

    /// Appends a bit to the back of the collection.
    ///
    /// If the stack capacity `N` is exceeded, this triggers a transparent spill to the heap.
    #[inline(always)]
    pub fn push(&mut self, value: bool) {
        unsafe {
            if self.on_stack {
                let stack = &mut *self.data.stack;
                // Attempt to push to stack
                if let Err(_) = stack.push(value) {
                    // Failure means full: Spill and retry
                    self.spill_to_heap();
                    (*self.data.heap).push(value);
                }
            } else {
                (*self.data.heap).push(value);
            }
        }
    }

    /// Removes the last bit from a vector and returns it, or `None` if it is empty.
    #[inline(always)]
    pub fn pop(&mut self) -> Option<bool> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).pop()
            } else {
                (*self.data.heap).pop()
            }
        }
    }

    /// Returns the bit value at the given index, or `None` if out of bounds.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<bool> {
        unsafe {
            if self.on_stack {
                self.data.stack.get(index)
            } else {
                self.data.heap.get(index)
            }
        }
    }

    /// Sets the bit value at the given index.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: bool) {
        unsafe {
            if self.on_stack {
                (*self.data.stack).set(index, value);
            } else {
                (*self.data.heap).set(index, value);
            }
        }
    }

    /// Returns a `BitSlice` view of the bits.
    pub fn as_bitslice(&self) -> &BitSlice<u8, O> {
        unsafe {
            if self.on_stack {
                self.data.stack.as_bitslice()
            } else {
                self.data.heap.as_bitslice()
            }
        }
    }

    /// Returns a mutable `BitSlice` view of the bits.
    pub fn as_bitslice_mut(&mut self) -> &mut BitSlice<u8, O> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).as_bitslice_mut()
            } else {
                (*self.data.heap).as_mut_bitslice()
            }
        }
    }

    /// The Critical Spill Function
    /// Moves data from HeaplessBitVec -> bitvec::BitVec
    #[inline(never)]
    unsafe fn spill_to_heap(&mut self) {
        unsafe {
            let stack = ManuallyDrop::take(&mut self.data.stack);

            // 2. Create BitVec from the raw bytes
            let mut heap_vec: BitVec<u8, O> = BitVec::from_slice(stack.as_raw_slice());

            // 3. Truncate to exact length
            heap_vec.truncate(stack.len());

            // 4. Reserve extra capacity
            heap_vec.reserve(stack.len());

            // 5. Update state
            // CRITICAL: We use ptr::write to avoid dropping the "old" bits (garbage data)
            // in the target union slot.
            ptr::write(&mut self.data.heap, ManuallyDrop::new(heap_vec));
            self.on_stack = false;
        }
    }
}

impl<const N: usize, O: BitOrder> Deref for SmallBitVec<N, O> {
    type Target = BitSlice<u8, O>;

    fn deref(&self) -> &Self::Target {
        self.as_bitslice()
    }
}

impl<const N: usize, O: BitOrder> DerefMut for SmallBitVec<N, O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_bitslice_mut()
    }
}

impl<const N: usize, O: BitOrder> Borrow<BitSlice<u8, O>> for SmallBitVec<N, O> {
    fn borrow(&self) -> &BitSlice<u8, O> {
        self.as_bitslice()
    }
}

impl<const N: usize, O: BitOrder> BorrowMut<BitSlice<u8, O>> for SmallBitVec<N, O> {
    fn borrow_mut(&mut self) -> &mut BitSlice<u8, O> {
        self.as_bitslice_mut()
    }
}

// --- Trait Impls ---

impl<const N: usize, O: BitOrder> AnyBitVec for SmallBitVec<N, O> {
    fn len(&self) -> usize {
        self.len()
    }

    fn get(&self, index: usize) -> Option<bool> {
        self.get(index)
    }
}

impl<const N: usize, O: BitOrder> Drop for SmallBitVec<N, O> {
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

impl<const N: usize, O: BitOrder> Clone for SmallBitVec<N, O> {
    fn clone(&self) -> Self {
        unsafe {
            if self.on_stack {
                SmallBitVec {
                    on_stack: true,
                    data: BitData {
                        stack: self.data.stack.clone(),
                    },
                }
            } else {
                SmallBitVec {
                    on_stack: false,
                    data: BitData {
                        heap: self.data.heap.clone(),
                    },
                }
            }
        }
    }
}

impl<const N: usize, O: BitOrder> core::fmt::Debug for SmallBitVec<N, O> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SmallBitVec [")?;
        for i in 0..self.len() {
            if self.get(i).unwrap_or(false) {
                f.write_str("1")?;
            } else {
                f.write_str("0")?;
            }
            if (i + 1) % 8 == 0 && i + 1 != self.len() {
                f.write_str("_")?;
            }
        }
        f.write_str("]")
    }
}

impl<const N: usize, O: BitOrder> Default for SmallBitVec<N, O> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod bitvec_basic_tests {
    use super::*;
    use bitvec::prelude::{Lsb0, Msb0};

    #[test]
    fn test_bitvec_traits_bitslice() {
        use std::borrow::BorrowMut;
        let mut sbv: SmallBitVec<1, Lsb0> = SmallBitVec::new();
        sbv.push(true);
        sbv.push(false);

        // Deref to BitSlice
        let slice: &bitvec::slice::BitSlice<u8, Lsb0> = &sbv;
        assert_eq!(slice.len(), 2);

        // BorrowMut
        let slice_mut: &mut bitvec::slice::BitSlice<u8, Lsb0> = sbv.borrow_mut();
        slice_mut.set(1, true);
        assert_eq!(sbv.get(1), Some(true));

        // Spill and test again
        for _ in 0..10 {
            sbv.push(true);
        }
        assert!(!sbv.is_on_stack());

        let slice_heap: &bitvec::slice::BitSlice<u8, Lsb0> = &sbv;
        assert_eq!(slice_heap.len(), 12);
    }

    #[test]
    fn test_bitvec_spill_trigger_lsb0() {
        let mut sbv: SmallBitVec<1, Lsb0> = SmallBitVec::new();
        for _ in 0..8 {
            sbv.push(true);
        }
        assert!(sbv.is_on_stack());
        assert_eq!(sbv.len(), 8);

        sbv.push(false); // Should spill
        assert!(!sbv.is_on_stack());
        assert_eq!(sbv.len(), 9);
        assert_eq!(sbv.get(0), Some(true));
        assert_eq!(sbv.get(8), Some(false));
    }

    #[test]
    fn test_bitvec_spill_trigger_msb0() {
        let mut sbv: SmallBitVec<1, Msb0> = SmallBitVec::new();
        sbv.push(true); // bit 0
        sbv.push(false); // bit 1
        for _ in 0..6 {
            sbv.push(true);
        }
        assert!(sbv.is_on_stack());

        sbv.push(true); // Should spill
        assert!(!sbv.is_on_stack());
        assert_eq!(sbv.get(0), Some(true));
        assert_eq!(sbv.get(1), Some(false));
    }

    #[test]
    fn test_bitvec_any_storage_pop_set() {
        let mut sbv: SmallBitVec<1, Lsb0> = SmallBitVec::new();
        sbv.push(true);
        sbv.push(false);
        assert_eq!(sbv.pop(), Some(false));
        sbv.set(0, false);
        assert_eq!(sbv.get(0), Some(false));

        for _ in 0..10 {
            sbv.push(true);
        }
        assert!(!sbv.is_on_stack());
        sbv.set(10, false);
        assert_eq!(sbv.get(10), Some(false));
        assert_eq!(sbv.pop(), Some(false));
    }

    #[test]
    fn test_bitvec_traits_debug_default() {
        let sbv: SmallBitVec<1, Lsb0> = SmallBitVec::default();
        assert!(sbv.is_empty());

        let mut sbv2 = SmallBitVec::<1, Lsb0>::new();
        sbv2.push(true);
        sbv2.push(false);
        let debug = format!("{:?}", sbv2);
        assert!(debug.contains("10"));
    }

    #[test]
    fn test_bitvec_any_storage_pop_empty() {
        let mut sbv: SmallBitVec<1, Lsb0> = SmallBitVec::new();
        assert_eq!(sbv.pop(), None);
    }

    #[test]
    fn test_bitvec_any_storage_get_none() {
        let mut sbv: SmallBitVec<1, Lsb0> = SmallBitVec::new();
        assert_eq!(sbv.get(10), None);

        for _ in 0..10 {
            sbv.push(true);
        }
        assert_eq!(sbv.get(100), None);
    }

    #[test]
    fn test_bitvec_traits_debug_multi_byte() {
        let mut sbv2: SmallBitVec<2, Lsb0> = SmallBitVec::new();
        for _ in 0..16 {
            sbv2.push(true);
        }
        let debug = format!("{:?}", sbv2);
        assert!(debug.contains("_"));
    }

    #[test]
    fn test_bitvec_traits_any_bitvec_impl() {
        let mut sbv: SmallBitVec<1, Lsb0> = SmallBitVec::new();
        sbv.push(true);
        sbv.push(false);

        let any: &dyn AnyBitVec = &sbv;
        assert_eq!(any.len(), 2);
        assert_eq!(any.get(0), Some(true));
        assert_eq!(any.get(1), Some(false));
    }
}

#[cfg(test)]
mod bitvec_coverage_tests {
    use super::*;
    use bitvec::prelude::Lsb0;

    #[test]
    fn test_any_bitvec_is_empty() {
        struct MockAny;
        impl AnyBitVec for MockAny {
            fn len(&self) -> usize {
                0
            }
            fn get(&self, _index: usize) -> Option<bool> {
                None
            }
        }
        let m = MockAny;
        assert!(m.is_empty());

        let mut b: BitVec<u8, Lsb0> = BitVec::new();
        {
            let any_b: &dyn AnyBitVec = &b;
            assert_eq!(any_b.len(), 0);
        }
        b.push(true);
        {
            let any_b: &dyn AnyBitVec = &b;
            assert_eq!(any_b.get(0), Some(true));
        }
    }

    #[test]
    fn test_small_bitvec_traits_and_heap() {
        use core::ops::DerefMut;
        use std::borrow::Borrow;

        let mut sbv: SmallBitVec<1, Lsb0> = SmallBitVec::new();
        sbv.push(true);

        // AnyBitVec len / get
        let any: &dyn AnyBitVec = &sbv;
        assert_eq!(any.len(), 1);
        assert_eq!(any.get(0), Some(true));

        // Borrow
        let borrowed: &bitvec::slice::BitSlice<u8, Lsb0> = sbv.borrow();
        assert_eq!(borrowed.len(), 1);

        // Clone stack
        let cloned_stack = sbv.clone();
        assert_eq!(cloned_stack.len(), 1);

        // DerefMut heap
        for _ in 0..10 {
            sbv.push(false);
        }
        let mut_slice = sbv.deref_mut();
        mut_slice.set(1, true);

        // Clone heap
        let cloned_heap = sbv.clone();
        assert_eq!(cloned_heap.len(), 11);
    }
}
