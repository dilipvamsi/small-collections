use bitvec::prelude::{BitOrder, Lsb0};
use core::marker::PhantomData;
use heapless::Vec as HVec;

use crate::bitvec::AnyBitVec;

/// A stack-allocated bit vector.
///
/// # Overview
/// `N` is the capacity in **BYTES**.
/// Total bit capacity = `N * 8`.
/// Uses `bitvec`'s `BitOrder` for defining how bits are arranged within bytes.
#[derive(Debug)]
pub struct HeaplessBitVec<const N: usize, O: BitOrder = Lsb0> {
    bytes: HVec<u8, N>,
    bit_len: usize,
    _order: PhantomData<O>,
}

impl<const N: usize, O: BitOrder> Clone for HeaplessBitVec<N, O> {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
            bit_len: self.bit_len,
            _order: PhantomData,
        }
    }
}

impl<const N: usize, O: BitOrder> HeaplessBitVec<N, O> {
    pub fn new() -> Self {
        Self {
            bytes: HVec::new(),
            bit_len: 0,
            _order: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.bit_len
    }

    pub fn is_empty(&self) -> bool {
        self.bit_len == 0
    }

    /// Push a bit. Returns `Err(val)` if the stack is full.
    pub fn push(&mut self, val: bool) -> Result<(), bool> {
        let bit_idx = self.bit_len % 8;

        // If we are starting a new byte, we need to push a zero-byte to the underlying Vec.
        if bit_idx == 0 {
            if self.bytes.push(0).is_err() {
                return Err(val); // Stack Full
            }
        }

        if val {
            self.set(self.bit_len, true);
        }

        self.bit_len += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<bool> {
        if self.bit_len == 0 {
            return None;
        }

        let val = self.get(self.bit_len - 1).unwrap();
        self.bit_len -= 1;

        // If we just removed the last bit of a byte, pop the byte to save space
        if self.bit_len % 8 == 0 {
            self.bytes.pop();
        }

        Some(val)
    }

    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.bit_len {
            return None;
        }
        let byte_idx = index / 8;
        let bit_idx = index % 8;

        use bitvec::slice::BitSlice;
        let byte = self.bytes[byte_idx];
        let slice = BitSlice::<u8, O>::from_element(&byte);
        slice.get(bit_idx).as_deref().copied()
    }

    pub fn set(&mut self, index: usize, value: bool) {
        if index >= self.bit_len && index != self.bit_len {
            panic!("Index out of bounds");
        }
        let byte_idx = index / 8;
        let bit_idx = index % 8;

        use bitvec::slice::BitSlice;
        let byte = &mut self.bytes[byte_idx];
        let slice = BitSlice::<u8, O>::from_element_mut(byte);
        slice.set(bit_idx, value);
    }

    /// Returns the raw underlying bytes.
    pub fn as_raw_slice(&self) -> &[u8] {
        &self.bytes
    }
}

impl<const N: usize, O: BitOrder> Default for HeaplessBitVec<N, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, O: BitOrder> AnyBitVec for HeaplessBitVec<N, O> {
    fn len(&self) -> usize {
        self.len()
    }

    fn get(&self, index: usize) -> Option<bool> {
        self.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::{Lsb0, Msb0};

    #[test]
    fn test_heapless_bitvec_stack_ops_lsb0() {
        let mut bv: HeaplessBitVec<1, Lsb0> = HeaplessBitVec::new();
        bv.push(true).unwrap(); // 0b00000001
        bv.push(false).unwrap(); // 0b00000001
        bv.push(true).unwrap(); // 0b00000101
        assert_eq!(bv.as_raw_slice()[0], 0b00000101);
        assert_eq!(bv.get(0), Some(true));
        assert_eq!(bv.get(1), Some(false));
        assert_eq!(bv.get(2), Some(true));
    }

    #[test]
    fn test_heapless_bitvec_stack_ops_msb0() {
        let mut bv: HeaplessBitVec<1, Msb0> = HeaplessBitVec::new();
        bv.push(true).unwrap(); // 0b10000000
        bv.push(false).unwrap(); // 0b10000000
        bv.push(true).unwrap(); // 0b10100000
        assert_eq!(bv.as_raw_slice()[0], 0b10100000);
        assert_eq!(bv.get(0), Some(true));
        assert_eq!(bv.get(1), Some(false));
        assert_eq!(bv.get(2), Some(true));
    }

    #[test]
    fn test_heapless_bitvec_stack_ops_set_get() {
        let mut bv: HeaplessBitVec<1, Lsb0> = HeaplessBitVec::new();
        for _ in 0..8 {
            bv.push(false).unwrap();
        }
        bv.set(0, true);
        bv.set(7, true);
        assert_eq!(bv.get(0), Some(true));
        assert_eq!(bv.get(7), Some(true));
        assert_eq!(bv.get(1), Some(false));
    }

    #[test]
    fn test_heapless_bitvec_stack_ops_empty_len() {
        let mut bv: HeaplessBitVec<1, Lsb0> = HeaplessBitVec::new();
        assert!(bv.is_empty());
        bv.push(true).unwrap();
        assert!(!bv.is_empty());
        assert_eq!(bv.len(), 1);
    }

    #[test]
    fn test_heapless_bitvec_traits_clone_default() {
        let bv1: HeaplessBitVec<1, Lsb0> = HeaplessBitVec::default();
        let mut bv2 = bv1.clone();
        bv2.push(true).unwrap();
        assert_eq!(bv2.len(), 1);
        assert_eq!(bv1.len(), 0);
    }

    #[test]
    fn test_heapless_bitvec_stack_ops_boundary_fill() {
        let mut bv: HeaplessBitVec<1, Lsb0> = HeaplessBitVec::new();
        for _ in 0..8 {
            bv.push(true).unwrap();
        }
        assert!(bv.push(true).is_err());
        assert_eq!(bv.len(), 8);
    }

    #[test]
    fn test_heapless_bitvec_stack_ops_pop() {
        let mut bv: HeaplessBitVec<1, Lsb0> = HeaplessBitVec::new();
        bv.push(true).unwrap();
        bv.push(false).unwrap();
        assert_eq!(bv.pop(), Some(false));
        assert_eq!(bv.pop(), Some(true));
        assert_eq!(bv.pop(), None);
    }

    #[test]
    fn test_heapless_bitvec_traits_any_bitvec_impl() {
        let mut bv: HeaplessBitVec<1, Lsb0> = HeaplessBitVec::new();
        bv.push(true).unwrap();

        let any: &dyn AnyBitVec = &bv;
        assert_eq!(any.len(), 1);
        assert_eq!(any.get(0), Some(true));
    }
}
