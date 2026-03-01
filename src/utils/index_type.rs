#![cfg(feature = "lru")]
//! Shared index types for stack-allocated collections.

use core::hash::Hash;
use std::fmt::Debug;

/// A sealed trait for integer types used as **compact doubly-linked-list node indices**.
///
/// Instead of pointer-based links, our heapless collections store indices into arrays.
/// This keeps the structure `#[no_std]`-friendly and saves 8 bytes per pointer on 64-bit
/// platforms.
pub trait IndexType: Copy + Eq + Hash + Debug + 'static {
    /// Sentinel value indicating "no node" (analogous to a null pointer).
    /// Used for representing the end of a linked list.
    const NONE: Self;

    /// The first valid index (typically 0).
    const ZERO: Self;

    /// Converts this index to a `usize` for array access.
    ///
    /// # Safety
    /// This is used for array indexing; ensure the value is within bounds.
    fn as_usize(self) -> usize;

    /// Converts a `usize` slot index to this compact type.
    ///
    /// # Panics
    /// May panic if `i` is too large for the underlying type (e.g., > 255 for `u8`).
    fn from_usize(i: usize) -> Self;

    /// Increments the index (self + 1).
    /// Used for iterating through slots or incrementing size counters.
    fn inc(self) -> Self;

    /// Decrements the index (self - 1).
    /// Used for decrementing size counters.
    fn dec(self) -> Self;

    /// Returns true if the index is zero.
    fn is_zero(self) -> bool;
}

impl IndexType for u8 {
    const NONE: Self = 255;
    const ZERO: Self = 0;
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn from_usize(i: usize) -> Self {
        i as u8
    }
    #[inline(always)]
    fn inc(self) -> Self {
        self + 1
    }
    #[inline(always)]
    fn dec(self) -> Self {
        self - 1
    }
    #[inline(always)]
    fn is_zero(self) -> bool {
        self == 0
    }
}

impl IndexType for u16 {
    const NONE: Self = 65535;
    const ZERO: Self = 0;
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn from_usize(i: usize) -> Self {
        i as u16
    }
    #[inline(always)]
    fn inc(self) -> Self {
        self + 1
    }
    #[inline(always)]
    fn dec(self) -> Self {
        self - 1
    }
    #[inline(always)]
    fn is_zero(self) -> bool {
        self == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_index_type<I: IndexType>() {
        let zero = I::ZERO;
        assert!(zero.is_zero());
        assert_eq!(zero.as_usize(), 0);

        let one = zero.inc();
        assert!(!one.is_zero());
        assert_eq!(one.as_usize(), 1);

        let zero_again = one.dec();
        assert!(zero_again.is_zero());

        let from = I::from_usize(10);
        assert_eq!(from.as_usize(), 10);

        let none = I::NONE;
        assert_ne!(none.as_usize(), 0);
    }

    #[test]
    fn test_u8_index() {
        test_index_type::<u8>();
    }

    #[test]
    fn test_u16_index() {
        test_index_type::<u16>();
    }
}
