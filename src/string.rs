//! UTF-8 string that lives on the stack and spills to the heap.
//!
//! Provides [`SmallString`] — backed by `heapless::String<N>` (stack) and
//! `std::string::String` (heap).  Spill uses `from_utf8_unchecked` to skip a redundant
//! UTF-8 scan since both sources already guarantee valid UTF-8, giving a memcpy-only
//! migration cost.
//!
//! Implements `Deref<Target = str>` so all `&str` methods are available directly.
//! [`AnyString`] provides an object-safe trait over both backends.

use core::mem::ManuallyDrop;
use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};

/// A trait for abstraction over different string types (Stack, Heap, Small).
pub trait AnyString {
    fn as_str(&self) -> &str;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn push_str(&mut self, s: &str);
    fn push(&mut self, ch: char);
    fn clear(&mut self);
    fn pop(&mut self) -> Option<char>;
    fn truncate(&mut self, new_len: usize);
}

impl AnyString for String {
    fn as_str(&self) -> &str {
        self.as_str()
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn push_str(&mut self, s: &str) {
        self.push_str(s);
    }
    fn push(&mut self, ch: char) {
        self.push(ch);
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn pop(&mut self) -> Option<char> {
        self.pop()
    }
    fn truncate(&mut self, new_len: usize) {
        self.truncate(new_len);
    }
}

/// A string that lives on the stack for `N` bytes, then spills to the heap.
///
/// # Overview
/// This collection uses a `heapless::String` for stack storage and a
/// `std::string::String` for heap storage.
///
/// # Safety
/// * `on_stack` tag determines which side of the `StringData` union is active.
/// * `SmallString` ensures all data remains valid UTF-8 by leveraging the invariants
///   of the underlying collections and performing manual UTF-8 checks only when necessary.
pub struct SmallString<const N: usize> {
    on_stack: bool,
    data: StringData<N>,
}

impl<const N: usize> AnyString for SmallString<N> {
    fn as_str(&self) -> &str {
        self.as_str()
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn push_str(&mut self, s: &str) {
        self.push_str(s);
    }
    fn push(&mut self, ch: char) {
        self.push(ch);
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn pop(&mut self) -> Option<char> {
        self.pop()
    }
    fn truncate(&mut self, new_len: usize) {
        self.truncate(new_len);
    }
}

/// The internal storage for `SmallString`.
///
/// We use `ManuallyDrop` because the compiler cannot know which field is active
/// and therefore cannot automatically drop the correct one.
union StringData<const N: usize> {
    stack: ManuallyDrop<heapless::String<N>>,
    heap: ManuallyDrop<std::string::String>,
}

impl<const N: usize> SmallString<N> {
    /// The maximum allowed stack size in bytes (16 KB).
    pub const MAX_STACK_SIZE: usize = 16 * 1024;

    /// Creates a new empty SmallString.
    ///
    /// # Compile-Time Safety
    /// **Size Limit:** Enforces a limit of 16 KB. Exceeding this fails the build.
    ///
    /// ## Test: Valid (Compiles)
    /// ```rust
    /// use small_collections::SmallString;
    /// let s: SmallString<64> = SmallString::new();
    /// ```
    ///
    /// ## Test: Invalid Size (Fails Compilation)
    /// ```rust,compile_fail
    /// use small_collections::SmallString;
    /// // 32 KB string -> Too big for stack guard
    /// let s: SmallString<32768> = SmallString::new();
    /// ```
    ///
    pub fn new() -> Self {
        // COMPILER GUARD
        const {
            assert!(
                std::mem::size_of::<Self>() <= Self::MAX_STACK_SIZE,
                "SmallString is too large! The struct size exceeds the 16KB limit. Reduce N."
            );
        }

        Self {
            on_stack: true,
            data: StringData {
                stack: ManuallyDrop::new(heapless::String::new()),
            },
        }
    }

    /// Creates a SmallString from a literal or slice
    pub fn from_str(s: &str) -> Self {
        let mut str = Self::new();
        str.push_str(s);
        str
    }

    /// Returns `true` if the string is currently storing data on the stack.
    #[inline(always)]
    pub fn is_on_stack(&self) -> bool {
        self.on_stack
    }

    // --- Core Operations ---

    /// Appends the given `char` to the end of this `SmallString`.
    ///
    /// If the stack capacity `N` is exceeded, this triggers a transparent spill to the heap.
    #[inline(always)]
    pub fn push(&mut self, ch: char) {
        unsafe {
            if self.on_stack {
                // 1. Get reference to the inner stack string
                let stack_str = &mut *self.data.stack;

                // 2. Pre-emptive Size Check
                // We check if adding the char's byte length exceeds capacity N.
                if stack_str.len() + ch.len_utf8() > N {
                    self.spill_to_heap_and_push_char(ch);
                } else {
                    // 3. Guaranteed Success
                    // Since we checked capacity, this push will never fail.
                    // We can safely ignore the Result.
                    match stack_str.push(ch) {
                        Ok(()) => return, // Success: exit early
                        Err(_) => unreachable!("Stack capacity check failed in push"),
                    }
                }
            } else {
                (*self.data.heap).push(ch);
            }
        }
    }

    /// Appends a given string slice onto the end of this `SmallString`.
    ///
    /// If the stack capacity `N` is exceeded, this triggers a transparent spill to the heap.
    #[inline(always)]
    pub fn push_str(&mut self, s: &str) {
        unsafe {
            if self.on_stack {
                let stack_str = &mut *self.data.stack;

                // 2. Pre-emptive Size Check
                // s.len() returns the byte length, which matches N's definition.
                if stack_str.len() + s.len() > N {
                    self.spill_to_heap_and_push_str(s);
                } else {
                    // 3. Guaranteed Success
                    match stack_str.push_str(s) {
                        Ok(()) => return, // Success: exit early
                        Err(_) => unreachable!("Stack capacity check failed in push str"),
                    }
                }
            } else {
                (*self.data.heap).push_str(s);
            }
        }
    }

    /// Returns the length of this `SmallString`, in bytes, not [`char`]s or graphemes.
    #[inline(always)]
    pub fn len(&self) -> usize {
        // We can use the Deref trait here for simplicity,
        // but explicit access is slightly faster in debug builds.
        unsafe {
            if self.on_stack {
                self.data.stack.len()
            } else {
                self.data.heap.len()
            }
        }
    }

    /// Returns `true` if this `SmallString` has a length of zero, and `false` otherwise.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Truncates this `SmallString`, removing all contents.
    ///
    /// While this means the `SmallString` will have a length of zero, it does not
    /// touch its capacity.
    #[inline(always)]
    pub fn clear(&mut self) {
        unsafe {
            if self.on_stack {
                (*self.data.stack).clear();
            } else {
                (*self.data.heap).clear();
            }
        }
    }

    // --- Spill Logic ---

    #[inline(never)]
    unsafe fn spill_to_heap_and_push_str(&mut self, pending_str: &str) {
        unsafe {
            // 1. Get raw bytes from stack (Instant access)
            let stack_bytes = self.data.stack.as_bytes();
            let pending_bytes = pending_str.as_bytes();

            // 2. Calculate exact capacity needed
            // We use N*2 strategy to prevent frequent re-allocations after spill
            let total_len = stack_bytes.len() + pending_bytes.len();
            let cap = std::cmp::max(total_len, N * 2);

            // 3. Allocate RAW MEMORY (Vec<u8>)
            // We do NOT allocate a String yet. This avoids String metadata overhead.
            let mut heap_vec = Vec::with_capacity(cap);

            // 4. Bitwise Copy (Memcpy)
            // extend_from_slice compiles down to a highly optimized 'rep movsb' or SIMD copy.
            heap_vec.extend_from_slice(stack_bytes);
            heap_vec.extend_from_slice(pending_bytes);

            // 5. ZERO-COST TRANSFORMATION
            // We skip the O(N) UTF-8 validation scan.
            // Safety: We know 'stack_bytes' came from a valid str, and 'pending_str' is a valid str.
            let new_heap = String::from_utf8_unchecked(heap_vec);

            // 6. State Switch
            ManuallyDrop::drop(&mut self.data.stack);
            self.data.heap = ManuallyDrop::new(new_heap);
            self.on_stack = false;
        }
    }

    #[inline(never)]
    unsafe fn spill_to_heap_and_push_char(&mut self, pending_char: char) {
        unsafe {
            let stack_bytes = self.data.stack.as_bytes();

            // char can be up to 4 bytes
            let cap = std::cmp::max(stack_bytes.len() + 4, N * 2);
            let mut heap_vec = Vec::with_capacity(cap);

            heap_vec.extend_from_slice(stack_bytes);

            // Encode char directly into the buffer without intermediate String allocation
            // 'encode_utf8' writes bytes directly to the end of the Vec
            let char_len = pending_char.len_utf8();
            let old_len = heap_vec.len();

            // Reserve space and write directly (unsafe speedup)
            heap_vec.set_len(old_len + char_len);
            pending_char.encode_utf8(&mut heap_vec[old_len..]);

            let new_heap = String::from_utf8_unchecked(heap_vec);

            ManuallyDrop::drop(&mut self.data.stack);
            self.data.heap = ManuallyDrop::new(new_heap);
            self.on_stack = false;
        }
    }
}

// --- Critical Trait: Deref ---
// This allows SmallString to be used exactly like &str
// You can call .trim(), .split(), .contains() on it automatically.

impl<const N: usize> Deref for SmallString<N> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        unsafe {
            if self.on_stack {
                self.data.stack.as_str()
            } else {
                self.data.heap.as_str()
            }
        }
    }
}

impl<const N: usize> DerefMut for SmallString<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            if self.on_stack {
                (*self.data.stack).as_mut_str()
            } else {
                (*self.data.heap).as_mut_str()
            }
        }
    }
}

impl<const N: usize> SmallString<N> {
    /// Extracts a string slice containing the entire `SmallString`.
    ///
    /// Equivalent to `&s[..]`.
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        // We can leverage the Deref trait we already wrote
        &**self
    }

    /// Extracts a mutable string slice containing the entire `SmallString`.
    ///
    /// Equivalent to `&mut s[..]`.
    #[inline(always)]
    pub fn as_mut_str(&mut self) -> &mut str {
        // Leverage DerefMut
        &mut **self
    }

    /// Converts a `SmallString` into a byte slice.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        self.as_str().as_bytes()
    }

    /// Converts a mutable `SmallString` into a mutable byte slice.
    ///
    /// # Safety
    /// The caller must ensure that the content of the slice remains valid UTF-8.
    /// If this invariant is violated, it is Undefined Behavior.
    #[inline(always)]
    pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { self.as_mut_str().as_bytes_mut() }
    }
}

// --- Safety & Standard Traits ---

// Manual implementation of Clone.
// We must check `on_stack` to know which field to clone.
impl<const N: usize> Clone for SmallString<N> {
    fn clone(&self) -> Self {
        unsafe {
            if self.on_stack {
                // Clone the Stack string
                let stack_clone = (*self.data.stack).clone();
                SmallString {
                    on_stack: true,
                    data: StringData {
                        stack: ManuallyDrop::new(stack_clone),
                    },
                }
            } else {
                // Clone the Heap string
                let heap_clone = (*self.data.heap).clone();
                SmallString {
                    on_stack: false,
                    data: StringData {
                        heap: ManuallyDrop::new(heap_clone),
                    },
                }
            }
        }
    }
}

impl<const N: usize> Drop for SmallString<N> {
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

impl<const N: usize> Default for SmallString<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> fmt::Display for SmallString<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f) // Delegate to str implementation
    }
}

impl<const N: usize> fmt::Debug for SmallString<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

// Allow: let s: SmallString<32> = "Hello".into();
impl<const N: usize> From<&str> for SmallString<N> {
    fn from(s: &str) -> Self {
        Self::from_str(s)
    }
}

// Allow writing to it like a buffer: write!(s, "Val: {}", 42)
impl<const N: usize> fmt::Write for SmallString<N> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }
}

// --- Equality (==) ---

// Added `const M: usize` to allow comparing SmallStrings of different capacities.
// e.g., SmallString<16> == SmallString<2>
impl<const N: usize, const M: usize> PartialEq<SmallString<M>> for SmallString<N> {
    fn eq(&self, other: &SmallString<M>) -> bool {
        self.as_str() == other.as_str()
    }
}

// Eq implies PartialEq<Self>, which is satisfied by the implementation above where M = N.
impl<const N: usize> Eq for SmallString<N> {}

// --- Equality with other types (unchanged) ---

impl<const N: usize> PartialEq<str> for SmallString<N> {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

// Allow: small_string == "hello" (where "hello" is &str)
impl<'a, const N: usize> PartialEq<&'a str> for SmallString<N> {
    fn eq(&self, other: &&'a str) -> bool {
        self.as_str() == *other
    }
}

impl<const N: usize> PartialEq<SmallString<N>> for &str {
    fn eq(&self, other: &SmallString<N>) -> bool {
        *self == other.as_str()
    }
}

impl<const N: usize> PartialEq<String> for SmallString<N> {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

// --- Ordering (<, >) ---

// FIXED: Added `const M: usize` here as well.
impl<const N: usize, const M: usize> PartialOrd<SmallString<M>> for SmallString<N> {
    fn partial_cmp(&self, other: &SmallString<M>) -> Option<Ordering> {
        Some(self.as_str().cmp(other.as_str()))
    }
}

impl<const N: usize> Ord for SmallString<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

// --- Hashing ---

impl<const N: usize> Hash for SmallString<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // IMPORTANT: Hash the string slice, not the struct fields.
        // This ensures hash("abc") == hash(SmallString::from("abc"))
        self.as_str().hash(state);
    }
}

// --- Borrowing --

impl<const N: usize> Borrow<str> for SmallString<N> {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> BorrowMut<str> for SmallString<N> {
    fn borrow_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<const N: usize> AsRef<str> for SmallString<N> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> AsRef<[u8]> for SmallString<N> {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

// --- Extend ---

// Allow: let s: SmallString<16> = chars.collect();
impl<const N: usize> FromIterator<char> for SmallString<N> {
    fn from_iter<I: IntoIterator<Item = char>>(iter: I) -> Self {
        let mut s = Self::new();
        for c in iter {
            s.push(c); // push handles spilling automatically
        }
        s
    }
}

// Allow: let s: SmallString<16> = strings.collect();
impl<'a, const N: usize> FromIterator<&'a str> for SmallString<N> {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let mut s = Self::new();
        for str_slice in iter {
            s.push_str(str_slice);
        }
        s
    }
}

// Allow: s.extend(chars)
impl<const N: usize> Extend<char> for SmallString<N> {
    fn extend<I: IntoIterator<Item = char>>(&mut self, iter: I) {
        for c in iter {
            self.push(c);
        }
    }
}

// Allow: s.extend(strings)
impl<'a, const N: usize> Extend<&'a str> for SmallString<N> {
    fn extend<I: IntoIterator<Item = &'a str>>(&mut self, iter: I) {
        for str_slice in iter {
            self.push_str(str_slice);
        }
    }
}

// --- Pop, Truncate ---

impl<const N: usize> SmallString<N> {
    /// Removes the last character from the string buffer and returns it.
    /// Returns None if the string is empty.
    #[inline(always)]
    pub fn pop(&mut self) -> Option<char> {
        unsafe {
            if self.on_stack {
                (*self.data.stack).pop()
            } else {
                (*self.data.heap).pop()
            }
        }
    }

    /// Shortens this String to the specified length.
    /// If new_len >= current length, this does nothing.
    /// Panics if new_len does not lie on a char boundary.
    #[inline(always)]
    pub fn truncate(&mut self, new_len: usize) {
        unsafe {
            if self.on_stack {
                (*self.data.stack).truncate(new_len);
            } else {
                (*self.data.heap).truncate(new_len);
            }
        }
    }

    /// Returns the total capacity (in bytes) of the string.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        unsafe {
            if self.on_stack {
                N // Stack capacity is fixed at N
            } else {
                (*self.data.heap).capacity()
            }
        }
    }
}

// --- Reserve ---

impl<const N: usize> SmallString<N> {
    /// Ensures that this string has at least the specified capacity.
    /// If the request exceeds N, this forces a spill to the heap immediately.
    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            if self.on_stack {
                let len = self.len();
                if len + additional > N {
                    // Force spill logic
                    // We cheat slightly by calling spill with an empty string,
                    // but we modify the logic to respect `capacity`.

                    // 1. Copy Stack
                    let stack_bytes = self.data.stack.as_bytes();

                    // 2. Alloc Heap with requested size
                    let cap = std::cmp::max(len + additional, N * 2);
                    let mut heap_vec = Vec::with_capacity(cap);
                    heap_vec.extend_from_slice(stack_bytes);

                    let new_heap = String::from_utf8_unchecked(heap_vec);

                    // 3. Switch
                    ManuallyDrop::drop(&mut self.data.stack);
                    self.data.heap = ManuallyDrop::new(new_heap);
                    self.on_stack = false;
                }
                // Else: it fits on stack, do nothing.
            } else {
                (*self.data.heap).reserve(additional);
            }
        }
    }
}

// -- u8 --

impl<const N: usize> SmallString<N> {
    /// Converts a vector of bytes to a SmallString.
    ///
    /// If the bytes are not valid UTF-8, this returns an error.
    /// If valid, it attempts to store them on the Stack if they fit.
    pub fn from_utf8(vec: Vec<u8>) -> Result<Self, std::string::FromUtf8Error> {
        // 1. Check UTF-8 validity using std helper
        let s_std = String::from_utf8(vec)?;

        // 2. Try to fit on Stack
        if s_std.len() <= N {
            let mut small = Self::new();
            small.push_str(&s_std);
            Ok(small)
        } else {
            // 3. Keep on Heap
            Ok(Self {
                on_stack: false,
                data: StringData {
                    heap: ManuallyDrop::new(s_std),
                },
            })
        }
    }

    /// Converts a byte slice to a SmallString.
    pub fn from_utf8_lossy(bytes: &[u8]) -> Self {
        let cow = String::from_utf8_lossy(bytes);
        match cow {
            // It was already valid utf8 and borrowed
            std::borrow::Cow::Borrowed(s) => Self::from(s),
            // It had invalid chars and was replaced/allocated
            std::borrow::Cow::Owned(s) => {
                if s.len() <= N {
                    Self::from(s.as_str())
                } else {
                    Self {
                        on_stack: false,
                        data: StringData {
                            heap: ManuallyDrop::new(s),
                        },
                    }
                }
            }
        }
    }
}

// -- interoperability --

impl<const N: usize> SmallString<N> {
    /// Consumes the SmallString and returns a heap-allocated `std::string::String`.
    ///
    /// If already on the heap, this is free (zero allocation).
    /// If on the stack, this allocates a new String.
    pub fn into_string(self) -> String {
        // Prevent Drop from running, we are taking ownership
        let mut this = ManuallyDrop::new(self);

        unsafe {
            if this.on_stack {
                let stack_str = &*this.data.stack;
                stack_str.as_str().to_string()
            } else {
                // Take ownership of the heap string
                ManuallyDrop::take(&mut this.data.heap)
            }
        }
    }

    /// Consumes the SmallString and returns a `Vec<u8>`.
    pub fn into_bytes(self) -> Vec<u8> {
        self.into_string().into_bytes()
    }
}

// -- retian --

impl<const N: usize> SmallString<N> {
    /// Retains only the characters specified by the predicate.
    /// In other words, remove all characters `c` such that `f(c)` returns `false`.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(char) -> bool,
    {
        unsafe {
            if self.on_stack {
                // heapless::String doesn't implement retain directly.
                // We simulate it by rebuilding the string in-place (simplest safe approach)
                // or doing a "scan and shift".
                // Since N is small, rebuilding is acceptable overhead.

                let stack_str = &mut *self.data.stack;
                // Note: This is an O(N) approach using a temporary buffer strategy
                // implicitly via string manipulation, or we simply rely on the fact
                // that we can't easily mutate the private buffer of heapless.

                // Safe Workaround: Copy chars that pass to a temp buffer, then clear and push back.
                // Since N is small (stack), this copy is cheap.
                let mut temp: heapless::String<N> = heapless::String::new();
                for c in stack_str.chars() {
                    if f(c) {
                        match temp.push(c) {
                            Ok(()) => continue,
                            Err(_) => unreachable!("temp string capacity check failed in push"),
                        }
                    }
                }
                *stack_str = temp;
            } else {
                (*self.data.heap).retain(f);
            }
        }
    }
}

// -- shrink --

impl<const N: usize> SmallString<N> {
    /// Demands that the underlying buffer releases any extra capacity.
    /// Only affects the Heap state. Stack state size is fixed at `N`.
    pub fn shrink_to_fit(&mut self) {
        unsafe {
            if !self.on_stack {
                (*self.data.heap).shrink_to_fit();
            }
        }
    }
}

#[cfg(test)]
mod string_basic_tests {

    use super::*;
    use std::borrow::Borrow;
    use std::collections::HashSet;
    use std::collections::hash_map::DefaultHasher;
    use std::fmt::Write; // Required for write! macro tests
    use std::hash::{Hash, Hasher};

    #[test]
    fn test_string_traits_borrow() {
        use std::borrow::{Borrow, BorrowMut};
        let mut s: SmallString<16> = SmallString::from("abc");

        // Test Borrow<str>
        let b: &str = s.borrow();
        assert_eq!(b, "abc");

        // Test BorrowMut<str>
        let b_mut: &mut str = s.borrow_mut();
        b_mut.make_ascii_uppercase();
        assert_eq!(s.as_str(), "ABC");
    }

    #[test]
    fn test_string_stack_ops_basic() {
        let mut s: SmallString<16> = SmallString::new();

        assert!(s.is_on_stack());
        assert!(s.is_empty());

        s.push_str("Hello");
        assert_eq!(s.len(), 5);
        assert_eq!(s.as_str(), "Hello");
        assert!(s.is_on_stack());

        s.push(' ');
        s.push_str("World");
        assert_eq!(s.len(), 11);
        assert_eq!(&*s, "Hello World"); // Test Deref
        assert!(s.is_on_stack());
    }

    // --- 2. Exact Boundary Tests ---
    #[test]
    fn test_string_spill_trigger_on_exact_capacity() {
        let mut s: SmallString<5> = SmallString::new();

        // Fill exactly to capacity
        s.push_str("12345");

        assert_eq!(s.len(), 5);
        assert!(s.is_on_stack(), "Should remain on stack at exact capacity");

        // Push one more char -> Spill
        s.push('6');
        assert!(!s.is_on_stack(), "Should spill after N+1");
        assert_eq!(s.as_str(), "123456");
    }

    // --- 3. Spill via push_str (Multi-char) ---
    #[test]
    fn test_string_spill_trigger_on_push_str() {
        let mut s: SmallString<4> = SmallString::new();
        s.push_str("Hi"); // Len 2 (Stack)

        // Push string that exceeds remaining space AND total capacity
        s.push_str(" there"); // "Hi there" = 8 chars

        assert!(!s.is_on_stack());
        assert_eq!(s.as_str(), "Hi there");
        assert_eq!(s.len(), 8);
    }

    // --- 4. Spill via push (Single char) ---
    #[test]
    fn test_string_spill_trigger_on_push_char() {
        let mut s: SmallString<3> = SmallString::new();
        s.push('A');
        s.push('B');
        s.push('C'); // Full (3/3)

        s.push('D'); // Spill

        assert!(!s.is_on_stack());
        assert_eq!(s.as_str(), "ABCD");
    }

    // --- 5. UTF-8 & Emojis (CRITICAL for unsafe blocks) ---
    #[test]
    fn test_string_spill_trigger_on_multibyte_char() {
        // Emojis are 4 bytes each.
        // Capacity 6: Can hold 1 emoji (4 bytes) + 2 bytes.
        // Cannot hold 2 emojis (8 bytes).
        let mut s: SmallString<6> = SmallString::new();

        s.push('🦀'); // 4 bytes
        assert!(s.is_on_stack());
        assert_eq!(s.len(), 4);

        // Push another emoji. 4 + 4 = 8 bytes. Spill required.
        // This tests the `push(char)` spill path with multibyte chars.
        s.push('🚀');

        assert!(!s.is_on_stack());
        assert_eq!(s.as_str(), "🦀🚀");
        assert_eq!(s.len(), 8);
    }

    #[test]
    fn test_string_spill_trigger_on_multibyte_str() {
        // Test `push_str` splitting logic (though we copy fully)
        let mut s: SmallString<5> = SmallString::new();
        s.push_str("hi");

        // Push mixed ascii and unicode that causes spill
        // "hi" (2) + "_👋" (1 + 4) = 7 bytes total
        s.push_str("_👋");

        assert!(!s.is_on_stack());
        assert_eq!(s.as_str(), "hi_👋");
    }

    // --- 6. Formatting Macro (fmt::Write) ---
    #[test]
    fn test_string_traits_fmt_write() {
        let mut s: SmallString<16> = SmallString::new();

        // Should fit on stack
        write!(s, "Value: {}", 100).unwrap();
        assert_eq!(s.as_str(), "Value: 100");
        assert!(s.is_on_stack());

        // Should cause spill
        write!(s, " and a very long suffix to force spill").unwrap();
        assert!(!s.is_on_stack());
        assert!(s.contains("Value: 100"));
        assert!(s.contains("suffix"));
    }

    // --- 7. Large Allocation Strategy ---
    #[test]
    fn test_string_spill_trigger_on_large_growth() {
        let mut s: SmallString<4> = SmallString::new();
        s.push_str("12");

        // Push a huge string immediately
        // The spill logic `max(needed, N*2)` should handle this without crashing
        let huge_chunk = "a".repeat(100);
        s.push_str(&huge_chunk);

        assert!(!s.is_on_stack());
        assert_eq!(s.len(), 102);
        assert!(s.ends_with("aaaa"));
    }

    // --- 8. Zero Capacity Edge Case ---
    #[test]
    fn test_string_any_storage_zero_capacity() {
        // A SmallString that holds nothing on the stack
        let mut s: SmallString<0> = SmallString::new();

        assert!(s.is_on_stack()); // Technically starts on "stack" (empty)

        s.push('a');
        assert!(!s.is_on_stack()); // Immediate spill
        assert_eq!(s.as_str(), "a");
    }

    // --- 9. Clear & Reuse ---
    #[test]
    fn test_string_any_storage_clear_reuse() {
        let mut s: SmallString<4> = SmallString::new();

        // Stack -> Clear -> Stack
        s.push_str("abc");
        s.clear();
        assert!(s.is_empty());
        assert!(s.is_on_stack());
        s.push_str("xyz");
        assert_eq!(s.as_str(), "xyz");

        // Stack -> Spill -> Clear -> Heap (Remains on heap usually)
        s.push_str("12345"); // Spill
        assert!(!s.is_on_stack());

        s.clear();
        assert!(s.is_empty());
        assert!(!s.is_on_stack()); // Implementation detail: keeps heap allocation

        // Reuse heap allocation
        s.push_str("HeapReuse");
        assert_eq!(s.as_str(), "HeapReuse");
    }

    // --- 10. Deref Methods ---
    #[test]
    fn test_string_traits_deref_methods() {
        let s: SmallString<10> = SmallString::from("hello");

        // These methods come from standard `str`, working via Deref
        assert!(s.starts_with("he"));
        assert!(s.contains("ll"));
        assert_eq!(s.to_uppercase(), "HELLO"); // Returns a new Std String
        assert_eq!(s.find('o'), Some(4));
    }

    #[test]
    fn test_string_traits_clone_on_stack() {
        // N=16. "Hello" (5 bytes) fits on stack.
        let mut original: SmallString<16> = SmallString::new();
        original.push_str("Hello");

        assert!(original.is_on_stack());

        // Clone it
        let mut copy = original.clone();

        // Verify state
        assert!(copy.is_on_stack());
        assert_eq!(original.as_str(), copy.as_str());

        // Modify Copy -> Ensure Original is untouched
        copy.push_str(" World");

        assert_eq!(original.as_str(), "Hello");
        assert_eq!(copy.as_str(), "Hello World");
    }

    #[test]
    fn test_string_traits_clone_on_heap() {
        // N=4. "Hello" (5 bytes) forces a spill to Heap.
        let mut original: SmallString<4> = SmallString::new();
        original.push_str("Hello");

        assert!(!original.is_on_stack());

        // Clone it
        let mut copy = original.clone();

        // Verify state
        assert!(!copy.is_on_stack());
        assert_eq!(original.as_str(), copy.as_str());

        // Modify Copy -> Ensure Original is untouched
        copy.push_str(" World");

        assert_eq!(original.as_str(), "Hello");
        assert_eq!(copy.as_str(), "Hello World");
    }

    // --- Helper to verify Hashing consistency ---
    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    // --- 1. Equality & Ordering Tests ---
    #[test]
    fn test_string_traits_equality() {
        // Stack vs Stack
        let s1: SmallString<16> = SmallString::from("hello");
        let s2: SmallString<16> = SmallString::from("hello");
        let s3: SmallString<16> = SmallString::from("world");
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);

        // Heap vs Heap (N=2 forces spill)
        let h1: SmallString<2> = SmallString::from("hello");
        let h2: SmallString<2> = SmallString::from("hello");
        assert_eq!(h1, h2);
        assert!(!h1.is_on_stack());

        // Stack vs Heap
        // s1 is on stack, h1 is on heap. Content is same.
        assert_eq!(s1, h1);

        // Comparison with &str and String
        assert_eq!(s1, "hello");
        assert_eq!("hello", s1);
        assert_eq!(s1, String::from("hello"));
    }

    #[test]
    fn test_string_traits_ordering() {
        let apple: SmallString<16> = SmallString::from("Apple");
        let banana: SmallString<16> = SmallString::from("Banana");

        // "Apple" < "Banana"
        assert!(apple < banana);

        // Sort a vector of SmallStrings
        let mut list = vec![banana.clone(), apple.clone()];
        list.sort();

        assert_eq!(list[0], "Apple");
        assert_eq!(list[1], "Banana");
    }

    // --- 2. Hashing Tests ---
    #[test]
    fn test_string_traits_hashing() {
        let s_stack: SmallString<16> = SmallString::from("testing");
        let s_heap: SmallString<2> = SmallString::from("testing"); // Spills
        let s_std: String = String::from("testing");
        let s_str: &str = "testing";

        // IMPORTANT: The hash must be identical to standard Rust strings
        // for HashMap interoperability.
        let h1 = calculate_hash(&s_stack);
        let h2 = calculate_hash(&s_heap);
        let h3 = calculate_hash(&s_std);
        let h4 = calculate_hash(&s_str);

        assert_eq!(h1, h2, "Stack and Heap hashing differ!");
        assert_eq!(h1, h3, "SmallString hash differs from String hash!");
        assert_eq!(h1, h4, "SmallString hash differs from &str hash!");

        // Verify use in HashSet
        let mut set = HashSet::new();
        set.insert(s_stack.clone());

        assert!(set.contains("testing")); // Look up using &str
        assert!(set.contains(s_heap.as_str())); // Look up using Heap SmallString
    }

    // --- 3. Borrow & AsRef Tests ---
    #[test]
    fn test_string_traits_borrow_as_ref() {
        let s: SmallString<16> = SmallString::from("hello");

        // Function expecting &str
        fn takes_str(_: &str) {}
        takes_str(s.as_ref());
        takes_str(s.borrow());

        // Function expecting AsRef<str>
        fn takes_as_ref<T: AsRef<str>>(_: T) {}
        takes_as_ref(s); // Pass by value
    }

    // --- 4. Iterator Tests (FromIterator / Extend) ---
    #[test]
    fn test_string_traits_from_iterator() {
        // Collect chars -> Stack
        let chars = vec!['a', 'b', 'c'];
        let s_stack: SmallString<16> = chars.into_iter().collect();
        assert_eq!(s_stack, "abc");
        assert!(s_stack.is_on_stack());

        // Collect chars -> Heap
        let many_chars = vec!['a'; 100];
        let s_heap: SmallString<16> = many_chars.into_iter().collect();
        assert_eq!(s_heap.len(), 100);
        assert!(!s_heap.is_on_stack());

        // Collect strings
        let strings = vec!["Hello", " ", "World"];
        let s_str: SmallString<32> = strings.into_iter().collect();
        assert_eq!(s_str, "Hello World");
    }

    #[test]
    fn test_string_traits_extend() {
        let mut s: SmallString<4> = SmallString::new();

        // Extend a little (Stack)
        s.extend(vec!['H', 'i']);
        assert_eq!(s, "Hi");
        assert!(s.is_on_stack());

        // Extend a lot (Spill)
        s.extend(vec!['!'; 20]);
        assert!(!s.is_on_stack());
        assert_eq!(s.len(), 22);
    }

    // --- 5. Manipulation Tests (Pop / Truncate) ---
    #[test]
    fn test_string_any_storage_pop() {
        // Stack Pop
        let mut s: SmallString<16> = SmallString::from("abc");
        assert_eq!(s.pop(), Some('c'));
        assert_eq!(s.pop(), Some('b'));
        assert_eq!(s, "a");

        // Heap Pop
        let mut h: SmallString<2> = SmallString::from("abc"); // Spills
        assert!(!h.is_on_stack());
        assert_eq!(h.pop(), Some('c'));
        assert_eq!(h, "ab");

        // Empty Pop
        let mut empty: SmallString<4> = SmallString::new();
        assert_eq!(empty.pop(), None);
    }

    #[test]
    fn test_string_any_storage_truncate_v2() {
        // Stack Truncate
        let mut s: SmallString<16> = SmallString::from("Hello World");
        s.truncate(5);
        assert_eq!(s, "Hello");
        assert_eq!(s.len(), 5);

        // Truncate to larger size (Should do nothing)
        s.truncate(100);
        assert_eq!(s, "Hello");

        // Heap Truncate
        let mut h: SmallString<2> = SmallString::from("Hello World"); // Spills
        assert!(!h.is_on_stack());
        h.truncate(2);
        assert_eq!(h, "He");
    }

    #[test]
    #[should_panic]
    fn test_string_any_storage_truncate_panic() {
        let mut s: SmallString<16> = SmallString::from("🦀"); // 4 bytes
        // Truncating inside a multibyte char panics in std, should panic here too
        s.truncate(2);
    }

    // --- 6. Reserve (Spill Control) ---
    #[test]
    fn test_string_any_storage_reserve() {
        // Case 1: Reserve fits in Stack
        let mut s: SmallString<16> = SmallString::new();
        s.push_str("Hi");
        s.reserve(4); // 2 + 4 = 6 <= 16.
        assert!(s.is_on_stack());

        // Case 2: Reserve forces Spill
        // We are at len 2. We request 20 more. Total 22 > 16.
        // This should force a move to heap.
        s.reserve(20);

        assert!(!s.is_on_stack());
        assert_eq!(s.as_str(), "Hi");
        assert!(s.capacity() >= 22); // Correct: 2 + 20

        // Case 3: Reserve on existing Heap
        // We request 100 MORE bytes than current length (2).
        // Total required = 102.
        s.reserve(100);

        // FIX: Check against 102, not 122.
        assert!(s.capacity() >= 102);
    }

    #[test]
    fn test_string_any_storage_into_string() {
        // Stack -> String
        let s_stack: SmallString<16> = SmallString::from("stack");
        let std_str = s_stack.into_string();
        assert_eq!(std_str, "stack");
        // Verify type is actually String
        let _: String = std_str;

        // Heap -> String (Zero cost move)
        let s_heap: SmallString<2> = SmallString::from("heap");
        let std_str2 = s_heap.into_string();
        assert_eq!(std_str2, "heap");
    }

    #[test]
    fn test_string_any_storage_from_utf8() {
        let bytes = vec![104, 101, 108, 108, 111]; // "hello"

        // Fits on Stack
        let s: SmallString<16> = SmallString::from_utf8(bytes.clone()).unwrap();
        assert!(s.is_on_stack());
        assert_eq!(s, "hello");

        // Force Heap
        let s2: SmallString<2> = SmallString::from_utf8(bytes).unwrap();
        assert!(!s2.is_on_stack());
        assert_eq!(s2, "hello");

        // Invalid UTF-8
        let invalid = vec![0, 159, 146, 150];
        assert!(SmallString::<16>::from_utf8(invalid).is_err());
    }

    #[test]
    fn test_string_any_storage_retain() {
        // Stack Retain
        let mut s: SmallString<16> = SmallString::from("AbCdEf");
        s.retain(|c| c.is_lowercase());
        assert_eq!(s, "bdf");
        assert!(s.is_on_stack());

        // Heap Retain
        let mut h: SmallString<2> = SmallString::from("AbCdEf"); // Spills
        h.retain(|c| c.is_uppercase());
        assert_eq!(h, "ACE");
        assert!(!h.is_on_stack());
    }

    #[test]
    fn test_string_any_storage_into_bytes() {
        let s: SmallString<16> = SmallString::from("ABC");
        let bytes = s.into_bytes();
        assert_eq!(bytes, vec![65, 66, 67]);
    }

    #[test]
    fn test_string_any_storage_from_utf8_lossy() {
        let bytes = b"hello \xF0\x90\x80world"; // Invalid UTF-8
        let s: SmallString<16> = SmallString::from_utf8_lossy(bytes);
        assert!(s.contains("hello "));
        assert!(s.contains("world"));
        assert!(s.is_on_stack());

        let huge_invalid = b"a".repeat(100);
        let s2: SmallString<16> = SmallString::from_utf8_lossy(&huge_invalid);
        assert!(!s2.is_on_stack());
    }

    #[test]
    fn test_string_any_storage_as_bytes_mut() {
        let mut s: SmallString<16> = SmallString::from("abc");
        unsafe {
            let bytes = s.as_bytes_mut();
            bytes[0] = b'z';
        }
        assert_eq!(s, "zbc");
    }

    #[test]
    fn test_string_traits_debug_display() {
        let h: SmallString<2> = SmallString::from("a");
        let debug = format!("{:?}", h);
        assert_eq!(debug, "\"a\"");
        let display = format!("{}", h);
        assert_eq!(display, "a");
    }

    #[test]
    fn test_string_any_storage_truncate() {
        // truncate stack
        let mut s: SmallString<16> = SmallString::from("hello");
        s.truncate(2);
        assert_eq!(s, "he");

        // truncate heap
        let mut h: SmallString<2> = SmallString::from("abc");
        h.truncate(1);
        assert_eq!(h, "a");
    }

    #[test]
    fn test_string_any_storage_deref_mut() {
        let mut s_mut = SmallString::<16>::from("abc");
        s_mut.as_mut_str().make_ascii_uppercase();
        assert_eq!(s_mut, "ABC");

        let mut h: SmallString<1> = SmallString::from_str("abc");
        h.as_mut_str().make_ascii_uppercase();
        assert_eq!(h, "ABC");
    }

    #[test]
    fn test_string_any_storage_partial_eq_variants() {
        let h: SmallString<1> = SmallString::from_str("abc");
        assert!(h == "abc");
        let s_ref = "abc";
        assert!(h == s_ref);
    }

    #[test]
    fn test_string_any_storage_extend_char() {
        let mut h: SmallString<1> = SmallString::from_str("abc");
        h.extend(['!', '?'].iter().cloned());
        assert_eq!(h, "abc!?");
    }

    #[test]
    fn test_string_any_storage_from_utf8_long() {
        let long_bytes = b"a".repeat(100);
        let s_long = SmallString::<16>::from_utf8(long_bytes).unwrap();
        assert!(!s_long.is_on_stack());
    }
}

#[cfg(test)]
mod string_coverage_tests {
    use super::*;

    #[test]
    fn test_any_string_trait_std_string_implementation() {
        let mut s = String::from("abc");
        let any: &mut dyn AnyString = &mut s;
        assert_eq!(any.as_str(), "abc");
        assert_eq!(any.len(), 3);
        assert!(!any.is_empty());
        any.push('d');
        any.push_str("ef");
        assert_eq!(any.as_str(), "abcdef");
        assert_eq!(any.pop(), Some('f'));
        any.truncate(3);
        assert_eq!(any.as_str(), "abc");
        any.clear();
        assert!(any.is_empty());
    }

    #[test]
    fn test_any_string_trait_small_string_implementation() {
        let mut s: SmallString<4> = SmallString::new();
        let any: &mut dyn AnyString = &mut s;
        any.push_str("abc");
        assert_eq!(any.len(), 3);
        any.push('d');
        assert_eq!(any.as_str(), "abcd");
        any.pop();
        any.truncate(1);
        any.clear();
        assert!(any.is_empty());
    }

    #[test]
    fn test_small_string_heap_storage_utf8_validity_and_retain() {
        let mut s: SmallString<2> = SmallString::new();
        s.push_str("abc"); // heap
        assert!(!s.is_on_stack());
        assert!(s.capacity() >= 2);

        // from_utf8 heap
        let long_vec = vec![b'a'; 10];
        let s2 = SmallString::<2>::from_utf8(long_vec).unwrap();
        assert!(!s2.is_on_stack());

        // from_utf8_lossy heap
        let s3 = SmallString::<2>::from_utf8_lossy(&[b'a'; 10]);
        assert!(!s3.is_on_stack());

        // retain heap
        let mut s4: SmallString<2> = SmallString::from("abcde");
        s4.retain(|c| c != 'b');
        assert_eq!(s4.as_str(), "acde");

        // shrink_to_fit heap
        s4.shrink_to_fit();
    }

    #[test]
    fn test_small_string_formatting_comparison_and_extension() {
        let mut s: SmallString<4> = SmallString::from("abc");

        // Display / Debug
        assert_eq!(format!("{}", s), "abc");
        assert_eq!(format!("{:?}", s), "\"abc\"");

        // PartialEq / PartialOrd
        let s2 = SmallString::<8>::from("abc");
        assert_eq!(s, s2);
        assert_eq!(s, "abc");
        assert_eq!("abc", s);
        assert_eq!(s, String::from("abc"));

        let s3 = SmallString::<4>::from("abd");
        assert!(s < s3);
        assert_eq!(s.cmp(&s3), std::cmp::Ordering::Less);

        // Hash
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut h);

        // Borrow / AsRef
        let b: &str = s.borrow();
        assert_eq!(b, "abc");
        let b_mut: &mut str = s.borrow_mut();
        b_mut.make_ascii_uppercase();
        assert_eq!(s.as_str(), "ABC");

        let r: &str = s.as_ref();
        assert_eq!(r, "ABC");
        let r_bytes: &[u8] = s.as_ref();
        assert_eq!(r_bytes, b"ABC");

        // Extend / FromIterator
        let mut s4: SmallString<4> = SmallString::new();
        s4.extend(['a', 'b']);
        s4.extend(["cd", "ef"]); // heap
        assert_eq!(s4.as_str(), "abcdef");

        let s5 = SmallString::<4>::from_iter(['x', 'y']);
        assert_eq!(s5.as_str(), "xy");
        let s6 = SmallString::<4>::from_iter(["hi", "ho"]);
        assert_eq!(s6.as_str(), "hiho");
    }

    #[test]
    fn test_small_string_fmt_write_trait() {
        use std::fmt::Write;
        let mut s: SmallString<4> = SmallString::new();
        write!(s, "{}", 12345).unwrap();
        assert_eq!(s.as_str(), "12345");
    }
}
