//! Benchmarks: HeaplessBitVec vs SmallBitVec vs bitvec::BitVec
//!
//! Three-way comparison.  Capacity is measured in **bytes** for `HeaplessBitVec`
//! and `SmallBitVec`; `bitvec::BitVec` grows dynamically on the heap.
//!
//! | Variant             | Storage         | Allocation |
//! |---------------------|-----------------|------------|
//! | `bitvec::BitVec`    | Heap            | Yes        |
//! | `SmallBitVec<8>`    | Stack (8 B) → Heap | No (until spill) |
//! | `HeaplessBitVec<8>` | Stack (8 B) only | Never     |
//!
//! 8 bytes = 64 bits of stack capacity.

use bitvec::prelude::{BitVec, Lsb0};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::{HeaplessBitVec, SmallBitVec};

/// Stack capacity in bytes → 64 bits.
const BYTES: usize = 8;
/// Number of bits to push in each benchmark.
const BITS: usize = BYTES * 8;

// ─── Push (sequential write) ──────────────────────────────────────────────────

fn bench_bitvec_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("BitVec Push (64 bits)");

    group.bench_function("bitvec::BitVec", |b| {
        b.iter(|| {
            let mut v: BitVec<u8, Lsb0> = BitVec::new();
            for i in 0..BITS {
                v.push(black_box(i % 2 == 0));
            }
            v
        })
    });

    group.bench_function("SmallBitVec<8>", |b| {
        b.iter(|| {
            let mut v: SmallBitVec<BYTES> = SmallBitVec::new();
            for i in 0..BITS {
                v.push(black_box(i % 2 == 0));
            }
            v
        })
    });

    group.bench_function("HeaplessBitVec<8>", |b| {
        b.iter(|| {
            let mut v: HeaplessBitVec<BYTES> = HeaplessBitVec::new();
            for i in 0..BITS {
                // Returns Err on overflow — all 64 bits fit in 8 bytes.
                let _ = v.push(black_box(i % 2 == 0));
            }
            v
        })
    });

    group.finish();
}

// ─── Get (random read) ────────────────────────────────────────────────────────

fn bench_bitvec_get(c: &mut Criterion) {
    // Pre-populate
    let mut bv_std: BitVec<u8, Lsb0> = BitVec::new();
    let mut bv_small: SmallBitVec<BYTES> = SmallBitVec::new();
    let mut bv_heapless: HeaplessBitVec<BYTES> = HeaplessBitVec::new();

    for i in 0..BITS {
        bv_std.push(i % 3 == 0);
        bv_small.push(i % 3 == 0);
        let _ = bv_heapless.push(i % 3 == 0);
    }

    let mut group = c.benchmark_group("BitVec Get (64 bits, all-hit)");

    group.bench_function("bitvec::BitVec", |b| {
        b.iter(|| {
            for i in 0..BITS {
                black_box(bv_std.get(black_box(i)));
            }
        })
    });

    group.bench_function("SmallBitVec<8>", |b| {
        b.iter(|| {
            for i in 0..BITS {
                black_box(bv_small.get(black_box(i)));
            }
        })
    });

    group.bench_function("HeaplessBitVec<8>", |b| {
        b.iter(|| {
            for i in 0..BITS {
                black_box(bv_heapless.get(black_box(i)));
            }
        })
    });

    group.finish();
}

// ─── Set (random write) ───────────────────────────────────────────────────────

fn bench_bitvec_set(c: &mut Criterion) {
    let mut bv_std: BitVec<u8, Lsb0> = (0..BITS).map(|_| false).collect();
    let mut bv_small: SmallBitVec<BYTES> = SmallBitVec::new();
    let mut bv_heapless: HeaplessBitVec<BYTES> = HeaplessBitVec::new();

    for _ in 0..BITS {
        bv_small.push(false);
        let _ = bv_heapless.push(false);
    }

    let mut group = c.benchmark_group("BitVec Set (64 bits)");

    group.bench_function("bitvec::BitVec", |b| {
        b.iter(|| {
            for i in 0..BITS {
                bv_std.set(black_box(i), black_box(i % 2 == 0));
            }
        })
    });

    group.bench_function("SmallBitVec<8>", |b| {
        b.iter(|| {
            for i in 0..BITS {
                bv_small.set(black_box(i), black_box(i % 2 == 0));
            }
        })
    });

    group.bench_function("HeaplessBitVec<8>", |b| {
        b.iter(|| {
            for i in 0..BITS {
                bv_heapless.set(black_box(i), black_box(i % 2 == 0));
            }
        })
    });

    group.finish();
}

// ─── Pop (sequential drain) ───────────────────────────────────────────────────

fn bench_bitvec_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("BitVec Pop (drain 64 bits)");

    group.bench_function("bitvec::BitVec", |b| {
        b.iter(|| {
            let mut v: BitVec<u8, Lsb0> = (0..BITS).map(|i| i % 2 == 0).collect();
            while let Some(b) = v.pop() {
                black_box(b);
            }
            v
        })
    });

    group.bench_function("SmallBitVec<8>", |b| {
        b.iter(|| {
            let mut v: SmallBitVec<BYTES> = SmallBitVec::new();
            for i in 0..BITS {
                v.push(i % 2 == 0);
            }
            while let Some(b) = v.pop() {
                black_box(b);
            }
            v
        })
    });

    group.bench_function("HeaplessBitVec<8>", |b| {
        b.iter(|| {
            let mut v: HeaplessBitVec<BYTES> = HeaplessBitVec::new();
            for i in 0..BITS {
                let _ = v.push(i % 2 == 0);
            }
            while let Some(b) = v.pop() {
                black_box(b);
            }
            v
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bitvec_push,
    bench_bitvec_get,
    bench_bitvec_set,
    bench_bitvec_pop
);
criterion_main!(benches);
