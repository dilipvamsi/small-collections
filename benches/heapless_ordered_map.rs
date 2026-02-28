//! Benchmarks: HeaplessOrderedMap vs SmallOrderedMap vs ordermap::OrderMap
//!
//! Three-way comparison at capacity `N = 16`.
//!
//! | Variant                  | Storage            | Lookup   | Order preserved |
//! |--------------------------|--------------------|----------|-----------------|
//! | `ordermap::OrderMap`     | Heap (hash table)  | O(1) avg | Yes             |
//! | `SmallOrderedMap`        | Stack → Heap       | O(N) / O(1) | Yes          |
//! | `HeaplessOrderedMap`     | Stack only         | O(N) linear | Yes          |
//!
//! `HeaplessOrderedMap` uses a `heapless::LinearMap` (linear scan) so lookups
//! are O(N) — this is intentional and should show clearly in the Get benchmark.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::{HeaplessOrderedMap, SmallOrderedMap};

const N: usize = 16;

// ─── Insert ───────────────────────────────────────────────────────────────────

fn bench_ordered_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("OrderedMap Insert (N=16)");

    group.bench_function("ordermap::OrderMap", |b| {
        b.iter(|| {
            let mut m = ordermap::OrderMap::with_capacity(N);
            for i in 0..N {
                m.insert(black_box(i as i32), black_box(i as i32));
            }
            m
        })
    });

    group.bench_function("SmallOrderedMap<i32,i32,16>", |b| {
        b.iter(|| {
            let mut m: SmallOrderedMap<i32, i32, N> = SmallOrderedMap::new();
            for i in 0..N {
                m.insert(black_box(i as i32), black_box(i as i32));
            }
            m
        })
    });

    group.bench_function("HeaplessOrderedMap<i32,i32,16>", |b| {
        b.iter(|| {
            let mut m: HeaplessOrderedMap<i32, i32, N> = HeaplessOrderedMap::new();
            for i in 0..N {
                // Returns Err on overflow — all N inserts fit here.
                let _ = m.insert(black_box(i as i32), black_box(i as i32));
            }
            m
        })
    });

    group.finish();
}

// ─── Get (linear-scan vs O(1) hash) ──────────────────────────────────────────

fn bench_ordered_get(c: &mut Criterion) {
    let mut m_std = ordermap::OrderMap::new();
    let mut m_small: SmallOrderedMap<i32, i32, N> = SmallOrderedMap::new();
    let mut m_heapless: HeaplessOrderedMap<i32, i32, N> = HeaplessOrderedMap::new();

    for i in 0..N {
        m_std.insert(i as i32, i as i32);
        m_small.insert(i as i32, i as i32);
        let _ = m_heapless.insert(i as i32, i as i32);
    }

    let mut group = c.benchmark_group("OrderedMap Get (N=16, all-hit)");

    group.bench_function("ordermap::OrderMap", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(m_std.get(&black_box(i as i32)));
            }
        })
    });

    group.bench_function("SmallOrderedMap<i32,i32,16>", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(m_small.get(&black_box(i as i32)));
            }
        })
    });

    group.bench_function("HeaplessOrderedMap<i32,i32,16> (O(N) scan)", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(m_heapless.get(&black_box(i as i32)));
            }
        })
    });

    group.finish();
}

// ─── Remove ───────────────────────────────────────────────────────────────────

fn bench_ordered_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("OrderedMap Remove (N=16, drain all)");

    group.bench_function("ordermap::OrderMap", |b| {
        b.iter(|| {
            let mut m = ordermap::OrderMap::with_capacity(N);
            for i in 0..N {
                m.insert(i as i32, i as i32);
            }
            for i in 0..N {
                black_box(m.remove(&(i as i32)));
            }
            m
        })
    });

    group.bench_function("SmallOrderedMap<i32,i32,16>", |b| {
        b.iter(|| {
            let mut m: SmallOrderedMap<i32, i32, N> = SmallOrderedMap::new();
            for i in 0..N {
                m.insert(i as i32, i as i32);
            }
            for i in 0..N {
                black_box(m.remove(&(i as i32)));
            }
            m
        })
    });

    group.bench_function("HeaplessOrderedMap<i32,i32,16>", |b| {
        b.iter(|| {
            let mut m: HeaplessOrderedMap<i32, i32, N> = HeaplessOrderedMap::new();
            for i in 0..N {
                let _ = m.insert(i as i32, i as i32);
            }
            for i in 0..N {
                black_box(m.remove(&(i as i32)));
            }
            m
        })
    });

    group.finish();
}

// ─── Iteration (insertion order) ─────────────────────────────────────────────

fn bench_ordered_iter(c: &mut Criterion) {
    let mut m_std = ordermap::OrderMap::new();
    let mut m_small: SmallOrderedMap<i32, i32, N> = SmallOrderedMap::new();
    let mut m_heapless: HeaplessOrderedMap<i32, i32, N> = HeaplessOrderedMap::new();

    for i in 0..N {
        m_std.insert(i as i32, i as i32);
        m_small.insert(i as i32, i as i32);
        let _ = m_heapless.insert(i as i32, i as i32);
    }

    let mut group = c.benchmark_group("OrderedMap Iter (N=16, insertion order)");

    group.bench_function("ordermap::OrderMap", |b| {
        b.iter(|| {
            let s: i32 = m_std.iter().map(|(_, v)| *v).sum();
            black_box(s)
        })
    });

    group.bench_function("SmallOrderedMap<i32,i32,16>", |b| {
        b.iter(|| {
            let s: i32 = m_small.iter().map(|(_, v)| *v).sum();
            black_box(s)
        })
    });

    group.bench_function("HeaplessOrderedMap<i32,i32,16>", |b| {
        b.iter(|| {
            let s: i32 = m_heapless.iter().map(|(_, v)| *v).sum();
            black_box(s)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ordered_insert,
    bench_ordered_get,
    bench_ordered_remove,
    bench_ordered_iter
);
criterion_main!(benches);
