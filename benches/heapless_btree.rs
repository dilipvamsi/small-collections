//! Benchmarks: HeaplessBTreeMap vs SmallBTreeMap vs std::collections::BTreeMap
//!
//! Three-way comparison at capacity `N = 16`.
//!
//! | Variant                | Storage            | Lookup       |
//! |------------------------|--------------------|--------------|
//! | `std::BTreeMap`        | Heap (B-tree)      | O(log n)     |
//! | `SmallBTreeMap`        | Stack then heap    | O(log N)     |
//! | `HeaplessBTreeMap`     | Stack only (vec)   | O(log N)     |
//!
//! `HeaplessBTreeMap::insert` returns `Err` on overflow so the bench wraps in
//! `unwrap_or_else` to ignore such cases (all inserts fit within N=16 here).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::{HeaplessBTreeMap, SmallBTreeMap};
use std::collections::BTreeMap;

const N: usize = 16;

// ─── Insert ───────────────────────────────────────────────────────────────────

fn bench_btree_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("BTreeMap Insert (N=16)");

    group.bench_function("std::BTreeMap", |b| {
        b.iter(|| {
            let mut m = BTreeMap::new();
            for i in 0..N {
                m.insert(black_box(i as i32), black_box(i as i32));
            }
            m
        })
    });

    group.bench_function("SmallBTreeMap<i32,i32,16>", |b| {
        b.iter(|| {
            let mut m: SmallBTreeMap<i32, i32, N> = SmallBTreeMap::new();
            for i in 0..N {
                m.insert(black_box(i as i32), black_box(i as i32));
            }
            m
        })
    });

    group.bench_function("HeaplessBTreeMap<i32,i32,16>", |b| {
        b.iter(|| {
            let mut m: HeaplessBTreeMap<i32, i32, N> = HeaplessBTreeMap::new();
            for i in 0..N {
                // Returns Err on overflow — all N inserts fit, so this is safe to ignore.
                let _ = m.insert(black_box(i as i32), black_box(i as i32));
            }
            m
        })
    });

    group.finish();
}

// ─── Get (binary-search lookup) ───────────────────────────────────────────────

fn bench_btree_get(c: &mut Criterion) {
    let mut m_std = BTreeMap::new();
    let mut m_small: SmallBTreeMap<i32, i32, N> = SmallBTreeMap::new();
    let mut m_heapless: HeaplessBTreeMap<i32, i32, N> = HeaplessBTreeMap::new();

    for i in 0..N {
        m_std.insert(i as i32, i as i32);
        m_small.insert(i as i32, i as i32);
        let _ = m_heapless.insert(i as i32, i as i32);
    }

    let mut group = c.benchmark_group("BTreeMap Get (N=16, all-hit)");

    group.bench_function("std::BTreeMap", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(m_std.get(&black_box(i as i32)));
            }
        })
    });

    group.bench_function("SmallBTreeMap<i32,i32,16>", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(m_small.get(&black_box(i as i32)));
            }
        })
    });

    group.bench_function("HeaplessBTreeMap<i32,i32,16>", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(m_heapless.get(&black_box(i as i32)));
            }
        })
    });

    group.finish();
}

// ─── Remove ───────────────────────────────────────────────────────────────────

fn bench_btree_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("BTreeMap Remove (N=16, drain all)");

    group.bench_function("std::BTreeMap", |b| {
        b.iter(|| {
            let mut m = BTreeMap::new();
            for i in 0..N {
                m.insert(i as i32, i as i32);
            }
            for i in 0..N {
                black_box(m.remove(&(i as i32)));
            }
            m
        })
    });

    group.bench_function("SmallBTreeMap<i32,i32,16>", |b| {
        b.iter(|| {
            let mut m: SmallBTreeMap<i32, i32, N> = SmallBTreeMap::new();
            for i in 0..N {
                m.insert(i as i32, i as i32);
            }
            for i in 0..N {
                black_box(m.remove(&(i as i32)));
            }
            m
        })
    });

    group.bench_function("HeaplessBTreeMap<i32,i32,16>", |b| {
        b.iter(|| {
            let mut m: HeaplessBTreeMap<i32, i32, N> = HeaplessBTreeMap::new();
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

// ─── Iteration (in-order) ─────────────────────────────────────────────────────

fn bench_btree_iter(c: &mut Criterion) {
    let mut m_std = BTreeMap::new();
    let mut m_small: SmallBTreeMap<i32, i32, N> = SmallBTreeMap::new();
    let mut m_heapless: HeaplessBTreeMap<i32, i32, N> = HeaplessBTreeMap::new();

    for i in 0..N {
        m_std.insert(i as i32, i as i32);
        m_small.insert(i as i32, i as i32);
        let _ = m_heapless.insert(i as i32, i as i32);
    }

    let mut group = c.benchmark_group("BTreeMap Iter (N=16)");

    group.bench_function("std::BTreeMap", |b| {
        b.iter(|| {
            let s: i32 = m_std.iter().map(|(_, v)| *v).sum();
            black_box(s)
        })
    });

    group.bench_function("SmallBTreeMap<i32,i32,16>", |b| {
        b.iter(|| {
            let s: i32 = m_small.iter().map(|(_, v)| *v).sum();
            black_box(s)
        })
    });

    group.bench_function("HeaplessBTreeMap<i32,i32,16>", |b| {
        b.iter(|| {
            // HeaplessBTreeMap::iter() yields &Entry<K,V>; .1 is the value.
            let s: i32 = m_heapless.iter().map(|e| e.1).sum();
            black_box(s)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_btree_insert,
    bench_btree_get,
    bench_btree_remove,
    bench_btree_iter
);
criterion_main!(benches);
