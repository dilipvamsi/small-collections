//! Benchmarks: HeaplessLruCache vs SmallLruCache vs lru::LruCache
//!
//! Three-way comparison at the same logical capacity `N = 16`:
//!
//! | Variant              | Storage         | Allocation |
//! |----------------------|-----------------|------------|
//! | `lru::LruCache`      | Heap (linked)   | Yes        |
//! | `SmallLruCache` (N=cap) | Stack SoA    | No         |
//! | `HeaplessLruCache`   | Stack SoA (raw) | No         |
//!
//! The `HeaplessLruCache` benchmark measures the *pure* stack path; `SmallLruCache`
//! benchmarks include the tagged-union dispatch overhead.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::{HeaplessLruCache, SmallLruCache};
use std::num::NonZeroUsize;

const N: usize = 16;

// ─── Put (insert) ─────────────────────────────────────────────────────────────

fn bench_lru_put(c: &mut Criterion) {
    let cap = NonZeroUsize::new(N).unwrap();
    let mut group = c.benchmark_group("LruCache Put (N=16)");

    group.bench_function("lru::LruCache", |b| {
        b.iter(|| {
            let mut l = lru::LruCache::new(cap);
            for i in 0..N {
                l.put(black_box(i as i32), black_box(i as i32));
            }
            l
        })
    });

    group.bench_function("SmallLruCache<i32,i32,16>", |b| {
        b.iter(|| {
            let mut l: SmallLruCache<i32, i32, N> = SmallLruCache::new(cap);
            for i in 0..N {
                l.put(black_box(i as i32), black_box(i as i32));
            }
            l
        })
    });

    group.bench_function("HeaplessLruCache<i32,i32,16>", |b| {
        b.iter(|| {
            let mut l: HeaplessLruCache<i32, i32, N> = HeaplessLruCache::new();
            for i in 0..N {
                let _ = l.put(black_box(i as i32), black_box(i as i32), N);
            }
            l
        })
    });

    group.finish();
}

// ─── Get (lookup with MRU promotion) ──────────────────────────────────────────

fn bench_lru_get(c: &mut Criterion) {
    let cap = NonZeroUsize::new(N).unwrap();

    let mut l_std = lru::LruCache::new(cap);
    let mut l_small: SmallLruCache<i32, i32, N> = SmallLruCache::new(cap);
    let mut l_heapless: HeaplessLruCache<i32, i32, N> = HeaplessLruCache::new();

    for i in 0..N {
        l_std.put(i as i32, i as i32);
        l_small.put(i as i32, i as i32);
        let _ = l_heapless.put(i as i32, i as i32, N);
    }

    let mut group = c.benchmark_group("LruCache Get (N=16, all-hit)");

    group.bench_function("lru::LruCache", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(l_std.get(&black_box(i as i32)));
            }
        })
    });

    group.bench_function("SmallLruCache<i32,i32,16>", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(l_small.get(&black_box(i as i32)));
            }
        })
    });

    group.bench_function("HeaplessLruCache<i32,i32,16>", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(l_heapless.get(&black_box(i as i32)));
            }
        })
    });

    group.finish();
}

// ─── Peek (lookup without MRU promotion) ──────────────────────────────────────

fn bench_lru_peek(c: &mut Criterion) {
    let cap = NonZeroUsize::new(N).unwrap();

    let mut l_std = lru::LruCache::new(cap);
    let mut l_small: SmallLruCache<i32, i32, N> = SmallLruCache::new(cap);
    let mut l_heapless: HeaplessLruCache<i32, i32, N> = HeaplessLruCache::new();

    for i in 0..N {
        l_std.put(i as i32, i as i32);
        l_small.put(i as i32, i as i32);
        let _ = l_heapless.put(i as i32, i as i32, N);
    }

    let mut group = c.benchmark_group("LruCache Peek (N=16, all-hit, no promote)");

    group.bench_function("lru::LruCache", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(l_std.peek(&black_box(i as i32)));
            }
        })
    });

    group.bench_function("SmallLruCache<i32,i32,16>", |b| {
        b.iter(|| {
            for i in 0..N {
                black_box(l_small.peek(&black_box(i as i32)));
            }
        })
    });

    group.bench_function("HeaplessLruCache<i32,i32,16>", |b| {
        b.iter(|| {
            for i in 0..N {
                // HeaplessLruCache::get promotes to MRU; use the map directly for peek semantics
                black_box(l_heapless.peek(&black_box(i as i32)));
            }
        })
    });

    group.finish();
}

// ─── LRU eviction churn ───────────────────────────────────────────────────────

fn bench_lru_eviction(c: &mut Criterion) {
    let cap = NonZeroUsize::new(N).unwrap();
    let total = N * 4; // insert 4x capacity → lots of evictions

    let mut group = c.benchmark_group("LruCache Eviction Churn (cap=16, insert 64)");

    group.bench_function("lru::LruCache", |b| {
        b.iter(|| {
            let mut l = lru::LruCache::new(cap);
            for i in 0..total {
                l.put(black_box(i as i32), black_box(i as i32));
            }
            l
        })
    });

    group.bench_function("SmallLruCache<i32,i32,16>", |b| {
        b.iter(|| {
            let mut l: SmallLruCache<i32, i32, N> = SmallLruCache::new(cap);
            for i in 0..total {
                l.put(black_box(i as i32), black_box(i as i32));
            }
            l
        })
    });

    group.bench_function("HeaplessLruCache<i32,i32,16>", |b| {
        b.iter(|| {
            let mut l: HeaplessLruCache<i32, i32, N> = HeaplessLruCache::new();
            for i in 0..total {
                let _ = l.put(black_box(i as i32), black_box(i as i32), N);
            }
            l
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_lru_put,
    bench_lru_get,
    bench_lru_peek,
    bench_lru_eviction
);
criterion_main!(benches);
