use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::SmallLruCache;
use std::num::NonZeroUsize;

fn bench_lru_cache(c: &mut Criterion) {
    let n_large = 128;
    let cap_large = NonZeroUsize::new(n_large).unwrap();
    let n_small = 8;
    let cap_small = NonZeroUsize::new(n_small).unwrap();

    // --- N = 8 ---
    {
        let mut group = c.benchmark_group("LruCache vs SmallLruCache (Put 8)");
        group.bench_function("lru::LruCache", |b| {
            b.iter(|| {
                let mut l = lru::LruCache::new(cap_small);
                for i in 0..n_small {
                    l.put(black_box(i as i32), black_box(i as i32));
                }
                l
            })
        });

        group.bench_function("SmallLruCache<i32, i32, 8>", |b| {
            b.iter(|| {
                let mut l: SmallLruCache<i32, i32, 8> = SmallLruCache::new(cap_small);
                for i in 0..n_small {
                    l.put(black_box(i as i32), black_box(i as i32));
                }
                l
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("LruCache vs SmallLruCache (Get 8)");
        let mut l_std = lru::LruCache::new(cap_small);
        let mut l_small: SmallLruCache<i32, i32, 8> = SmallLruCache::new(cap_small);
        for i in 0..n_small {
            l_std.put(i as i32, i as i32);
            l_small.put(i as i32, i as i32);
        }

        group.bench_function("lru::LruCache", |b| {
            b.iter(|| {
                for i in 0..n_small {
                    black_box(l_std.get(&black_box(i as i32)));
                }
            })
        });

        group.bench_function("SmallLruCache<i32, i32, 8>", |b| {
            b.iter(|| {
                for i in 0..n_small {
                    black_box(l_small.get(&black_box(i as i32)));
                }
            })
        });
        group.finish();
    }

    // --- N = 128 ---
    {
        let mut group = c.benchmark_group("LruCache vs SmallLruCache (Put 128)");
        group.bench_function("lru::LruCache", |b| {
            b.iter(|| {
                let mut l = lru::LruCache::new(cap_large);
                for i in 0..n_large {
                    l.put(black_box(i as i32), black_box(i as i32));
                }
                l
            })
        });

        group.bench_function("SmallLruCache<i32, i32, 128>", |b| {
            b.iter(|| {
                let mut l: SmallLruCache<i32, i32, 128> = SmallLruCache::new(cap_large);
                for i in 0..n_large {
                    l.put(black_box(i as i32), black_box(i as i32));
                }
                l
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("LruCache vs SmallLruCache (Get 128)");
        let mut l_std = lru::LruCache::new(cap_large);
        let mut l_small: SmallLruCache<i32, i32, 128> = SmallLruCache::new(cap_large);
        for i in 0..n_large {
            l_std.put(i as i32, i as i32);
            l_small.put(i as i32, i as i32);
        }

        group.bench_function("lru::LruCache", |b| {
            b.iter(|| {
                for i in 0..n_large {
                    black_box(l_std.get(&black_box(i as i32)));
                }
            })
        });

        group.bench_function("SmallLruCache<i32, i32, 128>", |b| {
            b.iter(|| {
                for i in 0..n_large {
                    black_box(l_small.get(&black_box(i as i32)));
                }
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_lru_cache);
criterion_main!(benches);
