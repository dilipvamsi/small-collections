use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::SmallLruCache;
use std::num::NonZeroUsize;

fn bench_lru_cache(c: &mut Criterion) {
    let n = 128;
    let cap = NonZeroUsize::new(n).unwrap();
    {
        let mut group = c.benchmark_group("LruCache vs SmallLruCache (Put 128)");
        group.bench_function("lru::LruCache", |b| {
            b.iter(|| {
                let mut l = lru::LruCache::new(cap);
                for i in 0..n {
                    l.put(black_box(i as i32), black_box(i as i32));
                }
                l
            })
        });

        group.bench_function("SmallLruCache<i32, i32, 128>", |b| {
            b.iter(|| {
                let mut l: SmallLruCache<i32, i32, 128> = SmallLruCache::new(cap);
                for i in 0..n {
                    l.put(black_box(i as i32), black_box(i as i32));
                }
                l
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("LruCache vs SmallLruCache (Get 128)");
        let mut l_std = lru::LruCache::new(cap);
        let mut l_small: SmallLruCache<i32, i32, 128> = SmallLruCache::new(cap);
        for i in 0..n {
            l_std.put(i as i32, i as i32);
            l_small.put(i as i32, i as i32);
        }

        group.bench_function("lru::LruCache", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(l_std.get(&black_box(i as i32)));
                }
            })
        });

        group.bench_function("SmallLruCache<i32, i32, 128>", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(l_small.get(&black_box(i as i32)));
                }
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_lru_cache);
criterion_main!(benches);
