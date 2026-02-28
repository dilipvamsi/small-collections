//! Benchmarks: Linear vs BTree vs Map LRU Cache
//!
//! Comparison across different capacities N = [8, 16, 32, 64, 128]:

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use small_collections::{HeaplessBTreeLruCache, HeaplessLinearLruCache, HeaplessLruCache};

const MAX_N: usize = 128;

fn bench_cache_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("LRU Put Comparison");

    for n in [8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("Map", n), &n, |b, &n| {
            b.iter(|| {
                let mut l: HeaplessLruCache<i32, i32, MAX_N> = HeaplessLruCache::new();
                for i in 0..n {
                    let _ = l.put(black_box(i as i32), black_box(i as i32), MAX_N);
                }
                l
            })
        });

        group.bench_with_input(BenchmarkId::new("Linear", n), &n, |b, &n| {
            b.iter(|| {
                let mut l: HeaplessLinearLruCache<i32, i32, MAX_N> = HeaplessLinearLruCache::new();
                for i in 0..n {
                    let _ = l.put(black_box(i as i32), black_box(i as i32), MAX_N);
                }
                l
            })
        });

        group.bench_with_input(BenchmarkId::new("BTree", n), &n, |b, &n| {
            b.iter(|| {
                let mut l: HeaplessBTreeLruCache<i32, i32, MAX_N> = HeaplessBTreeLruCache::new();
                for i in 0..n {
                    let _ = l.put(black_box(i as i32), black_box(i as i32), MAX_N);
                }
                l
            })
        });
    }
    group.finish();
}

fn bench_cache_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("LRU Get Comparison (Hit)");

    for n in [8, 16, 32, 64, 128] {
        let mut l_map: HeaplessLruCache<i32, i32, MAX_N> = HeaplessLruCache::new();
        let mut l_lin: HeaplessLinearLruCache<i32, i32, MAX_N> = HeaplessLinearLruCache::new();
        let mut l_bt: HeaplessBTreeLruCache<i32, i32, MAX_N> = HeaplessBTreeLruCache::new();

        for i in 0..n {
            let _ = l_map.put(i as i32, i as i32, MAX_N);
            let _ = l_lin.put(i as i32, i as i32, MAX_N);
            let _ = l_bt.put(i as i32, i as i32, MAX_N);
        }

        group.bench_with_input(BenchmarkId::new("Map", n), &n, |b, &n| {
            b.iter(|| {
                for i in 0..n {
                    black_box(l_map.get(&black_box(i as i32)));
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("Linear", n), &n, |b, &n| {
            b.iter(|| {
                for i in 0..n {
                    black_box(l_lin.get(&black_box(i as i32)));
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("BTree", n), &n, |b, &n| {
            b.iter(|| {
                for i in 0..n {
                    black_box(l_bt.get(&black_box(i as i32)));
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cache_put, bench_cache_get);
criterion_main!(benches);
