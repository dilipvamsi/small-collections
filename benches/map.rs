use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::{SmallMap, SmallBTreeMap, SmallOrderedMap};
use std::collections::{HashMap, BTreeMap};

fn bench_map(c: &mut Criterion) {
    let n = 8;
    {
        let mut group = c.benchmark_group("HashMap vs SmallMap (Insert 8)");
        group.bench_function("std::collections::HashMap", |b| {
            b.iter(|| {
                let mut m = HashMap::with_capacity(n);
                for i in 0..n {
                    m.insert(black_box(i as i32), black_box(i as i32));
                }
                m
            })
        });

        group.bench_function("SmallMap<i32, i32, 8>", |b| {
            b.iter(|| {
                let mut m: SmallMap<i32, i32, 8> = SmallMap::new();
                for i in 0..n {
                    m.insert(black_box(i as i32), black_box(i as i32));
                }
                m
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("HashMap vs SmallMap (Get 8)");
        let mut m_std = HashMap::new();
        let mut m_small: SmallMap<i32, i32, 8> = SmallMap::new();
        for i in 0..n {
            m_std.insert(i as i32, i as i32);
            m_small.insert(i as i32, i as i32);
        }

        group.bench_function("std::collections::HashMap", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(m_std.get(&black_box(i as i32)));
                }
            })
        });

        group.bench_function("SmallMap<i32, i32, 8>", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(m_small.get(&black_box(i as i32)));
                }
            })
        });
        group.finish();
    }
}

fn bench_btree_map(c: &mut Criterion) {
    let n = 8;
    {
        let mut group = c.benchmark_group("BTreeMap vs SmallBTreeMap (Insert 8)");
        group.bench_function("std::collections::BTreeMap", |b| {
            b.iter(|| {
                let mut m = BTreeMap::new();
                for i in 0..n {
                    m.insert(black_box(i as i32), black_box(i as i32));
                }
                m
            })
        });

        group.bench_function("SmallBTreeMap<i32, i32, 8>", |b| {
            b.iter(|| {
                let mut m: SmallBTreeMap<i32, i32, 8> = SmallBTreeMap::new();
                for i in 0..n {
                    m.insert(black_box(i as i32), black_box(i as i32));
                }
                m
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("BTreeMap vs SmallBTreeMap (Get 8)");
        let mut m_std = BTreeMap::new();
        let mut m_small: SmallBTreeMap<i32, i32, 8> = SmallBTreeMap::new();
        for i in 0..n {
            m_std.insert(i as i32, i as i32);
            m_small.insert(i as i32, i as i32);
        }

        group.bench_function("std::collections::BTreeMap", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(m_std.get(&black_box(i as i32)));
                }
            })
        });

        group.bench_function("SmallBTreeMap<i32, i32, 8>", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(m_small.get(&black_box(i as i32)));
                }
            })
        });
        group.finish();
    }
}

fn bench_ordered_map(c: &mut Criterion) {
    let n = 8;
    {
        let mut group = c.benchmark_group("OrderMap vs SmallOrderedMap (Insert 8)");
        group.bench_function("ordermap::OrderMap", |b| {
            b.iter(|| {
                let mut m = ordermap::OrderMap::with_capacity(n);
                for i in 0..n {
                    m.insert(black_box(i as i32), black_box(i as i32));
                }
                m
            })
        });

        group.bench_function("SmallOrderedMap<i32, i32, 8>", |b| {
            b.iter(|| {
                let mut m: SmallOrderedMap<i32, i32, 8> = SmallOrderedMap::new();
                for i in 0..n {
                    m.insert(black_box(i as i32), black_box(i as i32));
                }
                m
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("OrderMap vs SmallOrderedMap (Get 8)");
        let mut m_std = ordermap::OrderMap::new();
        let mut m_small: SmallOrderedMap<i32, i32, 8> = SmallOrderedMap::new();
        for i in 0..n {
            m_std.insert(i as i32, i as i32);
            m_small.insert(i as i32, i as i32);
        }

        group.bench_function("ordermap::OrderMap", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(m_std.get(&black_box(i as i32)));
                }
            })
        });

        group.bench_function("SmallOrderedMap<i32, i32, 8>", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(m_small.get(&black_box(i as i32)));
                }
            })
        });
        group.finish();
    }
}

fn bench_spill(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spill Overhead (N=8 -> 9)");
    let n_total = 9;

    group.bench_function("SmallMap Spill", |b| {
        b.iter(|| {
            let mut m: SmallMap<i32, i32, 8> = SmallMap::new();
            for i in 0..n_total {
                m.insert(black_box(i as i32), black_box(i as i32));
            }
            m
        })
    });
    group.finish();
}

criterion_group!(benches, bench_map, bench_btree_map, bench_ordered_map, bench_spill);
criterion_main!(benches);
