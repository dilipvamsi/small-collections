use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::SmallVec;

fn bench_vec(c: &mut Criterion) {
    let n = 16;
    {
        let mut group = c.benchmark_group("Vec vs SmallVec (Push 16)");
        group.bench_function("std::vec::Vec", |b| {
            b.iter(|| {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(black_box(i as i32));
                }
                v
            })
        });

        group.bench_function("SmallVec<i32, 16>", |b| {
            b.iter(|| {
                let mut v: SmallVec<i32, 16> = SmallVec::new();
                for i in 0..n {
                    v.push(black_box(i as i32));
                }
                v
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("Vec vs SmallVec (Access 16)");
        let v_std = vec![123i32; n];
        let mut v_small: SmallVec<i32, 16> = SmallVec::new();
        for _ in 0..n {
            v_small.push(123);
        }

        group.bench_function("std::vec::Vec", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(v_std.get(black_box(i)));
                }
            })
        });

        group.bench_function("SmallVec<i32, 16>", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(v_small.get(black_box(i)));
                }
            })
        });
        group.finish();
    }
}

fn bench_spill(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spill Overhead (N=8 -> 9)");
    let n_total = 9;

    group.bench_function("SmallVec Spill", |b| {
        b.iter(|| {
            let mut v: SmallVec<i32, 8> = SmallVec::new();
            for i in 0..n_total {
                v.push(black_box(i as i32));
            }
            v
        })
    });
    group.finish();
}

criterion_group!(benches, bench_vec, bench_spill);
criterion_main!(benches);
