use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::SmallDeque;
use std::collections::VecDeque;

fn bench_deque(c: &mut Criterion) {
    let n = 16;
    {
        let mut group = c.benchmark_group("VecDeque vs SmallDeque (PushBack 16)");
        group.bench_function("std::collections::VecDeque", |b| {
            b.iter(|| {
                let mut d = VecDeque::with_capacity(n);
                for i in 0..n {
                    d.push_back(black_box(i as i32));
                }
                d
            })
        });

        group.bench_function("SmallDeque<i32, 16>", |b| {
            b.iter(|| {
                let mut d: SmallDeque<i32, 16> = SmallDeque::new();
                for i in 0..n {
                    d.push_back(black_box(i as i32));
                }
                d
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("VecDeque vs SmallDeque (Get 16)");
        let mut d_std = VecDeque::new();
        let mut d_small: SmallDeque<i32, 16> = SmallDeque::new();
        for i in 0..n {
            d_std.push_back(i as i32);
            d_small.push_back(i as i32);
        }

        group.bench_function("std::collections::VecDeque", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(d_std.get(black_box(i)));
                }
            })
        });

        group.bench_function("SmallDeque<i32, 16>", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(d_small.get(black_box(i)));
                }
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_deque);
criterion_main!(benches);
