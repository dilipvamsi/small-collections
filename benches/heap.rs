use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::SmallBinaryHeap;
use std::collections::BinaryHeap;

fn bench_binary_heap(c: &mut Criterion) {
    let n = 8;
    {
        let mut group = c.benchmark_group("BinaryHeap vs SmallBinaryHeap (Push 8)");
        group.bench_function("std::collections::BinaryHeap", |b| {
            b.iter(|| {
                let mut h = BinaryHeap::with_capacity(n);
                for i in 0..n {
                    h.push(black_box(i as i32));
                }
                h
            })
        });

        group.bench_function("SmallBinaryHeap<i32, 8>", |b| {
            b.iter(|| {
                let mut h: SmallBinaryHeap<i32, 8> = SmallBinaryHeap::new();
                for i in 0..n {
                    h.push(black_box(i as i32));
                }
                h
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("BinaryHeap vs SmallBinaryHeap (Peek)");
        let mut h_std = BinaryHeap::new();
        let mut h_small: SmallBinaryHeap<i32, 8> = SmallBinaryHeap::new();
        for i in 0..n {
            h_std.push(i as i32);
            h_small.push(i as i32);
        }

        group.bench_function("std::collections::BinaryHeap", |b| {
            b.iter(|| {
                black_box(h_std.peek());
            })
        });

        group.bench_function("SmallBinaryHeap<i32, 8>", |b| {
            b.iter(|| {
                black_box(h_small.peek());
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_binary_heap);
criterion_main!(benches);
