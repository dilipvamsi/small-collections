use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::SmallBitVec;

fn bench_bitvec(c: &mut Criterion) {
    let n = 64;
    {
        let mut group = c.benchmark_group("BitVec vs SmallBitVec (Push 64)");
        group.bench_function("bitvec::BitVec", |b| {
            b.iter(|| {
                let mut v = bitvec::bitvec![u8, bitvec::prelude::Lsb0; 0; 0];
                for i in 0..n {
                    v.push(black_box(i % 2 == 0));
                }
                v
            })
        });

        group.bench_function("SmallBitVec<8>", |b| {
            b.iter(|| {
                let mut v: SmallBitVec<8> = SmallBitVec::new();
                for i in 0..n {
                    v.push(black_box(i % 2 == 0));
                }
                v
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("BitVec vs SmallBitVec (Get 64)");
        let mut v_std = bitvec::bitvec![u8, bitvec::prelude::Lsb0; 0; 0];
        let mut v_small: SmallBitVec<8> = SmallBitVec::new();
        for i in 0..n {
            v_std.push(i % 2 == 0);
            v_small.push(i % 2 == 0);
        }

        group.bench_function("bitvec::BitVec", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(v_std.get(black_box(i)));
                }
            })
        });

        group.bench_function("SmallBitVec<8>", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(v_small.get(black_box(i)));
                }
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_bitvec);
criterion_main!(benches);
