use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::SmallString;

fn bench_string(c: &mut Criterion) {
    let s = "short string 123";
    {
        let mut group = c.benchmark_group("String vs SmallString (Push 16)");
        group.bench_function("std::string::String", |b| {
            b.iter(|| {
                let mut st = String::with_capacity(32);
                st.push_str(black_box(s));
                st
            })
        });

        group.bench_function("SmallString<32>", |b| {
            b.iter(|| {
                let mut st: SmallString<32> = SmallString::new();
                st.push_str(black_box(s));
                st
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("String vs SmallString (Get)");
        let st_std = s.to_string();
        let mut st_small: SmallString<32> = SmallString::new();
        st_small.push_str(s);

        group.bench_function("std::string::String", |b| {
            b.iter(|| {
                black_box(&st_std[0..5]);
            })
        });

        group.bench_function("SmallString<32>", |b| {
            b.iter(|| {
                black_box(&st_small[0..5]);
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_string);
criterion_main!(benches);
