use criterion::{Criterion, black_box, criterion_group, criterion_main};
use small_collections::{
    SmallBTreeMap, SmallBinaryHeap, SmallBitVec, SmallDeque, SmallLruCache, SmallMap,
    SmallOrderedMap, SmallString, SmallVec,
};
use std::collections::{BTreeMap, BinaryHeap, HashMap, VecDeque};
use std::num::NonZeroUsize;

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
                    black_box(v_std[i]);
                }
            })
        });

        group.bench_function("SmallVec<i32, 16>", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(v_small[i]);
                }
            })
        });
        group.finish();
    }
}

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
        let mut group = c.benchmark_group("String vs SmallString (AsStr)");
        let st_std = s.to_string();
        let mut st_small: SmallString<32> = SmallString::new();
        st_small.push_str(s);

        group.bench_function("std::string::String", |b| {
            b.iter(|| {
                black_box(st_std.as_str());
            })
        });

        group.bench_function("SmallString<32>", |b| {
            b.iter(|| {
                black_box(st_small.as_str());
            })
        });
        group.finish();
    }
}

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

fn bench_lru_cache(c: &mut Criterion) {
    let n = 8;
    let cap = NonZeroUsize::new(n).unwrap();
    {
        let mut group = c.benchmark_group("LruCache vs SmallLruCache (Put 8)");
        group.bench_function("lru::LruCache", |b| {
            b.iter(|| {
                let mut l = lru::LruCache::new(cap);
                for i in 0..n {
                    l.put(black_box(i as i32), black_box(i as i32));
                }
                l
            })
        });

        group.bench_function("SmallLruCache<i32, i32, 8>", |b| {
            b.iter(|| {
                let mut l: SmallLruCache<i32, i32, 8> = SmallLruCache::new(cap);
                for i in 0..n {
                    l.put(black_box(i as i32), black_box(i as i32));
                }
                l
            })
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("LruCache vs SmallLruCache (Get 8)");
        let mut l_std = lru::LruCache::new(cap);
        let mut l_small: SmallLruCache<i32, i32, 8> = SmallLruCache::new(cap);
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

        group.bench_function("SmallLruCache<i32, i32, 8>", |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(l_small.get(&black_box(i as i32)));
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

    group.bench_function("SmallVec Spill", |b| {
        b.iter(|| {
            let mut v: SmallVec<i32, 8> = SmallVec::new();
            for i in 0..n_total {
                v.push(black_box(i as i32));
            }
            v
        })
    });

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

criterion_group!(
    benches,
    bench_vec,
    bench_map,
    bench_string,
    bench_deque,
    bench_bitvec,
    bench_btree_map,
    bench_binary_heap,
    bench_lru_cache,
    bench_ordered_map,
    bench_spill,
);
criterion_main!(benches);
