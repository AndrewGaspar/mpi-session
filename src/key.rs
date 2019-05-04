use std::marker::PhantomData;

pub trait Key {
    type Value;
}

pub struct RankKey<K>(PhantomData<K>);

impl<K> Key for RankKey<K> {
    type Value = mpi::topology::Rank;
}
