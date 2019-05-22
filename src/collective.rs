use std::marker::PhantomData;

use mpi::topology::Rank;
use mpi::traits::*;

use super::prelude::*;

pub struct AllGather<T: Equivalence, P: ProtocolPart>(PhantomData<(T, P)>);

unsafe impl<T: Equivalence, P: ProtocolPart> ProtocolPart for AllGather<T, P> {
    unsafe fn build_part() -> Self {
        Self(PhantomData)
    }
}

impl<T: Equivalence, P: ProtocolPart, C: Communicator> Session<AllGather<T, P>, C> {
    pub fn all_gather(self, input: &T, output: &mut [T]) -> Session<P, C> {
        unsafe {
            self.comm.all_gather_into(input, output);
            self.advance_next()
        }
    }
}

pub struct Gather<Source: RankSelect, T: Equivalence, P: ProtocolPart>(PhantomData<(Source, T, P)>);

unsafe impl<Source: RankSelect, T: Equivalence, P: ProtocolPart> ProtocolPart
    for Gather<Source, T, P>
{
    unsafe fn build_part() -> Self {
        Self(PhantomData)
    }
}

impl<Source: RankSelect, T: Equivalence, P: ProtocolPart, C: Communicator>
    Session<Gather<Source, T, P>, C>
{
    pub fn split(self) -> Split<Gatherer<T, P>, Gatheree<T, P>, C> {
        let rank = Source::get_rank(self.state());

        if self.comm.rank() == rank {
            Split::Left(unsafe { self.advance(Gatherer(rank, PhantomData)) })
        } else {
            Split::Right(unsafe { self.advance(Gatheree(rank, PhantomData)) })
        }
    }
}

pub struct Gatherer<T: Equivalence, P: ProtocolPart>(Rank, PhantomData<(T, P)>);

impl<T: Equivalence, P: ProtocolPart, C: Communicator> Session<Gatherer<T, P>, C> {
    pub fn gather(self, send: &T, receive: &mut [T]) -> Session<P, C> {
        unsafe {
            self.comm
                .process_at_rank(self.protocol().0)
                .gather_into_root(send, receive);
            self.advance_next()
        }
    }
}

pub struct Gatheree<T: Equivalence, P: ProtocolPart>(Rank, PhantomData<(T, P)>);

impl<T: Equivalence, P: ProtocolPart, C: Communicator> Session<Gatheree<T, P>, C> {
    pub fn gather(self, send: &T) -> Session<P, C> {
        unsafe {
            self.comm
                .process_at_rank(self.protocol().0)
                .gather_into(send);
            self.advance_next()
        }
    }
}

pub struct Publish<Source: RankSelect, K: super::key::Key, P: ProtocolPart>(
    PhantomData<(Source, K, P)>,
);

unsafe impl<Source: RankSelect, K: super::key::Key, P: ProtocolPart> ProtocolPart
    for Publish<Source, K, P>
{
    unsafe fn build_part() -> Self {
        Self(PhantomData)
    }
}

impl<Source: RankSelect, K: super::key::Key, P: ProtocolPart, C: Communicator>
    Session<Publish<Source, K, P>, C>
{
    pub fn split(self) -> Split<Publisher<K, P>, Publishee<K, P>, C> {
        let rank = Source::get_rank(self.state());
        if self.comm.rank() == rank {
            Split::Left(unsafe { self.advance(Publisher(rank, PhantomData)) })
        } else {
            Split::Right(unsafe { self.advance(Publishee(rank, PhantomData)) })
        }
    }
}

pub struct Publisher<K: super::key::Key, P: ProtocolPart>(Rank, PhantomData<(K, P)>);

impl<K: super::key::Key + 'static, P: ProtocolPart, C: Communicator> Session<Publisher<K, P>, C>
where
    K::Value: Equivalence,
{
    pub fn publish(mut self, mut value: K::Value) -> Session<P, C> {
        unsafe {
            self.comm
                .process_at_rank(self.protocol().0)
                .broadcast_into(&mut value);
            self.state_mut().insert::<K>(Box::new(value));
            self.advance_next()
        }
    }
}

pub struct Publishee<K: super::key::Key, P: ProtocolPart>(Rank, PhantomData<(K, P)>);

impl<K: super::key::Key + 'static, P: ProtocolPart, C: Communicator> Session<Publishee<K, P>, C>
where
    K::Value: Equivalence,
{
    pub fn receive(mut self) -> Session<P, C> {
        unsafe {
            let mut value: K::Value = std::mem::uninitialized();
            self.comm
                .process_at_rank(self.protocol().0)
                .broadcast_into(&mut value);
            self.state_mut().insert::<K>(Box::new(value));
            self.advance_next()
        }
    }
}
