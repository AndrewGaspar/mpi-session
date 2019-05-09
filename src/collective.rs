use std::marker::PhantomData;

use mpi::topology::Rank;
use mpi::traits::*;

use super::prelude::*;

pub struct AllGather<T: Equivalence, P: ProtocolPart>(PhantomData<(T, P)>);

unsafe impl<T: Equivalence, P: ProtocolPart> ProtocolPart for AllGather<T, P> {
    type State = ();

    unsafe fn build_part(_: Self::State) -> Self {
        Self(PhantomData)
    }
}

impl<T: Equivalence, P: ProtocolPart<State = ()>, C: Communicator> Session<AllGather<T, P>, C> {
    pub fn all_gather(self, input: &T, output: &mut [T]) -> Session<P, C> {
        unsafe {
            self.comm.all_gather_into(input, output);
            self.advance(())
        }
    }
}

pub struct Gather<Source: RankSelect, T: Equivalence, P: ProtocolPart>(PhantomData<(Source, T, P)>);

unsafe impl<Source: RankSelect, T: Equivalence, P: ProtocolPart> ProtocolPart
    for Gather<Source, T, P>
{
    type State = ();

    unsafe fn build_part(_: Self::State) -> Self {
        Self(PhantomData)
    }
}

impl<Source: RankSelect, T: Equivalence, P: ProtocolPart<State = ()>, C: Communicator>
    Session<Gather<Source, T, P>, C>
{
    pub fn split(self) -> GatherSplit<T, P, C> {
        let rank = Source::get_rank(self.state());

        if self.comm.rank() == rank {
            GatherSplit::Gatherer(unsafe { self.advance(rank) })
        } else {
            GatherSplit::Gatheree(unsafe { self.advance(rank) })
        }
    }
}

pub enum GatherSplit<T: Equivalence, P: ProtocolPart<State = ()>, C: Communicator> {
    Gatherer(Session<Gatherer<T, P>, C>),
    Gatheree(Session<Gatheree<T, P>, C>),
}

pub struct Gatherer<T: Equivalence, P: ProtocolPart<State = ()>>(Rank, PhantomData<(T, P)>);

unsafe impl<T: Equivalence, P: ProtocolPart<State = ()>> ProtocolPart for Gatherer<T, P> {
    type State = Rank;

    unsafe fn build_part(state: Self::State) -> Self {
        Self(state, PhantomData)
    }
}

impl<T: Equivalence, P: ProtocolPart<State = ()>, C: Communicator> Session<Gatherer<T, P>, C> {
    pub fn gather(self, send: &T, receive: &mut [T]) -> Session<P, C> {
        unsafe {
            self.comm
                .process_at_rank(self.protocol().0)
                .gather_into_root(send, receive);
            self.advance(())
        }
    }
}

pub struct Gatheree<T: Equivalence, P: ProtocolPart<State = ()>>(Rank, PhantomData<(T, P)>);

unsafe impl<T: Equivalence, P: ProtocolPart<State = ()>> ProtocolPart for Gatheree<T, P> {
    type State = Rank;

    unsafe fn build_part(state: Self::State) -> Self {
        Self(state, PhantomData)
    }
}

impl<T: Equivalence, P: ProtocolPart<State = ()>, C: Communicator> Session<Gatheree<T, P>, C> {
    pub fn gather(self, send: &T) -> Session<P, C> {
        unsafe {
            self.comm
                .process_at_rank(self.protocol().0)
                .gather_into(send);
            self.advance(())
        }
    }
}

pub struct Publish<Source: RankSelect, K: super::key::Key, P: ProtocolPart<State = ()>>(
    PhantomData<(Source, K, P)>,
);

unsafe impl<Source: RankSelect, K: super::key::Key, P: ProtocolPart<State = ()>> ProtocolPart
    for Publish<Source, K, P>
{
    type State = ();

    unsafe fn build_part(_: Self::State) -> Self {
        Self(PhantomData)
    }
}

impl<Source: RankSelect, K: super::key::Key, P: ProtocolPart<State = ()>, C: Communicator>
    Session<Publish<Source, K, P>, C>
{
    pub fn split(self) -> PublishSplit<K, P, C> {
        let rank = Source::get_rank(self.state());
        if self.comm.rank() == rank {
            PublishSplit::Publisher(unsafe { self.advance(rank) })
        } else {
            PublishSplit::Publishee(unsafe { self.advance(rank) })
        }
    }
}

pub enum PublishSplit<K: super::key::Key, P: ProtocolPart<State = ()>, C: Communicator> {
    Publisher(Session<Publisher<K, P>, C>),
    Publishee(Session<Publishee<K, P>, C>),
}

pub struct Publisher<K: super::key::Key, P: ProtocolPart<State = ()>>(Rank, PhantomData<(K, P)>);

unsafe impl<K: super::key::Key, P: ProtocolPart<State = ()>> ProtocolPart for Publisher<K, P> {
    type State = Rank;

    unsafe fn build_part(state: Self::State) -> Self {
        Self(state, PhantomData)
    }
}

impl<K: super::key::Key + 'static, P: ProtocolPart<State = ()>, C: Communicator>
    Session<Publisher<K, P>, C>
where
    K::Value: Equivalence,
{
    pub fn publish(mut self, mut value: K::Value) -> Session<P, C> {
        unsafe {
            self.comm
                .process_at_rank(self.protocol().0)
                .broadcast_into(&mut value);
            self.state_mut().insert::<K>(Box::new(value));
            self.advance(())
        }
    }
}

pub struct Publishee<K: super::key::Key, P: ProtocolPart<State = ()>>(Rank, PhantomData<(K, P)>);

unsafe impl<K: super::key::Key, P: ProtocolPart<State = ()>> ProtocolPart for Publishee<K, P> {
    type State = Rank;

    unsafe fn build_part(state: Self::State) -> Self {
        Self(state, PhantomData)
    }
}

impl<K: super::key::Key + 'static, P: ProtocolPart<State = ()>, C: Communicator>
    Session<Publishee<K, P>, C>
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
            self.advance(())
        }
    }
}
