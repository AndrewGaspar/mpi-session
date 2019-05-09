use std::marker::PhantomData;

use mpi::traits::*;

use super::prelude::*;

pub struct AllGather<T: Equivalence, P: ProtocolPart>(PhantomData<(T, P)>);

unsafe impl<T: Equivalence, P: ProtocolPart> ProtocolPart for AllGather<T, P> {}

impl<T: Equivalence, P: ProtocolPart, C: Communicator> Session<AllGather<T, P>, C> {
    pub fn all_gather(self, input: &T, output: &mut [T]) -> Session<P, C> {
        unsafe {
            self.comm.all_gather_into(input, output);
            self.advance()
        }
    }
}

pub struct Gather<Source: RankSelect, T: Equivalence, P: ProtocolPart>(PhantomData<(Source, T, P)>);

unsafe impl<Source: RankSelect, T: Equivalence, P: ProtocolPart> ProtocolPart
    for Gather<Source, T, P>
{
}

impl<Source: RankSelect, T: Equivalence, P: ProtocolPart, C: Communicator>
    Session<Gather<Source, T, P>, C>
{
    pub fn split(self) -> GatherSplit<Source, T, P, C> {
        if self.comm.rank() == Source::get_rank(self.state()) {
            GatherSplit::Gatherer(unsafe { self.advance() })
        } else {
            GatherSplit::Gatheree(unsafe { self.advance() })
        }
    }
}

pub enum GatherSplit<Source: RankSelect, T: Equivalence, P: ProtocolPart, C: Communicator> {
    Gatherer(Session<Gatherer<Source, T, P>, C>),
    Gatheree(Session<Gatheree<Source, T, P>, C>),
}

pub struct Gatherer<Source: RankSelect, T: Equivalence, P: ProtocolPart>(
    PhantomData<(Source, T, P)>,
);

unsafe impl<Source: RankSelect, T: Equivalence, P: ProtocolPart> ProtocolPart
    for Gatherer<Source, T, P>
{
}

impl<Source: RankSelect, T: Equivalence, P: ProtocolPart, C: Communicator>
    Session<Gatherer<Source, T, P>, C>
{
    pub fn gather(self, send: &T, receive: &mut [T]) -> Session<P, C> {
        unsafe {
            self.comm
                .process_at_rank(Source::get_rank(self.state()))
                .gather_into_root(send, receive);
            self.advance()
        }
    }
}

pub struct Gatheree<Source: RankSelect, T: Equivalence, P: ProtocolPart>(
    PhantomData<(Source, T, P)>,
);

unsafe impl<Source: RankSelect, T: Equivalence, P: ProtocolPart> ProtocolPart
    for Gatheree<Source, T, P>
{
}

impl<Source: RankSelect, T: Equivalence, P: ProtocolPart, C: Communicator>
    Session<Gatheree<Source, T, P>, C>
{
    pub fn gather(self, send: &T) -> Session<P, C> {
        unsafe {
            self.comm
                .process_at_rank(Source::get_rank(self.state()))
                .gather_into(send);
            self.advance()
        }
    }
}

pub struct Publish<Source: RankSelect, K: super::key::Key, P: ProtocolPart>(
    PhantomData<(Source, K, P)>,
);

unsafe impl<Source: RankSelect, K: super::key::Key, P: ProtocolPart> ProtocolPart
    for Publish<Source, K, P>
{
}

impl<Source: RankSelect, K: super::key::Key, P: ProtocolPart, C: Communicator>
    Session<Publish<Source, K, P>, C>
{
    pub fn split(self) -> PublishSplit<Source, K, P, C> {
        if self.comm.rank() == Source::get_rank(self.state()) {
            PublishSplit::Publisher(unsafe { self.advance() })
        } else {
            PublishSplit::Publishee(unsafe { self.advance() })
        }
    }
}

pub enum PublishSplit<Source: RankSelect, K: super::key::Key, P: ProtocolPart, C: Communicator> {
    Publisher(Session<Publisher<Source, K, P>, C>),
    Publishee(Session<Publishee<Source, K, P>, C>),
}

pub struct Publisher<Source: RankSelect, K: super::key::Key, P: ProtocolPart>(
    PhantomData<(Source, K, P)>,
);

unsafe impl<Source: RankSelect, K: super::key::Key, P: ProtocolPart> ProtocolPart
    for Publisher<Source, K, P>
{
}

impl<Source: RankSelect, K: super::key::Key + 'static, P: ProtocolPart, C: Communicator>
    Session<Publisher<Source, K, P>, C>
where
    K::Value: Equivalence,
{
    pub fn publish(mut self, mut value: K::Value) -> Session<P, C> {
        unsafe {
            self.comm
                .process_at_rank(Source::get_rank(self.state()))
                .broadcast_into(&mut value);
            self.state_mut().insert::<K>(Box::new(value));
            self.advance()
        }
    }
}

pub struct Publishee<Source: RankSelect, K: super::key::Key, P: ProtocolPart>(
    PhantomData<(Source, K, P)>,
);

unsafe impl<Source: RankSelect, K: super::key::Key, P: ProtocolPart> ProtocolPart
    for Publishee<Source, K, P>
{
}

impl<Source: RankSelect, K: super::key::Key + 'static, P: ProtocolPart, C: Communicator>
    Session<Publishee<Source, K, P>, C>
where
    K::Value: Equivalence,
{
    pub fn receive(mut self) -> Session<P, C> {
        unsafe {
            let mut value: K::Value = std::mem::uninitialized();
            self.comm
                .process_at_rank(Source::get_rank(self.state()))
                .broadcast_into(&mut value);
            self.state_mut().insert::<K>(Box::new(value));
            self.advance()
        }
    }
}
