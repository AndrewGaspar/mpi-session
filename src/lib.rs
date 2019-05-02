use std::marker::PhantomData;

use mpi::traits::*;

pub mod prelude {
    pub use super::AllGather;
    pub use super::Eps;
    pub use super::Protocol;
    pub use super::ProtocolPart;
}

pub struct Protocol<P, C> {
    comm: C,
    _phantom: PhantomData<P>,
}

impl<P, C> Protocol<P, C> {
    pub unsafe fn from_comm(comm: C) -> Self {
        Self {
            comm,
            _phantom: PhantomData,
        }
    }
}

pub unsafe trait ProtocolPart {}

pub struct Eps;

unsafe impl ProtocolPart for Eps {}

impl<C> Protocol<Eps, C> {
    pub fn done(self) -> C {
        self.comm
    }
}

pub struct AllGather<T: Equivalence, P: ProtocolPart>(PhantomData<(T, P)>);

impl<T: Equivalence, P: ProtocolPart, C: Communicator> Protocol<AllGather<T, P>, C> {
    pub fn all_gather(self, input: &T, output: &mut [T]) -> Protocol<P, C> {
        unsafe {
            self.comm.all_gather_into(input, output);
            Protocol::from_comm(self.comm)
        }
    }
}

unsafe impl<T: Equivalence, P: ProtocolPart> ProtocolPart for AllGather<T, P> {}
