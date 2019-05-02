use std::marker::PhantomData;

use mpi::traits::*;

use super::prelude::*;

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
