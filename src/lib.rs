use std::marker::PhantomData;

use mpi::traits::*;

pub unsafe trait ProtocolPart {
    type Dual;
    type Comm: mpi::traits::Communicator;

    unsafe fn from_comm(c: Self::Comm) -> Self;
}

pub struct Eps<C>(C);

impl<C> Eps<C> {
    pub fn done(self) -> C {
        self.0
    }
}

unsafe impl<C: mpi::traits::Communicator> ProtocolPart for Eps<C> {
    type Dual = Eps<C>;
    type Comm = C;

    unsafe fn from_comm(c: Self::Comm) -> Self {
        Self(c)
    }
}

pub struct AllGather<T, P: ProtocolPart>(P::Comm, PhantomData<(T, P)>);

impl<T: Equivalence, P: ProtocolPart> AllGather<T, P> {
    pub fn all_gather(self, input: &T, output: &mut [T]) -> P {
        unsafe {
            self.0.all_gather_into(input, output);
            ProtocolPart::from_comm(self.0)
        }
    }
}

unsafe impl<T, P: ProtocolPart> ProtocolPart for AllGather<T, P> {
    type Dual = AllGather<T, P>;
    type Comm = P::Comm;

    unsafe fn from_comm(c: Self::Comm) -> Self {
        Self(c, PhantomData)
    }
}

pub struct Communicator<C: mpi::traits::Communicator, P: ProtocolPart>(C, P);

impl<C: mpi::traits::Communicator, P: ProtocolPart> Communicator<C, P> {
    pub unsafe fn from_comm_protocol(comm: C, protocol: P) -> Self {
        Communicator(comm, protocol)
    }
}
