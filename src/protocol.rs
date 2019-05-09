use mpi::topology::Communicator;

use crate::prelude::*;

pub unsafe trait ProtocolPart {
    type State;

    unsafe fn build_part(state: Self::State) -> Self;
}

pub struct Eps;

unsafe impl ProtocolPart for Eps {
    type State = ();

    unsafe fn build_part(_: Self::State) -> Self {
        Self
    }
}

impl<C: Communicator> Session<Eps, C> {
    pub fn done(self) -> C {
        self.comm
    }
}
