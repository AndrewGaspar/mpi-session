use mpi::topology::Communicator;

use crate::prelude::*;

pub unsafe trait ProtocolPart {
    unsafe fn build_part() -> Self;
}

pub struct Eps;

unsafe impl ProtocolPart for Eps {
    unsafe fn build_part() -> Self {
        Self
    }
}

impl<C: Communicator> Session<Eps, C> {
    pub fn done(self) -> C {
        self.comm
    }
}
