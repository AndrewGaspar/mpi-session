use mpi::topology::Communicator;

use crate::prelude::*;

pub unsafe trait ProtocolPart {}

pub struct Eps;

unsafe impl ProtocolPart for Eps {}

impl<C: Communicator> Session<Eps, C> {
    pub fn done(self) -> C {
        self.comm
    }
}
