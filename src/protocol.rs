use crate::prelude::*;

pub unsafe trait ProtocolPart {}

pub struct Eps;

unsafe impl ProtocolPart for Eps {}

impl<C> Session<Eps, C> {
    pub fn done(self) -> C {
        self.comm
    }
}
