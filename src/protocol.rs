use std::marker::PhantomData;

pub struct Protocol<P, C> {
    pub(crate) comm: C,
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
