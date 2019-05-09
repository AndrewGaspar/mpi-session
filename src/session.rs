use std::any::{Any, TypeId};
use std::collections::BTreeMap;

use mpi::topology::Communicator;

use crate::prelude::*;

pub struct Session<P: ProtocolPart, C: Communicator> {
    pub(crate) comm: C,
    state: SessionState,
    protocol: P,
}

impl<P: ProtocolPart<State = ()>, C: Communicator> Session<P, C> {
    pub unsafe fn from_comm(comm: C) -> Self {
        Self {
            comm,
            state: SessionState::new(),
            protocol: P::build_part(()),
        }
    }
}

impl<P: ProtocolPart, C: Communicator> Session<P, C> {
    pub unsafe fn advance<N: ProtocolPart>(self, state: N::State) -> Session<N, C> {
        Session::<N, C> {
            comm: self.comm,
            state: self.state,
            protocol: N::build_part(state),
        }
    }

    pub fn protocol(&self) -> &P {
        &self.protocol
    }

    pub fn protocol_mut(&mut self) -> &mut P {
        &mut self.protocol
    }

    pub fn state(&self) -> &SessionState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut SessionState {
        &mut self.state
    }
}

pub struct SessionState {
    values: BTreeMap<TypeId, Box<dyn Any>>,
}

impl SessionState {
    fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    pub unsafe fn insert<K: 'static>(&mut self, value: Box<dyn Any>) {
        self.values.insert(TypeId::of::<K>(), value);
    }

    pub fn get_value<K: crate::key::Key + 'static>(&self) -> &K::Value {
        self.values
            .get(&TypeId::of::<K>())
            .expect("Failed to publish key!")
            .downcast_ref()
            .expect("Key stored with incorrect value!")
    }
}
