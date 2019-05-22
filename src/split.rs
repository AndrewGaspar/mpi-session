use mpi::traits::*;

use crate::prelude::*;

pub enum Split<L, R, C: Communicator> {
    Left(Session<L, C>),
    Right(Session<R, C>),
}
