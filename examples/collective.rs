use mpi::{self, traits::*};

use mpi_session::prelude::*;

fn main() -> Result<(), Box<std::error::Error>> {
    let universe = mpi::initialize().unwrap();

    let comm = universe.world();

    let size = comm.size();
    let rank = comm.rank();

    /// The Protocol type defines the communication protocol. This type provides a type safe
    /// communication pattern over an MPI communicator.
    ///
    /// This particular protocol performs an "MPI_Allgather" and is then done.
    type MyProtocol = AllGather<i32, Eps>;

    // Constructing a protocol is still unsafe since the starting state of the communicator cannot
    // be statically verified.
    let protocol = unsafe { Protocol::<MyProtocol, _>::from_comm(comm) };

    let mut ints = vec![0; size as usize];

    let eps = protocol.all_gather(&rank, &mut ints[..]);

    let comm = eps.done();

    assert_eq!(size, comm.size());
    assert_eq!(rank, comm.rank());

    Ok(())
}
