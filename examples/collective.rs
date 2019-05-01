use mpi::{self, topology::SystemCommunicator, traits::*};

use rsmpi_session::*;

fn main() -> Result<(), Box<std::error::Error>> {
    let universe = mpi::initialize().unwrap();

    let comm = universe.world();

    let size = comm.size();
    let rank = comm.rank();

    /// The Protocol type defines the communication protocol. This type provides a type safe
    /// communication pattern over an MPI communicator.
    ///
    /// This particular protocol performs an "MPI_Allgather" and is then done.
    type Protocol = AllGather<i32, Eps<SystemCommunicator>>;

    // Constructing a protocol is still unsafe since the starting state of the communicator cannot
    // be statically verified.
    let protocol = unsafe { Protocol::from_comm(comm) };

    let mut ints = vec![0; size as usize];

    let eps = protocol.all_gather(&rank, &mut ints[..]);

    let comm = eps.done();

    assert_eq!(size, comm.size());
    assert_eq!(rank, comm.rank());

    Ok(())
}
