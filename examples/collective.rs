use mpi::{self, traits::*};

use mpi_session::prelude::*;
use mpi_session::topology::ZeroRank;

struct Server;
type ServerRank = mpi_session::key::RankKey<Server>;

fn main() -> Result<(), Box<std::error::Error>> {
    let universe = mpi::initialize().unwrap();

    let comm = universe.world();

    let size = comm.size();
    let rank = comm.rank();

    /// The Session type defines the communication protocol. This type provides a type safe
    /// communication pattern over an MPI communicator.
    ///
    /// This particular protocol performs an "MPI_Allgather" and is then done.
    type MyProtocol = AllGather<i32, Publish<ZeroRank, ServerRank, Gather<ServerRank, i32, Eps>>>;

    // Constructing a session is still unsafe since the starting state of the communicator cannot
    // be statically verified.
    let session = unsafe { Session::<MyProtocol, _>::from_comm(comm) };

    let mut ints = vec![0; size as usize];

    let session = session.all_gather(&rank, &mut ints[..]);

    let session = match session.split() {
        PublishSplit::Publisher(p) => p.publish(1),
        PublishSplit::Publishee(p) => p.receive(),
    };

    let session = match session.split() {
        GatherSplit::Gatherer(g) => {
            let mut ranks = vec![0; comm.size() as usize];
            let session = g.gather(&comm.rank(), &mut ranks[..]);
            println!("{}: {:?}", comm.rank(), ranks);
            session
        }
        GatherSplit::Gatheree(g) => {
            g.gather(&comm.rank())
        }
    };

    let comm = session.done();

    assert_eq!(size, comm.size());
    assert_eq!(rank, comm.rank());

    Ok(())
}
