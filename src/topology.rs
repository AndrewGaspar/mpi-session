use mpi::topology::Rank;

use super::session::SessionState;

pub unsafe trait RankSelect {
    fn get_rank(state: &SessionState) -> Rank;
}

pub struct ZeroRank;

unsafe impl RankSelect for ZeroRank {
    fn get_rank(_: &SessionState) -> Rank {
        0
    }
}

unsafe impl<K: 'static> RankSelect for crate::key::RankKey<K> {
    fn get_rank(state: &SessionState) -> Rank {
        *state.get_value::<Self>()
    }
}
