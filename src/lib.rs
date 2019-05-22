pub mod collective;
pub mod error;
pub mod key;
pub mod protocol;
pub mod session;
pub mod split;
pub mod topology;

pub mod prelude {
    pub use super::collective::*;
    pub use super::protocol::Eps;
    pub use super::protocol::ProtocolPart;
    pub use super::session::Session;
    pub use super::split::Split;
    pub use super::topology::RankSelect;
}
