mod collective;
mod protocol;

pub mod prelude {
    pub use super::collective::AllGather;
    pub use super::protocol::Eps;
    pub use super::protocol::Protocol;
    pub use super::protocol::ProtocolPart;
}
