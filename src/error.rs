use std::any::TypeId;

use quick_error::quick_error;

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        UnpopulatedKey(typeid: TypeId) {}
    }
}
