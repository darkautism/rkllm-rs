#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// Keep the existing public surface intact while prototyping the next callback API.
// The legacy module remains the source of all current safe-wrapper types and raw re-exports.
#[path = "lib.rs"]
mod legacy;

pub use legacy::*;

mod callback_events;
pub use callback_events::{GenerationEvent, RkllmEventHandler};
