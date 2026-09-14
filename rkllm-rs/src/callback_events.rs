use crate::prelude::{
    LLMCallState, LLMHandle, RKLLMInferParam, RKLLMInput, RKLLMResult, RkllmCallbackHandler,
};
use std::error::Error;

/// A Rust-oriented view of RKLLM callback activity.
///
/// `Output` contains any result payload produced by the runtime. Control-only callback
/// states are exposed separately so callers do not need to understand RKLLM's numeric
/// callback state values.
#[derive(Debug)]
pub enum GenerationEvent<'a> {
    Output(RKLLMResult<'a>),
    Waiting,
    Finished,
    Error,
}

/// Handles high-level generation events.
///
/// Closures implement this trait automatically, so most callers can simply pass
/// `|event| { ... }` to `LLMHandle::run_events` or `run_events_async`.
pub trait RkllmEventHandler {
    fn handle_event(&mut self, event: GenerationEvent<'_>);
}

impl<F> RkllmEventHandler for F
where
    F: for<'a> FnMut(GenerationEvent<'a>),
{
    fn handle_event(&mut self, event: GenerationEvent<'_>) {
        self(event);
    }
}

struct EventHandlerAdapter<H> {
    handler: H,
}

impl<H> EventHandlerAdapter<H> {
    fn new(handler: H) -> Self {
        Self { handler }
    }
}

impl<H> RkllmCallbackHandler for EventHandlerAdapter<H>
where
    H: RkllmEventHandler,
{
    fn handle(&mut self, result: Option<RKLLMResult<'_>>, state: LLMCallState) {
        // Preserve every payload regardless of which low-level callback state carried it.
        // This avoids teaching callers that a terminal callback may also contain data.
        if let Some(result) = result {
            self.handler.handle_event(GenerationEvent::Output(result));
        }

        match state {
            LLMCallState::Normal | LLMCallState::GetLastHiddenLayer => {}
            LLMCallState::Waiting => self.handler.handle_event(GenerationEvent::Waiting),
            LLMCallState::Finish => self.handler.handle_event(GenerationEvent::Finished),
            LLMCallState::Error => self.handler.handle_event(GenerationEvent::Error),
        }
    }
}

impl LLMHandle {
    /// Run inference with Rust-oriented generation events.
    ///
    /// This is additive to the existing `run` API. Existing users can keep their
    /// `RkllmCallbackHandler` implementations unchanged and migrate when convenient.
    pub fn run_events<H>(
        &self,
        input: RKLLMInput,
        infer_params: Option<RKLLMInferParam>,
        handler: H,
    ) -> Result<(), Box<dyn Error + Send + Sync>>
    where
        H: RkllmEventHandler + Send + Sync + 'static,
    {
        self.run(input, infer_params, EventHandlerAdapter::new(handler))
    }

    /// Run asynchronous inference with Rust-oriented generation events.
    pub fn run_events_async<H>(
        &self,
        input: RKLLMInput,
        infer_params: Option<RKLLMInferParam>,
        handler: H,
    ) -> Result<(), Box<dyn Error + Send + Sync>>
    where
        H: RkllmEventHandler + Send + Sync + 'static,
    {
        self.run_async(input, infer_params, EventHandlerAdapter::new(handler))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::RKLLMPerfStatData;
    use std::borrow::Cow;
    use std::sync::{Arc, Mutex};

    fn result(text: &'static str) -> RKLLMResult<'static> {
        RKLLMResult {
            text: Cow::Borrowed(text),
            token_id: 1,
            last_hidden_layer: None,
            logits: None,
            perf: RKLLMPerfStatData {
                prefill_time_ms: 0.0,
                prefill_tokens: 0,
                generate_time_ms: 0.0,
                generate_tokens: 0,
                memory_usage_mb: 0.0,
            },
        }
    }

    #[test]
    fn terminal_payload_is_emitted_before_finished() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let output = Arc::clone(&seen);
        let mut adapter = EventHandlerAdapter::new(move |event| {
            let label = match event {
                GenerationEvent::Output(result) => format!("output:{}", result.text),
                GenerationEvent::Waiting => "waiting".into(),
                GenerationEvent::Finished => "finished".into(),
                GenerationEvent::Error => "error".into(),
            };
            output.lock().unwrap().push(label);
        });

        adapter.handle(Some(result("done")), LLMCallState::Finish);

        assert_eq!(
            *seen.lock().unwrap(),
            vec!["output:done".to_string(), "finished".to_string()]
        );
    }

    #[test]
    fn low_level_states_are_collapsed_to_simple_control_events() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let output = Arc::clone(&seen);
        let mut adapter = EventHandlerAdapter::new(move |event| {
            let label = match event {
                GenerationEvent::Output(_) => "output",
                GenerationEvent::Waiting => "waiting",
                GenerationEvent::Finished => "finished",
                GenerationEvent::Error => "error",
            };
            output.lock().unwrap().push(label);
        });

        adapter.handle(None, LLMCallState::Waiting);
        adapter.handle(None, LLMCallState::GetLastHiddenLayer);
        adapter.handle(None, LLMCallState::Error);

        assert_eq!(*seen.lock().unwrap(), vec!["waiting", "error"]);
    }
}
