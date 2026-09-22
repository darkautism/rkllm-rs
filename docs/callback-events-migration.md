# Callback event API migration

This branch prototypes an additive callback API. Existing `RkllmCallbackHandler`, `LLMHandle::run`, and `LLMHandle::run_async` users do not need to change code.

## Existing API remains valid

```rust
struct Handler;

impl RkllmCallbackHandler for Handler {
    fn handle(&mut self, result: Option<RKLLMResult<'_>>, state: LLMCallState) {
        if let (LLMCallState::Normal, Some(result)) = (state, result) {
            print!("{}", result.text);
        }
    }
}

handle.run(RKLLMInput::prompt("hello"), None, Handler)?;
```

## Opt-in event API

```rust
handle.run_events(RKLLMInput::prompt("hello"), None, |event| {
    match event {
        GenerationEvent::Output(result) => print!("{}", result.text),
        GenerationEvent::Waiting => {}
        GenerationEvent::Finished => println!(),
        GenerationEvent::Error => eprintln!("generation failed"),
    }
})?;
```

The event adapter preserves payloads carried by terminal callbacks: if RKLLM supplies a result together with `Finish` or `Error`, the adapter emits `Output(result)` first and then the terminal event.

## Migration policy

1. `0.1.x`: keep the legacy callback API source-compatible; introduce `run_events` and `run_events_async` as opt-in additions.
2. After the event API has real-world usage, update examples and README to prefer it. Do not remove the legacy API in the same release.
3. Only a future semver-breaking release may remove or redesign `RkllmCallbackHandler`. If that happens, publish a deprecation cycle first.

The important rule is that users are never forced to understand the upstream C callback state machine merely to update `rkllm-rs`.
