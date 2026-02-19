# Changelog

## Unreleased

### Changed
- Removed `LLMHandle::destroy()`. Handle cleanup is now managed by `Drop` only.
- Added `init_with_model_path(...)` and `rkllm_init_with_model_path(...)` so users can initialize with Rust strings without handling `CString` manually.
- Added `LLMConfig` and `init(...)` as the default Rust-first initialization API.
- Callback results now use `RKLLMResult<'_>` with borrowed slice accessors for logits/hidden states, avoiding raw pointer fields in user-facing output types.
- Kept low-level init entry points as explicit `unsafe` raw APIs (`init_raw(...)` / `rkllm_init_raw(...)`) for advanced usage.

### Added
- New UX-focused example: `examples/safe_api.rs`.

### Documentation
- Updated README (EN/zh_TW) with Rust-first safe wrapper usage guidance and sample code.
