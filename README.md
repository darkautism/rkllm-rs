<div align="center">
<picture>
   <source media="(prefers-color-scheme: dark)" srcset="images/header-dark.png">
   <img alt="Logo for Rkllm for Rust" src="images/header-light.png">
</picture>

# RKLLM-RS

`rkllm-rs` is a Rust FFI wrapper for the `librkllmrt` library.


[![dependency status](https://deps.rs/repo/github/darkautism/sensevoice-rs/status.svg)](https://deps.rs/repo/github/darkautism/sensevoice-rs)
[![][github-stars-shield]][github-stars-link]
[![][github-issues-shield]][github-issues-shield-link]
[![][github-contributors-shield]][github-contributors-link]
[![][license-shield]][license-shield-link]
[![][last-commit-shield]][last-commit-shield-link]

</div>

## System Requirements

Before using `rkllm-rs`, you need to install `librkllmrt`. Please download and install from the following link:

[Download librkllmrt.so](https://github.com/airockchip/rknn-llm/raw/refs/heads/main/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so)

Please install `librkllmrt.so` in one of the common Linux library paths:

- `/usr/lib`
- `/lib`
- `/usr/local/lib`
- `/opt/lib`

Alternatively, you can use the `LD_LIBRARY_PATH` environment variable to specify the library path. For example:

```bash
export LD_LIBRARY_PATH=/path/to/your/library:$LD_LIBRARY_PATH
```

The model used in this example can be found [here](https://huggingface.co/VRxiaojie/DeepSeek-R1-Distill-Qwen-7B-RK3588S-RKLLM1.1.4)

For devices with less memory, you can use [this model](https://huggingface.co/VRxiaojie/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4)

## Installation

### Install Rust

First, install Rust, or refer to [this guide](https://www.rust-lang.org/tools/install)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Lazy Way

```bash
# If you already installed git-lfs, skip this step
sudo apt install git-lfs

sudo curl -L https://github.com/airockchip/rknn-llm/raw/refs/heads/main/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so -o /usr/lib/librkllmrt.so
cargo install rkllm-rs --features bin
git clone https://huggingface.co/VRxiaojie/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4
rkllm ./DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4/deepseek-r1-1.5B-rkllm1.1.4.rkllm --model_type=deepseek
```

You should now see the LLM start up:

```
I rkllm: rkllm-runtime version: 1.1.4, rknpu driver version: 0.9.7, platform: RK3588

rkllm init success
Say something: Hello

Robot: 
<think>

</think>

Hello! How can I assist you today? 😊
Say something:
```

### Version Support

| Rkllm Version | Rkllm-rs version |
|---|---|
| v1.2.3 | 0.1.14 |
| v1.2.1 | 0.1.10 |
| v1.2.0 | 0.1.9 |
| v1.1.4 | 0.1.8 |


### Using as a Library

Add the following to your `Cargo.toml`:

```toml
[dependencies]
rkllm-rs = "0.1.14"
```

### Safe Wrapper API (Recommended)

`rkllm-rs` now provides Rust-first wrapper types so normal usage does not need C pointers:

- Initialize with `LLMConfig` + `init(...)`
- Callback result uses `RKLLMResult<'_>` with borrowed slices for large tensors
- Resource cleanup is automatic via `Drop` (no explicit `destroy()`)

```rust
use rkllm_rs::prelude::*;

struct Handler;

impl RkllmCallbackHandler for Handler {
    fn handle(&mut self, result: Option<RKLLMResult<'_>>, state: LLMCallState) {
        if let (LLMCallState::Normal, Some(result)) = (state, result) {
            print!("{}", result.text);
            if let Some(logits) = result.logits {
                println!("logits: {}", logits.logits().len());
            }
        }
    }
}

let mut config = LLMConfig::with_model_path("model.rkllm");
config.max_new_tokens = 128;
let handle = init(config)?;
```

For a full example, see `rkllm-rs/examples/safe_api.rs`.

### Using as a Binary

`rkllm-rs` also supports running as a binary, suitable for users who do not plan to do further development or prefer an out-of-the-box experience.

```bash
cargo install rkllm-rs --features bin
rkllm ~/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4/deepseek-r1-1.5B-rkllm1.1.4.rkllm --model_type=deepseek
```

Here is the help for the tool, with various parameters set according to the help:

```bash
Usage: rkllm [OPTIONS] [model]

Arguments:
  [model]  Rkllm model

Options:
      --model_type <model_type>
          Some module have special prefix in prompt, use this to fix [possible values: normal, deepseek]
  -c, --context_len <max_context_len>
          Maximum number of tokens in the context window
  -n, --new_tokens <max_new_tokens>
          Maximum number of new tokens to generate.
  -K, --top_k <top_k>
          Top-K sampling parameter for token generation.
  -P, --top_p <top_p>
          Top-P (nucleus) sampling parameter.
  -t, --temperature <temperature>
          Sampling temperature, affecting the randomness of token selection.
  -r, --repeat_penalty <repeat_penalty>
          Penalty for repeating tokens in generation.
  -f, --frequency_penalty <frequency_penalty>
          Penalizes frequent tokens during generation.
  -p, --presence_penalty <presence_penalty>
          Penalizes tokens based on their presence in the input.
      --prompt_cache <prompt_cache_path>
          Path to the prompt cache file.
      --skip_special_token
          Whether to skip special tokens during generation.
  -h, --help
          Print help (see more with '--help')
  -V, --version
          Print version
```

## Function Calling

`rkllm-rs` supports function calling. You can define your tools using Rust structs derived with `serde::Serialize` and pass them to the `set_function_tools` method.

First, ensure you have `serde` and `serde_json` dependencies in your `Cargo.toml`.

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rkllm-rs = "..."
```

Example usage:

```rust
use rkllm_rs::prelude::*;
use serde::{Serialize, Deserialize};

// Define your tool structures
#[derive(Serialize)]
struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ToolFunction,
}

// ... other struct definitions (ToolFunction, ToolParameters, etc.)

let tools = vec![
    Tool {
        tool_type: "function".to_string(),
        function: ToolFunction {
            name: "get_current_weather".to_string(),
            description: "Get the current weather".to_string(),
            parameters: ToolParameters { ... },
        },
    }
];

// Call set_function_tools
// handle.set_function_tools(
//     "System prompt here...",
//     &tools, // Pass the serializable tools struct
//     "<|tool_response|>"
// ).expect("Failed to set function tools");
```

For a complete example, please refer to `rkllm-rs/examples/function_call.rs`.

## Cross Attention (Experimental)

This library supports setting cross-attention parameters for multimodal models.
Since the C API expects raw pointers that must remain valid during inference, this wrapper uses a scoped closure approach to ensure safety.

```rust
let params = CrossAttnParam {
    encoder_k_cache: &k_cache,
    encoder_v_cache: &v_cache,
    encoder_mask: &mask,
    encoder_pos: &pos,
};

handle.with_cross_attn(&params, |h| {
    // The parameters are valid within this block.
    // Call h.run(...) here.
    // Do NOT rely on params being available after this block returns.
}).expect("Failed to set params");
```

See `rkllm-rs/examples/cross_attn.rs` for details.

## Online Tokenizer Config

Currently, the model types are hardcoded in the program, and unsupported models will not correctly generate `bos_token` and assistant prompts. Most models will produce incorrect responses without the correct prompts, such as irrelevant answers or self-dialogue (though, to be fair, they might still engage in self-dialogue even with the prompts).

Most models have `tokenizer_config.json` available online. Reading this configuration file can generate the correct prompts.

You can manually create the prompt using `tokenizer_config.json` or use Python's AutoTokenizer to generate it.

This library provides a method to automatically fetch the corresponding model's `tokenizer_config.json` from online sources.

```bash
cargo install rkllm-rs --features "bin, online_config"
rkllm ~/Tinnyllama-1.1B-rk3588-rkllm-1.1.4/TinyLlama-1.1B-Chat-v1.0-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm --model_type=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

This tool will fetch the `tokenizer_config.json` for TinyLlama online and attempt to correct the prompts.


## Support the Project

If this project has saved you time or helped you in your workflow, consider supporting its continued development. Your contribution helps me keep the project maintained and feature-rich!

[![][ko-fi-shield]][ko-fi-link]
[![][paypal-shield]][paypal-link]


<!-- Link Definitions -->
[release-shield]: https://img.shields.io/github/v/release/darkautism/rkllm-rs?color=369eff&labelColor=black&logo=github&style=flat-square
[release-link]: https://github.com/darkautism/rkllm-rs/releases
[license-shield]: https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square
[license-shield-link]: https://github.com/darkautism/rkllm-rs/blob/main/LICENSE
[last-commit-shield]: https://img.shields.io/github/last-commit/darkautism/rkllm-rs?color=c4f042&labelColor=black&style=flat-square
[last-commit-shield-link]: https://github.com/darkautism/rkllm-rs/commits/main
[github-stars-shield]: https://img.shields.io/github/stars/darkautism/rkllm-rs?labelColor&style=flat-square&color=ffcb47
[github-stars-link]: https://github.com/darkautism/rkllm-rs
[github-issues-shield]: https://img.shields.io/github/issues/darkautism/rkllm-rs?labelColor=black&style=flat-square&color=ff80eb
[github-issues-shield-link]: https://github.com/darkautism/rkllm-rs/issues
[github-contributors-shield]: https://img.shields.io/github/contributors/darkautism/rkllm-rs?color=c4f042&labelColor=black&style=flat-square
[github-contributors-link]: https://github.com/darkautism/rkllm-rs/graphs/contributors
[ko-fi-shield]: https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white
[ko-fi-link]: https://ko-fi.com/kautism
[paypal-shield]: https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white
[paypal-link]: https://paypal.me/kautism

