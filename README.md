# rkllm-rs

`rkllm-rs` is a Rust FFI wrapper for the `librkllmrt` library.

## README.md

- en [English](README.md)
- zh_TW [繁體中文](README.zh_TW.md)

## System Requirements

Before using `rkllm-rs`, you need to install `librkllmrt`. Please download and install from the following link:

[Download librkllmrt.so](https://github.com/airockchip/rknn-llm/raw/refs/heads/main/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so)

Please install `librkllmrt.so` in one of the common Linux library paths:

- `/usr/lib`
- `/lib`
- `/usr/local/lib`
- `/opt/lib`

Alternatively, you can use the `LD_LIBRARY_PATH` environment variable to specify the library path. For example:

```sh
export LD_LIBRARY_PATH=/path/to/your/library:$LD_LIBRARY_PATH
```

The model used in this example can be found [here](https://huggingface.co/VRxiaojie/DeepSeek-R1-Distill-Qwen-7B-RK3588S-RKLLM1.1.4)

For devices with less memory, you can use [this model](https://huggingface.co/VRxiaojie/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4)

## Installation

### Install Rust

First, install Rust, or refer to [this guide](https://www.rust-lang.org/tools/install)

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Lazy Way

```
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

### Using as a Library

Add the following to your `Cargo.toml`:

```
[dependencies]
rkllm-rs = "0.1.0"
```

### Using as a Binary

`rkllm-rs` also supports running as a binary, suitable for users who do not plan to do further development or prefer an out-of-the-box experience.

```
cargo install rkllm-rs --features bin
rkllm ~/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4/deepseek-r1-1.5B-rkllm1.1.4.rkllm --model_type=deepseek
```

Here is the help for the tool, with various parameters set according to the help:

```
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