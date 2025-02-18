# rkllm-rs

`rkllm-rs` 是一個使用 Rust FFI 包裝 `librkllmrt` 的函式庫。

## README.md

- en [English](README.md)
- zh_TW [繁體中文](readme/README.zh_TW.md)

## 系統需求

在使用 `rkllm-rs` 之前，需要先安裝 `librkllmrt`。請從以下鏈接下載並安裝：

[下載 librkllmrt.so](https://github.com/airockchip/rknn-llm/raw/refs/heads/main/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so)

請將 `librkllmrt.so` 安裝至以下的常見 Linux 函式庫路徑之一：

- `/usr/lib`
- `/lib`
- `/usr/local/lib`
- `/opt/lib`

或者您可以使用 `LD_LIBRARY_PATH` 環境變數來指定函式庫路徑。例如：

```sh
export LD_LIBRARY_PATH=/path/to/your/library:$LD_LIBRARY_PATH

```

本例使用的模型[在此](https://huggingface.co/VRxiaojie/DeepSeek-R1-Distill-Qwen-7B-RK3588S-RKLLM1.1.4)

對於記憶體比較小的板子，可以改使用[這個](https://huggingface.co/VRxiaojie/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4)

## 安裝

### Install Rust

不管怎麼樣先安裝rust，或者[參考](https://www.rust-lang.org/tools/install)

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 懶

```
# If you already installed git-lfs, skip this steps
sudo apt install gitlfs


sudo curl -L https://github.com/airockchip/rknn-llm/raw/refs/heads/main/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so -o /usr/lib/librkllmrt.so
cargo install rkllm-rs --features bin
git clone https://huggingface.co/VRxiaojie/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4
rkllm ./DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4/deepseek-r1-1.5B-rkllm1.1.4.rkllm --model_type=deepseek
```

這樣你就能看到llm啟動了

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

### 以library使用

在你的cargo.toml加入

```

[dependencies]
rkllm-rs = "0.1.0"

```

### 以binary使用

rkllm-rs也支持以binary方式啟動，適合不打算進行二次開發或者打算開箱即用的使用者

```
cargo install rkllm-rs --features bin
rkllm ~/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4/deepseek-r1-1.5B-rkllm1.1.4.rkllm --model_type=deepseek
```

這是工具的help，各種參數依照help設定

```
Usage: rkllm [OPTIONS] [model]

Arguments:
  [model]  Rkllm model

Options:
      --model_type <model_type>
          some module have special prefix in prompt, use this to fix [possible values: normal, deepseek]
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
