# rkllm-rs

`rkllm-rs` 是一個使用 Rust FFI 包裝 `librkllmrt` 的函式庫。

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

## 安裝

在你的cargo.toml加入

```

[dependencies]
rkllm-rs = "0.1.0"

```

## Example

直接執行demo
本例使用的模型[在此](https://huggingface.co/VRxiaojie/DeepSeek-R1-Distill-Qwen-7B-RK3588S-RKLLM1.1.4)
對於記憶體比較小的板子，可以改使用[這個](https://huggingface.co/VRxiaojie/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4)
```
cargo run --example demo ~/DeepSeek-R1-Distill-Qwen-7B-RK3588S-RKLLM1.1.4/deepseek-r1-7B-rkllm1.1.4.rkllm
```
