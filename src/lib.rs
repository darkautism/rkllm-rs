#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub mod prelude {
    use serde::Serialize;
    use std::ffi::CStr;
    use std::ptr::null_mut;
    use std::sync::Arc;
    use std::sync::Mutex;

    pub use super::RKLLMExtendParam;
    pub use super::RKLLMLoraParam;
    pub use super::RKLLMParam;
    pub use super::RKLLMResultLastHiddenLayer;

    /// Represents the state of an LLM call.
    #[derive(Debug, PartialEq, Eq)]
    pub enum LLMCallState {
        /// The LLM call is in a normal running state.
        Normal = 0,
        /// The LLM call is waiting for a complete UTF-8 encoded character.
        Waiting = 1,
        /// The LLM call has finished execution.
        Finish = 2,
        /// An error occurred during the LLM call.
        Error = 3,
        /// Retrieve the last hidden layer during inference.
        GetLastHiddenLayer = 4,
    }

    #[derive(Debug, Clone, Default)]
    pub enum KeepHistory {
        #[default]
        /// Do not keep the history of the conversation.
        NoKeepHistory = 0,
        /// Keep the history of the conversation.
        KeepHistory = 1,
    }

    /// Structure for defining parameters during inference.
    #[derive(Debug, Clone, Default)]
    pub struct RKLLMInferParam {
        /// Inference mode, such as generating text or getting the last hidden layer.
        pub mode: RKLLMInferMode,
        /// Optional Lora adapter parameters.
        pub lora_params: Option<String>,
        /// Optional prompt cache parameters.
        pub prompt_cache_params: Option<RKLLMPromptCacheParam>,
        pub keep_history: KeepHistory,
    }

    /// Defines the inference mode for the LLM.
    #[derive(Debug, Copy, Clone, Default)]
    pub enum RKLLMInferMode {
        /// The LLM generates text based on the input. This is the default mode.
        #[default]
        InferGenerate = 0,
        /// The LLM retrieves the last hidden layer for further processing.
        InferGetLastHiddenLayer = 1,
    }

    impl Into<u32> for RKLLMInferMode {
        /// Converts the enum variant to its underlying u32 value.
        fn into(self) -> u32 {
            self as u32
        }
    }

    /// Structure to define parameters for caching prompts.
    #[derive(Debug, Clone)]
    pub struct RKLLMPromptCacheParam {
        /// Indicates whether to save the prompt cache. If `true`, the cache is saved.
        pub save_prompt_cache: bool,
        /// Path to the prompt cache file.
        pub prompt_cache_path: String,
    }

    impl Default for super::RKLLMParam {
        /// Creates a default `RKLLMParam` by calling the underlying C function.
        fn default() -> Self {
            unsafe { super::rkllm_createDefaultParam() }
        }
    }

    /// Represents the result of an LLM inference.
    #[derive(Debug, Clone)]
    pub struct RKLLMResult {
        /// The generated text from the LLM.
        pub text: String,
        /// The ID of the generated token.
        pub token_id: i32,
        /// The last hidden layer's states if requested during inference.
        pub last_hidden_layer: RKLLMResultLastHiddenLayer,
    }

    #[derive(Debug, Clone)]
    pub struct RKLLMLoraAdapter {
        pub lora_adapter_path: String,
        pub lora_adapter_name: String,
        pub scale: f32,
    }

    /// Structure holding parameters for cross-attention inference.
    ///
    /// This structure is used when performing cross-attention in the decoder.
    /// It provides the encoder output (key/value caches), position indices,
    /// and attention mask.
    pub struct CrossAttnParam<'a> {
        /// Slice to encoder key cache (size: num_layers * num_tokens * num_kv_heads * head_dim).
        pub encoder_k_cache: &'a [f32],
        /// Slice to encoder value cache (size: num_layers * num_kv_heads * head_dim * num_tokens).
        pub encoder_v_cache: &'a [f32],
        /// Slice to encoder attention mask (array of size num_tokens).
        pub encoder_mask: &'a [f32],
        /// Slice to encoder token positions (array of size num_tokens).
        pub encoder_pos: &'a [i32],
    }

    /// Handle to an LLM instance.
    #[derive(Clone, Debug, Copy)]
    pub struct LLMHandle {
        handle: super::LLMHandle,
    }

    unsafe impl Send for LLMHandle {} // Asserts that the handle is safe to send across threads.
    unsafe impl Sync for LLMHandle {} // Asserts that the handle is safe to share across threads.

    /// Trait for handling callbacks from LLM operations.
    pub trait RkllmCallbackHandler {
        /// Handles the result and state of an LLM call.
        fn handle(&mut self, result: Option<RKLLMResult>, state: LLMCallState);
    }

    /// Internal structure to hold the callback handler.
    pub struct InstanceData {
        /// The callback handler wrapped in `Arc` and `Mutex` for thread safety.
        pub callback_handler: Arc<Mutex<dyn RkllmCallbackHandler + Send + Sync>>,
    }

    impl LLMHandle {
        /// Destroys the LLM instance and releases its resources.
        pub fn destroy(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let ret = unsafe { super::rkllm_destroy(self.handle) };

            if ret == 0 {
                return Ok(());
            } else {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_run returned non-zero: {}", ret),
                )));
            }
        }

        /// Runs an LLM inference task asynchronously.
        ///
        /// # Parameters
        /// - `rkllm_input`: The input data for the LLM.
        /// - `rkllm_infer_params`: Optional parameters for the inference task.
        /// - `user_data`: The callback handler to process the results.
        ///
        /// # Returns
        /// This function does not return a value directly. Instead, it starts an asynchronous operation and processes results via the provided callback handler.
        pub fn run(
            &self,
            rkllm_input: RKLLMInput,
            rkllm_infer_params: Option<RKLLMInferParam>,
            user_data: impl RkllmCallbackHandler + Send + Sync + 'static,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let instance_data = Arc::new(InstanceData {
                callback_handler: Arc::new(Mutex::new(user_data)),
            });

            let userdata_ptr = Arc::into_raw(instance_data) as *mut std::ffi::c_void;
            let prompt_cstring;
            let prompt_cstring_ptr;
            let role_text;
            let role_text_ptr;
            let mut input = match rkllm_input.input_type {
                RKLLMInputType::Prompt(prompt) => {
                    prompt_cstring = std::ffi::CString::new(prompt).unwrap();
                    prompt_cstring_ptr = prompt_cstring.as_ptr() as *const std::os::raw::c_char;

                    role_text = match rkllm_input.role {
                        RKLLMInputRole::User => "user",
                        RKLLMInputRole::Tool => "tool",
                    };
                    role_text_ptr = role_text.as_ptr() as *const std::os::raw::c_char;

                    super::RKLLMInput {
                        input_type: super::RKLLMInputType_RKLLM_INPUT_PROMPT,
                        enable_thinking: rkllm_input.enable_thinking,
                        role: role_text_ptr,
                        __bindgen_anon_1: super::RKLLMInput__bindgen_ty_1 {
                            prompt_input: prompt_cstring_ptr,
                        },
                    }
                }
                RKLLMInputType::Token(_) => todo!(),
                RKLLMInputType::Embed(_) => todo!(),
                RKLLMInputType::Multimodal(_) => todo!(),
            };

            let prompt_cache_cstring;
            let prompt_cache_cstring_ptr;

            let lora_adapter_name;
            let lora_adapter_name_ptr;
            let mut loraparam;

            let new_rkllm_infer_params: *mut super::RKLLMInferParam =
                if let Some(rkllm_infer_params) = rkllm_infer_params {
                    &mut super::RKLLMInferParam {
                        keep_history: rkllm_infer_params.keep_history as i32,
                        mode: rkllm_infer_params.mode.into(),
                        lora_params: match rkllm_infer_params.lora_params {
                            Some(a) => {
                                lora_adapter_name = a;
                                lora_adapter_name_ptr =
                                    lora_adapter_name.as_ptr() as *const std::os::raw::c_char;
                                loraparam = RKLLMLoraParam {
                                    lora_adapter_name: lora_adapter_name_ptr,
                                };
                                &mut loraparam
                            }
                            None => null_mut(),
                        },
                        prompt_cache_params: if let Some(cache_params) =
                            rkllm_infer_params.prompt_cache_params
                        {
                            prompt_cache_cstring =
                                std::ffi::CString::new(cache_params.prompt_cache_path).unwrap();
                            prompt_cache_cstring_ptr =
                                prompt_cache_cstring.as_ptr() as *const std::os::raw::c_char;

                            &mut super::RKLLMPromptCacheParam {
                                save_prompt_cache: if cache_params.save_prompt_cache {
                                    1
                                } else {
                                    0
                                },
                                prompt_cache_path: prompt_cache_cstring_ptr,
                            }
                        } else {
                            null_mut()
                        },
                    }
                } else {
                    null_mut()
                };

            let ret = unsafe {
                super::rkllm_run(
                    self.handle,
                    &mut input,
                    new_rkllm_infer_params,
                    userdata_ptr,
                )
            };

            let _ = unsafe {
                // C 語言端不會管理這個記憶體
                // 我們必須把它轉回 Arc，讓 Rust 的機制自動 Drop 它
                // 這樣包在裡面的 Sender 才會被關閉
                Arc::from_raw(userdata_ptr as *const InstanceData)
            };
            if ret == 0 {
                return Ok(());
            } else {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_run returned non-zero: {}", ret),
                )));
            }
        }

        /// Loads a prompt cache from a file.
        ///
        /// # Parameters
        /// - `cache_path`: The path to the prompt cache file.
        pub fn load_prompt_cache(
            &self,
            cache_path: &str,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let prompt_cache_path = std::ffi::CString::new(cache_path).unwrap();
            let prompt_cache_path_ptr = prompt_cache_path.as_ptr() as *const std::os::raw::c_char;
            let ret = unsafe { super::rkllm_load_prompt_cache(self.handle, prompt_cache_path_ptr) };
            if ret == 0 {
                return Ok(());
            } else {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_load_prompt_cache returned non-zero: {}", ret),
                )));
            }
        }

        /// Release a prompt cache from a file.
        pub fn release_prompt_cache(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let ret = unsafe { super::rkllm_release_prompt_cache(self.handle) };
            if ret == 0 {
                return Ok(());
            } else {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_release_prompt_cache returned non-zero: {}", ret),
                )));
            }
        }

        /// Aborts an ongoing LLM task
        pub fn abort(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let ret = unsafe { super::rkllm_abort(self.handle) };
            if ret == 0 {
                return Ok(());
            } else {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_abort returned non-zero: {}", ret),
                )));
            }
        }

        /// Checks if an LLM task is currently running.
        pub fn is_running(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let ret = unsafe { super::rkllm_is_running(self.handle) };
            if ret == 0 {
                return Ok(());
            } else {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_is_running returned non-zero: {}", ret),
                )));
            }
        }

        /// Loads a Lora adapter into the LLM.
        pub fn load_lora(
            &self,
            lora_cfg: &RKLLMLoraAdapter,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let lora_adapter_name_cstring =
                std::ffi::CString::new(lora_cfg.lora_adapter_name.clone()).unwrap();
            let lora_adapter_name_cstring_ptr =
                lora_adapter_name_cstring.as_ptr() as *const std::os::raw::c_char;
            let lora_adapter_path_cstring =
                std::ffi::CString::new(lora_cfg.lora_adapter_path.clone()).unwrap();
            let lora_adapter_path_cstring_ptr =
                lora_adapter_path_cstring.as_ptr() as *const std::os::raw::c_char;
            let mut param = super::RKLLMLoraAdapter {
                lora_adapter_path: lora_adapter_path_cstring_ptr,
                lora_adapter_name: lora_adapter_name_cstring_ptr,
                scale: lora_cfg.scale,
            };
            let ret = unsafe { super::rkllm_load_lora(self.handle, &mut param) };
            if ret == 0 {
                return Ok(());
            } else {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_load_lora returned non-zero: {}", ret),
                )));
            }
        }

        /// Clear the key-value cache for a given LLM handle.
        ///
        /// This function is used to clear part or all of the KV cache.
        ///
        /// # Parameters
        /// - `keep_system_prompt`: Flag indicating whether to retain the system prompt in the cache (true to retain, false to clear).
        ///   This flag is ignored if a specific range [start_pos, end_pos) is provided.
        /// - `start_pos`: Slice of start positions (inclusive) of the KV cache ranges to clear, one per batch.
        /// - `end_pos`: Slice of end positions (exclusive) of the KV cache ranges to clear, one per batch.
        ///   If both start_pos and end_pos are None, the entire cache will be cleared and keep_system_prompt will take effect.
        ///   If start_pos[i] < end_pos[i], only the specified range will be cleared, and keep_system_prompt will be ignored.
        ///
        /// # Note
        /// start_pos or end_pos is only valid when keep_history == 0 and the generation has been paused by returning 1 in the callback.
        pub fn clear_kv_cache(
            &self,
            keep_system_prompt: bool,
            start_pos: Option<&[i32]>,
            end_pos: Option<&[i32]>,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let start_ptr = start_pos.map_or(std::ptr::null_mut(), |s| s.as_ptr() as *mut i32);
            let end_ptr = end_pos.map_or(std::ptr::null_mut(), |s| s.as_ptr() as *mut i32);

            let ret = unsafe {
                super::rkllm_clear_kv_cache(
                    self.handle,
                    if keep_system_prompt { 1 } else { 0 },
                    start_ptr,
                    end_ptr,
                )
            };

            if ret == 0 {
                Ok(())
            } else {
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_clear_kv_cache returned non-zero: {}", ret),
                )))
            }
        }

        /// Get the current size of the key-value cache for a given LLM handle.
        ///
        /// This function returns the total number of positions currently stored in the model's KV cache.
        ///
        /// # Parameters
        /// - `cache_sizes`: Mutable slice where the per-batch cache sizes will be stored.
        ///   The slice must be preallocated with space for `n_batch` elements.
        pub fn get_kv_cache_size(
            &self,
            cache_sizes: &mut [i32],
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let ret =
                unsafe { super::rkllm_get_kv_cache_size(self.handle, cache_sizes.as_mut_ptr()) };

            if ret == 0 {
                Ok(())
            } else {
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_get_kv_cache_size returned non-zero: {}", ret),
                )))
            }
        }

        /// Sets the chat template for the LLM, including system prompt, prefix, and postfix.
        ///
        /// This function allows you to customize the chat template by providing a system prompt, a prompt prefix, and a prompt postfix.
        /// The system prompt is typically used to define the behavior or context of the language model,
        /// while the prefix and postfix are used to format the user input and output respectively.
        ///
        /// # Parameters
        /// - `system_prompt`: The system prompt that defines the context or behavior of the language model.
        /// - `prompt_prefix`: The prefix added before the user input in the chat.
        /// - `prompt_postfix`: The postfix added after the user input in the chat.
        pub fn set_chat_template(
            &self,
            system_prompt: &str,
            prompt_prefix: &str,
            prompt_postfix: &str,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let sys_prompt_c = std::ffi::CString::new(system_prompt).unwrap();
            let prefix_c = std::ffi::CString::new(prompt_prefix).unwrap();
            let postfix_c = std::ffi::CString::new(prompt_postfix).unwrap();

            let ret = unsafe {
                super::rkllm_set_chat_template(
                    self.handle,
                    sys_prompt_c.as_ptr(),
                    prefix_c.as_ptr(),
                    postfix_c.as_ptr(),
                )
            };

            if ret == 0 {
                Ok(())
            } else {
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_set_chat_template returned non-zero: {}", ret),
                )))
            }
        }

        /// Sets the function calling configuration for the LLM, including system prompt, tool definitions, and tool response token.
        ///
        /// # Parameters
        /// - `system_prompt`: The system prompt that defines the context or behavior of the language model.
        /// - `tools`: A serializable struct that defines the available functions, including their names, descriptions, and parameters.
        ///   It will be serialized to JSON before passing to C API.
        /// - `tool_response_str`: A unique tag used to identify function call results within a conversation. It acts as the marker tag,
        ///   allowing tokenizer to recognize tool outputs separately from normal dialogue turns.
        pub fn set_function_tools<T: Serialize>(
            &self,
            system_prompt: &str,
            tools: &T,
            tool_response_str: &str,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let sys_prompt_c = std::ffi::CString::new(system_prompt).unwrap();
            let tools_json = serde_json::to_string(tools).map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Failed to serialize tools: {}", e),
                ))
            })?;
            let tools_c = std::ffi::CString::new(tools_json).unwrap();
            let tool_response_c = std::ffi::CString::new(tool_response_str).unwrap();

            let ret = unsafe {
                super::rkllm_set_function_tools(
                    self.handle,
                    sys_prompt_c.as_ptr(),
                    tools_c.as_ptr(),
                    tool_response_c.as_ptr(),
                )
            };

            if ret == 0 {
                Ok(())
            } else {
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_set_function_tools returned non-zero: {}", ret),
                )))
            }
        }

        /// Sets the cross-attention parameters for the LLM decoder within a scoped closure.
        ///
        /// This function temporarily sets the cross-attention parameters for the duration of the provided closure.
        /// This ensures that the pointers to the Rust slices provided in `cross_attn_params` remain valid
        /// while the LLM is using them (e.g., during `run`).
        ///
        /// # Parameters
        /// - `cross_attn_params`: Structure containing encoder-related input data used for cross-attention.
        /// - `func`: A closure that will be executed while the parameters are set. You should call `run` inside this closure.
        ///
        /// # Example
        /// ```rust,no_run
        /// # use rkllm_rs::prelude::*;
        /// # let handle: LLMHandle = unsafe { std::mem::zeroed() };
        /// # let params = CrossAttnParam {
        /// #     encoder_k_cache: &[], encoder_v_cache: &[], encoder_mask: &[], encoder_pos: &[]
        /// # };
        /// handle.with_cross_attn(&params, |h| {
        ///     // Safe to call run here, params are valid
        ///     // h.run(..., ..., ...).unwrap();
        /// }).expect("Failed to set params");
        /// ```
        pub fn with_cross_attn<F, R>(
            &self,
            cross_attn_params: &CrossAttnParam,
            func: F,
        ) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
        where
            F: FnOnce(&LLMHandle) -> R,
        {
            // Internal guard to ensure params are unset when the scope exits
            struct ResetGuard {
                handle: super::LLMHandle,
            }

            impl Drop for ResetGuard {
                fn drop(&mut self) {
                    unsafe {
                        // Pass NULL to clear/unset the parameters
                        super::rkllm_set_cross_attn_params(self.handle, std::ptr::null_mut());
                    }
                }
            }

            // Validate that lengths match num_tokens implied by encoder_pos or others if necessary.
            // We assume the user has set up the slices correctly.
            let num_tokens = cross_attn_params.encoder_pos.len() as i32;

            let mut c_params = super::RKLLMCrossAttnParam {
                encoder_k_cache: cross_attn_params.encoder_k_cache.as_ptr() as *mut f32,
                encoder_v_cache: cross_attn_params.encoder_v_cache.as_ptr() as *mut f32,
                encoder_mask: cross_attn_params.encoder_mask.as_ptr() as *mut f32,
                encoder_pos: cross_attn_params.encoder_pos.as_ptr() as *mut i32,
                num_tokens: num_tokens,
            };

            // Set the parameters
            let ret = unsafe { super::rkllm_set_cross_attn_params(self.handle, &mut c_params) };

            if ret != 0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("rkllm_set_cross_attn_params returned non-zero: {}", ret),
                )));
            }

            // Create the guard to unset params on exit
            let _guard = ResetGuard {
                handle: self.handle,
            };

            // Execute the closure
            Ok(func(self))
        }

    }

    /// Internal callback function to handle LLM results from the C library.
    unsafe extern "C" fn callback_passtrough(
        result: *mut super::RKLLMResult,
        userdata: *mut ::std::os::raw::c_void,
        state: super::LLMCallState,
    ) -> i32 {
        Arc::increment_strong_count(userdata); // We don't actually want to free it
        let instance_data = unsafe { Arc::from_raw(userdata as *const InstanceData) };
        let new_state = match state {
            0 => LLMCallState::Normal,
            1 => LLMCallState::Waiting,
            2 => LLMCallState::Finish,
            3 => LLMCallState::Error,
            4 => LLMCallState::GetLastHiddenLayer,
            _ => panic!("Unexpected LLMCallState"),
        };

        let new_result = if result.is_null() {
            None
        } else {
            Some(RKLLMResult {
                text: if (*result).text.is_null() {
                    String::new()
                } else {
                    (unsafe { CStr::from_ptr((*result).text) })
                        .to_str()
                        .expect("Failed to convert C string")
                        .to_owned()
                        .clone()
                },
                token_id: (*result).token_id,
                last_hidden_layer: (*result).last_hidden_layer,
            })
        };

        instance_data
            .callback_handler
            .lock()
            .unwrap()
            .handle(new_result, new_state);
        0
    }

    /// Initializes the LLM with the given parameters.
    ///
    /// # Parameters
    /// - `param`: A pointer to the LLM configuration parameters.
    ///
    /// # Returns
    /// If successful, returns a `Result` containing the `LLMHandle`; otherwise, returns an error.
    pub fn rkllm_init(
        param: *mut super::RKLLMParam,
    ) -> Result<LLMHandle, Box<dyn std::error::Error + Send + Sync>> {
        let mut handle = LLMHandle {
            handle: std::ptr::null_mut(),
        };

        let callback: Option<
            unsafe extern "C" fn(
                *mut super::RKLLMResult,
                *mut ::std::os::raw::c_void,
                super::LLMCallState,
            ) -> i32,
        > = Some(callback_passtrough);
        let ret = unsafe { super::rkllm_init(&mut handle.handle, param, callback) };
        if ret == 0 {
            return Ok(handle);
        } else {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("rkllm_init returned non-zero: {}", ret),
            )));
        }
    }

    /// Represents different types of input that can be provided to the LLM.
    pub struct RKLLMInput {
        /// The type of input being provided to the LLM.
        pub input_type: RKLLMInputType,
        /// Whether to enable thinking during the inference.
        pub enable_thinking: bool,
        /// The role of the user providing the input.
        pub role: RKLLMInputRole,
    }

    /// The type of input being provided to the LLM.
    pub enum RKLLMInputType {
        /// Input is a text prompt.
        Prompt(String),
        /// Input is a sequence of tokens.
        Token(String),
        /// Input is an embedding vector.
        Embed(String),
        /// Input is multimodal, such as text and image.
        Multimodal(String),
    }

    /// The role of the user providing the input.
    pub enum RKLLMInputRole {
        /// User
        User,
        /// Tool
        Tool,
    }
}
