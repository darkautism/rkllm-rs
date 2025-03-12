#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub mod prelude {
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

    /// Structure for defining parameters during inference.
    #[derive(Debug, Clone, Default)]
    pub struct RKLLMInferParam {
        /// Inference mode, such as generating text or getting the last hidden layer.
        pub mode: RKLLMInferMode,
        /// Optional Lora adapter parameters.
        pub lora_params: Option<String>,
        /// Optional prompt cache parameters.
        pub prompt_cache_params: Option<RKLLMPromptCacheParam>,
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
        ///
        /// # Returns
        /// A status code: 0 for success, non-zero for failure.
        pub fn destroy(&self) -> i32 {
            unsafe { super::rkllm_destroy(self.handle) }
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
        ) {
            let instance_data = Arc::new(InstanceData {
                callback_handler: Arc::new(Mutex::new(user_data)),
            });

            let userdata_ptr = Arc::into_raw(instance_data) as *mut std::ffi::c_void;
            let prompt_cstring;
            let prompt_cstring_ptr;
            let mut input = match rkllm_input {
                RKLLMInput::Prompt(prompt) => {
                    prompt_cstring = std::ffi::CString::new(prompt).unwrap();
                    prompt_cstring_ptr = prompt_cstring.as_ptr() as *const std::os::raw::c_char;
                    super::RKLLMInput {
                        input_type: super::RKLLMInputType_RKLLM_INPUT_PROMPT,
                        __bindgen_anon_1: super::RKLLMInput__bindgen_ty_1 {
                            prompt_input: prompt_cstring_ptr,
                        },
                    }
                }
                RKLLMInput::Token(_) => todo!(),
                RKLLMInput::Embed(_) => todo!(),
                RKLLMInput::Multimodal(_) => todo!(),
            };

            let prompt_cache_cstring;
            let prompt_cache_cstring_ptr;

            let lora_adapter_name;
            let lora_adapter_name_ptr;
            let mut loraparam;

            let new_rkllm_infer_params: *mut super::RKLLMInferParam =
                if let Some(rkllm_infer_params) = rkllm_infer_params {
                    &mut super::RKLLMInferParam {
                        mode: rkllm_infer_params.mode.into(),
                        lora_params: match rkllm_infer_params.lora_params {
                            Some(a) => {
                                lora_adapter_name = a;
                                lora_adapter_name_ptr = lora_adapter_name.as_ptr() as *const std::os::raw::c_char;
                                loraparam = RKLLMLoraParam{
                                    lora_adapter_name: lora_adapter_name_ptr
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
                            prompt_cache_cstring_ptr = prompt_cache_cstring.as_ptr() as *const std::os::raw::c_char;

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

            unsafe {
                super::rkllm_run(
                    self.handle,
                    &mut input,
                    new_rkllm_infer_params,
                    userdata_ptr,
                )
            };
        }

        /// Loads a prompt cache from a file.
        ///
        /// # Parameters
        /// - `cache_path`: The path to the prompt cache file.
        ///
        /// # Returns
        /// This function does not return a value directly. Instead, it loads the cache into the LLM instance.
        pub fn load_prompt_cache(&self, cache_path: &str) {
            let prompt_cache_path = std::ffi::CString::new(cache_path).unwrap();
            let prompt_cache_path_ptr = prompt_cache_path.as_ptr() as *const std::os::raw::c_char;
            unsafe { super::rkllm_load_prompt_cache(self.handle, prompt_cache_path_ptr) };
        }
    }

    /// Internal callback function to handle LLM results from the C library.
    unsafe extern "C" fn callback_passtrough(
        result: *mut super::RKLLMResult,
        userdata: *mut ::std::os::raw::c_void,
        state: super::LLMCallState,
    ) {
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

        instance_data.callback_handler.lock().unwrap().handle(new_result, new_state);
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
            ),
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
    pub enum RKLLMInput {
        /// Input is a text prompt.
        Prompt(String),
        /// Input is a sequence of tokens.
        Token(String),
        /// Input is an embedding vector.
        Embed(String),
        /// Input is multimodal, such as text and image.
        Multimodal(String),
    }
}