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

    #[derive(Debug, PartialEq, Eq)]
    pub enum LLMCallState {
        #[doc = "< The LLM call is in a normal running state."]
        Normal = 0,
        #[doc = "< The LLM call is waiting for complete UTF-8 encoded character."]
        Waiting = 1,
        #[doc = "< The LLM call has finished execution."]
        Finish = 2,
        #[doc = "< An error occurred during the LLM call."]
        Error = 3,
        #[doc = "< Retrieve the last hidden layer during inference."]
        GetLastHiddenLayer = 4,
    }

    #[doc = " @struct RKLLMInferParam\n @brief Structure for defining parameters during inference."]
    #[derive(Debug, Clone, Default)]
    pub struct RKLLMInferParam {
        #[doc = "< Inference mode (e.g., generate or get last hidden layer)."]
        pub mode: RKLLMInferMode,
        #[doc = "< Pointer to Lora adapter parameters."]
        pub lora_params: Option<String>,
        #[doc = "< Pointer to prompt cache parameters."]
        pub prompt_cache_params: Option<RKLLMPromptCacheParam>,
    }

    #[derive(Debug, Copy, Clone, Default)]
    pub enum RKLLMInferMode {
        #[doc = "< The LLM generates text based on input."]
        #[default]
        InferGenerate = 0,
        #[doc = "< The LLM retrieves the last hidden layer for further processing."]
        InferGetLastHiddenLayer = 1,
    }
    impl Into<u32> for RKLLMInferMode {
        fn into(self) -> u32 {
            self as u32
        }
    }

    #[doc = " @struct RKLLMPromptCacheParam\n @brief Structure to define parameters for caching prompts."]
    #[derive(Debug, Clone)]
    pub struct RKLLMPromptCacheParam {
        #[doc = "< Flag to indicate whether to save the prompt cache (0 = don't save, 1 = save)."]
        pub save_prompt_cache: bool,
        #[doc = "< Path to the prompt cache file."]
        pub prompt_cache_path: String,
    }

    impl Default for super::RKLLMParam {
        fn default() -> Self {
            unsafe { super::rkllm_createDefaultParam() }
        }
    }

    #[doc = " @struct RKLLMResult\n @brief Structure to represent the result of LLM inference."]
    #[derive(Debug, Clone)]
    pub struct RKLLMResult {
        #[doc = "< Generated text result."]
        pub text: String,
        #[doc = "< ID of the generated token."]
        pub token_id: i32,
        #[doc = "< Hidden states of the last layer (if requested)."]
        pub last_hidden_layer: RKLLMResultLastHiddenLayer,
    }

    #[doc = " @struct LLMHandle\n @brief LLMHandle."]
    #[derive(Clone, Debug, Copy)]
    pub struct LLMHandle {
        handle: super::LLMHandle,
    }

    unsafe impl Send for LLMHandle {} // Asserts the pointer is safe to send
    unsafe impl Sync for LLMHandle {} // Asserts the pointer is safe to share

    pub trait RkllmCallbackHandler {
        fn handle(&mut self, result: Option<RKLLMResult>, state: LLMCallState);
    }

    pub struct InstanceData {
        pub callback_handler: Arc<Mutex<dyn RkllmCallbackHandler + Send + Sync>>,
    }

    impl LLMHandle {
        #[doc = " @brief Destroys the LLM instance and releases resources.\n @param handle LLM handle.\n @return Status code (0 for success, non-zero for failure)."]
        pub fn destroy(&self) -> i32 {
            unsafe { super::rkllm_destroy(self.handle) }
        }

        #[doc = " @brief Runs an LLM inference task asynchronously.\n @param handle LLM handle.\n @param rkllm_input Input data for the LLM.\n @param rkllm_infer_params Parameters for the inference task.\n @param userdata Pointer to user data for the callback.\n @return Status code (0 for success, non-zero for failure)."]
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
                    prompt_cstring_ptr = prompt_cstring.as_ptr();
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
                                lora_adapter_name_ptr = lora_adapter_name.as_ptr();
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
                            prompt_cache_cstring_ptr = prompt_cache_cstring.as_ptr();

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

        #[doc = " @brief Loads a prompt cache from a file.\n @param handle LLM handle.\n @param prompt_cache_path Path to the prompt cache file.\n @return Status code (0 for success, non-zero for failure)."]
        pub fn load_prompt_cache(&self, cache_path: &str) {
            let prompt_cache_path = std::ffi::CString::new(cache_path).unwrap();
            let prompt_cache_path_ptr = prompt_cache_path.as_ptr();
            unsafe { super::rkllm_load_prompt_cache(self.handle, prompt_cache_path_ptr) };
        }
    }

    unsafe extern "C" fn callback_passtrough(
        result: *mut super::RKLLMResult,
        userdata: *mut ::std::os::raw::c_void,
        state: super::LLMCallState,
    ) {
        Arc::increment_strong_count(userdata); // 我們沒有真的要free掉它
        let instance_data = unsafe { Arc::from_raw(userdata as *const InstanceData) };
        let new_state = match state {
            0 => LLMCallState::Normal,
            1 => LLMCallState::Waiting,
            2 => LLMCallState::Finish,
            3 => LLMCallState::Error,
            4 => LLMCallState::GetLastHiddenLayer,
            _ => panic!("Not expect LLMCallState"),
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
                        .expect("Convert cstr failed")
                        .to_owned()
                        .clone()
                },
                token_id: (*result).token_id,
                last_hidden_layer: (*result).last_hidden_layer,
            })
        };

        instance_data.callback_handler.lock().unwrap().handle(new_result, new_state);
    }

    #[doc = " @brief Initializes the LLM with the given parameters.\n @param handle Pointer to the LLM handle.\n @param param Configuration parameters for the LLM.\n @param callback Callback function to handle LLM results.\n @return Status code (0 for success, non-zero for failure)."]
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
                format!("rkllm_init ret non zero: {}", ret),
            )));
        }
    }

    pub enum RKLLMInput {
        #[doc = "< Input is a text prompt."]
        Prompt(String),
        #[doc = "< Input is a sequence of tokens."]
        Token(String),
        #[doc = "< Input is an embedding vector."]
        Embed(String),
        #[doc = "< Input is multimodal (e.g., text and image)."]
        Multimodal(String),
    }
}
