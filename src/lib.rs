#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub mod prelude {

    use std::ffi::CStr;
    use std::ptr::null_mut;

    pub use super::RKLLMExtendParam;
    pub use super::RKLLMLoraParam;
    pub use super::RKLLMParam;
    pub use super::RKLLMResultLastHiddenLayer;

    type rkllm_callback =
        fn(result: Option<RKLLMResult>, userdata: *mut ::std::os::raw::c_void, state: LLMCallState);

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
    #[derive(Debug, Clone)]
    pub struct RKLLMInferParam {
        #[doc = "< Inference mode (e.g., generate or get last hidden layer)."]
        pub mode: RKLLMInferMode,
        #[doc = "< Pointer to Lora adapter parameters."]
        pub lora_params: Option<*mut RKLLMLoraParam>,
        #[doc = "< Pointer to prompt cache parameters."]
        pub prompt_cache_params: Option<RKLLMPromptCacheParam>,
    }

    #[derive(Debug, Copy, Clone)]
    pub enum RKLLMInferMode {
        #[doc = "< The LLM generates text based on input."]
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

    #[derive(Debug)]
    pub struct RKLLMResult {
        pub text: String,
        pub token_id: i32,
        pub last_hidden_layer: RKLLMResultLastHiddenLayer,
    }

    pub struct LLMHandle {
        handle: super::LLMHandle,
    }

    impl LLMHandle {
        pub fn destroy(&self) {
            unsafe { super::rkllm_destroy(self.handle) };
        }

        pub fn run(
            &self,
            rkllm_input: RKLLMInput,
            rkllm_infer_params: Option<RKLLMInferParam>,
            userdata: *mut ::std::os::raw::c_void,
        ) {
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

            let new_rkllm_infer_params: *mut super::RKLLMInferParam =
                if let Some(rkllm_infer_params) = rkllm_infer_params {
                    &mut super::RKLLMInferParam {
                        mode: rkllm_infer_params.mode.into(),
                        lora_params: match rkllm_infer_params.lora_params {
                            Some(a) => a,
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

            print!(
                "\nsuper::rkllm_run: {:?}, {:?}\n",
                self.handle, new_rkllm_infer_params
            );
            unsafe { super::rkllm_run(self.handle, &mut input, new_rkllm_infer_params, userdata) };
            print!("\nend::rkllm_run: \n");
        }

        pub fn load_prompt_cache(&self, cache_path: &str) {
            let prompt_cache_path = std::ffi::CString::new(cache_path).unwrap();
            let prompt_cache_path_ptr = prompt_cache_path.as_ptr();
            unsafe { super::rkllm_load_prompt_cache(self.handle, prompt_cache_path_ptr) };
        }
    }

    static mut CALLBACK: Option<rkllm_callback> = None;

    unsafe extern "C" fn callback_passtrough(
        result: *mut super::RKLLMResult,
        userdata: *mut ::std::os::raw::c_void,
        state: super::LLMCallState,
    ) {
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

        if let Some(callback) = CALLBACK {
            callback(new_result, userdata, new_state);
        }
    }

    pub fn rkllm_init(
        param: *mut super::RKLLMParam,
        callback: rkllm_callback,
    ) -> Result<LLMHandle, i32> {
        let mut handle = LLMHandle {
            handle: std::ptr::null_mut(),
        };
        unsafe { CALLBACK = Some(callback) };
        let callback: Option<
            unsafe extern "C" fn(
                *mut super::RKLLMResult,
                *mut ::std::os::raw::c_void,
                super::LLMCallState,
            ),
        > = Some(callback_passtrough);
        let ret = unsafe { super::rkllm_init(&mut handle.handle, param, callback) };

        print!("rkllm_init: {:?} \n", handle.handle);
        if ret == 0 {
            return Ok(handle);
        } else {
            return Err(ret);
        }
    }

    pub enum RKLLMInput {
        Prompt(String),
        Token(String),
        Embed(String),
        Multimodal(String),
    }
}
