#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub use rkllm_sys_rs::*;

pub mod prelude {
    use serde::Serialize;
    use std::borrow::Cow;
    use std::ffi::{c_void, CStr, CString};
    use std::io;
    use std::os::raw::c_char;
    use std::ptr::null_mut;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    type BoxError = Box<dyn std::error::Error + Send + Sync>;

    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    pub enum LLMCallState {
        Normal = 0,
        Waiting = 1,
        Finish = 2,
        Error = 3,
        GetLastHiddenLayer = 4,
    }

    #[derive(Debug, Clone, Default)]
    pub enum KeepHistory {
        #[default]
        NoKeepHistory = 0,
        KeepHistory = 1,
    }

    #[derive(Debug, Clone, Default)]
    pub struct RKLLMInferParam {
        pub mode: RKLLMInferMode,
        pub lora_params: Option<String>,
        pub prompt_cache_params: Option<RKLLMPromptCacheParam>,
        pub keep_history: KeepHistory,
    }

    #[derive(Debug, Copy, Clone, Default)]
    pub enum RKLLMInferMode {
        #[default]
        InferGenerate = 0,
        InferGetLastHiddenLayer = 1,
        InferGetLogits = 2,
    }

    impl From<RKLLMInferMode> for u32 {
        fn from(value: RKLLMInferMode) -> Self {
            value as u32
        }
    }

    #[derive(Debug, Clone)]
    pub struct RKLLMPromptCacheParam {
        pub save_prompt_cache: bool,
        pub prompt_cache_path: String,
    }

    #[derive(Debug, Clone, Default)]
    pub struct LLMExtendParam {
        pub base_domain_id: i32,
        pub embed_flash: i8,
        pub enabled_cpus_num: i8,
        pub enabled_cpus_mask: u32,
        pub n_batch: u8,
        pub use_cross_attn: i8,
    }

    impl From<super::RKLLMExtendParam> for LLMExtendParam {
        fn from(value: super::RKLLMExtendParam) -> Self {
            Self {
                base_domain_id: value.base_domain_id,
                embed_flash: value.embed_flash,
                enabled_cpus_num: value.enabled_cpus_num,
                enabled_cpus_mask: value.enabled_cpus_mask,
                n_batch: value.n_batch,
                use_cross_attn: value.use_cross_attn,
            }
        }
    }

    impl From<&LLMExtendParam> for super::RKLLMExtendParam {
        fn from(value: &LLMExtendParam) -> Self {
            Self {
                base_domain_id: value.base_domain_id,
                embed_flash: value.embed_flash,
                enabled_cpus_num: value.enabled_cpus_num,
                enabled_cpus_mask: value.enabled_cpus_mask,
                n_batch: value.n_batch,
                use_cross_attn: value.use_cross_attn,
                reserved: [0; 104],
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct LLMConfig {
        pub model_path: Option<String>,
        pub max_context_len: i32,
        pub max_new_tokens: i32,
        pub top_k: i32,
        pub n_keep: i32,
        pub top_p: f32,
        pub temperature: f32,
        pub repeat_penalty: f32,
        pub frequency_penalty: f32,
        pub presence_penalty: f32,
        pub mirostat: i32,
        pub mirostat_tau: f32,
        pub mirostat_eta: f32,
        pub skip_special_token: bool,
        pub is_async: bool,
        pub img_start: Option<String>,
        pub img_end: Option<String>,
        pub img_content: Option<String>,
        pub extend_param: LLMExtendParam,
    }

    fn c_string_ptr_to_option(ptr: *const c_char) -> Option<String> {
        if ptr.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }

    impl Default for LLMConfig {
        fn default() -> Self {
            let raw = super::RKLLMParam::default();
            Self {
                model_path: None,
                max_context_len: raw.max_context_len,
                max_new_tokens: raw.max_new_tokens,
                top_k: raw.top_k,
                n_keep: raw.n_keep,
                top_p: raw.top_p,
                temperature: raw.temperature,
                repeat_penalty: raw.repeat_penalty,
                frequency_penalty: raw.frequency_penalty,
                presence_penalty: raw.presence_penalty,
                mirostat: raw.mirostat,
                mirostat_tau: raw.mirostat_tau,
                mirostat_eta: raw.mirostat_eta,
                skip_special_token: raw.skip_special_token,
                is_async: raw.is_async,
                img_start: c_string_ptr_to_option(raw.img_start),
                img_end: c_string_ptr_to_option(raw.img_end),
                img_content: c_string_ptr_to_option(raw.img_content),
                extend_param: raw.extend_param.into(),
            }
        }
    }

    impl LLMConfig {
        pub fn with_model_path(model_path: impl Into<String>) -> Self {
            let mut config = Self::default();
            config.model_path = Some(model_path.into());
            config
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct RKLLMResultLastHiddenLayerData<'a> {
        hidden_states: &'a [f32],
        pub embd_size: i32,
        pub num_tokens: i32,
    }

    impl<'a> RKLLMResultLastHiddenLayerData<'a> {
        pub fn hidden_states(&self) -> &'a [f32] {
            self.hidden_states
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct RKLLMResultLogitsData<'a> {
        logits: &'a [f32],
        pub vocab_size: i32,
        pub num_tokens: i32,
    }

    impl<'a> RKLLMResultLogitsData<'a> {
        pub fn logits(&self) -> &'a [f32] {
            self.logits
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct RKLLMPerfStatData {
        pub prefill_time_ms: f32,
        pub prefill_tokens: i32,
        pub generate_time_ms: f32,
        pub generate_tokens: i32,
        pub memory_usage_mb: f32,
    }

    #[derive(Debug)]
    pub struct RKLLMResult<'a> {
        pub text: Cow<'a, str>,
        pub token_id: i32,
        pub last_hidden_layer: Option<RKLLMResultLastHiddenLayerData<'a>>,
        pub logits: Option<RKLLMResultLogitsData<'a>>,
        pub perf: RKLLMPerfStatData,
    }

    fn checked_len(a: i32, b: i32) -> Option<usize> {
        let a = usize::try_from(a).ok()?;
        let b = usize::try_from(b).ok()?;
        a.checked_mul(b)
    }

    unsafe fn optional_f32_slice<'a>(ptr: *const f32, len: usize) -> Option<&'a [f32]> {
        if ptr.is_null() || len == 0 {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(ptr, len) })
        }
    }

    impl<'a> RKLLMResult<'a> {
        unsafe fn from_raw(result: &'a super::RKLLMResult) -> Self {
            let text = if result.text.is_null() {
                Cow::Borrowed("")
            } else {
                unsafe { CStr::from_ptr(result.text) }.to_string_lossy()
            };

            let last_hidden_layer = checked_len(
                result.last_hidden_layer.num_tokens,
                result.last_hidden_layer.embd_size,
            )
            .and_then(|len| unsafe {
                optional_f32_slice(result.last_hidden_layer.hidden_states, len)
            })
            .map(|hidden_states| RKLLMResultLastHiddenLayerData {
                hidden_states,
                embd_size: result.last_hidden_layer.embd_size,
                num_tokens: result.last_hidden_layer.num_tokens,
            });

            let logits = checked_len(result.logits.num_tokens, result.logits.vocab_size)
                .and_then(|len| unsafe { optional_f32_slice(result.logits.logits, len) })
                .map(|logits| RKLLMResultLogitsData {
                    logits,
                    vocab_size: result.logits.vocab_size,
                    num_tokens: result.logits.num_tokens,
                });

            Self {
                text,
                token_id: result.token_id,
                last_hidden_layer,
                logits,
                perf: RKLLMPerfStatData {
                    prefill_time_ms: result.perf.prefill_time_ms,
                    prefill_tokens: result.perf.prefill_tokens,
                    generate_time_ms: result.perf.generate_time_ms,
                    generate_tokens: result.perf.generate_tokens,
                    memory_usage_mb: result.perf.memory_usage_mb,
                },
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct RKLLMLoraAdapter {
        pub lora_adapter_path: String,
        pub lora_adapter_name: String,
        pub scale: f32,
    }

    pub struct CrossAttnParam<'a> {
        pub encoder_k_cache: &'a [f32],
        pub encoder_v_cache: &'a [f32],
        pub encoder_mask: &'a [f32],
        pub encoder_pos: &'a [i32],
    }

    #[derive(Debug)]
    pub struct LLMHandle {
        handle: super::LLMHandle,
        is_destroyed: AtomicBool,
        _owned_param_strings: InitParamStrings,
    }

    unsafe impl Send for LLMHandle {}
    unsafe impl Sync for LLMHandle {}

    pub trait RkllmCallbackHandler {
        fn handle(&mut self, result: Option<RKLLMResult<'_>>, state: LLMCallState);
    }

    #[derive(Debug, Default)]
    struct InitParamStrings {
        model_path: Option<CString>,
        img_start: Option<CString>,
        img_end: Option<CString>,
        img_content: Option<CString>,
    }

    fn raw_param_from_config(
        config: &LLMConfig,
    ) -> Result<(super::RKLLMParam, InitParamStrings), BoxError> {
        let strings = InitParamStrings {
            model_path: config
                .model_path
                .as_ref()
                .map(|value| CString::new(value.as_str()))
                .transpose()?,
            img_start: config
                .img_start
                .as_ref()
                .map(|value| CString::new(value.as_str()))
                .transpose()?,
            img_end: config
                .img_end
                .as_ref()
                .map(|value| CString::new(value.as_str()))
                .transpose()?,
            img_content: config
                .img_content
                .as_ref()
                .map(|value| CString::new(value.as_str()))
                .transpose()?,
        };

        let raw = super::RKLLMParam {
            model_path: strings
                .model_path
                .as_ref()
                .map_or(std::ptr::null(), |path| path.as_ptr()),
            max_context_len: config.max_context_len,
            max_new_tokens: config.max_new_tokens,
            top_k: config.top_k,
            n_keep: config.n_keep,
            top_p: config.top_p,
            temperature: config.temperature,
            repeat_penalty: config.repeat_penalty,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            mirostat: config.mirostat,
            mirostat_tau: config.mirostat_tau,
            mirostat_eta: config.mirostat_eta,
            skip_special_token: config.skip_special_token,
            is_async: config.is_async,
            img_start: strings
                .img_start
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
            img_end: strings
                .img_end
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
            img_content: strings
                .img_content
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
            extend_param: (&config.extend_param).into(),
        };

        Ok((raw, strings))
    }

    struct RunArguments {
        input: Box<super::RKLLMInput>,
        infer_param: Option<Box<super::RKLLMInferParam>>,
        _role: CString,
        _prompt: Option<CString>,
        _lora_adapter_name: Option<CString>,
        _lora_param: Option<Box<super::RKLLMLoraParam>>,
        _prompt_cache_path: Option<CString>,
        _prompt_cache_param: Option<Box<super::RKLLMPromptCacheParam>>,
    }

    impl RunArguments {
        fn new(input: RKLLMInput, infer_param: Option<RKLLMInferParam>) -> Result<Self, BoxError> {
            let role = CString::new(match input.role {
                RKLLMInputRole::User => "user",
                RKLLMInputRole::Tool => "tool",
            })?;

            let prompt = match input.input_type {
                RKLLMInputType::Prompt(prompt) => Some(CString::new(prompt)?),
                RKLLMInputType::Token(_) => {
                    return Err(Box::new(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "RKLLM_INPUT_TOKEN is not wrapped yet",
                    )))
                }
                RKLLMInputType::Embed(_) => {
                    return Err(Box::new(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "RKLLM_INPUT_EMBED is not wrapped yet",
                    )))
                }
                RKLLMInputType::Multimodal(_) => {
                    return Err(Box::new(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "RKLLM_INPUT_MULTIMODAL is not wrapped yet",
                    )))
                }
            };

            let raw_input = Box::new(super::RKLLMInput {
                input_type: super::RKLLMInputType_RKLLM_INPUT_PROMPT,
                enable_thinking: input.enable_thinking,
                role: role.as_ptr(),
                __bindgen_anon_1: super::RKLLMInput__bindgen_ty_1 {
                    prompt_input: prompt.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()),
                },
            });

            let mut lora_adapter_name = None;
            let mut lora_param = None;
            let mut prompt_cache_path = None;
            let mut prompt_cache_param = None;

            let raw_infer_param = if let Some(infer_param) = infer_param {
                lora_adapter_name = infer_param.lora_params.map(CString::new).transpose()?;
                lora_param = lora_adapter_name.as_ref().map(|name| {
                    Box::new(super::RKLLMLoraParam {
                        lora_adapter_name: name.as_ptr(),
                    })
                });

                prompt_cache_path = infer_param
                    .prompt_cache_params
                    .as_ref()
                    .map(|cache| CString::new(cache.prompt_cache_path.clone()))
                    .transpose()?;
                prompt_cache_param = infer_param.prompt_cache_params.as_ref().map(|cache| {
                    Box::new(super::RKLLMPromptCacheParam {
                        save_prompt_cache: if cache.save_prompt_cache { 1 } else { 0 },
                        prompt_cache_path: prompt_cache_path
                            .as_ref()
                            .map_or(std::ptr::null(), |path| path.as_ptr()),
                    })
                });

                Some(Box::new(super::RKLLMInferParam {
                    keep_history: infer_param.keep_history as i32,
                    mode: infer_param.mode.into(),
                    lora_params: lora_param
                        .as_mut()
                        .map_or(null_mut(), |param| param.as_mut() as *mut _),
                    prompt_cache_params: prompt_cache_param
                        .as_mut()
                        .map_or(null_mut(), |param| param.as_mut() as *mut _),
                }))
            } else {
                None
            };

            Ok(Self {
                input: raw_input,
                infer_param: raw_infer_param,
                _role: role,
                _prompt: prompt,
                _lora_adapter_name: lora_adapter_name,
                _lora_param: lora_param,
                _prompt_cache_path: prompt_cache_path,
                _prompt_cache_param: prompt_cache_param,
            })
        }

        fn input_ptr(&mut self) -> *mut super::RKLLMInput {
            self.input.as_mut() as *mut _
        }

        fn infer_param_ptr(&mut self) -> *mut super::RKLLMInferParam {
            self.infer_param
                .as_mut()
                .map_or(null_mut(), |param| param.as_mut() as *mut _)
        }
    }

    struct InstanceData {
        callback_handler: Arc<Mutex<dyn RkllmCallbackHandler + Send + Sync>>,
        finished: AtomicBool,
        run_args: Option<RunArguments>,
    }

    impl InstanceData {
        fn new(
            callback_handler: impl RkllmCallbackHandler + Send + Sync + 'static,
            run_args: Option<RunArguments>,
        ) -> Self {
            Self {
                callback_handler: Arc::new(Mutex::new(callback_handler)),
                finished: AtomicBool::new(false),
                run_args,
            }
        }

        fn run_args_ptrs(
            &mut self,
        ) -> Result<(*mut super::RKLLMInput, *mut super::RKLLMInferParam), BoxError> {
            let args = self
                .run_args
                .as_mut()
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "missing run arguments"))?;
            Ok((args.input_ptr(), args.infer_param_ptr()))
        }
    }

    struct CallbackContextGuard {
        userdata_ptr: *const InstanceData,
        reclaim_on_drop: bool,
    }

    impl CallbackContextGuard {
        fn new(instance_data: &Arc<InstanceData>) -> Self {
            Self {
                userdata_ptr: Arc::into_raw(Arc::clone(instance_data)),
                reclaim_on_drop: true,
            }
        }

        fn userdata_ptr(&self) -> *mut c_void {
            self.userdata_ptr as *mut c_void
        }

        fn disarm(&mut self) {
            self.reclaim_on_drop = false;
        }
    }

    impl Drop for CallbackContextGuard {
        fn drop(&mut self) {
            if !self.reclaim_on_drop {
                return;
            }
            let _ = unsafe { Arc::from_raw(self.userdata_ptr) };
        }
    }

    fn status_to_result(api: &str, ret: i32) -> Result<(), BoxError> {
        if ret == 0 {
            Ok(())
        } else {
            Err(Box::new(io::Error::new(
                io::ErrorKind::Other,
                format!("{} returned non-zero: {}", api, ret),
            )))
        }
    }

    impl LLMHandle {
        fn ensure_alive(&self) -> Result<(), BoxError> {
            if self.handle.is_null() {
                return Err(Box::new(io::Error::new(
                    io::ErrorKind::Other,
                    "rkllm handle is null",
                )));
            }
            if self.is_destroyed.load(Ordering::Acquire) {
                return Err(Box::new(io::Error::new(
                    io::ErrorKind::Other,
                    "rkllm handle is already destroyed",
                )));
            }
            Ok(())
        }

        pub fn run(
            &self,
            rkllm_input: RKLLMInput,
            rkllm_infer_params: Option<RKLLMInferParam>,
            user_data: impl RkllmCallbackHandler + Send + Sync + 'static,
        ) -> Result<(), BoxError> {
            self.ensure_alive()?;

            let run_args = RunArguments::new(rkllm_input, rkllm_infer_params)?;
            let mut instance_data = Arc::new(InstanceData::new(user_data, Some(run_args)));
            let (input_ptr, infer_param_ptr) = {
                let data = Arc::get_mut(&mut instance_data).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        "failed to get mutable callback context",
                    )
                })?;
                data.run_args_ptrs()?
            };
            let mut callback_context = CallbackContextGuard::new(&instance_data);

            let ret = unsafe {
                super::rkllm_run(
                    self.handle,
                    input_ptr,
                    infer_param_ptr,
                    callback_context.userdata_ptr(),
                )
            };

            let finished = instance_data.finished.load(Ordering::Acquire);
            if finished {
                callback_context.disarm();
            }

            status_to_result("rkllm_run", ret)?;
            if !finished {
                return Err(Box::new(io::Error::new(
                    io::ErrorKind::Other,
                    "rkllm_run returned before terminal callback; callback context remains owned by runtime",
                )));
            }
            Ok(())
        }

        pub fn run_sync(
            &self,
            rkllm_input: RKLLMInput,
            rkllm_infer_params: Option<RKLLMInferParam>,
            user_data: impl RkllmCallbackHandler + Send + Sync + 'static,
        ) -> Result<(), BoxError> {
            self.run(rkllm_input, rkllm_infer_params, user_data)
        }

        pub fn run_async(
            &self,
            rkllm_input: RKLLMInput,
            rkllm_infer_params: Option<RKLLMInferParam>,
            user_data: impl RkllmCallbackHandler + Send + Sync + 'static,
        ) -> Result<(), BoxError> {
            self.ensure_alive()?;

            let run_args = RunArguments::new(rkllm_input, rkllm_infer_params)?;
            let mut instance_data = Arc::new(InstanceData::new(user_data, Some(run_args)));
            let (input_ptr, infer_param_ptr) = {
                let data = Arc::get_mut(&mut instance_data).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        "failed to get mutable callback context",
                    )
                })?;
                data.run_args_ptrs()?
            };
            let mut callback_context = CallbackContextGuard::new(&instance_data);

            let ret = unsafe {
                super::rkllm_run_async(
                    self.handle,
                    input_ptr,
                    infer_param_ptr,
                    callback_context.userdata_ptr(),
                )
            };

            if ret == 0 || instance_data.finished.load(Ordering::Acquire) {
                callback_context.disarm();
            }

            status_to_result("rkllm_run_async", ret)
        }

        pub fn load_prompt_cache(&self, cache_path: &str) -> Result<(), BoxError> {
            self.ensure_alive()?;
            let prompt_cache_path = CString::new(cache_path)?;
            let ret =
                unsafe { super::rkllm_load_prompt_cache(self.handle, prompt_cache_path.as_ptr()) };
            status_to_result("rkllm_load_prompt_cache", ret)
        }

        pub fn release_prompt_cache(&self) -> Result<(), BoxError> {
            self.ensure_alive()?;
            let ret = unsafe { super::rkllm_release_prompt_cache(self.handle) };
            status_to_result("rkllm_release_prompt_cache", ret)
        }

        pub fn abort(&self) -> Result<(), BoxError> {
            self.ensure_alive()?;
            let ret = unsafe { super::rkllm_abort(self.handle) };
            status_to_result("rkllm_abort", ret)
        }

        pub fn is_running(&self) -> Result<(), BoxError> {
            self.ensure_alive()?;
            let ret = unsafe { super::rkllm_is_running(self.handle) };
            status_to_result("rkllm_is_running", ret)
        }

        pub fn load_lora(&self, lora_cfg: &RKLLMLoraAdapter) -> Result<(), BoxError> {
            self.ensure_alive()?;
            let lora_adapter_name = CString::new(lora_cfg.lora_adapter_name.clone())?;
            let lora_adapter_path = CString::new(lora_cfg.lora_adapter_path.clone())?;
            let mut param = super::RKLLMLoraAdapter {
                lora_adapter_path: lora_adapter_path.as_ptr(),
                lora_adapter_name: lora_adapter_name.as_ptr(),
                scale: lora_cfg.scale,
            };
            let ret = unsafe { super::rkllm_load_lora(self.handle, &mut param) };
            status_to_result("rkllm_load_lora", ret)
        }

        pub fn clear_kv_cache(
            &self,
            keep_system_prompt: bool,
            start_pos: Option<&[i32]>,
            end_pos: Option<&[i32]>,
        ) -> Result<(), BoxError> {
            self.ensure_alive()?;
            if let (Some(start), Some(end)) = (start_pos, end_pos) {
                if start.len() != end.len() {
                    return Err(Box::new(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "start_pos and end_pos length mismatch",
                    )));
                }
            }

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
            status_to_result("rkllm_clear_kv_cache", ret)
        }

        pub fn get_kv_cache_size(&self, cache_sizes: &mut [i32]) -> Result<(), BoxError> {
            self.ensure_alive()?;
            let ret =
                unsafe { super::rkllm_get_kv_cache_size(self.handle, cache_sizes.as_mut_ptr()) };
            status_to_result("rkllm_get_kv_cache_size", ret)
        }

        pub fn set_chat_template(
            &self,
            system_prompt: &str,
            prompt_prefix: &str,
            prompt_postfix: &str,
        ) -> Result<(), BoxError> {
            self.ensure_alive()?;
            let system_prompt = CString::new(system_prompt)?;
            let prompt_prefix = CString::new(prompt_prefix)?;
            let prompt_postfix = CString::new(prompt_postfix)?;
            let ret = unsafe {
                super::rkllm_set_chat_template(
                    self.handle,
                    system_prompt.as_ptr(),
                    prompt_prefix.as_ptr(),
                    prompt_postfix.as_ptr(),
                )
            };
            status_to_result("rkllm_set_chat_template", ret)
        }

        pub fn set_function_tools<T: Serialize>(
            &self,
            system_prompt: &str,
            tools: &T,
            tool_response_str: &str,
        ) -> Result<(), BoxError> {
            self.ensure_alive()?;
            let system_prompt = CString::new(system_prompt)?;
            let tools_json = serde_json::to_string(tools).map_err(|err| {
                Box::new(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("failed to serialize tools: {}", err),
                )) as BoxError
            })?;
            let tools_json = CString::new(tools_json)?;
            let tool_response = CString::new(tool_response_str)?;

            let ret = unsafe {
                super::rkllm_set_function_tools(
                    self.handle,
                    system_prompt.as_ptr(),
                    tools_json.as_ptr(),
                    tool_response.as_ptr(),
                )
            };
            status_to_result("rkllm_set_function_tools", ret)
        }

        pub fn with_cross_attn<F, R>(
            &self,
            cross_attn_params: &CrossAttnParam,
            func: F,
        ) -> Result<R, BoxError>
        where
            F: FnOnce(&LLMHandle) -> R,
        {
            self.ensure_alive()?;

            struct ResetGuard {
                handle: super::LLMHandle,
            }

            impl Drop for ResetGuard {
                fn drop(&mut self) {
                    unsafe {
                        super::rkllm_set_cross_attn_params(self.handle, std::ptr::null_mut());
                    }
                }
            }

            let mut c_params = super::RKLLMCrossAttnParam {
                encoder_k_cache: cross_attn_params.encoder_k_cache.as_ptr() as *mut f32,
                encoder_v_cache: cross_attn_params.encoder_v_cache.as_ptr() as *mut f32,
                encoder_mask: cross_attn_params.encoder_mask.as_ptr() as *mut f32,
                encoder_pos: cross_attn_params.encoder_pos.as_ptr() as *mut i32,
                num_tokens: cross_attn_params.encoder_pos.len() as i32,
            };

            let ret = unsafe { super::rkllm_set_cross_attn_params(self.handle, &mut c_params) };
            status_to_result("rkllm_set_cross_attn_params", ret)?;

            let _guard = ResetGuard {
                handle: self.handle,
            };
            Ok(func(self))
        }
    }

    impl Drop for LLMHandle {
        fn drop(&mut self) {
            if self.handle.is_null() || self.is_destroyed.swap(true, Ordering::AcqRel) {
                return;
            }
            let _ = unsafe { super::rkllm_destroy(self.handle) };
        }
    }

    unsafe extern "C" fn callback_passtrough(
        result: *mut super::RKLLMResult,
        userdata: *mut c_void,
        state: super::LLMCallState,
    ) -> i32 {
        if userdata.is_null() {
            return 0;
        }

        let instance_data = Arc::from_raw(userdata as *const InstanceData);

        let state = match state {
            0 => LLMCallState::Normal,
            1 => LLMCallState::Waiting,
            2 => LLMCallState::Finish,
            3 => LLMCallState::Error,
            4 => LLMCallState::GetLastHiddenLayer,
            _ => LLMCallState::Error,
        };

        let result = if result.is_null() {
            None
        } else {
            Some(unsafe { RKLLMResult::from_raw(&*result) })
        };

        if let Ok(mut handler) = instance_data.callback_handler.lock() {
            handler.handle(result, state);
        }

        // The runtime owns one raw Arc pointer across callbacks; once we hit a terminal state
        // we stop restoring that raw pointer so Rust can drop callback state automatically.
        if matches!(state, LLMCallState::Finish | LLMCallState::Error) {
            instance_data.finished.store(true, Ordering::Release);
        } else {
            let _ = Arc::into_raw(instance_data);
        }

        0
    }

    pub fn init(config: LLMConfig) -> Result<LLMHandle, BoxError> {
        let (mut raw_param, owned_strings) = raw_param_from_config(&config)?;
        unsafe { rkllm_init_raw_with_strings(&mut raw_param as *mut _, owned_strings) }
    }

    pub fn init_with_model_path(model_path: impl Into<String>) -> Result<LLMHandle, BoxError> {
        init(LLMConfig::with_model_path(model_path))
    }

    #[deprecated(since = "0.1.15", note = "Use init_with_model_path(...) instead.")]
    pub fn rkllm_init_with_model_path(
        model_path: impl Into<String>,
    ) -> Result<LLMHandle, BoxError> {
        init_with_model_path(model_path)
    }

    pub unsafe fn init_raw(param: &mut super::RKLLMParam) -> Result<LLMHandle, BoxError> {
        unsafe { rkllm_init_raw_with_strings(param as *mut _, InitParamStrings::default()) }
    }

    #[deprecated(
        since = "0.1.15",
        note = "Use init_raw(...) for explicit low-level usage."
    )]
    pub unsafe fn rkllm_init_raw(param: &mut super::RKLLMParam) -> Result<LLMHandle, BoxError> {
        unsafe { init_raw(param) }
    }

    unsafe fn rkllm_init_raw_with_strings(
        param: *mut super::RKLLMParam,
        owned_param_strings: InitParamStrings,
    ) -> Result<LLMHandle, BoxError> {
        let mut handle = std::ptr::null_mut();
        let callback: Option<
            unsafe extern "C" fn(*mut super::RKLLMResult, *mut c_void, super::LLMCallState) -> i32,
        > = Some(callback_passtrough);

        let ret = unsafe { super::rkllm_init(&mut handle, param, callback) };
        if ret == 0 {
            Ok(LLMHandle {
                handle,
                is_destroyed: AtomicBool::new(false),
                _owned_param_strings: owned_param_strings,
            })
        } else {
            Err(Box::new(io::Error::new(
                io::ErrorKind::Other,
                format!("rkllm_init returned non-zero: {}", ret),
            )))
        }
    }

    pub struct RKLLMInput {
        pub input_type: RKLLMInputType,
        pub enable_thinking: bool,
        pub role: RKLLMInputRole,
    }

    impl RKLLMInput {
        pub fn prompt(prompt: impl Into<String>) -> Self {
            Self {
                input_type: RKLLMInputType::Prompt(prompt.into()),
                enable_thinking: false,
                role: RKLLMInputRole::User,
            }
        }

        #[allow(non_snake_case)]
        #[deprecated(since = "0.1.14", note = "Use RKLLMInput::prompt(...) instead.")]
        pub fn Prompt(prompt: impl Into<String>) -> Self {
            Self::prompt(prompt)
        }
    }

    pub enum RKLLMInputType {
        Prompt(String),
        Token(String),
        Embed(String),
        Multimodal(String),
    }

    pub enum RKLLMInputRole {
        User,
        Tool,
    }
}
