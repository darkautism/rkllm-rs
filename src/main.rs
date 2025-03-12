use clap::builder::PossibleValue;
use clap::{Arg, ArgAction, Command, ValueEnum};
use rkllm_rs::prelude::*;
use std::io;
use std::io::Write;

#[cfg(feature = "online_config")]
use autotokenizer::AutoTokenizer;
#[cfg(feature = "online_config")]
use autotokenizer::DefaultPromptMessage;

struct UserDataWithCallBack {
    userdata: String,
}
impl RkllmCallbackHandler for UserDataWithCallBack {
    fn handle(&mut self, result: Option<RKLLMResult>, state: LLMCallState) {
        match state {
            LLMCallState::Normal => {
                print!("{}", result.unwrap().text);
                io::stdout().flush().expect("Flushing failed");
            }
            LLMCallState::Waiting => {}
            LLMCallState::Finish => {
                println!("\n{}", self.userdata);
            }
            LLMCallState::Error => {
                print!("\\run error\n");
            }
            LLMCallState::GetLastHiddenLayer => {
                print!("GetLastHiddenLayer\n");
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ModelType {
    Normal,
    DeepSeek,
}

impl ValueEnum for ModelType {
    fn value_variants<'a>() -> &'a [Self] {
        &[ModelType::Normal, ModelType::DeepSeek]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            ModelType::Normal => PossibleValue::new("normal").help("Normal model"),
            ModelType::DeepSeek => PossibleValue::new("deepseek").help("DeepSeek"),
        })
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}

impl std::str::FromStr for ModelType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        for variant in Self::value_variants() {
            if variant.to_possible_value().unwrap().matches(s, false) {
                return Ok(*variant);
            }
        }
        Err(format!("invalid variant: {s}"))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    const VERSION: &str = env!("CARGO_PKG_VERSION");
    let matches = Command::new("rkllm")
        .about("llm runner for rockchip")
        .version(VERSION)
        .arg_required_else_help(true)
        .arg(
            Arg::new("model")
                .help("Rkllm model")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("model_type")
                .long("model_type")
                .help("some module have special prefix in prompt, use this to fix")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("max_context_len")
                .short('c')
                .long("context_len")
                .help("Maximum number of tokens in the context window")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("max_new_tokens")
                .short('n')
                .long("new_tokens")
                .help("Maximum number of new tokens to generate.")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("top_k")
                .short('K')
                .long("top_k")
                .help("Top-K sampling parameter for token generation.")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("top_p")
                .short('P')
                .long("top_p")
                .help("Top-P (nucleus) sampling parameter.")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("temperature")
                .short('t')
                .long("temperature")
                .help("Sampling temperature, affecting the randomness of token selection.")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("repeat_penalty")
                .short('r')
                .long("repeat_penalty")
                .help("Penalty for repeating tokens in generation.")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("frequency_penalty")
                .short('f')
                .long("frequency_penalty")
                .help("Penalizes frequent tokens during generation.")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("presence_penalty")
                .short('p')
                .long("presence_penalty")
                .help("Penalizes tokens based on their presence in the input.")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("prompt_cache_path")
                .long("prompt_cache")
                .help("Path to the prompt cache file.")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .arg(
            Arg::new("skip_special_token")
                .long("skip_special_token")
                .help("Whether to skip special tokens during generation.")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let mut param = RKLLMParam {
        ..Default::default()
    };
    // These variable will pass into c so keep this ptr prevent crash.
    let mut cache_path = None;
    let model_path;
    let model_path_ptr;
    let mut modeltype = ModelType::Normal;
    #[cfg(feature = "online_config")]
    let mut atoken = None;

    if let Some(value) = matches.get_one::<String>("model_type") {
        #[cfg(not(feature = "online_config"))]
        {
            if value == "deepseek" {
                modeltype = ModelType::DeepSeek;
            }
        }
        #[cfg(feature = "online_config")]
        {
            if let Ok(_atoken) = AutoTokenizer::from_pretrained(value.clone(), None) {
                atoken = Some(_atoken);
            };
        }
    }

    if let Some(value) = matches.get_one::<String>("model") {
        model_path = std::ffi::CString::new(value.clone()).unwrap();
        model_path_ptr = model_path.as_ptr() as *const std::os::raw::c_char;
        param.model_path = model_path_ptr;
    }
    if let Some(value) = matches.get_one::<i32>("max_context_len") {
        param.max_context_len = *value;
    }
    if let Some(value) = matches.get_one::<i32>("max_new_tokens") {
        param.max_new_tokens = *value;
    }
    if let Some(value) = matches.get_one::<i32>("top_k") {
        param.top_k = *value;
    }
    if let Some(value) = matches.get_one::<f32>("top_p") {
        param.top_p = *value;
    }
    if let Some(value) = matches.get_one::<f32>("temperature") {
        param.temperature = *value;
    }
    if let Some(value) = matches.get_one::<f32>("repeat_penalty") {
        param.repeat_penalty = *value;
    }
    if let Some(value) = matches.get_one::<f32>("frequency_penalty") {
        param.frequency_penalty = *value;
    }
    if let Some(value) = matches.get_one::<f32>("presence_penalty") {
        param.presence_penalty = *value;
    }
    if let Some(value) = matches.get_one::<String>("prompt_cache_path") {
        cache_path = Some(value);
    }
    if matches.get_flag("skip_special_token") {
        param.skip_special_token = true;
    }

    let llm_handle = rkllm_init(&mut param)?;

    let rkllm_infer_params = RKLLMInferParam {
        mode: RKLLMInferMode::InferGenerate,
        lora_params: None,
        prompt_cache_params: if let Some(cache_path) = cache_path {
            Some(RKLLMPromptCacheParam {
                save_prompt_cache: true,
                prompt_cache_path: cache_path.to_owned(),
            })
        } else {
            None
        },
    };
    if let Some(cache_path) = cache_path {
        let _ = llm_handle.load_prompt_cache(cache_path);
    }

    loop {
        print!("Say something: ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Cannot read input");
        if input.trim() == "exit" {
            println!("Exit!");
            break;
        } else {
            // Change prompt for different type model
            input = match modeltype {
                ModelType::Normal => input,
                ModelType::DeepSeek => {
                    format!("<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜>", input)
                }
            };
            #[cfg(feature = "online_config")]
            {
                // 定義對話上下文
                let ctx = vec![
                    DefaultPromptMessage::new(
                        "system",
                        "You are a smart speaker, please help users with questions.",
                    ),
                    DefaultPromptMessage::new("user", &input),
                ];

                if let Some(ref real_atoken) = atoken {
                    if let Ok(parsed) = real_atoken.apply_chat_template(ctx, true) {
                        input = parsed;
                    } else {
                        println!("apply_chat_template failed.");
                    };
                }
            }
            // For AutoTokenizer debug
            // println!("{}", input);

            print!("\nRobot: \n");
            let _ = llm_handle.run(
                RKLLMInput::Prompt(input),
                Some(rkllm_infer_params.clone()),
                UserDataWithCallBack {
                    userdata: "This is an example for user pass custom data into callback"
                        .to_owned(),
                },
            );
        }
    }

    let _ = llm_handle.destroy();
    Ok(())
}
