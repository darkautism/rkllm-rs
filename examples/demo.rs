use std::io::{self, Write};
use std::os::unix::ffi::OsStrExt; // 引入標準輸入/輸出的模組

use rkllm_rs::prelude::*;

fn callback(result: Option<RKLLMResult>, _userdata: *mut ::std::os::raw::c_void, state: LLMCallState) {
    match state {
        LLMCallState::Normal => {
            print!("{}", result.unwrap().text);
            io::stdout().flush().expect("Flushing failed");
        }
        LLMCallState::Waiting => {
            print!("Waiting\n");
        }
        LLMCallState::Finish => {
            print!("\n");
        }
        LLMCallState::Error => {
            print!("\\run error\n");
        }
        LLMCallState::GetLastHiddenLayer => {
            print!("GetLastHiddenLayer\n");
        }
    }
}

fn main() {
    let mut param = RKLLMParam {
        ..Default::default()
    };

    let mut argv_iter = argv::iter();
    argv_iter.next();
    let Some(model_path) = argv_iter.next() else {
        panic!("faild");
    };
    param.model_path = model_path.as_bytes().as_ptr();
    //Set sampling parameters
    param.top_k = 1;
    param.top_p = 0.95;
    param.temperature = 0.8;
    param.repeat_penalty = 1.1;
    param.frequency_penalty = 0.0;
    param.presence_penalty = 0.0;
    param.max_new_tokens = 2048;
    param.max_context_len = 2048;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;
    let Ok(llm_handle) = rkllm_init(&mut param, callback) else {
        panic!("Init rkllm failed.");
    };
    print!("rkllm init success\n");

    // Prompt Cache
    let cache_path = "./prompt_cache.bin";

    let rkllm_infer_params = RKLLMInferParam {
        mode: RKLLMInferMode::InferGenerate,
        lora_params: None,
        prompt_cache_params: None,
        // prompt_cache_params: Some(RKLLMPromptCacheParam {
        //     save_prompt_cache: true,
        //     prompt_cache_path: cache_path.to_owned(),
        // }),
    };

    //llm_handle.load_prompt_cache(cache_path);

    loop {
        print!("Say something: ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Cannot read input");
        if input == "exit" {
            println!("Exit!");
            break;
        } else {
            input = format!("<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜>", input);
            print!("\nRobot: \n");
            llm_handle.run(
                RKLLMInput::Prompt(input),
                Some(rkllm_infer_params.clone()),
                std::ptr::null_mut(),
            );
        }
    }

    llm_handle.destroy();
}
