use rkllm_rs::prelude::*;

struct PrintStats;

impl RkllmCallbackHandler for PrintStats {
    fn handle(&mut self, result: Option<RKLLMResult<'_>>, state: LLMCallState) {
        if !matches!(state, LLMCallState::Normal) {
            return;
        }
        let Some(result) = result else {
            return;
        };

        print!("{}", result.text);
        if let Some(logits) = result.logits {
            println!(" logits_count={}", logits.logits().len());
        }
        if let Some(last_hidden_layer) = result.last_hidden_layer {
            println!(
                " hidden_state_count={}",
                last_hidden_layer.hidden_states().len()
            );
        }
    }
}

fn main() {
    let mut config = LLMConfig::with_model_path("/path/to/model.rkllm");
    config.max_new_tokens = 64;

    // let handle = init(config).expect("init failed");
    // let infer = RKLLMInferParam::default();
    // handle
    //     .run(RKLLMInput::prompt("Hello"), Some(infer), PrintStats)
    //     .expect("run failed");
    let _ = config;
    let _ = PrintStats;
}
