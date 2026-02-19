use rkllm_rs::prelude::*;

struct Data1 {
    userdata: String,
}
impl RkllmCallbackHandler for Data1 {
    fn handle(&mut self, _result: Option<RKLLMResult<'_>>, _state: LLMCallState) {
        match _state {
            LLMCallState::Normal => {
                if let Some(ret) = _result {
                    print!("{}", ret.text);
                }
            }
            LLMCallState::Waiting => {}
            LLMCallState::Finish => {
                println!("\n{}", self.userdata);
            }
            LLMCallState::Error => todo!(),
            LLMCallState::GetLastHiddenLayer => todo!(),
        }
    }
}

struct Data2 {}
impl RkllmCallbackHandler for Data2 {
    fn handle(&mut self, _result: Option<RKLLMResult<'_>>, _state: LLMCallState) {
        print!("2");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = "/home/kautism/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4/deepseek-r1-1.5B-rkllm1.1.4.rkllm";
    let llm_handle1 = async move { init_with_model_path(path) };
    let llm_handle2 = async move { init_with_model_path(path) };

    let joined_future = futures::future::try_join(llm_handle1, llm_handle2);
    let (llm_handle1, llm_handle2) = futures::executor::block_on(joined_future)?;

    let a = async move {
        let rkllm_infer_params = RKLLMInferParam::default();
        let input = format!(
            "<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜>",
            "哈哈是我喇"
        );

        let _ = llm_handle1.run(
            RKLLMInput::prompt(input),
            Some(rkllm_infer_params.clone()),
            Data1 {
                userdata: "This is an example for user pass custom data into callback".to_owned(),
            },
        );
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    };
    let b = async move {
        let rkllm_infer_params = RKLLMInferParam::default();
        let input = format!(
            "<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜>",
            "哈哈是我喇"
        );
        let _ = llm_handle2.run(
            RKLLMInput::prompt(input),
            Some(rkllm_infer_params.clone()),
            Data2 {},
        );
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    };

    let ra = Box::pin(a);
    let rb = Box::pin(b);
    let joined_future = futures::future::try_join(ra, rb);
    futures::executor::block_on(joined_future)?;

    Ok(())
}
