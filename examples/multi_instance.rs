use std::{
    pin::{pin, Pin},
};

use rkllm_rs::prelude::*;


struct Data1{
    userdata: String,
}
impl RkllmCallbackHandler for Data1 {
    fn handle(&self, _result: Option<RKLLMResult>, _state: LLMCallState) {
        match _state {
            LLMCallState::Normal => {
                if let Some(ret) = _result {
                    print!("{}", ret.text);
                }
            },
            LLMCallState::Waiting => {},
            LLMCallState::Finish => {
                println!("\n{}", self.userdata);
            },
            LLMCallState::Error => todo!(),
            LLMCallState::GetLastHiddenLayer => todo!(),
        }
    }
}

struct Data2{

}
impl RkllmCallbackHandler for Data2 {
    fn handle(&self, _result: Option<RKLLMResult>, _state: LLMCallState) {
        print!("2");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut param = RKLLMParam {
        ..Default::default()
    };

    let path = "/home/kautism/DeepSeek-R1-Distill-Qwen-1.5B-RK3588S-RKLLM1.1.4/deepseek-r1-1.5B-rkllm1.1.4.rkllm".to_owned();
    let path_ptr = path.as_ptr();
    param.model_path = path_ptr;


    let mut llm_handle1 = async move {rkllm_init(&mut param)};
    let mut llm_handle2 = async move {rkllm_init(&mut param)};
    
    let joined_future = futures::future::try_join(llm_handle1, llm_handle2);
    let (mut llm_handle1, mut llm_handle2) = futures::executor::block_on(joined_future)?;

    let a = async move {
        let rkllm_infer_params = RKLLMInferParam::default();
        let input = format!(
            "<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜>",
            "哈哈是我喇"
        );

        llm_handle1.run::<Data1>(
            RKLLMInput::Prompt(input),
            Some(rkllm_infer_params.clone()),
            Data1{
                userdata: "This is an example for user pass custom data into callback".to_owned(),
            },
        );
        llm_handle1.destroy();
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    };
    let b = async move {
        let rkllm_infer_params = RKLLMInferParam::default();
        let input = format!(
            "<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜>",
            "哈哈是我喇"
        );
        llm_handle2.run::<Data2>(
            RKLLMInput::Prompt(input),
            Some(rkllm_infer_params.clone()),
            Data2{},
        );
        llm_handle2.destroy();
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    };

    let ra = Box::pin(a);
    let rb = Box::pin(b);
    let joined_future = futures::future::try_join(ra, rb);
    futures::executor::block_on(joined_future)?;

    Ok(())
}
