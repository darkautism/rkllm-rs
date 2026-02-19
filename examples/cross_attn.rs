use rkllm_rs::prelude::*;

fn main() {
    // Since we cannot link against the real library, we just demonstrate the syntax.
    // In a real app, you would init the handle properly.
    let _handle = unsafe { std::mem::zeroed::<LLMHandle>() };

    // Mock data for cross attention
    // In reality, these would be large tensors from an encoder model
    let k_cache = vec![0.0f32; 100];
    let v_cache = vec![0.0f32; 100];
    let mask = vec![0.0f32; 10];
    let pos = vec![0; 10];

    let _params = CrossAttnParam {
        encoder_k_cache: &k_cache,
        encoder_v_cache: &v_cache,
        encoder_mask: &mask,
        encoder_pos: &pos,
    };

    println!("Setting up cross attention scope...");

    // The `with_cross_attn` method ensures that `params` are set before the closure runs
    // and unset (set to NULL) after the closure finishes (or panics).
    // This prevents dangling pointers if `params` were to be dropped while `handle` still referred to them.

    /*
    // Uncomment to run (requires linking)
    let result = handle.with_cross_attn(&params, |h| {
        println!("Inside scope: Params are valid. Calling run()...");

        // Call h.run(...) here
        // h.run(input, infer_params, callback).unwrap();

        "Run completed"
    });
    */

    println!("Scope finished. Params are now unset in the LLM handle.");
    println!("Example finished successfully (dry run).");
}
