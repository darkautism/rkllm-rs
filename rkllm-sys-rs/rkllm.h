#ifndef _RKLLM_H_
#define _RKLLM_H_
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define CPU0 (1 << 0)  // 0x01
#define CPU1 (1 << 1)  // 0x02
#define CPU2 (1 << 2)  // 0x04
#define CPU3 (1 << 3)  // 0x08
#define CPU4 (1 << 4)  // 0x10
#define CPU5 (1 << 5)  // 0x20
#define CPU6 (1 << 6)  // 0x40
#define CPU7 (1 << 7)  // 0x80

/**
 * @typedef LLMHandle
 * @brief A handle used to manage and interact with the large language model.
 */
typedef void* LLMHandle;

/**
 * @enum LLMCallState
 * @brief Describes the possible states of an LLM call.
 */
typedef enum {
    RKLLM_RUN_NORMAL  = 0, /**< The LLM call is in a normal running state. */
    RKLLM_RUN_WAITING = 1, /**< The LLM call is waiting for complete UTF-8 encoded character. */
    RKLLM_RUN_FINISH  = 2, /**< The LLM call has finished execution. */
    RKLLM_RUN_ERROR   = 3, /**< An error occurred during the LLM call. */
} LLMCallState;

/**
 * @enum RKLLMInputType
 * @brief Defines the types of inputs that can be fed into the LLM.
 */
typedef enum {
    RKLLM_INPUT_PROMPT      = 0, /**< Input is a text prompt. */
    RKLLM_INPUT_TOKEN       = 1, /**< Input is a sequence of tokens. */
    RKLLM_INPUT_EMBED       = 2, /**< Input is an embedding vector. */
    RKLLM_INPUT_MULTIMODAL  = 3, /**< Input is multimodal (e.g., text and image). */
} RKLLMInputType;

/**
 * @enum RKLLMInferMode
 * @brief Specifies the inference modes of the LLM.
 */
typedef enum {
    RKLLM_INFER_GENERATE                    = 0, /**< The LLM generates text based on input. */
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER       = 1, /**< The LLM retrieves the last hidden layer for further processing. */
    RKLLM_INFER_GET_LOGITS                  = 2, /**< The LLM retrieves logits for further processing. */
} RKLLMInferMode;

/**
 * @brief Function pointer type for callback to get LLM embeddings
 * @param userdata Pointer to user-defined data
 * @param tokens Array of token IDs
 * @param num_tokens Number of tokens in the tokens array
 * @param embed Pointer to buffer that will store the embedding output
 * @param len Length of the embedding buffer in bytes
 * @return Returns 0 on success, non-zero value on failure
 */
typedef int (*LLMGetEmbedCallback)(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len);

/**
 * @typedef LLMTokenizerCallback
 * @brief Callback function to handle tokenization.
 * @param userdata Pointer to user data for the callback.
 * @param text Pointer to the input text.
 * @param text_len Length of input text in bytes.
 * @param tokens Pointer to the array of token IDs.
 * @param n_tokens_max Max number of output tokens.
 * @return Return token count (>=0) on success, negative value on error.
 */
typedef int (*LLMTokenizerCallback)(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max);

/**
 * @struct RKLLMExtendParam
 * @brief The extend parameters for configuring an LLM instance.
 */
typedef struct {
    int32_t      base_domain_id;
    int8_t       embed_flash;
    int8_t       enabled_cpus_num;
    uint32_t     enabled_cpus_mask;
    uint8_t      n_batch;
    int8_t       use_cross_attn;
    uint8_t      reserved[104];
} RKLLMExtendParam;

/**
 * @struct RKLLMParam
 * @brief Defines the parameters for configuring an LLM instance.
 */
typedef struct {
    const char* model_path;
    int32_t max_context_len;
    int32_t max_new_tokens;
    int32_t top_k;
    int32_t n_keep;
    float top_p;
    float temperature;
    float repeat_penalty;
    float frequency_penalty;
    float presence_penalty;
    int32_t mirostat;
    float mirostat_tau;
    float mirostat_eta;
    bool skip_special_token;
    bool ignore_eos_token;
    bool is_async;
    RKLLMExtendParam extend_param;
} RKLLMParam;

typedef struct {
    const char* lora_adapter_path;
    const char* lora_adapter_name;
    float scale;
} RKLLMLoraAdapter;

typedef struct {
    float* embed;
    size_t n_tokens;
} RKLLMEmbedInput;

typedef struct {
    int32_t* input_ids;
    size_t n_tokens;
} RKLLMTokenInput;

typedef struct {
    char* prompt;
    struct {
        float* image_embed;
        size_t n_image_tokens;
        size_t n_image;
        const char* image_start;
        const char* image_end;
        const char* image_content;
        size_t image_width;
        size_t image_height;
    } image;
    struct {
        float* video_embed;
        size_t n_frame_tokens;
        size_t n_frame_per_video;
        size_t n_video;
        const char* video_start;
        const char* video_end;
        const char* video_content;
        size_t frame_width;
        size_t frame_height;
    } video;
} RKLLMMultiModalInput;

typedef struct {
    const char* role;
    bool enable_thinking;
    RKLLMInputType input_type;
    union {
        const char* prompt_input;
        RKLLMEmbedInput embed_input;
        RKLLMTokenInput token_input;
        RKLLMMultiModalInput multimodal_input;
    };
} RKLLMInput;

typedef struct {
    const char* lora_adapter_name;
} RKLLMLoraParam;

typedef struct {
    int save_prompt_cache;
    const char* prompt_cache_path;
} RKLLMPromptCacheParam;

typedef struct {
    float* encoder_k_cache;
    float* encoder_v_cache;
    float* encoder_mask;
    int32_t* encoder_pos;
    int num_tokens;
} RKLLMCrossAttnParam;

typedef struct {
  int32_t top_k;
  float top_p;
  float temperature;
  float repeat_penalty;
  float frequency_penalty;
  float presence_penalty;
  int32_t mirostat;
  float mirostat_tau;
  float mirostat_eta;
} RKLLMSamplingParam;

typedef struct {
    RKLLMInferMode mode;
    RKLLMLoraParam* lora_params;
    RKLLMPromptCacheParam* prompt_cache_params;
    RKLLMSamplingParam* sampling_params;
    int keep_history;
    int32_t max_new_tokens;
} RKLLMInferParam;

typedef struct {
    const float* hidden_states;
    int embd_size;
    int num_tokens;
} RKLLMResultLastHiddenLayer;

typedef struct {
    const float* logits;
    int vocab_size;
    int num_tokens;
} RKLLMResultLogits;

typedef struct {
    float prefill_time_ms;
    int prefill_tokens;
    float generate_time_ms;
    int generate_tokens;
    float memory_usage_mb;
} RKLLMPerfStat;

typedef struct {
    const char* text;
    int32_t token_id;
    RKLLMResultLastHiddenLayer last_hidden_layer;
    RKLLMResultLogits logits;
    RKLLMPerfStat perf;
} RKLLMResult;

typedef int(*LLMResultCallback)(RKLLMResult* result, void* userdata, LLMCallState state);

typedef struct
{
    LLMResultCallback result_callback;
    void*             result_userdata;
    LLMTokenizerCallback tokenizer_callback;
    void*                tokenizer_userdata;
    LLMGetEmbedCallback embed_callback;
    void*               embed_userdata;
} RKLLMCallback;

RKLLMParam rkllm_createDefaultParam();
int rkllm_init(LLMHandle* handle, RKLLMParam* param, RKLLMCallback* callback);
int rkllm_load_lora(LLMHandle handle, RKLLMLoraAdapter* lora_adapter);
int rkllm_load_prompt_cache(LLMHandle handle, const char* prompt_cache_path);
int rkllm_release_prompt_cache(LLMHandle handle);
int rkllm_destroy(LLMHandle handle);
int rkllm_run(LLMHandle handle, RKLLMInput* rkllm_input, RKLLMInferParam* rkllm_infer_params, void* userdata);
int rkllm_run_async(LLMHandle handle, RKLLMInput* rkllm_input, RKLLMInferParam* rkllm_infer_params, void* userdata);
int rkllm_abort(LLMHandle handle);
int rkllm_is_running(LLMHandle handle);
int rkllm_clear_kv_cache(LLMHandle handle, int keep_system_prompt, int* start_pos, int* end_pos);
int rkllm_get_kv_cache_size(LLMHandle handle, int* cache_sizes);
int rkllm_set_chat_template(LLMHandle handle, const char* system_prompt, const char* prompt_prefix, const char* prompt_postfix);
int rkllm_set_function_tools(LLMHandle handle, const char* system_prompt, const char* tools, const char* tool_response_str);
int rkllm_set_cross_attn_params(LLMHandle handle, RKLLMCrossAttnParam* cross_attn_params);

#ifdef __cplusplus
}
#endif

#endif