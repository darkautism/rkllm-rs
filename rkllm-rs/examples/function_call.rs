use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct ToolParameter {
    #[serde(rename = "type")]
    param_type: String,
    description: String,
    #[serde(default)]
    enum_values: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: ToolParameters,
}

#[derive(Serialize, Deserialize)]
struct ToolParameters {
    #[serde(rename = "type")]
    param_type: String,
    properties: std::collections::HashMap<String, ToolParameter>,
    required: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ToolFunction,
}

fn main() {
    // This example demonstrates how to construct the tools JSON and call set_function_tools.
    // Note: Since we don't have a real model to run, we just show the setup.

    let weather_tool = Tool {
        tool_type: "function".to_string(),
        function: ToolFunction {
            name: "get_current_weather".to_string(),
            description: "Get the current weather in a given location".to_string(),
            parameters: ToolParameters {
                param_type: "object".to_string(),
                properties: {
                    let mut m = std::collections::HashMap::new();
                    m.insert(
                        "location".to_string(),
                        ToolParameter {
                            param_type: "string".to_string(),
                            description: "The city and state, e.g. San Francisco, CA".to_string(),
                            enum_values: None,
                        },
                    );
                    m.insert(
                        "unit".to_string(),
                        ToolParameter {
                            param_type: "string".to_string(),
                            description: "The temperature unit".to_string(),
                            enum_values: Some(vec![
                                "celsius".to_string(),
                                "fahrenheit".to_string(),
                            ]),
                        },
                    );
                    m
                },
                required: vec!["location".to_string()],
            },
        },
    };

    let tools = vec![weather_tool];

    // In a real application, you would initialize LLMHandle here.
    // let handle = ...;

    // For demonstration, we just show how you would call the function if you had a handle.
    // The following code is commented out because we cannot link against the actual library in this environment.
    /*
    let handle = LLMHandle { handle: std::ptr::null_mut() }; // Mock handle for compilation check
    // NOTE: This will crash if run because the handle is null and functions call C API,
    // but we just want to show the API usage.

    // handle.set_function_tools(
    //     "You are a helpful assistant.",
    //     &tools,
    //     "<|tool_response|>"
    // ).expect("Failed to set function tools");
    */

    println!("Tools JSON would look like:");
    println!("{}", serde_json::to_string_pretty(&tools).unwrap());

    println!("Example finished successfully (dry run).");
}
