// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Microsoft.SemanticKernel.Connectors.OllamaAI.Client;

/// <summary>
/// Request for chat completion.
/// </summary>
internal sealed class ChatCompletionRequest
{
    [JsonPropertyName("model")]
    public string Model { get; set; }

    [JsonPropertyName("messages")]
    public IList<OllamaChatMessage> Messages { get; set; } = [];

    [JsonPropertyName("temperature")]
    public double Temperature { get; set; } = 0.7;

    [JsonPropertyName("top_p")]
    public double TopP { get; set; } = 1;

    [JsonPropertyName("max_tokens")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? MaxTokens { get; set; }

    [JsonPropertyName("stream")]
    public bool Stream { get; set; } = false;

    [JsonPropertyName("safe_prompt")]
    public bool SafePrompt { get; set; } = false;

    [JsonPropertyName("tools")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IList<OllamaTool>? Tools { get; set; }

    [JsonPropertyName("tool_choice")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ToolChoice { get; set; }

    [JsonPropertyName("random_seed")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Construct an instance of <see cref="ChatCompletionRequest"/>.
    /// </summary>
    /// <param name="model">ID of the model to use.</param>
    [JsonConstructor]
    internal ChatCompletionRequest(string model)
    {
        this.Model = model;
    }

    /// <summary>
    /// Add a tool to the request.
    /// </summary>
    internal void AddTool(OllamaTool tool)
    {
        this.Tools ??= [];
        this.Tools.Add(tool);
    }

    /// <summary>
    /// Add a message to the request.
    /// </summary>
    /// <param name="message"></param>
    internal void AddMessage(OllamaChatMessage message)
    {
        this.Messages.Add(message);
    }
}
