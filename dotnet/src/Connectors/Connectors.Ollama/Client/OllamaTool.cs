// Copyright (c) Microsoft. All rights reserved.

using System.Text.Json.Serialization;

namespace Microsoft.SemanticKernel.Connectors.OllamaAI.Client;

/// <summary>
/// A tool to be used in the chat completion request.
/// </summary>
internal sealed class OllamaTool
{
    /// <summary>
    /// The type of the tool. Currently, only function is supported.
    /// </summary>
    [JsonPropertyName("type")]
    public string Type { get; set; }

    /// <summary>
    /// The associated function.
    /// </summary>
    [JsonPropertyName("function")]
    public OllamaFunction Function { get; set; }

    /// <summary>
    /// Construct an instance of <see cref="OllamaTool"/>.
    /// </summary>
    [JsonConstructorAttribute]
    public OllamaTool(string type, OllamaFunction function)
    {
        this.Type = type;
        this.Function = function;
    }
}
