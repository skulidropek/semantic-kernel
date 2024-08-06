// Copyright (c) Microsoft. All rights reserved.

using System.Text.Json.Serialization;

namespace Microsoft.SemanticKernel.Connectors.OllamaAI.Client;

/// <summary>
/// Base class for Ollama response.
/// </summary>
internal abstract class OllamaResponseBase
{
    [JsonPropertyName("id")]
    public string? Id { get; set; }

    [JsonPropertyName("object")]
    public string? Object { get; set; }

    [JsonPropertyName("model")]
    public string? Model { get; set; }

    [JsonPropertyName("usage")]
    public OllamaUsage? Usage { get; set; }
}
