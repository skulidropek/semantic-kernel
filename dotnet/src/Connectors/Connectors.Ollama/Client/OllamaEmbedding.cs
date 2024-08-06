// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Microsoft.SemanticKernel.Connectors.OllamaAI.Client;

/// <summary>
/// Ollama embedding data.
/// </summary>
internal sealed class OllamaEmbedding
{
    [JsonPropertyName("object")]
    public string? Object { get; set; }

    [JsonPropertyName("embedding")]
    public IList<float>? Embedding { get; set; }

    [JsonPropertyName("index")]
    public int? Index { get; set; }
}
