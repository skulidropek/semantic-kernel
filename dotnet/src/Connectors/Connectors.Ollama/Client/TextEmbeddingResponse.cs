// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Microsoft.SemanticKernel.Connectors.OllamaAI.Client;

/// <summary>
/// Response for text embedding.
/// </summary>
internal sealed class TextEmbeddingResponse : OllamaResponseBase
{
    [JsonPropertyName("data")]
    public IList<OllamaEmbedding>? Data { get; set; }
}
