// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.Text;
using System.Text.Json.Serialization;

namespace Microsoft.SemanticKernel.Connectors.OllamaAI.Client;

/// <summary>
/// Represents a chat completion chunk from Ollama.
/// </summary>
internal sealed class OllamaChatCompletionChunk
{
    [JsonPropertyName("model")]
    public string? Model { get; set; }

    [JsonPropertyName("created_at")]
    public string? CreatedAt { get; set; }

    [JsonPropertyName("message")]
    public OllamaChatMessage? Message { get; set; }

    [JsonPropertyName("done_reason")]
    public string? DoneReason { get; set; }

    [JsonPropertyName("done")]
    public bool Done { get; set; }

    [JsonPropertyName("total_duration")]
    public long? TotalDuration { get; set; }

    [JsonPropertyName("load_duration")]
    public long? LoadDuration { get; set; }

    [JsonPropertyName("prompt_eval_count")]
    public int? PromptEvalCount { get; set; }

    [JsonPropertyName("prompt_eval_duration")]
    public long? PromptEvalDuration { get; set; }

    [JsonPropertyName("eval_count")]
    public int? EvalCount { get; set; }

    [JsonPropertyName("eval_duration")]
    public long? EvalDuration { get; set; }

    internal IReadOnlyDictionary<string, object?>? GetMetadata() =>
        this._metadata ??= new Dictionary<string, object?>(8)
        {
                { nameof(OllamaChatCompletionChunk.Model), this.Model },
                { nameof(OllamaChatCompletionChunk.CreatedAt), this.CreatedAt },
                { nameof(OllamaChatCompletionChunk.Message), this.Message },
                { nameof(OllamaChatCompletionChunk.DoneReason), this.DoneReason },
                { nameof(OllamaChatCompletionChunk.Done), this.Done },
                { nameof(OllamaChatCompletionChunk.TotalDuration), this.TotalDuration },
                { nameof(OllamaChatCompletionChunk.LoadDuration), this.LoadDuration },
                { nameof(OllamaChatCompletionChunk.PromptEvalCount), this.PromptEvalCount },
                { nameof(OllamaChatCompletionChunk.PromptEvalDuration), this.PromptEvalDuration },
                { nameof(OllamaChatCompletionChunk.EvalCount), this.EvalCount },
                { nameof(OllamaChatCompletionChunk.EvalDuration), this.EvalDuration },
        };

    internal string? GetRole() => this.Message?.Role;

    internal string? GetContent() => this.Message.IsToolCall ? "" : this.Message?.Content;

    internal Encoding? GetEncoding() => null;

    private IReadOnlyDictionary<string, object?>? _metadata;

    internal OllamaUsage Usage => new()
    {
        PromptTokens = this.PromptEvalCount,
        CompletionTokens = this.EvalCount,
        TotalTokens = this.PromptEvalCount + this.EvalCount,
    };
}
