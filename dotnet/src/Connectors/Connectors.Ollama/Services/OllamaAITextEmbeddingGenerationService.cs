// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.SemanticKernel.Connectors.OllamaAI.Client;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Http;
using Microsoft.SemanticKernel.Services;

namespace Microsoft.SemanticKernel.Connectors.OllamaAI;

/// <summary>
/// Ollama text embedding service.
/// </summary>
public sealed class OllamaAITextEmbeddingGenerationService : ITextEmbeddingGenerationService
{
    /// <summary>
    /// Initializes a new instance of the <see cref="OllamaAITextEmbeddingGenerationService"/> class.
    /// </summary>
    /// <param name="modelId">The Ollama modelId for the text generation service.</param>
    /// <param name="apiKey">API key for accessing the OllamaAI service.</param>
    /// <param name="endpoint">Optional  uri endpoint including the port where OllamaAI server is hosted. Default is https://api.Ollama.ai.</param>
    /// <param name="httpClient">Optional HTTP client to be used for communication with the OllamaAI API.</param>
    /// <param name="loggerFactory">Optional logger factory to be used for logging.</param>
    public OllamaAITextEmbeddingGenerationService(string modelId, string apiKey, Uri? endpoint = null, HttpClient? httpClient = null, ILoggerFactory? loggerFactory = null)
    {
        this.Client = new OllamaClient(
            modelId: modelId,
            endpoint: endpoint ?? httpClient?.BaseAddress,
            apiKey: apiKey,
            httpClient: HttpClientProvider.GetHttpClient(httpClient),
            logger: loggerFactory?.CreateLogger(this.GetType()) ?? NullLogger.Instance
        );

        this.AttributesInternal.Add(AIServiceExtensions.ModelIdKey, modelId);
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object?> Attributes => this.AttributesInternal;

    /// <inheritdoc/>
    public Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(IList<string> data, Kernel? kernel = null, CancellationToken cancellationToken = default)
        => this.Client.GenerateEmbeddingsAsync(data, cancellationToken, executionSettings: null, kernel);

    #region private
    private Dictionary<string, object?> AttributesInternal { get; } = [];
    private OllamaClient Client { get; }
    #endregion
}
