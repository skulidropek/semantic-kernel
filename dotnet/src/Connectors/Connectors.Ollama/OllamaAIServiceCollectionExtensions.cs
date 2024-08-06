// Copyright (c) Microsoft. All rights reserved.

using System;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OllamaAI;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Http;

namespace Microsoft.SemanticKernel;

/// <summary>
/// Provides extension methods for the <see cref="IServiceCollection"/> interface to configure Ollama connectors.
/// </summary>
public static class OllamaAIServiceCollectionExtensions
{
    /// <summary>
    /// Adds an Ollama chat completion service with the specified configuration.
    /// </summary>
    /// <param name="services">The <see cref="IServiceCollection"/> instance to augment.</param>
    /// <param name="modelId">The name of the Ollama modelId.</param>
    /// <param name="apiKey">The API key required for accessing the Ollama service.</param>
    /// <param name="endpoint">Optional  uri endpoint including the port where OllamaAI server is hosted. Default is https://api.Ollama.ai.</param>
    /// <param name="serviceId">A local identifier for the given AI service.</param>
    /// <returns>The same instance as <paramref name="services"/>.</returns>
    public static IServiceCollection AddOllamaChatCompletion(
        this IServiceCollection services,
        string modelId,
        string apiKey,
        Uri? endpoint = null,
        string? serviceId = null)
    {
        Verify.NotNull(services);

        return services.AddKeyedSingleton<IChatCompletionService>(serviceId, (serviceProvider, _) =>
            new OllamaAIChatCompletionService(modelId, apiKey, endpoint, HttpClientProvider.GetHttpClient(serviceProvider)));
    }

    /// <summary>
    /// Adds an Ollama text embedding generation service with the specified configuration.
    /// </summary>
    /// <param name="services">The <see cref="IServiceCollection"/> instance to augment.</param>
    /// <param name="modelId">The name of theOllama modelId.</param>
    /// <param name="apiKey">The API key required for accessing the Ollama service.</param>
    /// <param name="endpoint">Optional  uri endpoint including the port where OllamaAI server is hosted. Default is https://api.Ollama.ai.</param>
    /// <param name="serviceId">A local identifier for the given AI service.</param>
    /// <returns>The same instance as <paramref name="services"/>.</returns>
    public static IServiceCollection AddOllamaTextEmbeddingGeneration(
        this IServiceCollection services,
        string modelId,
        string apiKey,
        Uri? endpoint = null,
        string? serviceId = null)
    {
        Verify.NotNull(services);

        return services.AddKeyedSingleton<ITextEmbeddingGenerationService>(serviceId, (serviceProvider, _) =>
            new OllamaAITextEmbeddingGenerationService(modelId, apiKey, endpoint, HttpClientProvider.GetHttpClient(serviceProvider)));
    }
}
