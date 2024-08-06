// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Azure;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Diagnostics;
using Microsoft.SemanticKernel.Http;
using Microsoft.SemanticKernel.Text;

namespace Microsoft.SemanticKernel.Connectors.OllamaAI.Client;

/// <summary>
/// The Ollama client.
/// </summary>
internal sealed class OllamaClient
{
    internal OllamaClient(
        string modelId,
        HttpClient httpClient,
        string apiKey,
        Uri? endpoint = null,
        ILogger? logger = null)
    {
        Verify.NotNullOrWhiteSpace(modelId);
        Verify.NotNullOrWhiteSpace(apiKey);
        Verify.NotNull(httpClient);

        this._endpoint = endpoint;
        this._modelId = modelId;
        this._apiKey = apiKey;
        this._httpClient = httpClient;
        this._logger = logger ?? NullLogger.Instance;
        this._streamJsonParser = new StreamJsonParser();
    }

    internal async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(ChatHistory chatHistory, CancellationToken cancellationToken, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null)
    {
        this.ValidateChatHistory(chatHistory);

        string modelId = executionSettings?.ModelId ?? this._modelId;
        var OllamaExecutionSettings = OllamaAIPromptExecutionSettings.FromExecutionSettings(executionSettings);
        var chatRequest = this.CreateChatCompletionRequest(modelId, stream: false, chatHistory, OllamaExecutionSettings, kernel);
        var endpoint = this.GetEndpoint(OllamaExecutionSettings, path: "chat/completions");
        var autoInvoke = kernel is not null && OllamaExecutionSettings.ToolCallBehavior?.MaximumAutoInvokeAttempts > 0 && s_inflightAutoInvokes.Value < MaxInflightAutoInvokes;

        for (int requestIndex = 1; ; requestIndex++)
        {
            OllamaChatCompletionChunk? responseData = null;
            List<ChatMessageContent> responseContent;
            using (var activity = ModelDiagnostics.StartCompletionActivity(this._endpoint, this._modelId, ModelProvider, chatHistory, OllamaExecutionSettings))
            {
                try
                {
                    using var httpRequestMessage = this.CreatePost(chatRequest, endpoint, this._apiKey, stream: false);
                    responseData = await this.SendRequestAsync<OllamaChatCompletionChunk>(httpRequestMessage, cancellationToken).ConfigureAwait(false);
                    this.LogUsage(responseData?.Usage);
                    if (responseData is null || responseData.Message is null)
                    {
                        throw new KernelException("Chat completions not found");
                    }
                }
                catch (Exception ex) when (activity is not null)
                {
                    activity.SetError(ex);

                    // Capture available metadata even if the operation failed.
                    if (responseData is not null)
                    {
                        if (responseData.CreatedAt is string id)
                        {
                            activity.SetResponseId(id);
                        }

                        if (responseData.Usage is OllamaUsage usage)
                        {
                            if (usage.PromptTokens is int promptTokens)
                            {
                                activity.SetPromptTokenUsage(promptTokens);
                            }
                            if (usage.CompletionTokens is int completionTokens)
                            {
                                activity.SetCompletionTokenUsage(completionTokens);
                            }
                        }
                    }

                    throw;
                }

                responseContent = this.ToChatMessageContent(modelId, responseData);
                activity?.SetCompletionResponse(responseContent, responseData.Usage?.PromptTokens, responseData.Usage?.CompletionTokens);
            }

            // If we don't want to attempt to invoke any functions, just return the result.
            // Or if we are auto-invoking but we somehow end up with other than 1 choice even though only 1 was requested, similarly bail.
            if (!autoInvoke)// || responseData.Choices.Count != 1)
            {
                return responseContent;
            }

            // Get our single result and extract the function call information. If this isn't a function call, or if it is
            // but we're unable to find the function or extract the relevant information, just return the single result.
            // Note that we don't check the FinishReason and instead check whether there are any tool calls, as the service
            // may return a FinishReason of "stop" even if there are tool calls to be made, in particular if a required tool
            // is specified.
            var chatChoice = responseData.Message; // TODO Handle multiple choices
            if (!chatChoice.IsToolCall)
            {
                return responseContent;
            }

            if (this._logger.IsEnabled(LogLevel.Debug))
            {
                this._logger.LogDebug("Tool requests: {Requests}", chatChoice.ToolCallCount);
            }
            if (this._logger.IsEnabled(LogLevel.Trace))
            {
                this._logger.LogTrace("Function call requests: {Requests}", string.Join(", ", chatChoice.ToolCalls!.Select(tc => $"{tc.Function?.Name}({tc.Function?.Parameters})")));
            }

            Debug.Assert(kernel is not null);

            // Add the original assistant message to the chatRequest; this is required for the service
            // to understand the tool call responses. Also add the result message to the caller's chat
            // history: if they don't want it, they can remove it, but this makes the data available,
            // including metadata like usage.
            chatRequest.AddMessage(chatChoice);

            var chatMessageContent = this.ToChatMessageContent(modelId, responseData, chatChoice);
            chatHistory.Add(chatMessageContent);

            // We must send back a response for every tool call, regardless of whether we successfully executed it or not.
            // If we successfully execute it, we'll add the result. If we don't, we'll add an error.
            for (int toolCallIndex = 0; toolCallIndex < chatChoice.ToolCallCount; toolCallIndex++)
            {
                var toolCall = chatChoice.ToolCalls![toolCallIndex];

                // We currently only know about function tool calls. If it's anything else, we'll respond with an error.
                if (toolCall.Function is null)
                {
                    this.AddResponseMessage(chatRequest, chatHistory, toolCall, result: null, "Error: Tool call was not a function call.");
                    continue;
                }

                // Make sure the requested function is one we requested. If we're permitting any kernel function to be invoked,
                // then we don't need to check this, as it'll be handled when we look up the function in the kernel to be able
                // to invoke it. If we're permitting only a specific list of functions, though, then we need to explicitly check.
                if (OllamaExecutionSettings.ToolCallBehavior?.AllowAnyRequestedKernelFunction is not true &&
                    !IsRequestableTool(chatRequest, toolCall.Function!))
                {
                    this.AddResponseMessage(chatRequest, chatHistory, toolCall, result: null, "Error: Function call chatRequest for a function that wasn't defined.");
                    continue;
                }

                // Find the function in the kernel and populate the arguments.
                if (!kernel!.Plugins.TryGetFunctionAndArguments(toolCall.Function, out KernelFunction? function, out KernelArguments? functionArgs))
                {
                    this.AddResponseMessage(chatRequest, chatHistory, toolCall, result: null, "Error: Requested function could not be found.");
                    continue;
                }

                // Now, invoke the function, and add the resulting tool call message to the chat options.
                FunctionResult functionResult = new(function) { Culture = kernel.Culture };
                AutoFunctionInvocationContext invocationContext = new(kernel, function, functionResult, chatHistory, chatMessageContent)
                {
                    ToolCallId = toolCall.Id,
                    Arguments = functionArgs,
                    RequestSequenceIndex = requestIndex - 1,
                    FunctionSequenceIndex = toolCallIndex,
                    FunctionCount = chatChoice.ToolCalls.Count,
                    CancellationToken = cancellationToken
                };
                s_inflightAutoInvokes.Value++;
                try
                {
                    invocationContext = await OnAutoFunctionInvocationAsync(kernel, invocationContext, async (context) =>
                    {
                        // Check if filter requested termination.
                        if (context.Terminate)
                        {
                            return;
                        }

                        // Note that we explicitly do not use executionSettings here; those pertain to the all-up operation and not necessarily to any
                        // further calls made as part of this function invocation. In particular, we must not use function calling settings naively here,
                        // as the called function could in turn telling the model about itself as a possible candidate for invocation.
                        context.Result = await function.InvokeAsync(kernel, invocationContext.Arguments, cancellationToken: cancellationToken).ConfigureAwait(false);
                    }).ConfigureAwait(false);
                }
#pragma warning disable CA1031 // Do not catch general exception types
                catch (Exception e)
#pragma warning restore CA1031
                {
                    this.AddResponseMessage(chatRequest, chatHistory, toolCall, result: null, $"Error: Exception while invoking function. {e.Message}");
                    continue;
                }
                finally
                {
                    s_inflightAutoInvokes.Value--;
                }

                // Apply any changes from the auto function invocation filters context to final result.
                functionResult = invocationContext.Result;

                object functionResultValue = functionResult.GetValue<object>() ?? string.Empty;
                var stringResult = ProcessFunctionResult(functionResultValue, OllamaExecutionSettings.ToolCallBehavior);

                this.AddResponseMessage(chatRequest, chatHistory, toolCall, result: stringResult, errorMessage: null);

                // If filter requested termination, returning latest function result.
                if (invocationContext.Terminate)
                {
                    if (this._logger.IsEnabled(LogLevel.Debug))
                    {
                        this._logger.LogDebug("Filter requested termination of automatic function invocation.");
                    }

                    return [chatHistory.Last()];
                }
            }

            // Update tool use information for the next go-around based on having completed another requestIndex.
            Debug.Assert(OllamaExecutionSettings.ToolCallBehavior is not null);

            // Set the tool choice to none. If we end up wanting to use tools, we'll reset it to the desired value.
            chatRequest.ToolChoice = "none";
            chatRequest.Tools?.Clear();

            if (requestIndex >= OllamaExecutionSettings.ToolCallBehavior!.MaximumUseAttempts)
            {
                // Don't add any tools as we've reached the maximum attempts limit.
                if (this._logger.IsEnabled(LogLevel.Debug))
                {
                    this._logger.LogDebug("Maximum use ({MaximumUse}) reached; removing the tool.", OllamaExecutionSettings.ToolCallBehavior!.MaximumUseAttempts);
                }
            }
            else
            {
                // Regenerate the tool list as necessary. The invocation of the function(s) could have augmented
                // what functions are available in the kernel.
                OllamaExecutionSettings.ToolCallBehavior.ConfigureRequest(kernel, chatRequest);
            }

            // Disable auto invocation if we've exceeded the allowed limit.
            if (requestIndex >= OllamaExecutionSettings.ToolCallBehavior!.MaximumAutoInvokeAttempts)
            {
                autoInvoke = false;
                if (this._logger.IsEnabled(LogLevel.Debug))
                {
                    this._logger.LogDebug("Maximum auto-invoke ({MaximumAutoInvoke}) reached.", OllamaExecutionSettings.ToolCallBehavior!.MaximumAutoInvokeAttempts);
                }
            }
        }
    }

    internal async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(ChatHistory chatHistory, [EnumeratorCancellation] CancellationToken cancellationToken, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null)
    {
        this.ValidateChatHistory(chatHistory);

        var OllamaExecutionSettings = OllamaAIPromptExecutionSettings.FromExecutionSettings(executionSettings);
        string modelId = OllamaExecutionSettings.ModelId ?? this._modelId;
        var chatRequest = this.CreateChatCompletionRequest(modelId, stream: true, chatHistory, OllamaExecutionSettings, kernel);
        var autoInvoke = kernel is not null && OllamaExecutionSettings.ToolCallBehavior?.MaximumAutoInvokeAttempts > 0 && s_inflightAutoInvokes.Value < MaxInflightAutoInvokes;

        List<OllamaToolCall>? toolCalls = null;
        for (int requestIndex = 1; ; requestIndex++)
        {
            // Reset state
            toolCalls?.Clear();

            // Stream the responses
            using (var activity = ModelDiagnostics.StartCompletionActivity(this._endpoint, this._modelId, ModelProvider, chatHistory, OllamaExecutionSettings))
            {
                // Make the request.
                IAsyncEnumerable<StreamingChatMessageContent> response;
                try
                {
                    response = this.StreamChatMessageContentsAsync(chatHistory, OllamaExecutionSettings, chatRequest, modelId, cancellationToken);
                }
                catch (Exception e) when (activity is not null)
                {
                    activity.SetError(e);
                    throw;
                }

                var responseEnumerator = response.ConfigureAwait(false).GetAsyncEnumerator();
                List<StreamingKernelContent>? streamedContents = activity is not null ? [] : null;
                string? streamedRole = null;
                try
                {
                    while (true)
                    {
                        try
                        {
                            if (!await responseEnumerator.MoveNextAsync())
                            {
                                break;
                            }
                        }
                        catch (Exception ex) when (activity is not null)
                        {
                            activity.SetError(ex);
                            throw;
                        }

                        StreamingChatMessageContent update = responseEnumerator.Current;

                        // If we're intending to invoke function calls, we need to consume that function call information.
                        if (autoInvoke)
                        {
                            if (update.InnerContent is not OllamaChatCompletionChunk completionChunk || completionChunk.Message is null)
                            {
                                continue;
                            }

                            OllamaChatMessage chatChoice = completionChunk.Message; // TODO Handle multiple choices
                            streamedRole ??= chatChoice!.Role;
                            if (chatChoice.ToolCallCount > 0)
                            {
                                // Create a copy of the tool calls to avoid modifying the original list
                                toolCalls = new List<OllamaToolCall>(chatChoice.ToolCalls!);

                                // Add the original assistant message to the chatRequest; this is required for the service
                                // to understand the tool call responses. Also add the result message to the caller's chat
                                // history: if they don't want it, they can remove it, but this makes the data available,
                                // including metadata like usage.
                                chatRequest.AddMessage(new OllamaChatMessage(streamedRole, completionChunk.GetContent()) { ToolCalls = chatChoice.ToolCalls });
                                chatHistory.Add(this.ToChatMessageContent(modelId, streamedRole!, completionChunk, chatChoice));
                            }
                        }

                        streamedContents?.Add(update);
                        yield return update;
                    }
                }
                finally
                {
                    activity?.EndStreaming(streamedContents);
                    await responseEnumerator.DisposeAsync();
                }
            }

            // If we don't have a function to invoke, we're done.
            // Note that we don't check the FinishReason and instead check whether there are any tool calls, as the service
            // may return a FinishReason of "stop" even if there are tool calls to be made, in particular if a required tool
            // is specified.
            if (!autoInvoke ||
                toolCalls is not { Count: > 0 })
            {
                yield break;
            }

            // Log the requests
            if (this._logger.IsEnabled(LogLevel.Trace))
            {
                this._logger.LogTrace("Function call requests: {Requests}", string.Join(", ", toolCalls.Select(mtc => $"{mtc.Function?.Name}({mtc.Function?.Parameters})")));
            }
            else if (this._logger.IsEnabled(LogLevel.Debug))
            {
                this._logger.LogDebug("Function call requests: {Requests}", toolCalls.Count);
            }

            // We must send back a response for every tool call, regardless of whether we successfully executed it or not.
            // If we successfully execute it, we'll add the result. If we don't, we'll add an error.
            // TODO Check are we missing code here?

            for (int toolCallIndex = 0; toolCallIndex < toolCalls.Count; toolCallIndex++)
            {
                var toolCall = toolCalls[toolCallIndex];

                // We currently only know about function tool calls. If it's anything else, we'll respond with an error.
                if (toolCall.Function is null)
                {
                    this.AddResponseMessage(chatRequest, chatHistory, toolCall, result: null, "Error: Tool call was not a function call.");
                    continue;
                }

                // Make sure the requested function is one we requested. If we're permitting any kernel function to be invoked,
                // then we don't need to check this, as it'll be handled when we look up the function in the kernel to be able
                // to invoke it. If we're permitting only a specific list of functions, though, then we need to explicitly check.
                if (OllamaExecutionSettings.ToolCallBehavior?.AllowAnyRequestedKernelFunction is not true &&
                    !IsRequestableTool(chatRequest, toolCall.Function!))
                {
                    this.AddResponseMessage(chatRequest, chatHistory, toolCall, result: null, "Error: Function call chatRequest for a function that wasn't defined.");
                    continue;
                }

                // Find the function in the kernel and populate the arguments.
                if (!kernel!.Plugins.TryGetFunctionAndArguments(toolCall.Function, out KernelFunction? function, out KernelArguments? functionArgs))
                {
                    this.AddResponseMessage(chatRequest, chatHistory, toolCall, result: null, "Error: Requested function could not be found.");
                    continue;
                }

                // Now, invoke the function, and add the resulting tool call message to the chat options.
                FunctionResult functionResult = new(function) { Culture = kernel.Culture };
                AutoFunctionInvocationContext invocationContext = new(kernel, function, functionResult, chatHistory, chatHistory.Last())
                {
                    ToolCallId = toolCall.Id,
                    Arguments = functionArgs,
                    RequestSequenceIndex = requestIndex - 1,
                    FunctionSequenceIndex = toolCallIndex,
                    FunctionCount = toolCalls.Count,
                    CancellationToken = cancellationToken
                };
                s_inflightAutoInvokes.Value++;
                try
                {
                    invocationContext = await OnAutoFunctionInvocationAsync(kernel, invocationContext, async (context) =>
                    {
                        // Check if filter requested termination.
                        if (context.Terminate)
                        {
                            return;
                        }

                        // Note that we explicitly do not use executionSettings here; those pertain to the all-up operation and not necessarily to any
                        // further calls made as part of this function invocation. In particular, we must not use function calling settings naively here,
                        // as the called function could in turn telling the model about itself as a possible candidate for invocation.
                        context.Result = await function.InvokeAsync(kernel, invocationContext.Arguments, cancellationToken: cancellationToken).ConfigureAwait(false);
                    }).ConfigureAwait(false);
                }
#pragma warning disable CA1031 // Do not catch general exception types
                catch (Exception e)
#pragma warning restore CA1031
                {
                    this.AddResponseMessage(chatRequest, chatHistory, toolCall, result: null, $"Error: Exception while invoking function. {e.Message}");
                    continue;
                }
                finally
                {
                    s_inflightAutoInvokes.Value--;
                }

                // Apply any changes from the auto function invocation filters context to final result.
                functionResult = invocationContext.Result;

                object functionResultValue = functionResult.GetValue<object>() ?? string.Empty;
                var stringResult = ProcessFunctionResult(functionResultValue, OllamaExecutionSettings.ToolCallBehavior);

                this.AddResponseMessage(chatRequest, chatHistory, toolCall, result: stringResult, errorMessage: null);

                // If filter requested termination, returning latest function result and breaking request iteration loop.
                if (invocationContext.Terminate)
                {
                    if (this._logger.IsEnabled(LogLevel.Debug))
                    {
                        this._logger.LogDebug("Filter requested termination of automatic function invocation.");
                    }

                    var lastChatMessage = chatHistory.Last();

                    yield return new StreamingChatMessageContent(lastChatMessage.Role, lastChatMessage.Content);
                    yield break;
                }
            }

            // Update tool use information for the next go-around based on having completed another requestIndex.
            Debug.Assert(OllamaExecutionSettings.ToolCallBehavior is not null);

            // Set the tool choice to none. If we end up wanting to use tools, we'll reset it to the desired value.
            chatRequest.ToolChoice = "none";
            chatRequest.Tools?.Clear();

            if (requestIndex >= OllamaExecutionSettings.ToolCallBehavior!.MaximumUseAttempts)
            {
                // Don't add any tools as we've reached the maximum attempts limit.
                if (this._logger.IsEnabled(LogLevel.Debug))
                {
                    this._logger.LogDebug("Maximum use ({MaximumUse}) reached; removing the tool.", OllamaExecutionSettings.ToolCallBehavior!.MaximumUseAttempts);
                }
            }
            else
            {
                // Regenerate the tool list as necessary. The invocation of the function(s) could have augmented
                // what functions are available in the kernel.
                OllamaExecutionSettings.ToolCallBehavior.ConfigureRequest(kernel, chatRequest);
            }

            // Disable auto invocation if we've exceeded the allowed limit.
            if (requestIndex >= OllamaExecutionSettings.ToolCallBehavior!.MaximumAutoInvokeAttempts)
            {
                autoInvoke = false;
                if (this._logger.IsEnabled(LogLevel.Debug))
                {
                    this._logger.LogDebug("Maximum auto-invoke ({MaximumAutoInvoke}) reached.", OllamaExecutionSettings.ToolCallBehavior!.MaximumAutoInvokeAttempts);
                }
            }
        }
    }

    private async IAsyncEnumerable<StreamingChatMessageContent> StreamChatMessageContentsAsync(ChatHistory chatHistory, OllamaAIPromptExecutionSettings executionSettings, ChatCompletionRequest chatRequest, string modelId, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        this.ValidateChatHistory(chatHistory);

        var endpoint = this.GetEndpoint(executionSettings, path: "api/chat");
        using var httpRequestMessage = this.CreatePost(chatRequest, endpoint, this._apiKey, stream: true);
        using var response = await this.SendStreamingRequestAsync(httpRequestMessage, cancellationToken).ConfigureAwait(false);
        using var responseStream = await response.Content.ReadAsStreamAndTranslateExceptionAsync().ConfigureAwait(false);
        await foreach (var streamingChatContent in this.ProcessChatResponseStreamAsync(responseStream, modelId, cancellationToken).ConfigureAwait(false))
        {
            yield return streamingChatContent;
        }
    }

    private async IAsyncEnumerable<StreamingChatMessageContent> ProcessChatResponseStreamAsync(Stream stream, string modelId, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        IAsyncEnumerator<StreamTrueModel>? responseEnumerator = null;

        try
        {
            var responseEnumerable = this.ParseChatResponseStreamAsync(stream, cancellationToken);
            responseEnumerator = responseEnumerable.GetAsyncEnumerator(cancellationToken);

            string? currentRole = null;

            List<StreamTrueModel> toolMessages = new List<StreamTrueModel>();

            while (await responseEnumerator.MoveNextAsync().ConfigureAwait(false))
            {
                var chunk = responseEnumerator.Current!;

                if(chunk.Message.Content == "[TOOL_CALLS]")
                {
                    toolMessages.Add(chunk);
                    continue;
                }
                else if(toolMessages.Count > 0)
                {
                    toolMessages.Add(chunk);
                    continue;
                }

                var OllamaChunk = ConvertStreamTrueToFalse(new List<StreamTrueModel>() { chunk });
                currentRole ??= OllamaChunk.GetRole();

                yield return new(role: new AuthorRole(currentRole ?? "assistant"),
                    content: OllamaChunk.GetContent(),
                    choiceIndex: 0,
                    modelId: modelId,
                    encoding: OllamaChunk.GetEncoding(),
                    innerContent: OllamaChunk,
                    metadata: OllamaChunk.GetMetadata());
            }

            if(toolMessages.Count > 0)
            {
                var OllamaChunk = ConvertStreamTrueToFalse(toolMessages);
                yield return new(role: AuthorRole.Tool,
                  content: OllamaChunk.GetContent(),
                  choiceIndex: 0,
                  modelId: modelId,
                  encoding: OllamaChunk.GetEncoding(),
                  innerContent: OllamaChunk,
                  metadata: OllamaChunk.GetMetadata());
            }
        }
        finally
        {
            if (responseEnumerator != null)
            {
                await responseEnumerator.DisposeAsync().ConfigureAwait(false);
            }
        }
    }

    public static OllamaChatCompletionChunk ConvertStreamTrueToFalse(List<StreamTrueModel> streamTrue)
    {
        //Console.WriteLine(JsonSerializer.Serialize(streamTrue));
        // Объединение контента
        var combinedContent = string.Concat(
            streamTrue
                .Select(msg => msg.Message.Content)
        );

        // Удаление лишних пробелов и прочих символов
        List<OllamaToolCall> tools = null;

        if (combinedContent.Contains("[TOOL_CALLS]"))
        {
            combinedContent = combinedContent.Trim().Replace("[TOOL_CALLS]", "");

            // Регулярное выражение для замены arguments
            string pattern = @"(""arguments"":\s*)\{([^}]*)\}";

            combinedContent = Regex.Replace(combinedContent, pattern, match =>
            {
                string argumentsContent = match.Groups[2].Value;
                string escapedArgumentsContent = argumentsContent.Replace("\"", "\\\"");
                return match.Groups[1].Value + "\"{" + escapedArgumentsContent + "}\"";
            }, RegexOptions.Singleline);

            Console.WriteLine(combinedContent);
            var OllamaFunctions = JsonSerializer.Deserialize<List<OllamaFunction>>(combinedContent);

            // Обработка deserialized данных, если нужно, например:
            var toolCalls = OllamaFunctions.Select(f => new OllamaToolCall
            {
                Id = Guid.NewGuid().ToString(),
                Function = f
            }).ToList();
            tools = toolCalls;
        }

        // Создание структуры stream = false
        try
        {
            var streamFalse = new OllamaChatCompletionChunk
            {
                Model = streamTrue[0].Model,
                CreatedAt = streamTrue[0].CreatedAt.ToString(),
                Message = new OllamaChatMessage(streamTrue[0].Message.Role, combinedContent) //tools == null ? : ""
                {
                    ToolCalls = tools,
                },
                DoneReason = "stop",
                Done = true,
                TotalDuration = streamTrue.Sum(msg => msg.TotalDuration ?? 0),
                LoadDuration = streamTrue.Sum(msg => msg.LoadDuration ?? 0),
                PromptEvalCount = streamTrue.Sum(msg => msg.PromptEvalCount ?? 0),
                PromptEvalDuration = streamTrue.Sum(msg => msg.PromptEvalDuration ?? 0),
                EvalCount = streamTrue.Sum(msg => msg.EvalCount ?? 0),
                EvalDuration = streamTrue.Sum(msg => msg.EvalDuration ?? 0)
            };

            return streamFalse;

        }
        catch (Exception ex)
        {
            Console.WriteLine(JsonSerializer.Serialize(ex));
        }

        return null;

    }


    private async IAsyncEnumerable<StreamTrueModel> ParseChatResponseStreamAsync(Stream responseStream, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await foreach (var json in this._streamJsonParser.ParseAsync(responseStream, cancellationToken: cancellationToken).ConfigureAwait(false))
        {
            yield return DeserializeResponse<StreamTrueModel>(json);
        }
    }

    internal async Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(IList<string> data, CancellationToken cancellationToken, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null)
    {
        var request = new TextEmbeddingRequest(this._modelId, data);
        var OllamaExecutionSettings = OllamaAIPromptExecutionSettings.FromExecutionSettings(executionSettings);
        var endpoint = this.GetEndpoint(OllamaExecutionSettings, path: "embeddings");
        using var httpRequestMessage = this.CreatePost(request, endpoint, this._apiKey, false);

        var response = await this.SendRequestAsync<TextEmbeddingResponse>(httpRequestMessage, cancellationToken).ConfigureAwait(false);

        return response.Data!.Select(item => new ReadOnlyMemory<float>([.. item.Embedding])).ToList();
    }

    #region private
    private readonly string _modelId;
    private readonly string _apiKey;
    private readonly Uri? _endpoint;
    private readonly HttpClient _httpClient;
    private readonly ILogger _logger;
    private readonly StreamJsonParser _streamJsonParser;

    /// <summary>Provider name used for diagnostics.</summary>
    private const string ModelProvider = "Ollamaai";

    /// <summary>
    /// The maximum number of auto-invokes that can be in-flight at any given time as part of the current
    /// asynchronous chain of execution.
    /// </summary>
    /// <remarks>
    /// This is a fail-safe mechanism. If someone accidentally manages to set up execution settings in such a way that
    /// auto-invocation is invoked recursively, and in particular where a prompt function is able to auto-invoke itself,
    /// we could end up in an infinite loop. This const is a backstop against that happening. We should never come close
    /// to this limit, but if we do, auto-invoke will be disabled for the current flow in order to prevent runaway execution.
    /// With the current setup, the way this could possibly happen is if a prompt function is configured with built-in
    /// execution settings that opt-in to auto-invocation of everything in the kernel, in which case the invocation of that
    /// prompt function could advertise itself as a candidate for auto-invocation. We don't want to outright block that,
    /// if that's something a developer has asked to do (e.g. it might be invoked with different arguments than its parent
    /// was invoked with), but we do want to limit it. This limit is arbitrary and can be tweaked in the future and/or made
    /// configurable should need arise.
    /// </remarks>
    private const int MaxInflightAutoInvokes = 5;

    /// <summary>Tracking <see cref="AsyncLocal{Int32}"/> for <see cref="MaxInflightAutoInvokes"/>.</summary>
    private static readonly AsyncLocal<int> s_inflightAutoInvokes = new();

    private static readonly string s_namespace = typeof(OllamaAIChatCompletionService).Namespace!;

    /// <summary>
    /// Instance of <see cref="Meter"/> for metrics.
    /// </summary>
    private static readonly Meter s_meter = new(s_namespace);

    /// <summary>
    /// Instance of <see cref="Counter{T}"/> to keep track of the number of prompt tokens used.
    /// </summary>
    private static readonly Counter<int> s_promptTokensCounter =
        s_meter.CreateCounter<int>(
            name: $"{s_namespace}.tokens.prompt",
            unit: "{token}",
            description: "Number of prompt tokens used");

    /// <summary>
    /// Instance of <see cref="Counter{T}"/> to keep track of the number of completion tokens used.
    /// </summary>
    private static readonly Counter<int> s_completionTokensCounter =
        s_meter.CreateCounter<int>(
            name: $"{s_namespace}.tokens.completion",
            unit: "{token}",
            description: "Number of completion tokens used");

    /// <summary>
    /// Instance of <see cref="Counter{T}"/> to keep track of the total number of tokens used.
    /// </summary>
    private static readonly Counter<int> s_totalTokensCounter =
        s_meter.CreateCounter<int>(
            name: $"{s_namespace}.tokens.total",
            unit: "{token}",
            description: "Number of tokens used");

    /// <summary>Log token usage to the logger and metrics.</summary>
    private void LogUsage(OllamaUsage? usage)
    {
        if (usage is null || usage.PromptTokens is null || usage.CompletionTokens is null || usage.TotalTokens is null)
        {
            this._logger.LogDebug("Usage information unavailable.");
            return;
        }

        if (this._logger.IsEnabled(LogLevel.Information))
        {
            this._logger.LogInformation(
                "Prompt tokens: {PromptTokens}. Completion tokens: {CompletionTokens}. Total tokens: {TotalTokens}.",
                usage.PromptTokens,
                usage.CompletionTokens,
                usage.TotalTokens);
        }

        s_promptTokensCounter.Add(usage.PromptTokens.Value);
        s_completionTokensCounter.Add(usage.CompletionTokens.Value);
        s_totalTokensCounter.Add(usage.TotalTokens.Value);
    }

    /// <summary>
    /// Messages are required and the first prompt role should be user or system.
    /// </summary>
    private void ValidateChatHistory(ChatHistory chatHistory)
    {
        Verify.NotNull(chatHistory);

        if (chatHistory.Count == 0)
        {
            throw new ArgumentException("Chat history must contain at least one message", nameof(chatHistory));
        }
        var firstRole = chatHistory[0].Role.ToString();
        if (firstRole is not "system" and not "user")
        {
            throw new ArgumentException("The first message in chat history must have either the system or user role", nameof(chatHistory));
        }
    }

    private ChatCompletionRequest CreateChatCompletionRequest(string modelId, bool stream, ChatHistory chatHistory, OllamaAIPromptExecutionSettings executionSettings, Kernel? kernel = null)
    {
        if (this._logger.IsEnabled(LogLevel.Trace))
        {
            this._logger.LogTrace("ChatHistory: {ChatHistory}, Settings: {Settings}",
                JsonSerializer.Serialize(chatHistory),
                JsonSerializer.Serialize(executionSettings));
        }

        var request = new ChatCompletionRequest(modelId)
        {
            Stream = stream,
            Messages = chatHistory.SelectMany(chatMessage => this.ToOllamaChatMessages(chatMessage, executionSettings?.ToolCallBehavior)).ToList(),
            Temperature = executionSettings.Temperature,
            TopP = executionSettings.TopP,
            MaxTokens = executionSettings.MaxTokens,
            SafePrompt = executionSettings.SafePrompt,
            RandomSeed = executionSettings.RandomSeed
        };

        executionSettings.ToolCallBehavior?.ConfigureRequest(kernel, request);

        return request;
    }

    internal List<OllamaChatMessage> ToOllamaChatMessages(ChatMessageContent content, OllamaAIToolCallBehavior? toolCallBehavior)
    {
        if (content.Role == AuthorRole.Assistant)
        {
            // Handling function calls supplied via ChatMessageContent.Items collection elements of the FunctionCallContent type.
            var message = new OllamaChatMessage(content.Role.ToString(), content.Content ?? string.Empty);
            Dictionary<string, OllamaToolCall> toolCalls = [];
            foreach (var item in content.Items)
            {
                if (item is not FunctionCallContent callRequest)
                {
                    continue;
                }

                if (callRequest.Id is null || toolCalls.ContainsKey(callRequest.Id))
                {
                    continue;
                }

                var arguments = JsonSerializer.Serialize(callRequest.Arguments);
                var toolCall = new OllamaToolCall()
                {
                    Id = callRequest.Id,
                    Function = new OllamaFunction(
                        callRequest.FunctionName,
                        callRequest.PluginName)
                    {
                        Arguments = arguments
                    }
                };
                toolCalls.Add(callRequest.Id, toolCall);
            }
            if (toolCalls.Count > 0)
            {
                message.ToolCalls = [.. toolCalls.Values];
            }
            return [message];
        }

        if (content.Role == AuthorRole.Tool)
        {
            List<OllamaChatMessage>? messages = null;
            foreach (var item in content.Items)
            {
                if (item is not FunctionResultContent resultContent)
                {
                    continue;
                }

                messages ??= [];

                var stringResult = ProcessFunctionResult(resultContent.Result ?? string.Empty, toolCallBehavior);
                messages.Add(new OllamaChatMessage(content.Role.ToString(), stringResult));
            }
            if (messages is not null)
            {
                return messages;
            }

            throw new NotSupportedException("No function result provided in the tool message.");
        }

        return [new OllamaChatMessage(content.Role.ToString(), content.Content ?? string.Empty)];
    }

    private HttpRequestMessage CreatePost(object requestData, Uri endpoint, string apiKey, bool stream)
    {
        var httpRequestMessage = HttpRequest.CreatePostRequest(endpoint, requestData);
        this.SetRequestHeaders(httpRequestMessage, apiKey, stream);

        return httpRequestMessage;
    }

    private void SetRequestHeaders(HttpRequestMessage request, string apiKey, bool stream)
    {
        request.Headers.Add("User-Agent", HttpHeaderConstant.Values.UserAgent);
        request.Headers.Add(HttpHeaderConstant.Names.SemanticKernelVersion, HttpHeaderConstant.Values.GetAssemblyVersion(this.GetType()));
        request.Headers.Add("Accept", stream ? "text/event-stream" : "application/json");
        request.Headers.Add("Authorization", $"Bearer {apiKey}");
        request.Content!.Headers.ContentType = new MediaTypeHeaderValue("application/json");
    }

    private async Task<T> SendRequestAsync<T>(HttpRequestMessage httpRequestMessage, CancellationToken cancellationToken)
    {
        using var response = await this._httpClient.SendWithSuccessCheckAsync(httpRequestMessage, cancellationToken).ConfigureAwait(false);

        var body = await response.Content.ReadAsStringWithExceptionMappingAsync().ConfigureAwait(false);

        return DeserializeResponse<T>(body);
    }

    private async Task<HttpResponseMessage> SendStreamingRequestAsync(HttpRequestMessage httpRequestMessage, CancellationToken cancellationToken)
    {
        return await this._httpClient.SendWithSuccessCheckAsync(httpRequestMessage, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
    }

    private Uri GetEndpoint(OllamaAIPromptExecutionSettings executionSettings, string path)
    {
        var endpoint = this._endpoint ?? new Uri($"https://api.Ollama.ai/{executionSettings.ApiVersion}");
        var separator = endpoint.AbsolutePath.EndsWith("/", StringComparison.InvariantCulture) ? string.Empty : "/";
        return new Uri($"{endpoint}{separator}{path}");
    }

    /// <summary>Checks if a tool call is for a function that was defined.</summary>
    private static bool IsRequestableTool(ChatCompletionRequest request, OllamaFunction func)
    {
        var tools = request.Tools;
        for (int i = 0; i < tools?.Count; i++)
        {
            if (string.Equals(tools[i].Function.Name, func.Name, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static T DeserializeResponse<T>(string body)
    {
        try
        {
            T? deserializedResponse = JsonSerializer.Deserialize<T>(body);
            return deserializedResponse ?? throw new JsonException("Response is null");
        }
        catch (JsonException exc)
        {
            throw new KernelException("Unexpected response from model", exc)
            {
                Data = { { "ResponseData", body } },
            };
        }
    }


    private List<ChatMessageContent> ToChatMessageContent(string modelId, OllamaChatCompletionChunk response)
    {
        // ToDo убрать
        return new List<ChatMessageContent>() { this.ToChatMessageContent(modelId, response, response.Message) };
    }

    private ChatMessageContent ToChatMessageContent(string modelId, OllamaChatCompletionChunk response, OllamaChatMessage chatChoice)
    {
        var message = new ChatMessageContent(new AuthorRole(chatChoice!.Role!), response!.GetContent(), modelId, chatChoice, Encoding.UTF8, GetChatChoiceMetadata(response, chatChoice));

        if (chatChoice.ToolCalls.Count > 0)
        {
            foreach (var toolCall in chatChoice.ToolCalls!)
            {
                this.AddFunctionCallContent(message, toolCall);
            }
        }

        return message;
    }

    private ChatMessageContent ToChatMessageContent(string modelId, string streamedRole, OllamaChatCompletionChunk chunk, OllamaChatMessage chatChoice)
    {
        var message = new ChatMessageContent(new AuthorRole(chatChoice!.Role!), chunk!.GetContent(), modelId, chatChoice, Encoding.UTF8, GetChatChoiceMetadata(chunk, chatChoice));

        if (chatChoice.IsToolCall)
        {
            foreach (var toolCall in chatChoice.ToolCalls!)
            {
                this.AddFunctionCallContent(message, toolCall);
            }
        }

        return message;
    }

    private void AddFunctionCallContent(ChatMessageContent message, OllamaToolCall toolCall)
    {
        if (toolCall.Function is null)
        {
            return;
        }

        // Adding items of 'FunctionCallContent' type to the 'Items' collection even though the function calls are available via the 'ToolCalls' property.
        // This allows consumers to work with functions in an LLM-agnostic way.
        Exception? exception = null;
        KernelArguments? arguments = null;
        if (toolCall.Function.Arguments is not null)
        {
            try
            {
                arguments = JsonSerializer.Deserialize<KernelArguments>(toolCall.Function.Arguments);
                if (arguments is not null)
                {
                    // Iterate over copy of the names to avoid mutating the dictionary while enumerating it
                    var names = arguments.Names.ToArray();
                    foreach (var name in names)
                    {
                        arguments[name] = arguments[name]?.ToString();
                    }
                }
            }
            catch (JsonException ex)
            {
                exception = new KernelException("Error: Function call arguments were invalid JSON.", ex);

                if (this._logger.IsEnabled(LogLevel.Debug))
                {
                    this._logger.LogDebug(ex, "Failed to deserialize function arguments ({FunctionName}/{FunctionId}).", toolCall.Function.Name, toolCall.Id);
                }
            }
        }

        var functionCallContent = new FunctionCallContent(
            functionName: toolCall.Function.FunctionName,
            pluginName: toolCall.Function.PluginName,
            id: toolCall.Id,
            arguments: arguments)
        {
            InnerContent = toolCall,
            Exception = exception
        };

        message.Items.Add(functionCallContent);
    }

    private void AddResponseMessage(ChatCompletionRequest chatRequest, ChatHistory chat, OllamaToolCall toolCall, string? result, string? errorMessage)
    {
        // Log any error
        if (errorMessage is not null && this._logger.IsEnabled(LogLevel.Debug))
        {
            Debug.Assert(result is null);
            this._logger.LogDebug("Failed to handle tool request ({ToolId}). {Error}", toolCall.Function?.Name, errorMessage);
        }

        // Add the tool response message to both the chat options
        result ??= errorMessage ?? string.Empty;
        chatRequest.AddMessage(new OllamaChatMessage(AuthorRole.Tool.ToString(), result));

        // Add the tool response message to the chat history
        var message = new ChatMessageContent(AuthorRole.Tool, result, metadata: new Dictionary<string, object?> { { nameof(OllamaToolCall.Function), toolCall.Function } });

        // Add an item of type FunctionResultContent to the ChatMessageContent.Items collection in addition to the function result stored as a string in the ChatMessageContent.Content property.  
        // This will enable migration to the new function calling model and facilitate the deprecation of the current one in the future.
        if (toolCall.Function is not null)
        {
            message.Items.Add(new FunctionResultContent(
                toolCall.Function.FunctionName,
                toolCall.Function.PluginName,
                toolCall.Id,
                result));
        }

        chat.Add(message);
    }

    private static Dictionary<string, object?> GetChatChoiceMetadata(OllamaChatCompletionChunk completionResponse, OllamaChatMessage chatChoice)
    {
        return new Dictionary<string, object?>(6)
        {
            //{ nameof(completionResponse), completionResponse },
            { nameof(completionResponse.Model), completionResponse.Model },
            { nameof(completionResponse.Usage), completionResponse.Usage },
            { nameof(completionResponse.CreatedAt), completionResponse.CreatedAt },
            //{ nameof(chatChoice.Index), chatChoice.Index },
            //{ nameof(chatChoice.FinishReason), chatChoice.FinishReason },
        };
    }

    private static Dictionary<string, object?> GetChatChoiceMetadata(OllamaChatCompletionChunk completionChunk, OllamaChatCompletionChoice chatChoice)
    {
        return new Dictionary<string, object?>(7)
        {
        //    { nameof(completionChunk.Id), completionChunk.Id },
        //    { nameof(completionChunk.Object), completionChunk.Object },
            { nameof(completionChunk.Model), completionChunk.Model },
            { nameof(completionChunk.Usage), completionChunk.Usage },
            { nameof(completionChunk.CreatedAt), completionChunk.CreatedAt },
            { nameof(chatChoice.Index), chatChoice.Index },
            { nameof(chatChoice.FinishReason), chatChoice.FinishReason },
        };
    }

    /// <summary>
    /// Processes the function result.
    /// </summary>
    /// <param name="functionResult">The result of the function call.</param>
    /// <param name="toolCallBehavior">The ToolCallBehavior object containing optional settings like JsonSerializerOptions.TypeInfoResolver.</param>
    /// <returns>A string representation of the function result.</returns>
    private static string? ProcessFunctionResult(object functionResult, OllamaAIToolCallBehavior? toolCallBehavior)
    {
        if (functionResult is string stringResult)
        {
            return stringResult;
        }

        // This is an optimization to use ChatMessageContent content directly  
        // without unnecessary serialization of the whole message content class.  
        if (functionResult is ChatMessageContent chatMessageContent)
        {
            return chatMessageContent.ToString();
        }

        // For polymorphic serialization of unknown in advance child classes of the KernelContent class,  
        // a corresponding JsonTypeInfoResolver should be provided via the JsonSerializerOptions.TypeInfoResolver property.  
        // For more details about the polymorphic serialization, see the article at:  
        // https://learn.microsoft.com/en-us/dotnet/standard/serialization/system-text-json/polymorphism?pivots=dotnet-8-0
        return JsonSerializer.Serialize(functionResult, toolCallBehavior?.ToolCallResultSerializerOptions);
    }

    /// <summary>
    /// Executes auto function invocation filters and/or function itself.
    /// This method can be moved to <see cref="Kernel"/> when auto function invocation logic will be extracted to common place.
    /// </summary>
    private static async Task<AutoFunctionInvocationContext> OnAutoFunctionInvocationAsync(
        Kernel kernel,
        AutoFunctionInvocationContext context,
        Func<AutoFunctionInvocationContext, Task> functionCallCallback)
    {
        await InvokeFilterOrFunctionAsync(kernel.AutoFunctionInvocationFilters, functionCallCallback, context).ConfigureAwait(false);

        return context;
    }

    /// <summary>
    /// This method will execute auto function invocation filters and function recursively.
    /// If there are no registered filters, just function will be executed.
    /// If there are registered filters, filter on <paramref name="index"/> position will be executed.
    /// Second parameter of filter is callback. It can be either filter on <paramref name="index"/> + 1 position or function if there are no remaining filters to execute.
    /// Function will be always executed as last step after all filters.
    /// </summary>
    private static async Task InvokeFilterOrFunctionAsync(
        IList<IAutoFunctionInvocationFilter>? autoFunctionInvocationFilters,
        Func<AutoFunctionInvocationContext, Task> functionCallCallback,
        AutoFunctionInvocationContext context,
        int index = 0)
    {
        if (autoFunctionInvocationFilters is { Count: > 0 } && index < autoFunctionInvocationFilters.Count)
        {
            await autoFunctionInvocationFilters[index].OnAutoFunctionInvocationAsync(context,
                (context) => InvokeFilterOrFunctionAsync(autoFunctionInvocationFilters, functionCallCallback, context, index + 1)).ConfigureAwait(false);
        }
        else
        {
            await functionCallCallback(context).ConfigureAwait(false);
        }
    }
    #endregion
}
