package tech.kayys.gollek.server.api.v1.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import tech.kayys.gollek.spi.tool.ToolCall;

import java.util.List;

@JsonInclude(JsonInclude.Include.NON_NULL)
public record ChatCompletionMessage(
    String role,
    String content,
    String toolCallId,
    List<ToolCall> toolCalls
) {}
