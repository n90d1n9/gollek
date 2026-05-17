package tech.kayys.gollek.safetensor.engine.prompt;

import java.util.List;
import java.util.Locale;
import tech.kayys.gollek.models.core.ChatTemplateFormatter;
import tech.kayys.gollek.spi.Message;

/**
 * Runtime prompt-template compatibility layer for checkpoints whose prompt
 * formatting must not depend on whatever tokenizer-core artifact happens to be
 * on the classpath.
 */
public final class PromptTemplateCompat {

    private PromptTemplateCompat() {
    }

    public static String format(List<Message> messages, String modelType) {
        if (isGemma4(modelType)) {
            return formatGemma4(messages);
        }
        return ChatTemplateFormatter.format(messages, modelType);
    }

    private static boolean isGemma4(String modelType) {
        if (modelType == null || modelType.isBlank()) {
            return false;
        }
        String normalized = modelType.trim().toLowerCase(Locale.ROOT);
        return normalized.startsWith("gemma4");
    }

    private static String formatGemma4(List<Message> messages) {
        var sb = new StringBuilder("<bos>");
        String start = "<|turn>";
        String end = "<turn|>";
        String firstUserPrefix = "";
        int startIdx = 0;
        if (!messages.isEmpty() && messages.getFirst().getRole() == Message.Role.SYSTEM) {
            firstUserPrefix = messages.getFirst().getContent().trim();
            if (!firstUserPrefix.isEmpty()) {
                firstUserPrefix += "\n\n";
            }
            startIdx = 1;
        }

        boolean firstRendered = true;
        for (int i = startIdx; i < messages.size(); i++) {
            var message = messages.get(i);
            String roleName = message.getRole() == Message.Role.ASSISTANT
                    ? "model"
                    : message.getRole().name().toLowerCase(Locale.ROOT);
            sb.append(start).append(roleName).append("\n");
            if (firstRendered && !firstUserPrefix.isEmpty() && message.getRole() == Message.Role.USER) {
                sb.append(firstUserPrefix);
            }
            sb.append(message.getContent().trim()).append(end).append("\n");
            firstRendered = false;
        }
        sb.append(start).append("model\n");
        return sb.toString();
    }
}
