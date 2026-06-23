package tech.kayys.gollek.safetensor.engine.prompt;

import java.util.List;
import tech.kayys.gollek.models.core.ChatTemplateFormatter;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

/**
 * Runtime prompt-template compatibility layer for checkpoints whose prompt
 * formatting must not depend on whatever tokenizer-core artifact happens to be
 * on the classpath.
 */
public final class PromptTemplateCompat {

    private PromptTemplateCompat() {
    }

    public static String format(List<Message> messages, String modelType) {
        return format(messages, modelType, null);
    }

    public static String format(List<Message> messages, String modelType, ModelRuntimeTraits runtimeTraits) {
        if (runtimeTraits != null) {
            if (runtimeTraits.gemma4Text()) {
                return ChatTemplateFormatter.format(messages, "gemma4_text");
            }
            if (runtimeTraits.qwenText()) {
                return ChatTemplateFormatter.format(messages, "qwen");
            }
            if (runtimeTraits.gemma3Text()) {
                return ChatTemplateFormatter.format(messages, "gemma3_text");
            }
        }
        return ChatTemplateFormatter.format(messages, modelType);
    }
}
