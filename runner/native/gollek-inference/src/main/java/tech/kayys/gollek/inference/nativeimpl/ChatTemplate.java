package tech.kayys.gollek.inference.nativeimpl;

import tech.kayys.gollek.spi.Message;
import java.util.List;

/**
 * Interface for chat templates that normalize message lists into a single prompt string.
 */
public interface ChatTemplate {
    
    String apply(List<Message> messages);

    static ChatTemplate forArchitecture(String arch) {
        if (arch == null) return new DefaultTemplate();
        String lower = arch.toLowerCase();
        if (lower.contains("qwen") || lower.contains("llama-3")) {
            return new ChatMLTemplate();
        }
        if (lower.contains("gemma")) {
            return new GemmaTemplate();
        }
        // Add more templates as needed
        return new DefaultTemplate();
    }
}

/**
 * Gemma chat template used by Google Gemma/Gemma-2.
 */
class GemmaTemplate implements ChatTemplate {
    @Override
    public String apply(List<Message> messages) {
        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            String role = msg.getRole() != null ? msg.getRole().name().toLowerCase() : "user";
            if (role.equals("assistant")) role = "model";
            sb.append("<start_of_turn>").append(role).append("\n")
              .append(msg.getContent())
              .append("<end_of_turn>\n");
        }
        sb.append("<start_of_turn>model\n");
        return sb.toString();
    }
}

/**
 * Standard ChatML template used by Qwen, Llama3-Instruct, and others.
 */
class ChatMLTemplate implements ChatTemplate {
    @Override
    public String apply(List<Message> messages) {
        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            String role = msg.getRole() != null ? msg.getRole().name().toLowerCase() : "user";
            sb.append("<|im_start|>").append(role).append("\n")
              .append(msg.getContent())
              .append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }
}

/**
 * Fallback template that just concatenates contents.
 */
class DefaultTemplate implements ChatTemplate {
    @Override
    public String apply(List<Message> messages) {
        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            sb.append(msg.getContent()).append("\n");
        }
        return sb.toString();
    }
}
