package tech.kayys.gollek.inference.nativeimpl;

import tech.kayys.gollek.spi.Message;
import java.util.List;

/**
 * Interface for chat templates that normalize message lists into a single prompt string.
 */
public interface ChatTemplate {
    
    String apply(List<Message> messages, java.util.Map<String, Integer> specialTokens);
    
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
    public String apply(List<Message> messages, java.util.Map<String, Integer> specialTokens) {
        String start = specialTokens.containsKey("<|turn>") ? "<|turn>" : "<start_of_turn>";
        String end = specialTokens.containsKey("<turn|>") ? "<turn|>" : "<end_of_turn>";
        
        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            String role = msg.getRole() != null ? msg.getRole().name().toLowerCase() : "user";
            if (role.equals("assistant")) role = "model";
            sb.append(start).append(role).append("\n")
              .append(msg.getContent())
              .append(end).append("\n");
        }
        sb.append(start).append("model\n");
        return sb.toString();
    }
}

/**
 * Standard ChatML template used by Qwen, Llama3-Instruct, and others.
 */
class ChatMLTemplate implements ChatTemplate {
    @Override
    public String apply(List<Message> messages, java.util.Map<String, Integer> specialTokens) {
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
    public String apply(List<Message> messages, java.util.Map<String, Integer> specialTokens) {
        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            sb.append(msg.getContent()).append("\n");
        }
        return sb.toString();
    }
}
