package tech.kayys.gollek.provider.openai;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * OpenAI chat completions response.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class OpenAIResponse {

    @JsonProperty("id")
    private String id;

    @JsonProperty("object")
    private String object;

    @JsonProperty("created")
    private Long created;

    @JsonProperty("model")
    private String model;

    @JsonProperty("choices")
    private java.util.List<OpenAiChoice> choices;

    @JsonProperty("usage")
    private OpenAIUsage usage;

    public OpenAIResponse() {
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getObject() {
        return object;
    }

    public void setObject(String object) {
        this.object = object;
    }

    public Long getCreated() {
        return created;
    }

    public void setCreated(Long created) {
        this.created = created;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public java.util.List<OpenAiChoice> getChoices() {
        return choices;
    }

    public void setChoices(java.util.List<OpenAiChoice> choices) {
        this.choices = choices;
    }

    public OpenAIUsage getUsage() {
        return usage;
    }

    public void setUsage(OpenAIUsage usage) {
        this.usage = usage;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class OpenAiChoice {
        @JsonProperty("index")
        private Integer index;

        @JsonProperty("message")
        private OpenAIMessage message;

        @JsonProperty("delta")
        private OpenAIMessage delta;

        @JsonProperty("finish_reason")
        private String finishReason;

        public Integer getIndex() {
            return index;
        }

        public void setIndex(Integer index) {
            this.index = index;
        }

        public OpenAIMessage getMessage() {
            return message;
        }

        public void setMessage(OpenAIMessage message) {
            this.message = message;
        }

        public OpenAIMessage getDelta() {
            return delta;
        }

        public void setDelta(OpenAIMessage delta) {
            this.delta = delta;
        }

        public String getFinishReason() {
            return finishReason;
        }

        public void setFinishReason(String finishReason) {
            this.finishReason = finishReason;
        }
    }
}
