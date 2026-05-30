package tech.kayys.gollek.spi.pipeline;

import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Request envelope passed from a backend provider into a feature pipeline.
 *
 * <p>The original {@link ProviderRequest} stays intact, while provider-specific
 * model facts and execution attributes are exposed in neutral maps. This keeps
 * feature projects independent from internal provider diagnostic classes.</p>
 */
public final class ModelPipelineRequest {
    private final ProviderRequest request;
    private final String providerId;
    private final Path modelPath;
    private final Map<String, Object> modelFacts;
    private final Map<String, Object> attributes;

    public ModelPipelineRequest(
            ProviderRequest request,
            String providerId,
            Path modelPath,
            Map<String, Object> modelFacts,
            Map<String, Object> attributes) {
        this.request = Objects.requireNonNull(request, "request");
        this.providerId = providerId;
        this.modelPath = modelPath;
        this.modelFacts = modelFacts == null ? Map.of() : Map.copyOf(modelFacts);
        this.attributes = attributes == null ? Map.of() : Map.copyOf(attributes);
    }

    public ProviderRequest request() {
        return request;
    }

    public String providerId() {
        return providerId;
    }

    public Optional<Path> modelPath() {
        return Optional.ofNullable(modelPath);
    }

    public Map<String, Object> modelFacts() {
        return modelFacts;
    }

    public Map<String, Object> attributes() {
        return attributes;
    }

    public Optional<String> factText(String key) {
        return text(modelFacts.get(key));
    }

    public boolean factBoolean(String key) {
        Object value = modelFacts.get(key);
        if (value instanceof Boolean bool) {
            return bool;
        }
        return text(value)
                .map(Boolean::parseBoolean)
                .orElse(false);
    }

    public Optional<String> parameterText(String key) {
        return text(request.getParameters().get(key));
    }

    public Optional<Integer> parameterInt(String key) {
        Object value = request.getParameters().get(key);
        if (value instanceof Number number) {
            return Optional.of(number.intValue());
        }
        return text(value).flatMap(ModelPipelineRequest::parseInt);
    }

    public boolean parameterBoolean(String key) {
        Object value = request.getParameters().get(key);
        if (value instanceof Boolean bool) {
            return bool;
        }
        return text(value)
                .map(Boolean::parseBoolean)
                .orElse(false);
    }

    public Optional<Integer> attributeInt(String key) {
        Object value = attributes.get(key);
        if (value instanceof Number number) {
            return Optional.of(number.intValue());
        }
        return text(value).flatMap(ModelPipelineRequest::parseInt);
    }

    public static Builder builder(ProviderRequest request) {
        return new Builder(request);
    }

    private static Optional<String> text(Object value) {
        if (value == null) {
            return Optional.empty();
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? Optional.empty() : Optional.of(text);
    }

    private static Optional<Integer> parseInt(String text) {
        try {
            return Optional.of(Integer.parseInt(text));
        } catch (NumberFormatException ignored) {
            return Optional.empty();
        }
    }

    public static final class Builder {
        private final ProviderRequest request;
        private String providerId;
        private Path modelPath;
        private final Map<String, Object> modelFacts = new LinkedHashMap<>();
        private final Map<String, Object> attributes = new LinkedHashMap<>();

        private Builder(ProviderRequest request) {
            this.request = Objects.requireNonNull(request, "request");
        }

        public Builder providerId(String providerId) {
            this.providerId = providerId;
            return this;
        }

        public Builder modelPath(Path modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        public Builder fact(String key, Object value) {
            if (key != null && value != null) {
                this.modelFacts.put(key, value);
            }
            return this;
        }

        public Builder facts(Map<String, Object> values) {
            if (values != null) {
                values.forEach(this::fact);
            }
            return this;
        }

        public Builder attribute(String key, Object value) {
            if (key != null && value != null) {
                this.attributes.put(key, value);
            }
            return this;
        }

        public Builder attributes(Map<String, Object> values) {
            if (values != null) {
                values.forEach(this::attribute);
            }
            return this;
        }

        public ModelPipelineRequest build() {
            return new ModelPipelineRequest(request, providerId, modelPath, modelFacts, attributes);
        }
    }
}
