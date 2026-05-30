package tech.kayys.gollek.spi.model;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Result of resolving a model config claim to attached model-family plugins.
 */
public record ModelFamilyResolution(
        String modelType,
        String architectureClassName,
        Status status,
        List<String> familyIds,
        List<ModelFamilySupportReport> supportReports,
        List<ModelTokenizerDescriptor> tokenizerDescriptors) {

    public enum Status {
        RESOLVED,
        NOT_FOUND,
        AMBIGUOUS
    }

    public ModelFamilyResolution {
        modelType = Objects.toString(modelType, "").trim();
        architectureClassName = Objects.toString(architectureClassName, "").trim();
        status = status == null ? Status.NOT_FOUND : status;
        familyIds = List.copyOf(familyIds == null ? List.of() : familyIds);
        supportReports = List.copyOf(supportReports == null ? List.of() : supportReports);
        tokenizerDescriptors = List.copyOf(tokenizerDescriptors == null ? List.of() : tokenizerDescriptors);
    }

    public boolean resolved() {
        return status == Status.RESOLVED;
    }

    public boolean ambiguous() {
        return status == Status.AMBIGUOUS;
    }

    public boolean notFound() {
        return status == Status.NOT_FOUND;
    }

    public Optional<String> primaryFamilyId() {
        return familyIds.stream().findFirst();
    }

    public Optional<ModelFamilySupportReport> primarySupportReport() {
        return supportReports.stream().findFirst();
    }

    public boolean requiresAttention() {
        return !problemCodes().isEmpty();
    }

    public List<String> problemCodes() {
        if (notFound()) {
            return List.of("model_family_not_found");
        }
        if (ambiguous()) {
            return List.of("model_family_ambiguous");
        }
        if (resolved() && supportReports.isEmpty()) {
            return List.of("model_family_support_report_unavailable");
        }
        return List.of();
    }

    public List<String> remediationHints() {
        if (notFound()) {
            return List.of(
                    "Attach or install a model-family plugin that claims " + querySummary() + ".",
                    "Run `gollek modules --json | jq '.modelFamilyPlugins.plugins[].id'` to inspect attached families.");
        }
        if (ambiguous()) {
            return List.of(
                    "Detach one of the overlapping model-family plugins or narrow the model-family bundle selector.",
                    "Run `gollek modules --json | jq '.modelFamilyPlugins.conflicts'` to inspect duplicate claims.");
        }
        if (resolved() && supportReports.isEmpty()) {
            return List.of(
                    "The matched model-family plugin did not publish a support report; check its plugin implementation.");
        }
        return List.of();
    }

    public String summary() {
        String query = querySummary();
        return switch (status) {
            case RESOLVED -> "resolved " + query + " to " + String.join(", ", familyIds);
            case AMBIGUOUS -> "ambiguous " + query + " matched " + String.join(", ", familyIds);
            case NOT_FOUND -> "no model family matched " + query;
        };
    }

    private String querySummary() {
        if (!modelType.isBlank() && !architectureClassName.isBlank()) {
            return "model_type=" + modelType + ", architecture=" + architectureClassName;
        }
        if (!modelType.isBlank()) {
            return "model_type=" + modelType;
        }
        if (!architectureClassName.isBlank()) {
            return "architecture=" + architectureClassName;
        }
        return "empty model-family query";
    }
}
