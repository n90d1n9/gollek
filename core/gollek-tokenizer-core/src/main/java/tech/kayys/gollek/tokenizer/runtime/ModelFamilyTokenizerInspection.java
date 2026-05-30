package tech.kayys.gollek.tokenizer.runtime;

import tech.kayys.gollek.spi.model.ModelFamilyResolution;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * File-system inspection for tokenizer descriptors advertised by a model family.
 */
public record ModelFamilyTokenizerInspection(
        Path modelDir,
        ModelFamilyResolution modelFamily,
        List<DescriptorStatus> descriptors) {

    public ModelFamilyTokenizerInspection {
        descriptors = List.copyOf(descriptors == null ? List.of() : descriptors);
    }

    public static ModelFamilyTokenizerInspection inspect(Path modelDir, ModelFamilyResolution modelFamily) {
        List<DescriptorStatus> descriptors = modelFamily == null
                ? List.of()
                : modelFamily.tokenizerDescriptors().stream()
                        .map(descriptor -> DescriptorStatus.inspect(modelDir, descriptor))
                        .toList();
        return new ModelFamilyTokenizerInspection(modelDir, modelFamily, descriptors);
    }

    public List<ModelTokenizerDescriptor> usableDescriptors() {
        return descriptors.stream()
                .filter(DescriptorStatus::usable)
                .map(DescriptorStatus::descriptor)
                .toList();
    }

    public List<String> usableDescriptorIds() {
        return usableDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList();
    }

    public boolean requiresAttention() {
        return !problemCodes().isEmpty();
    }

    public List<String> problemCodes() {
        List<String> codes = new ArrayList<>();
        if (modelFamily != null) {
            codes.addAll(modelFamily.problemCodes());
            if (modelFamily.resolved() && descriptors.isEmpty()) {
                codes.add("model_family_tokenizer_descriptors_missing");
            } else if (modelFamily.resolved()
                    && !descriptors.isEmpty()
                    && descriptors.stream().noneMatch(DescriptorStatus::usable)) {
                codes.add("model_family_tokenizer_files_missing");
            }
        }
        return List.copyOf(codes);
    }

    public List<String> remediationHints() {
        List<String> hints = new ArrayList<>();
        if (modelFamily != null) {
            hints.addAll(modelFamily.remediationHints());
        }
        if (problemCodes().contains("model_family_tokenizer_descriptors_missing")) {
            hints.add("Publish at least one tokenizer descriptor from the matched model-family plugin.");
        }
        if (problemCodes().contains("model_family_tokenizer_files_missing")) {
            String requirements = descriptors.stream()
                    .map(DescriptorStatus::requirementSummary)
                    .reduce((left, right) -> left + "; " + right)
                    .orElse("no tokenizer descriptor requirements were available");
            hints.add("Add one required tokenizer file group for the matched family: " + requirements + ".");
        }
        return List.copyOf(hints);
    }

    public String summary() {
        if (modelFamily == null) {
            return "no model-family tokenizer inspection was available";
        }
        if (descriptors.isEmpty()) {
            return "model family " + String.join(", ", modelFamily.familyIds())
                    + " did not advertise tokenizer descriptors";
        }
        return "model family tokenizer descriptors usable="
                + usableDescriptorIds()
                + ", inspected="
                + descriptors.stream().map(DescriptorStatus::id).toList();
    }

    public record DescriptorStatus(
            ModelTokenizerDescriptor descriptor,
            String id,
            ModelTokenizerKind kind,
            boolean usable,
            List<String> existingFileGroup,
            List<List<String>> requiredFileGroups,
            List<List<String>> missingFileGroups) {

        public DescriptorStatus {
            existingFileGroup = List.copyOf(existingFileGroup == null ? List.of() : existingFileGroup);
            requiredFileGroups = requiredFileGroups == null
                    ? List.of()
                    : requiredFileGroups.stream()
                            .map(group -> group == null ? List.<String>of() : List.copyOf(group))
                            .toList();
            missingFileGroups = missingFileGroups == null
                    ? List.of()
                    : missingFileGroups.stream()
                            .map(group -> group == null ? List.<String>of() : List.copyOf(group))
                            .toList();
        }

        public static DescriptorStatus inspect(Path modelDir, ModelTokenizerDescriptor descriptor) {
            List<String> existing = descriptor.firstExistingFileGroup(modelDir)
                    .map(paths -> paths.stream()
                            .map(path -> modelDir.relativize(path).toString())
                            .toList())
                    .orElse(List.of());
            List<List<String>> missing = descriptor.requiredFileGroups().stream()
                    .filter(group -> group.stream().anyMatch(relative -> !Files.exists(modelDir.resolve(relative))))
                    .toList();
            return new DescriptorStatus(
                    descriptor,
                    descriptor.id(),
                    descriptor.kind(),
                    !existing.isEmpty(),
                    existing,
                    descriptor.requiredFileGroups(),
                    missing);
        }

        public String requirementSummary() {
            return id + " requires " + requiredFileGroups;
        }
    }
}
