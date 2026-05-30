package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Trainer-facing validation for multimodal datasets.
 */
public final class MultimodalDatasetValidator {
    public static final String CODE_DATASET_TOO_SMALL = "DATASET_TOO_SMALL";
    public static final String CODE_MISSING_REQUIRED_MODALITY = "MISSING_REQUIRED_MODALITY";
    public static final String CODE_MISSING_NON_TEXT_ASSET = "MISSING_NON_TEXT_ASSET";
    public static final String CODE_BLANK_TEXT = "BLANK_TEXT";
    public static final String CODE_UNRESOLVED_ASSET = "UNRESOLVED_ASSET";
    public static final String CODE_DUPLICATE_SOURCE_PATH = "DUPLICATE_SOURCE_PATH";

    private final int minSamples;
    private final Set<ModalityType> requiredModalities;
    private final boolean requireAnyNonTextAsset;
    private final boolean requireNonBlankText;
    private final IssuePolicy unresolvedAssetPolicy;
    private final IssuePolicy duplicateSourcePathPolicy;

    private MultimodalDatasetValidator(Builder builder) {
        this.minSamples = builder.minSamples;
        this.requiredModalities = Collections.unmodifiableSet(builder.requiredModalities.isEmpty()
                ? EnumSet.noneOf(ModalityType.class)
                : EnumSet.copyOf(builder.requiredModalities));
        this.requireAnyNonTextAsset = builder.requireAnyNonTextAsset;
        this.requireNonBlankText = builder.requireNonBlankText;
        this.unresolvedAssetPolicy = builder.unresolvedAssetPolicy;
        this.duplicateSourcePathPolicy = builder.duplicateSourcePathPolicy;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static MultimodalDatasetValidator textAssetTraining(ModalityType assetModality) {
        Objects.requireNonNull(assetModality, "assetModality must not be null");
        if (assetModality == ModalityType.TEXT) {
            throw new IllegalArgumentException("assetModality must be a non-text modality");
        }
        return builder()
                .minSamples(2)
                .require(ModalityType.TEXT, assetModality)
                .requireNonBlankText(true)
                .unresolvedAssets(IssuePolicy.ERROR)
                .duplicateSourcePaths(IssuePolicy.WARNING)
                .build();
    }

    public MultimodalValidationResult validate(Dataset<? extends List<MultimodalContent>> dataset) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        List<List<MultimodalContent>> samples = new ArrayList<>(dataset.size());
        for (int i = 0; i < dataset.size(); i++) {
            samples.add(dataset.get(i));
        }
        return validate(samples);
    }

    public MultimodalValidationResult validate(List<? extends List<MultimodalContent>> samples) {
        Objects.requireNonNull(samples, "samples must not be null");
        MultimodalDatasetReport report = samples.isEmpty()
                ? emptyReport()
                : MultimodalDatasetDiagnostics.inspect(samples);
        List<MultimodalDatasetIssue> issues = new ArrayList<>();

        if (report.sampleCount() < minSamples) {
            issues.add(error(
                    CODE_DATASET_TOO_SMALL,
                    "dataset has fewer samples than this trainer profile requires",
                    allSampleIndices(report.sampleCount()),
                    Map.of("minSamples", minSamples, "actualSamples", report.sampleCount())));
        }

        for (ModalityType modality : requiredModalities) {
            List<Integer> missing = samplesMissingModality(samples, modality);
            if (!missing.isEmpty()) {
                issues.add(error(
                        CODE_MISSING_REQUIRED_MODALITY,
                        "one or more samples are missing required modality " + modality,
                        missing,
                        Map.of("modality", modality.name())));
            }
        }

        if (requireAnyNonTextAsset && !report.samplesWithoutNonTextAsset().isEmpty()) {
            issues.add(error(
                    CODE_MISSING_NON_TEXT_ASSET,
                    "one or more samples do not contain any non-text asset",
                    report.samplesWithoutNonTextAsset(),
                    Map.of("required", "non-text asset")));
        }

        if (requireNonBlankText && report.hasBlankText()) {
            issues.add(error(
                    CODE_BLANK_TEXT,
                    "one or more samples contain blank text content",
                    report.samplesWithBlankText(),
                    Map.of("blankTextPartCount", report.blankTextPartCount())));
        }

        if (unresolvedAssetPolicy != IssuePolicy.IGNORE && report.hasUnresolvedAssets()) {
            issues.add(policyIssue(
                    unresolvedAssetPolicy,
                    CODE_UNRESOLVED_ASSET,
                    "one or more non-text assets have no URI, base64 data, or raw bytes",
                    samplesWithUnresolvedAssets(samples),
                    Map.of("unresolvedAssetCount", report.unresolvedAssetCount())));
        }

        if (duplicateSourcePathPolicy != IssuePolicy.IGNORE && report.hasDuplicateSourcePaths()) {
            Map<String, List<Integer>> duplicateSamples = duplicateSourcePathSamples(samples, report.duplicateSourcePaths());
            issues.add(policyIssue(
                    duplicateSourcePathPolicy,
                    CODE_DUPLICATE_SOURCE_PATH,
                    "one or more source assets appear in multiple samples; use source-grouped splits to avoid leakage",
                    flattenSampleIndices(duplicateSamples),
                    Map.of(
                            "sourcePaths", report.duplicateSourcePaths(),
                            "samplesBySourcePath", duplicateSamples)));
        }

        return new MultimodalValidationResult(report, issues);
    }

    public enum IssuePolicy {
        IGNORE,
        WARNING,
        ERROR
    }

    public static final class Builder {
        private int minSamples = 1;
        private final EnumSet<ModalityType> requiredModalities = EnumSet.noneOf(ModalityType.class);
        private boolean requireAnyNonTextAsset;
        private boolean requireNonBlankText;
        private IssuePolicy unresolvedAssetPolicy = IssuePolicy.WARNING;
        private IssuePolicy duplicateSourcePathPolicy = IssuePolicy.IGNORE;

        private Builder() {
        }

        public Builder minSamples(int minSamples) {
            if (minSamples < 1) {
                throw new IllegalArgumentException("minSamples must be at least 1");
            }
            this.minSamples = minSamples;
            return this;
        }

        public Builder require(ModalityType first, ModalityType... rest) {
            requiredModalities.add(Objects.requireNonNull(first, "first modality must not be null"));
            if (rest != null) {
                for (ModalityType modality : rest) {
                    requiredModalities.add(Objects.requireNonNull(modality, "required modality must not be null"));
                }
            }
            return this;
        }

        public Builder requireAnyNonTextAsset(boolean requireAnyNonTextAsset) {
            this.requireAnyNonTextAsset = requireAnyNonTextAsset;
            return this;
        }

        public Builder requireNonBlankText(boolean requireNonBlankText) {
            this.requireNonBlankText = requireNonBlankText;
            return this;
        }

        public Builder unresolvedAssets(IssuePolicy policy) {
            this.unresolvedAssetPolicy = Objects.requireNonNull(policy, "policy must not be null");
            return this;
        }

        public Builder duplicateSourcePaths(IssuePolicy policy) {
            this.duplicateSourcePathPolicy = Objects.requireNonNull(policy, "policy must not be null");
            return this;
        }

        public MultimodalDatasetValidator build() {
            return new MultimodalDatasetValidator(this);
        }
    }

    private static MultimodalDatasetIssue error(
            String code,
            String message,
            List<Integer> sampleIndices,
            Map<String, Object> details) {
        return MultimodalDatasetIssue.error(code, message, sampleIndices, details);
    }

    private static MultimodalDatasetIssue policyIssue(
            IssuePolicy policy,
            String code,
            String message,
            List<Integer> sampleIndices,
            Map<String, Object> details) {
        return switch (policy) {
            case ERROR -> MultimodalDatasetIssue.error(code, message, sampleIndices, details);
            case WARNING -> MultimodalDatasetIssue.warning(code, message, sampleIndices, details);
            case IGNORE -> throw new IllegalArgumentException("ignored issues should not be materialized");
        };
    }

    private static List<Integer> allSampleIndices(int size) {
        List<Integer> indices = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            indices.add(i);
        }
        return indices;
    }

    private static MultimodalDatasetReport emptyReport() {
        return new MultimodalDatasetReport(
                0,
                0,
                Map.of(),
                Map.of(),
                Map.of(),
                Map.of(),
                0,
                0,
                0,
                0,
                0,
                0,
                List.of(),
                List.of(),
                List.of(),
                List.of());
    }

    private static List<Integer> samplesMissingModality(
            List<? extends List<MultimodalContent>> samples,
            ModalityType modality) {
        List<Integer> missing = new ArrayList<>();
        for (int index = 0; index < samples.size(); index++) {
            List<MultimodalContent> sample = MultimodalDatasetSupport.immutableSample(
                    samples.get(index),
                    "sample " + index);
            if (!MultimodalDatasetSupport.modalitySet(sample).contains(modality)) {
                missing.add(index);
            }
        }
        return missing;
    }

    private static List<Integer> samplesWithUnresolvedAssets(List<? extends List<MultimodalContent>> samples) {
        List<Integer> indices = new ArrayList<>();
        for (int index = 0; index < samples.size(); index++) {
            if (hasUnresolvedAsset(samples.get(index))) {
                indices.add(index);
            }
        }
        return indices;
    }

    private static boolean hasUnresolvedAsset(List<MultimodalContent> sample) {
        for (MultimodalContent content : MultimodalDatasetSupport.immutableSample(sample, "sample")) {
            if (content.getModality() != ModalityType.TEXT && !hasAssetPayload(content)) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasAssetPayload(MultimodalContent content) {
        return content.isRemote()
                || (content.getBase64Data() != null && !content.getBase64Data().isBlank())
                || (content.getRawBytes() != null && content.getRawBytes().length > 0);
    }

    private static Map<String, List<Integer>> duplicateSourcePathSamples(
            List<? extends List<MultimodalContent>> samples,
            List<String> duplicateSourcePaths) {
        Set<String> duplicates = new LinkedHashSet<>(duplicateSourcePaths);
        Map<String, List<Integer>> samplesByPath = new LinkedHashMap<>();
        for (String path : duplicates) {
            samplesByPath.put(path, new ArrayList<>());
        }
        for (int sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
            for (MultimodalContent content : MultimodalDatasetSupport.immutableSample(
                    samples.get(sampleIndex),
                    "sample " + sampleIndex)) {
                if (content.getModality() == ModalityType.TEXT) {
                    continue;
                }
                String sourcePath = sourcePath(content);
                if (sourcePath != null && samplesByPath.containsKey(sourcePath)) {
                    List<Integer> indices = samplesByPath.get(sourcePath);
                    if (indices.isEmpty() || indices.get(indices.size() - 1) != sampleIndex) {
                        indices.add(sampleIndex);
                    }
                }
            }
        }
        Map<String, List<Integer>> immutable = new LinkedHashMap<>();
        samplesByPath.forEach((path, indices) -> immutable.put(path, List.copyOf(indices)));
        return Collections.unmodifiableMap(immutable);
    }

    private static List<Integer> flattenSampleIndices(Map<String, List<Integer>> samplesByPath) {
        Set<Integer> unique = new LinkedHashSet<>();
        samplesByPath.values().forEach(unique::addAll);
        return List.copyOf(unique);
    }

    private static String sourcePath(MultimodalContent content) {
        Map<String, Object> metadata = content.getMetadata();
        Object value = metadata == null ? null : metadata.get("sourcePath");
        if (value != null) {
            String sourcePath = value.toString().trim();
            if (!sourcePath.isEmpty()) {
                return sourcePath;
            }
        }
        return content.isRemote() ? content.getUri().trim() : null;
    }
}
