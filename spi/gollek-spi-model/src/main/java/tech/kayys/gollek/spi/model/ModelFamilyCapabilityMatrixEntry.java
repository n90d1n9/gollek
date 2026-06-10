package tech.kayys.gollek.spi.model;

import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Flattened support matrix row for one detachable model-family plugin.
 */
public record ModelFamilyCapabilityMatrixEntry(
        String id,
        String displayName,
        String bundleProfile,
        boolean defaultBundle,
        boolean causalLm,
        boolean encoder,
        boolean decoder,
        boolean embedding,
        boolean tokenizer,
        boolean chatTemplate,
        boolean vision,
        boolean audio,
        boolean multimodal,
        boolean moe,
        boolean training,
        boolean gguf,
        boolean onnx,
        List<String> architectureAdapterIds,
        ModelFamilyDirectSupport directSafetensorStatus,
        String directSafetensorReason,
        Map<String, String> directSafetensorCaveats) {

    public ModelFamilyCapabilityMatrixEntry {
        id = id == null || id.isBlank() ? "unknown" : id.trim();
        displayName = displayName == null || displayName.isBlank() ? id : displayName.trim();
        bundleProfile = bundleProfile == null || bundleProfile.isBlank()
                ? ModelFamilyBundleProfile.OPTIONAL.key()
                : bundleProfile.trim();
        directSafetensorStatus = directSafetensorStatus == null
                ? ModelFamilyDirectSupport.NOT_ADVERTISED
                : directSafetensorStatus;
        directSafetensorReason = directSafetensorReason == null || directSafetensorReason.isBlank()
                ? directSafetensorStatus.label()
                : directSafetensorReason.trim();
        directSafetensorCaveats = directSafetensorCaveats == null ? Map.of() : Map.copyOf(directSafetensorCaveats);
        architectureAdapterIds = architectureAdapterIds == null ? List.of() : architectureAdapterIds.stream()
                .map(adapterId -> Objects.toString(adapterId, "").trim())
                .filter(adapterId -> !adapterId.isBlank())
                .distinct()
                .toList();
    }

    public static ModelFamilyCapabilityMatrixEntry from(ModelFamilySupportReport report) {
        EnumSet<ModelFamilyCapability> capabilities = report.capabilities().isEmpty()
                ? EnumSet.noneOf(ModelFamilyCapability.class)
                : EnumSet.copyOf(report.capabilities());
        return new ModelFamilyCapabilityMatrixEntry(
                report.id(),
                report.displayName(),
                report.bundleProfile().key(),
                report.defaultBundle(),
                capabilities.contains(ModelFamilyCapability.CAUSAL_LM),
                capabilities.contains(ModelFamilyCapability.ENCODER),
                capabilities.contains(ModelFamilyCapability.DECODER),
                capabilities.contains(ModelFamilyCapability.EMBEDDING),
                capabilities.contains(ModelFamilyCapability.TOKENIZER),
                capabilities.contains(ModelFamilyCapability.CHAT_TEMPLATE),
                capabilities.contains(ModelFamilyCapability.VISION),
                capabilities.contains(ModelFamilyCapability.AUDIO),
                capabilities.contains(ModelFamilyCapability.MULTIMODAL),
                capabilities.contains(ModelFamilyCapability.MOE),
                capabilities.contains(ModelFamilyCapability.TRAINING),
                capabilities.contains(ModelFamilyCapability.GGUF),
                capabilities.contains(ModelFamilyCapability.ONNX),
                report.architectureAdapterIds(),
                report.directSafetensorStatus(),
                report.directSafetensorReason(),
                report.directSafetensorCaveats());
    }

    public boolean directSafetensorReady() {
        return directSafetensorStatus.ready();
    }

    public boolean architectureAdapterPresent() {
        return !architectureAdapterIds.isEmpty();
    }

    public int architectureAdapterCount() {
        return architectureAdapterIds.size();
    }

    public String compactSummary() {
        return id + "[" + bundleProfile + "]("
                + "tok=" + yesNo(tokenizer)
                + ",gguf=" + yesNo(gguf)
                + ",safetensor=" + directSafetensorStatus.label()
                + ",adapters=" + architectureAdapterCount()
                + ",onnx=" + yesNo(onnx)
                + ",train=" + yesNo(training)
                + ",vlm=" + yesNo(multimodal)
                + ",moe=" + yesNo(moe)
                + ")";
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
    }
}
