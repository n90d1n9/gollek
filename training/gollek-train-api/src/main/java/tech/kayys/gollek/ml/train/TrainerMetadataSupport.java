package tech.kayys.gollek.ml.train;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

/**
 * Shared metadata formatting helpers for trainer summaries, reports, and checkpoints.
 */
final class TrainerMetadataSupport {
    private TrainerMetadataSupport() {
    }

    static boolean filePresent(Path path) {
        return path != null && Files.isRegularFile(path);
    }

    static String normalizeDevice(String deviceId) {
        if (deviceId == null || deviceId.isBlank()) {
            return "auto";
        }
        String value = deviceId.trim().toLowerCase(Locale.ROOT).replace('_', '-');
        return switch (value) {
            case "auto", "gpu", "accelerated", "default" -> "auto";
            case "cpu", "java", "none", "off", "disable", "disabled" -> "cpu";
            case "metal", "mps", "apple", "apple-metal", "apple-gpu", "metal-gpu" -> "metal";
            case "cuda", "nvidia", "gpu-nvidia", "nvidia-cuda" -> "cuda";
            default -> value;
        };
    }

    static int readInt(Object value, int fallback) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String text) {
            try {
                return Integer.parseInt(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    static Object immutableSnapshot(Object value) {
        return TrainerMetricSnapshots.immutableSnapshot(value);
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> stateSnapshot(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return Map.of();
        }
        Object snapshot = immutableSnapshot(state);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }

    static void flatten(
            Map<String, Object> target,
            String prefix,
            Map<String, ?> values) {
        for (Map.Entry<String, ?> entry : values.entrySet()) {
            target.put(prefix + entry.getKey(), entry.getValue());
        }
    }

    static String parameterSignature(Map<String, Parameter> parameters) {
        StringBuilder signature = new StringBuilder();
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            appendParameterSignature(signature, entry.getKey(), entry.getValue());
        }
        return signature.toString();
    }

    static String parameterSignature(List<Parameter> parameters) {
        StringBuilder signature = new StringBuilder();
        for (int index = 0; index < parameters.size(); index++) {
            appendParameterSignature(signature, Integer.toString(index), parameters.get(index));
        }
        return signature.toString();
    }

    private static void appendParameterSignature(
            StringBuilder signature,
            String name,
            Parameter parameter) {
        GradTensor tensor = parameter.data();
        signature
                .append(name)
                .append(':')
                .append(Arrays.toString(tensor.shape()))
                .append(':')
                .append(tensor.numel())
                .append(';');
    }
}
