package tech.kayys.gollek.spi.feature;

import java.util.List;
import java.util.Map;

/**
 * Generic descriptor for Gollek feature adapters.
 *
 * <p>The descriptor intentionally uses strings for inputs, outputs, formats,
 * targets, and metadata so it can cover inference runners, training jobs,
 * quantization passes, backend bridges, conversion tools, and future adapter
 * families without forcing every domain into one enum.</p>
 */
public record FeatureAdapterDescriptor(
        String id,
        String kind,
        String name,
        String version,
        String family,
        List<String> inputs,
        List<String> outputs,
        List<String> formats,
        List<String> targets,
        List<String> tags,
        Map<String, Object> metadata) {

    public FeatureAdapterDescriptor {
        id = normalize(id);
        kind = FeatureAdapterKinds.normalize(kind);
        name = normalize(name);
        version = normalize(version);
        family = normalize(family);
        inputs = inputs == null ? List.of() : List.copyOf(inputs);
        outputs = outputs == null ? List.of() : List.copyOf(outputs);
        formats = formats == null ? List.of() : List.copyOf(formats);
        targets = targets == null ? List.of() : List.copyOf(targets);
        tags = tags == null ? List.of() : List.copyOf(tags);
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim();
    }
}
