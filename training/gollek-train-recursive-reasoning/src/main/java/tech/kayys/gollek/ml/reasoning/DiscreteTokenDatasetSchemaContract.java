package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reusable schema contract descriptor for recursive-reasoning training metadata.
 */
public record DiscreteTokenDatasetSchemaContract(
        Class<?> resourceOwner,
        String name,
        String label,
        String payloadKind,
        String payloadSchemaVersion,
        String jsonSchemaId,
        String jsonSchemaResource,
        List<String> inspectorSections) {
    private static final String DIGEST_ALGORITHM = "SHA-256";

    public DiscreteTokenDatasetSchemaContract {
        resourceOwner = Objects.requireNonNull(resourceOwner, "resourceOwner must not be null");
        name = DiscreteTokenDatasetMetadataSupport.requireText(name, "name");
        label = DiscreteTokenDatasetMetadataSupport.requireText(label, "label");
        payloadKind = DiscreteTokenDatasetMetadataSupport.requireText(payloadKind, "payloadKind");
        payloadSchemaVersion = DiscreteTokenDatasetMetadataSupport.requireText(payloadSchemaVersion, "payloadSchemaVersion");
        jsonSchemaId = DiscreteTokenDatasetMetadataSupport.requireText(jsonSchemaId, "jsonSchemaId");
        jsonSchemaResource = DiscreteTokenDatasetMetadataSupport.requireText(jsonSchemaResource, "jsonSchemaResource");
        inspectorSections = List.copyOf(Objects.requireNonNull(
                inspectorSections,
                "inspectorSections must not be null"));
    }

    public String jsonSchemaText() {
        return resourceText(resourceOwner, jsonSchemaResource, label + " JSON schema");
    }

    public Map<String, Object> jsonSchemaMetadata() {
        return DiscreteTokenDatasetCheckpointMetadataJson.fromJson(jsonSchemaText());
    }

    public String jsonSchemaSha256() {
        return sha256Hex(jsonSchemaText());
    }

    public int jsonSchemaByteCount() {
        return utf8ByteCount(jsonSchemaText());
    }

    public Map<String, Object> toMetadata(boolean includeSchema) {
        String schemaText = jsonSchemaText();
        Map<String, Object> schema = DiscreteTokenDatasetCheckpointMetadataJson.fromJson(schemaText);
        Map<String, Object> contract = new LinkedHashMap<>();
        contract.put("name", name);
        contract.put("label", label);
        contract.put("payloadKind", payloadKind);
        contract.put("payloadSchemaVersion", payloadSchemaVersion);
        contract.put("jsonSchemaId", jsonSchemaId);
        contract.put("jsonSchemaResource", jsonSchemaResource);
        contract.put("jsonSchemaSha256", sha256Hex(schemaText));
        contract.put("jsonSchemaByteCount", utf8ByteCount(schemaText));
        contract.put("jsonSchemaDraft", schema.get("$schema"));
        contract.put("title", schema.get("title"));
        contract.put("description", schema.get("description"));
        contract.put("inspectorSections", inspectorSections);
        if (includeSchema) {
            contract.put("schema", schema);
        }
        return Collections.unmodifiableMap(contract);
    }

    public static String resourceText(Class<?> owner, String resource, String label) {
        Objects.requireNonNull(owner, "owner must not be null");
        String normalizedResource = DiscreteTokenDatasetMetadataSupport.requireText(resource, "resource");
        String normalizedLabel = DiscreteTokenDatasetMetadataSupport.requireText(label, "label");
        try (InputStream stream = owner.getClassLoader().getResourceAsStream(normalizedResource)) {
            if (stream == null) {
                throw new IllegalStateException(normalizedLabel + " resource not found: " + normalizedResource);
            }
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read " + normalizedLabel + " resource", e);
        }
    }

    public static Map<String, Object> jsonMetadata(Class<?> owner, String resource, String label) {
        return DiscreteTokenDatasetCheckpointMetadataJson.fromJson(resourceText(owner, resource, label));
    }

    public static int utf8ByteCount(String text) {
        return Objects.requireNonNull(text, "text must not be null")
                .getBytes(StandardCharsets.UTF_8)
                .length;
    }

    public static String sha256Hex(String text) {
        return hex(sha256().digest(Objects.requireNonNull(text, "text must not be null")
                .getBytes(StandardCharsets.UTF_8)));
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance(DIGEST_ALGORITHM);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(DIGEST_ALGORITHM + " is not available", e);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder builder = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            builder.append(Character.forDigit((value >>> 4) & 0x0f, 16));
            builder.append(Character.forDigit(value & 0x0f, 16));
        }
        return builder.toString();
    }

}
