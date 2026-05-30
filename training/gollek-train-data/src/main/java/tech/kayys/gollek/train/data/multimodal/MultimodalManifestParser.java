package tech.kayys.gollek.train.data.multimodal;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

final class MultimodalManifestParser {
    private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};

    private final ObjectMapper mapper;
    private final MultimodalManifestAssetResolver assetResolver;

    MultimodalManifestParser(ObjectMapper mapper, MultimodalManifestAssetResolver assetResolver) {
        this.mapper = Objects.requireNonNull(mapper, "mapper must not be null");
        this.assetResolver = Objects.requireNonNull(assetResolver, "assetResolver must not be null");
    }

    List<MultimodalContent> parse(JsonNode row, int lineNumber) throws IOException {
        if (row == null || !row.isObject()) {
            throw new IllegalArgumentException("manifest line " + lineNumber + " must be a JSON object");
        }
        Map<String, Object> sampleMetadata = objectMap(row.get("metadata"));
        List<MultimodalContent> parts = row.has("parts")
                ? parseParts(row.get("parts"), sampleMetadata, lineNumber)
                : parseShorthand(row, sampleMetadata, lineNumber);
        if (parts.isEmpty()) {
            throw new IllegalArgumentException("manifest line " + lineNumber + " does not contain multimodal content");
        }
        return parts;
    }

    private List<MultimodalContent> parseParts(
            JsonNode partsNode,
            Map<String, Object> sampleMetadata,
            int lineNumber) throws IOException {
        if (partsNode == null || !partsNode.isArray()) {
            throw new IllegalArgumentException("manifest line " + lineNumber + " parts must be an array");
        }
        List<MultimodalContent> parts = new ArrayList<>();
        int partIndex = 0;
        for (JsonNode partNode : partsNode) {
            if (!partNode.isObject()) {
                throw new IllegalArgumentException("manifest line " + lineNumber + " part " + partIndex
                        + " must be an object");
            }
            parts.add(parsePart(partNode, sampleMetadata, lineNumber, partIndex));
            partIndex++;
        }
        return parts;
    }

    private List<MultimodalContent> parseShorthand(
            JsonNode row,
            Map<String, Object> sampleMetadata,
            int lineNumber) throws IOException {
        List<MultimodalContent> parts = new ArrayList<>();
        addTextPart(parts, row, sampleMetadata);
        addBinaryPart(parts, row, "image", ModalityType.IMAGE, sampleMetadata, lineNumber);
        addBinaryPart(parts, row, "audio", ModalityType.AUDIO, sampleMetadata, lineNumber);
        addBinaryPart(parts, row, "video", ModalityType.VIDEO, sampleMetadata, lineNumber);
        addBinaryPart(parts, row, "document", ModalityType.DOCUMENT, sampleMetadata, lineNumber);
        if (row.has("embedding")) {
            parts.add(withMetadata(
                    MultimodalContent.ofEmbedding(floatArray(row.get("embedding"), "embedding")),
                    sampleMetadata));
        }
        if (row.has("timeSeries")) {
            parts.add(withMetadata(
                    MultimodalContent.ofTimeSeries(
                            doubleArray(row.get("timeSeries"), "timeSeries"),
                            optionalLong(row, "samplingRateHz", 1L)),
                    sampleMetadata));
        }
        return parts;
    }

    private MultimodalContent parsePart(
            JsonNode partNode,
            Map<String, Object> sampleMetadata,
            int lineNumber,
            int partIndex) throws IOException {
        ModalityType modality = modality(partNode, lineNumber, partIndex);
        Map<String, Object> partMetadata = objectMap(partNode.get("metadata"));
        partMetadata.putAll(sampleMetadata);
        return switch (modality) {
            case TEXT -> parseTextPart(partNode, partMetadata, lineNumber, partIndex);
            case IMAGE, AUDIO, VIDEO, DOCUMENT -> parseBinaryPart(partNode, modality, partMetadata);
            case EMBEDDING -> withMetadata(
                    MultimodalContent.ofEmbedding(floatArray(partNode.get("embedding"), "embedding")),
                    partMetadata);
            case TIME_SERIES -> withMetadata(
                    MultimodalContent.ofTimeSeries(
                            doubleArray(partNode.get("timeSeries"), "timeSeries"),
                            optionalLong(partNode, "samplingRateHz", 1L)),
                    partMetadata);
        };
    }

    private MultimodalContent parseTextPart(
            JsonNode partNode,
            Map<String, Object> metadata,
            int lineNumber,
            int partIndex) {
        String text = firstText(partNode, "text", "value", "content");
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("manifest line " + lineNumber + " part " + partIndex
                    + " TEXT requires non-blank text");
        }
        return withMetadata(MultimodalContent.ofText(text.trim()), metadata);
    }

    private MultimodalContent parseBinaryPart(
            JsonNode partNode,
            ModalityType modality,
            Map<String, Object> metadata) throws IOException {
        String mimeType = textValue(partNode, "mimeType");
        String documentFormat = textValue(partNode, "documentFormat");
        if (documentFormat == null) {
            documentFormat = textValue(partNode, "format");
        }
        if (hasText(partNode, "base64Data")) {
            return withMetadata(binaryBuilder(modality)
                    .base64Data(textValue(partNode, "base64Data"))
                    .mimeType(mimeType)
                    .documentFormat(documentFormat)
                    .build(), metadata);
        }
        if (hasText(partNode, "path")) {
            return contentFromAsset(assetResolver.resolvePath(
                    textValue(partNode, "path"),
                    modality,
                    mimeType,
                    documentFormat), modality, metadata);
        }
        if (hasText(partNode, "uri")) {
            return contentFromAsset(assetResolver.resolveUri(
                    textValue(partNode, "uri"),
                    modality,
                    mimeType,
                    documentFormat), modality, metadata);
        }
        throw new IllegalArgumentException(modality + " manifest part requires path, uri, or base64Data");
    }

    private void addTextPart(List<MultimodalContent> parts, JsonNode row, Map<String, Object> sampleMetadata) {
        String text = firstText(row, "text", "prompt", "caption");
        if (text != null && !text.isBlank()) {
            parts.add(withMetadata(MultimodalContent.ofText(text.trim()), sampleMetadata));
        }
    }

    private void addBinaryPart(
            List<MultimodalContent> parts,
            JsonNode row,
            String field,
            ModalityType modality,
            Map<String, Object> sampleMetadata,
            int lineNumber) throws IOException {
        if (!row.has(field)) {
            return;
        }
        JsonNode node = row.get(field);
        Map<String, Object> metadata = objectMap(row.get(field + "Metadata"));
        metadata.putAll(sampleMetadata);
        if (node.isTextual()) {
            parts.add(contentFromAsset(assetResolver.resolvePath(node.asText(), modality, null, null), modality, metadata));
            return;
        }
        if (node.isObject()) {
            parts.add(parseBinaryPart(node, modality, metadata));
            return;
        }
        throw new IllegalArgumentException("manifest line " + lineNumber + " field " + field
                + " must be a string path or object");
    }

    private MultimodalContent contentFromAsset(
            MultimodalManifestAssetResolver.Asset asset,
            ModalityType modality,
            Map<String, Object> metadata) {
        MultimodalContent.Builder builder = binaryBuilder(modality)
                .mimeType(asset.mimeType())
                .documentFormat(asset.documentFormat());
        if (asset.bytes() != null) {
            builder.base64Data(java.util.Base64.getEncoder().encodeToString(asset.bytes()));
        } else {
            builder.uri(asset.uri());
        }
        if (asset.path() != null) {
            builder.meta("sourcePath", asset.path().toString());
        }
        return withMetadata(builder.build(), metadata);
    }

    private static MultimodalContent.Builder binaryBuilder(ModalityType modality) {
        return MultimodalContent.builder(modality);
    }

    private static ModalityType modality(JsonNode partNode, int lineNumber, int partIndex) {
        String value = firstText(partNode, "modality", "type");
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("manifest line " + lineNumber + " part " + partIndex
                    + " requires modality");
        }
        try {
            return ModalityType.valueOf(value.trim().replace('-', '_').toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException error) {
            throw new IllegalArgumentException("manifest line " + lineNumber + " part " + partIndex
                    + " has unsupported modality: " + value, error);
        }
    }

    private MultimodalContent withMetadata(MultimodalContent content, Map<String, Object> metadata) {
        Map<String, Object> merged = new java.util.LinkedHashMap<>();
        if (content.getMetadata() != null) {
            merged.putAll(content.getMetadata());
        }
        merged.putAll(metadata);
        if (merged.isEmpty()) {
            return content;
        }
        MultimodalContent.Builder builder = MultimodalContent.builder(content.getModality())
                .text(content.getText())
                .rawBytes(content.getRawBytes())
                .mimeType(content.getMimeType())
                .base64Data(content.getBase64Data())
                .uri(content.getUri())
                .documentFormat(content.getDocumentFormat())
                .embedding(content.getEmbedding())
                .timeSeries(content.getTimeSeries())
                .samplingRateHz(content.getSamplingRateHz());
        merged.forEach(builder::meta);
        return builder.build();
    }

    private Map<String, Object> objectMap(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return new java.util.LinkedHashMap<>();
        }
        if (!node.isObject()) {
            throw new IllegalArgumentException("metadata must be an object");
        }
        return new java.util.LinkedHashMap<>(mapper.convertValue(node, MAP_TYPE));
    }

    private static String firstText(JsonNode node, String... names) {
        for (String name : names) {
            String value = textValue(node, name);
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    private static String textValue(JsonNode node, String name) {
        JsonNode value = node == null ? null : node.get(name);
        return value == null || value.isNull() ? null : value.asText();
    }

    private static boolean hasText(JsonNode node, String name) {
        String value = textValue(node, name);
        return value != null && !value.isBlank();
    }

    private static long optionalLong(JsonNode node, String name, long defaultValue) {
        JsonNode value = node == null ? null : node.get(name);
        return value == null || value.isNull() ? defaultValue : value.asLong();
    }

    private static float[] floatArray(JsonNode node, String name) {
        if (node == null || !node.isArray()) {
            throw new IllegalArgumentException(name + " must be an array");
        }
        float[] values = new float[node.size()];
        Iterator<JsonNode> iterator = node.elements();
        int index = 0;
        while (iterator.hasNext()) {
            values[index++] = (float) iterator.next().asDouble();
        }
        return values;
    }

    private static double[] doubleArray(JsonNode node, String name) {
        if (node == null || !node.isArray()) {
            throw new IllegalArgumentException(name + " must be an array");
        }
        double[] values = new double[node.size()];
        Iterator<JsonNode> iterator = node.elements();
        int index = 0;
        while (iterator.hasNext()) {
            values[index++] = iterator.next().asDouble();
        }
        return values;
    }
}
