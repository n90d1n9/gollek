package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Dependency-free JSON codec for checkpoint and resume metadata maps.
 */
public final class DiscreteTokenDatasetCheckpointMetadataJson {
    private static final int INDENT_WIDTH = 2;

    private DiscreteTokenDatasetCheckpointMetadataJson() {}

    public static String toJson(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        StringBuilder builder = new StringBuilder();
        writeValue(builder, metadata, false, 0);
        return builder.toString();
    }

    public static String toPrettyJson(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        StringBuilder builder = new StringBuilder();
        writeValue(builder, metadata, true, 0);
        return builder.toString();
    }

    public static Map<String, Object> fromJson(String json) {
        Objects.requireNonNull(json, "json must not be null");
        Parser parser = new Parser(json);
        Object value = parser.parse();
        if (value instanceof Map<?, ?> map) {
            return immutableStringMap(map);
        }
        throw new IllegalArgumentException("checkpoint metadata JSON must be an object");
    }

    public static void write(Path path, Map<?, ?> metadata) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Files.writeString(path, toPrettyJson(metadata) + System.lineSeparator(), StandardCharsets.UTF_8);
    }

    public static void write(Path path, DiscreteTokenDatasetCheckpointManifest manifest) throws IOException {
        Objects.requireNonNull(manifest, "manifest must not be null");
        write(path, manifest.toMetadata());
    }

    public static Map<String, Object> read(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        return fromJson(Files.readString(path, StandardCharsets.UTF_8));
    }

    public static DiscreteTokenDatasetCheckpointManifestSnapshot readSnapshot(Path path) throws IOException {
        return DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(read(path));
    }

    public static DiscreteTokenDatasetPlanReport readPlanReport(Path path) throws IOException {
        return DiscreteTokenDatasetPlanReport.fromMetadata(read(path));
    }

    public static DiscreteTokenDatasetCheckpointResumeReportSnapshot readResumeReportSnapshot(
            Path path) throws IOException {
        return DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(read(path));
    }

    private static void writeValue(StringBuilder builder, Object value, boolean pretty, int indent) {
        if (value == null) {
            builder.append("null");
            return;
        }
        if (value instanceof Map<?, ?> map) {
            writeMap(builder, map, pretty, indent);
            return;
        }
        if (value instanceof CharSequence text) {
            writeString(builder, text.toString());
            return;
        }
        if (value instanceof Number number) {
            writeNumber(builder, number);
            return;
        }
        if (value instanceof Boolean flag) {
            builder.append(flag);
            return;
        }
        if (value instanceof Iterable<?> iterable) {
            writeArray(builder, iterableItems(iterable), pretty, indent);
            return;
        }
        if (value.getClass().isArray()) {
            writeArray(builder, arrayItems(value), pretty, indent);
            return;
        }
        throw new IllegalArgumentException("unsupported checkpoint metadata value type: "
                + value.getClass().getName());
    }

    private static void writeMap(
            StringBuilder builder,
            Map<?, ?> map,
            boolean pretty,
            int indent) {
        List<MetadataEntry> entries = sortedEntries(map);
        if (entries.isEmpty()) {
            builder.append("{}");
            return;
        }

        builder.append('{');
        if (pretty) {
            builder.append('\n');
        }
        for (int index = 0; index < entries.size(); index++) {
            MetadataEntry entry = entries.get(index);
            if (pretty) {
                appendIndent(builder, indent + 1);
            }
            writeString(builder, entry.key());
            builder.append(pretty ? ": " : ":");
            writeValue(builder, entry.value(), pretty, indent + 1);
            if (index + 1 < entries.size()) {
                builder.append(',');
            }
            if (pretty) {
                builder.append('\n');
            }
        }
        if (pretty) {
            appendIndent(builder, indent);
        }
        builder.append('}');
    }

    private static void writeArray(
            StringBuilder builder,
            List<?> values,
            boolean pretty,
            int indent) {
        if (values.isEmpty()) {
            builder.append("[]");
            return;
        }

        builder.append('[');
        if (pretty) {
            builder.append('\n');
        }
        for (int index = 0; index < values.size(); index++) {
            if (pretty) {
                appendIndent(builder, indent + 1);
            }
            writeValue(builder, values.get(index), pretty, indent + 1);
            if (index + 1 < values.size()) {
                builder.append(',');
            }
            if (pretty) {
                builder.append('\n');
            }
        }
        if (pretty) {
            appendIndent(builder, indent);
        }
        builder.append(']');
    }

    private static void writeString(StringBuilder builder, String value) {
        builder.append('"');
        for (int index = 0; index < value.length(); index++) {
            char ch = value.charAt(index);
            switch (ch) {
                case '"' -> builder.append("\\\"");
                case '\\' -> builder.append("\\\\");
                case '\b' -> builder.append("\\b");
                case '\f' -> builder.append("\\f");
                case '\n' -> builder.append("\\n");
                case '\r' -> builder.append("\\r");
                case '\t' -> builder.append("\\t");
                default -> {
                    if (ch < 0x20) {
                        builder.append("\\u");
                        appendHex(builder, ch);
                    } else {
                        builder.append(ch);
                    }
                }
            }
        }
        builder.append('"');
    }

    private static void writeNumber(StringBuilder builder, Number number) {
        if (number instanceof Double doubleValue && !Double.isFinite(doubleValue)) {
            throw new IllegalArgumentException("checkpoint metadata number must be finite: " + number);
        }
        if (number instanceof Float floatValue && !Float.isFinite(floatValue)) {
            throw new IllegalArgumentException("checkpoint metadata number must be finite: " + number);
        }

        String text = number.toString();
        try {
            new BigDecimal(text);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("checkpoint metadata number is not JSON-compatible: " + text, e);
        }
        builder.append(text);
    }

    private static List<MetadataEntry> sortedEntries(Map<?, ?> map) {
        List<MetadataEntry> entries = new ArrayList<>(map.size());
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            Object key = entry.getKey();
            if (!(key instanceof CharSequence text)) {
                throw new IllegalArgumentException("checkpoint metadata object keys must be strings");
            }
            String normalized = text.toString();
            if (normalized.isBlank()) {
                throw new IllegalArgumentException("checkpoint metadata object keys must not be blank");
            }
            entries.add(new MetadataEntry(normalized, entry.getValue()));
        }
        entries.sort(Comparator.comparing(MetadataEntry::key));
        return entries;
    }

    private static List<Object> iterableItems(Iterable<?> iterable) {
        List<Object> values = new ArrayList<>();
        for (Object item : iterable) {
            values.add(item);
        }
        return values;
    }

    private static List<Object> arrayItems(Object array) {
        int length = Array.getLength(array);
        List<Object> values = new ArrayList<>(length);
        for (int index = 0; index < length; index++) {
            values.add(Array.get(array, index));
        }
        return values;
    }

    private static void appendIndent(StringBuilder builder, int indent) {
        builder.append(" ".repeat(indent * INDENT_WIDTH));
    }

    private static void appendHex(StringBuilder builder, char ch) {
        for (int shift = 12; shift >= 0; shift -= 4) {
            builder.append(Character.forDigit((ch >>> shift) & 0x0f, 16));
        }
    }

    private static Map<String, Object> immutableStringMap(Map<?, ?> map) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            Object key = entry.getKey();
            if (!(key instanceof CharSequence text)) {
                throw new IllegalArgumentException("checkpoint metadata object keys must be strings");
            }
            copy.put(text.toString(), entry.getValue());
        }
        return Collections.unmodifiableMap(copy);
    }

    private record MetadataEntry(String key, Object value) {}

    private static final class Parser {
        private final String json;
        private int offset;

        private Parser(String json) {
            this.json = json;
        }

        private Object parse() {
            skipWhitespace();
            Object value = parseValue();
            skipWhitespace();
            if (!end()) {
                throw error("unexpected trailing content");
            }
            return value;
        }

        private Object parseValue() {
            skipWhitespace();
            if (end()) {
                throw error("unexpected end of JSON");
            }

            char ch = json.charAt(offset);
            return switch (ch) {
                case '{' -> parseObject();
                case '[' -> parseArray();
                case '"' -> parseString();
                case 't' -> parseLiteral("true", Boolean.TRUE);
                case 'f' -> parseLiteral("false", Boolean.FALSE);
                case 'n' -> parseLiteral("null", null);
                default -> {
                    if (ch == '-' || isDigit(ch)) {
                        yield parseNumber();
                    }
                    throw error("unexpected character '" + ch + "'");
                }
            };
        }

        private Map<String, Object> parseObject() {
            expect('{');
            skipWhitespace();
            if (peek('}')) {
                offset++;
                return Map.of();
            }

            Map<String, Object> values = new LinkedHashMap<>();
            while (true) {
                skipWhitespace();
                if (!peek('"')) {
                    throw error("object keys must be strings");
                }
                String key = parseString();
                if (values.containsKey(key)) {
                    throw error("duplicate object key '" + key + "'");
                }
                skipWhitespace();
                expect(':');
                values.put(key, parseValue());
                skipWhitespace();
                if (peek('}')) {
                    offset++;
                    return Collections.unmodifiableMap(values);
                }
                expect(',');
            }
        }

        private List<Object> parseArray() {
            expect('[');
            skipWhitespace();
            if (peek(']')) {
                offset++;
                return List.of();
            }

            List<Object> values = new ArrayList<>();
            while (true) {
                values.add(parseValue());
                skipWhitespace();
                if (peek(']')) {
                    offset++;
                    return List.copyOf(values);
                }
                expect(',');
            }
        }

        private String parseString() {
            expect('"');
            StringBuilder builder = new StringBuilder();
            while (!end()) {
                char ch = json.charAt(offset++);
                if (ch == '"') {
                    return builder.toString();
                }
                if (ch == '\\') {
                    builder.append(parseEscape());
                    continue;
                }
                if (ch < 0x20) {
                    throw error("unescaped control character in string");
                }
                builder.append(ch);
            }
            throw error("unterminated string");
        }

        private char parseEscape() {
            if (end()) {
                throw error("unterminated escape sequence");
            }
            char ch = json.charAt(offset++);
            return switch (ch) {
                case '"' -> '"';
                case '\\' -> '\\';
                case '/' -> '/';
                case 'b' -> '\b';
                case 'f' -> '\f';
                case 'n' -> '\n';
                case 'r' -> '\r';
                case 't' -> '\t';
                case 'u' -> parseUnicodeEscape();
                default -> throw error("unsupported escape sequence '\\" + ch + "'");
            };
        }

        private char parseUnicodeEscape() {
            if (offset + 4 > json.length()) {
                throw error("incomplete unicode escape");
            }
            int value = 0;
            for (int index = 0; index < 4; index++) {
                char ch = json.charAt(offset++);
                int digit = Character.digit(ch, 16);
                if (digit < 0) {
                    throw error("invalid unicode escape");
                }
                value = (value << 4) | digit;
            }
            return (char) value;
        }

        private Object parseNumber() {
            int start = offset;
            if (peek('-')) {
                offset++;
            }
            parseIntegerPart();

            boolean decimal = false;
            if (peek('.')) {
                decimal = true;
                offset++;
                parseDigits("fraction");
            }
            if (peek('e') || peek('E')) {
                decimal = true;
                offset++;
                if (peek('+') || peek('-')) {
                    offset++;
                }
                parseDigits("exponent");
            }

            String text = json.substring(start, offset);
            if (decimal) {
                double value = Double.parseDouble(text);
                if (!Double.isFinite(value)) {
                    throw error("number must be finite");
                }
                return value;
            }
            try {
                return Long.parseLong(text);
            } catch (NumberFormatException e) {
                try {
                    return new BigInteger(text).longValueExact();
                } catch (ArithmeticException ignored) {
                    throw error("integer is outside long range");
                }
            }
        }

        private void parseIntegerPart() {
            if (end()) {
                throw error("number requires an integer part");
            }
            if (peek('0')) {
                offset++;
                if (!end() && isDigit(json.charAt(offset))) {
                    throw error("number must not contain leading zeroes");
                }
                return;
            }
            parseDigits("integer");
        }

        private void parseDigits(String part) {
            int start = offset;
            while (!end() && isDigit(json.charAt(offset))) {
                offset++;
            }
            if (start == offset) {
                throw error("number requires " + part + " digits");
            }
        }

        private Object parseLiteral(String literal, Object value) {
            if (!json.startsWith(literal, offset)) {
                throw error("expected '" + literal + "'");
            }
            offset += literal.length();
            return value;
        }

        private void expect(char expected) {
            if (end() || json.charAt(offset) != expected) {
                throw error("expected '" + expected + "'");
            }
            offset++;
        }

        private boolean peek(char expected) {
            return !end() && json.charAt(offset) == expected;
        }

        private void skipWhitespace() {
            while (!end()) {
                char ch = json.charAt(offset);
                if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') {
                    return;
                }
                offset++;
            }
        }

        private boolean end() {
            return offset >= json.length();
        }

        private boolean isDigit(char ch) {
            return ch >= '0' && ch <= '9';
        }

        private IllegalArgumentException error(String message) {
            return new IllegalArgumentException(message + " at offset " + offset);
        }
    }
}
