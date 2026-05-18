package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class TrainerJsonParser {
    private final String json;
    private int index;

    private TrainerJsonParser(String json) {
        this.json = json;
    }

    static Object parse(String json) {
        return new TrainerJsonParser(json).parse();
    }

    private Object parse() {
        Object value = parseValue();
        skipWhitespace();
        if (index != json.length()) {
            throw error("Trailing characters");
        }
        return value;
    }

    private Object parseValue() {
        skipWhitespace();
        if (index >= json.length()) {
            throw error("Expected value");
        }
        char ch = json.charAt(index);
        return switch (ch) {
            case '{' -> parseObject();
            case '[' -> parseArray();
            case '"' -> parseString();
            case 't' -> parseLiteral("true", Boolean.TRUE);
            case 'f' -> parseLiteral("false", Boolean.FALSE);
            case 'n' -> parseLiteral("null", null);
            default -> parseNumber();
        };
    }

    private Map<String, Object> parseObject() {
        expect('{');
        Map<String, Object> values = new LinkedHashMap<>();
        skipWhitespace();
        if (consume('}')) {
            return Collections.unmodifiableMap(values);
        }
        while (true) {
            skipWhitespace();
            if (index >= json.length() || json.charAt(index) != '"') {
                throw error("Expected object key");
            }
            String key = parseString();
            skipWhitespace();
            expect(':');
            values.put(key, parseValue());
            skipWhitespace();
            if (consume('}')) {
                return Collections.unmodifiableMap(values);
            }
            expect(',');
        }
    }

    private List<Object> parseArray() {
        expect('[');
        List<Object> values = new ArrayList<>();
        skipWhitespace();
        if (consume(']')) {
            return Collections.unmodifiableList(values);
        }
        while (true) {
            values.add(parseValue());
            skipWhitespace();
            if (consume(']')) {
                return Collections.unmodifiableList(values);
            }
            expect(',');
        }
    }

    private String parseString() {
        expect('"');
        StringBuilder value = new StringBuilder();
        while (index < json.length()) {
            char ch = json.charAt(index++);
            if (ch == '"') {
                return value.toString();
            }
            if (ch != '\\') {
                value.append(ch);
                continue;
            }
            if (index >= json.length()) {
                throw error("Unterminated escape");
            }
            char escaped = json.charAt(index++);
            switch (escaped) {
                case '"' -> value.append('"');
                case '\\' -> value.append('\\');
                case '/' -> value.append('/');
                case 'b' -> value.append('\b');
                case 'f' -> value.append('\f');
                case 'n' -> value.append('\n');
                case 'r' -> value.append('\r');
                case 't' -> value.append('\t');
                case 'u' -> value.append(parseUnicodeEscape());
                default -> throw error("Invalid escape");
            }
        }
        throw error("Unterminated string");
    }

    private char parseUnicodeEscape() {
        if (index + 4 > json.length()) {
            throw error("Invalid unicode escape");
        }
        int codePoint = 0;
        for (int i = 0; i < 4; i++) {
            int digit = Character.digit(json.charAt(index++), 16);
            if (digit < 0) {
                throw error("Invalid unicode escape");
            }
            codePoint = (codePoint << 4) + digit;
        }
        return (char) codePoint;
    }

    private Object parseLiteral(String literal, Object value) {
        if (!json.startsWith(literal, index)) {
            throw error("Expected " + literal);
        }
        index += literal.length();
        return value;
    }

    private Number parseNumber() {
        int start = index;
        if (consume('-') && index >= json.length()) {
            throw error("Invalid number");
        }
        readDigits();
        boolean floatingPoint = false;
        if (consume('.')) {
            floatingPoint = true;
            readDigits();
        }
        if (index < json.length() && (json.charAt(index) == 'e' || json.charAt(index) == 'E')) {
            floatingPoint = true;
            index++;
            if (index < json.length() && (json.charAt(index) == '+' || json.charAt(index) == '-')) {
                index++;
            }
            readDigits();
        }
        String number = json.substring(start, index);
        try {
            if (floatingPoint) {
                return Double.parseDouble(number);
            }
            long parsed = Long.parseLong(number);
            if (parsed >= Integer.MIN_VALUE && parsed <= Integer.MAX_VALUE) {
                return (int) parsed;
            }
            return parsed;
        } catch (NumberFormatException error) {
            throw error("Invalid number");
        }
    }

    private void readDigits() {
        int start = index;
        while (index < json.length() && Character.isDigit(json.charAt(index))) {
            index++;
        }
        if (index == start) {
            throw error("Expected digit");
        }
    }

    private void expect(char expected) {
        if (!consume(expected)) {
            throw error("Expected '" + expected + "'");
        }
    }

    private boolean consume(char expected) {
        if (index < json.length() && json.charAt(index) == expected) {
            index++;
            return true;
        }
        return false;
    }

    private void skipWhitespace() {
        while (index < json.length() && Character.isWhitespace(json.charAt(index))) {
            index++;
        }
    }

    private IllegalArgumentException error(String message) {
        return new IllegalArgumentException(message + " at offset " + index);
    }
}
