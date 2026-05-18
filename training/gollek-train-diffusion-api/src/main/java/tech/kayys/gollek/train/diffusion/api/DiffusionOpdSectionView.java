package tech.kayys.gollek.train.diffusion.api;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Typed view over a named normalized DiffusionOPD report section.
 */
public record DiffusionOpdSectionView(
        String name,
        Map<String, Object> values) {

    public DiffusionOpdSectionView {
        values = Map.copyOf(new LinkedHashMap<>(values));
    }

    public Set<String> keys() {
        return values.keySet();
    }

    public Object value(String key) {
        return values.get(key);
    }

    public String string(String key) {
        Object value = values.get(key);
        return value == null ? null : String.valueOf(value);
    }

    public Double number(String key) {
        Object value = values.get(key);
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value == null) {
            return null;
        }
        try {
            return Double.parseDouble(String.valueOf(value));
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> objectMap(String key) {
        Object value = values.get(key);
        if (value instanceof Map<?, ?> map) {
            LinkedHashMap<String, Object> normalized = new LinkedHashMap<>();
            map.forEach((entryKey, entryValue) -> normalized.put(String.valueOf(entryKey), entryValue));
            return Map.copyOf(normalized);
        }
        return Map.of();
    }
}
