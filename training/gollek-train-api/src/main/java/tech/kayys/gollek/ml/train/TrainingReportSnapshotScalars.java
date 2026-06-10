package tech.kayys.gollek.ml.train;

import java.net.URI;
import java.net.URL;
import java.nio.file.Path;
import java.time.temporal.TemporalAccessor;
import java.util.Date;

/**
 * Scalar normalization rules for JSON-shaped training report snapshots.
 */
final class TrainingReportSnapshotScalars {
    private TrainingReportSnapshotScalars() {
    }

    static boolean isScalar(Object value) {
        return value == null
                || value instanceof String
                || value instanceof Number
                || value instanceof Boolean
                || value instanceof Character
                || value instanceof Enum<?>
                || value instanceof Path
                || value instanceof TemporalAccessor
                || value instanceof Date
                || value instanceof URI
                || value instanceof URL;
    }

    static Object snapshot(Object value) {
        if (value == null
                || value instanceof String
                || value instanceof Boolean) {
            return value;
        }
        if (value instanceof Double doubleValue) {
            return Double.isFinite(doubleValue.doubleValue()) ? value : value.toString();
        }
        if (value instanceof Float floatValue) {
            return Float.isFinite(floatValue.floatValue()) ? value : value.toString();
        }
        if (value instanceof Number) {
            return value;
        }
        if (value instanceof Character character) {
            return character.toString();
        }
        if (value instanceof Enum<?> enumValue) {
            return enumValue.name();
        }
        if (value instanceof Path path) {
            return path.toString();
        }
        if (value instanceof TemporalAccessor temporal) {
            return temporal.toString();
        }
        if (value instanceof Date date) {
            return date.toInstant().toString();
        }
        if (value instanceof URI uri) {
            return uri.toString();
        }
        if (value instanceof URL url) {
            return url.toString();
        }
        return String.valueOf(value);
    }
}
