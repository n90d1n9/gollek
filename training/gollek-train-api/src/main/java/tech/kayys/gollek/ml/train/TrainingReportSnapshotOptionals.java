package tech.kayys.gollek.ml.train;

import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;
import java.util.function.Function;

/**
 * Optional normalization for JSON-shaped training report snapshots.
 */
final class TrainingReportSnapshotOptionals {
    private TrainingReportSnapshotOptionals() {
    }

    static boolean isOptional(Object value) {
        return value instanceof Optional<?>
                || value instanceof OptionalDouble
                || value instanceof OptionalInt
                || value instanceof OptionalLong;
    }

    static Object snapshot(Object value, Function<Object, Object> nestedSnapshot) {
        if (value instanceof Optional<?> optional) {
            return optional.isPresent() ? nestedSnapshot.apply(optional.get()) : null;
        }
        if (value instanceof OptionalDouble optional) {
            return optional.isPresent() ? TrainingReportSnapshotScalars.snapshot(optional.getAsDouble()) : null;
        }
        if (value instanceof OptionalInt optional) {
            return optional.isPresent() ? optional.getAsInt() : null;
        }
        if (value instanceof OptionalLong optional) {
            return optional.isPresent() ? optional.getAsLong() : null;
        }
        throw new IllegalArgumentException("value is not an Optional type");
    }
}
