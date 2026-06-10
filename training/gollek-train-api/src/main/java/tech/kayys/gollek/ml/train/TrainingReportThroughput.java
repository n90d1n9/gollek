package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalLong;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.OptionalLong;

/**
 * Type-safe view over trainer throughput counters captured during training and validation.
 */
public record TrainingReportThroughput(
        Phase train,
        Phase validation) {
    public TrainingReportThroughput {
        train = train == null ? Phase.empty("train") : train;
        validation = validation == null ? Phase.empty("validation") : validation;
    }

    public static TrainingReportThroughput fromMetadata(Map<String, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new TrainingReportThroughput(
                Phase.fromMetadata(metadata, "train"),
                Phase.fromMetadata(metadata, "validation"));
    }

    public boolean available() {
        return train.available() || validation.available();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("available", available());
        map.put("train", train.toMap());
        map.put("validation", validation.toMap());
        return Map.copyOf(map);
    }

    /**
     * Throughput counters and derived rates for one trainer phase.
     */
    public record Phase(
            String name,
            OptionalLong batchCount,
            OptionalLong sampleCount,
            OptionalLong inputElementCount,
            OptionalLong labelElementCount,
            OptionalDouble computeMillis,
            OptionalDouble samplesPerSecond,
            OptionalDouble batchesPerSecond,
            OptionalDouble averageBatchMillis) {
        public Phase {
            name = name == null || name.isBlank() ? "unknown" : name.trim();
            batchCount = batchCount == null ? OptionalLong.empty() : batchCount;
            sampleCount = sampleCount == null ? OptionalLong.empty() : sampleCount;
            inputElementCount = inputElementCount == null ? OptionalLong.empty() : inputElementCount;
            labelElementCount = labelElementCount == null ? OptionalLong.empty() : labelElementCount;
            computeMillis = computeMillis == null ? OptionalDouble.empty() : computeMillis;
            samplesPerSecond = samplesPerSecond == null ? OptionalDouble.empty() : samplesPerSecond;
            batchesPerSecond = batchesPerSecond == null ? OptionalDouble.empty() : batchesPerSecond;
            averageBatchMillis = averageBatchMillis == null ? OptionalDouble.empty() : averageBatchMillis;
        }

        static Phase empty(String name) {
            return new Phase(
                    name,
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty());
        }

        static Phase fromMetadata(Map<String, ?> metadata, String phase) {
            return new Phase(
                    phase,
                    optionalLong(metadata.get(phase + "BatchCount")),
                    optionalLong(metadata.get(phase + "SampleCount")),
                    optionalLong(metadata.get(phase + "InputElementCount")),
                    optionalLong(metadata.get(phase + "LabelElementCount")),
                    optionalDouble(metadata.get(phase + "ComputeMillis")),
                    optionalDouble(metadata.get(phase + "SamplesPerSecond")),
                    optionalDouble(metadata.get(phase + "BatchesPerSecond")),
                    optionalDouble(metadata.get(phase + "AverageBatchMillis")));
        }

        public boolean available() {
            return batchCount.isPresent()
                    || sampleCount.isPresent()
                    || inputElementCount.isPresent()
                    || labelElementCount.isPresent()
                    || computeMillis.isPresent()
                    || samplesPerSecond.isPresent()
                    || batchesPerSecond.isPresent()
                    || averageBatchMillis.isPresent();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("available", available());
            map.put("name", name);
            batchCount.ifPresent(value -> map.put("batchCount", value));
            sampleCount.ifPresent(value -> map.put("sampleCount", value));
            inputElementCount.ifPresent(value -> map.put("inputElementCount", value));
            labelElementCount.ifPresent(value -> map.put("labelElementCount", value));
            computeMillis.ifPresent(value -> map.put("computeMillis", value));
            samplesPerSecond.ifPresent(value -> map.put("samplesPerSecond", value));
            batchesPerSecond.ifPresent(value -> map.put("batchesPerSecond", value));
            averageBatchMillis.ifPresent(value -> map.put("averageBatchMillis", value));
            return Map.copyOf(map);
        }
    }
}
