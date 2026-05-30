package tech.kayys.gollek.ml.train;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.DataLoader.Batch;

/**
 * Captures opt-in dataset distribution diagnostics without consuming generic
 * one-shot training iterables.
 */
final class TrainerDataDistributionMetadata {
    private TrainerDataDistributionMetadata() {
    }

    static Map<String, Object> capture(
            boolean enabled,
            Iterable<Batch> trainLoader,
            Iterable<Batch> validationLoader) {
        if (!enabled) {
            return Map.of();
        }

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("dataDistributionDiagnosticsEnabled", Boolean.TRUE);
        DistributionSnapshot train = putLoader(metadata, "trainDataDistribution", trainLoader);
        DistributionSnapshot validation = putLoader(metadata, "validationDataDistribution", validationLoader);
        putDrift(metadata, train, validation);
        return Map.copyOf(metadata);
    }

    private static DistributionSnapshot putLoader(
            Map<String, Object> metadata,
            String prefix,
            Iterable<Batch> loader) {
        if (loader == null) {
            metadata.put(prefix + ".available", Boolean.FALSE);
            metadata.put(prefix + ".skipped", Boolean.TRUE);
            metadata.put(prefix + ".skipReason", "no-loader");
            return DistributionSnapshot.unavailable("no-loader");
        }
        if (!(loader instanceof DataLoader.TensorDataLoader tensorLoader)) {
            metadata.put(prefix + ".available", Boolean.FALSE);
            metadata.put(prefix + ".skipped", Boolean.TRUE);
            metadata.put(prefix + ".skipReason", "unsupported-loader");
            metadata.put(prefix + ".loaderType", loader.getClass().getName());
            return DistributionSnapshot.unavailable("unsupported-loader");
        }

        try {
            long diagnosticEpoch = tensorLoader.initialEpoch();
            Iterable<Batch> diagnosticLoader = tensorLoader.epoch(diagnosticEpoch);
            DistributionKind kind = inferKind(diagnosticLoader);
            metadata.put(prefix + ".available", Boolean.TRUE);
            metadata.put(prefix + ".skipped", Boolean.FALSE);
            metadata.put(prefix + ".kind", kind.metadataValue());
            metadata.put(prefix + ".diagnosticEpoch", diagnosticEpoch);
            metadata.put(prefix + ".diagnosticEpochViewUsed", Boolean.TRUE);
            if (kind == DistributionKind.MULTI_LABEL) {
                DataLoader.MultiLabelDistributionReport report = DataLoader.multiLabelDistribution(diagnosticLoader);
                metadata.putAll(report.toMetadata(prefix));
                return DistributionSnapshot.multiLabel(report);
            } else {
                DataLoader.ClassificationDistributionReport report = DataLoader.classificationDistribution(
                        diagnosticLoader);
                metadata.putAll(report.toMetadata(prefix));
                return DistributionSnapshot.classification(report);
            }
        } catch (RuntimeException e) {
            metadata.put(prefix + ".available", Boolean.FALSE);
            metadata.put(prefix + ".skipped", Boolean.FALSE);
            metadata.put(prefix + ".errorType", e.getClass().getSimpleName());
            metadata.put(prefix + ".error", e.getMessage());
            return DistributionSnapshot.error(e);
        }
    }

    private static void putDrift(
            Map<String, Object> metadata,
            DistributionSnapshot train,
            DistributionSnapshot validation) {
        if (!train.available() || !validation.available()) {
            metadata.put("dataDistributionDrift.available", Boolean.FALSE);
            metadata.put("dataDistributionDrift.skipped", Boolean.TRUE);
            metadata.put("dataDistributionDrift.skipReason", "missing-compatible-distributions");
            return;
        }
        if (train.kind() != validation.kind()) {
            metadata.put("dataDistributionDrift.available", Boolean.FALSE);
            metadata.put("dataDistributionDrift.skipped", Boolean.TRUE);
            metadata.put("dataDistributionDrift.skipReason", "distribution-kind-mismatch");
            metadata.put("dataDistributionDrift.trainKind", train.kind().metadataValue());
            metadata.put("dataDistributionDrift.validationKind", validation.kind().metadataValue());
            return;
        }

        try {
            metadata.put("dataDistributionDrift.skipped", Boolean.FALSE);
            if (train.kind() == DistributionKind.MULTI_LABEL) {
                metadata.putAll(DataLoader.multiLabelDistributionDrift(
                        train.multiLabel(),
                        validation.multiLabel()).toMetadata("dataDistributionDrift"));
            } else {
                metadata.putAll(DataLoader.classificationDistributionDrift(
                        train.classification(),
                        validation.classification()).toMetadata("dataDistributionDrift"));
            }
        } catch (RuntimeException e) {
            metadata.put("dataDistributionDrift.available", Boolean.FALSE);
            metadata.put("dataDistributionDrift.skipped", Boolean.FALSE);
            metadata.put("dataDistributionDrift.errorType", e.getClass().getSimpleName());
            metadata.put("dataDistributionDrift.error", e.getMessage());
        }
    }

    private static DistributionKind inferKind(Iterable<Batch> loader) {
        Iterator<Batch> iterator = loader.iterator();
        if (!iterator.hasNext()) {
            throw new IllegalArgumentException("loader must contain at least one batch");
        }

        long[] labelShape = iterator.next().labels().shape();
        if (labelShape.length >= 2 && labelShape[labelShape.length - 1] > 1) {
            return DistributionKind.MULTI_LABEL;
        }
        return DistributionKind.CLASSIFICATION;
    }

    private enum DistributionKind {
        CLASSIFICATION("classification"),
        MULTI_LABEL("multi-label");

        private final String metadataValue;

        DistributionKind(String metadataValue) {
            this.metadataValue = metadataValue;
        }

        String metadataValue() {
            return metadataValue;
        }
    }

    private record DistributionSnapshot(
            DistributionKind kind,
            DataLoader.ClassificationDistributionReport classification,
            DataLoader.MultiLabelDistributionReport multiLabel,
            boolean available,
            String unavailableReason) {
        static DistributionSnapshot classification(DataLoader.ClassificationDistributionReport report) {
            return new DistributionSnapshot(DistributionKind.CLASSIFICATION, report, null, true, null);
        }

        static DistributionSnapshot multiLabel(DataLoader.MultiLabelDistributionReport report) {
            return new DistributionSnapshot(DistributionKind.MULTI_LABEL, null, report, true, null);
        }

        static DistributionSnapshot unavailable(String reason) {
            return new DistributionSnapshot(null, null, null, false, reason);
        }

        static DistributionSnapshot error(RuntimeException error) {
            return unavailable(error.getClass().getSimpleName());
        }
    }
}
