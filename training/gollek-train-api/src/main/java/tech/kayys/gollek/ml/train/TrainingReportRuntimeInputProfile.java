package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalLong;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.OptionalLong;

/**
 * Compact typed view of runtime-profile input pipeline timings.
 */
record TrainingReportRuntimeInputProfile(
        Stage trainIterator,
        Stage trainHasNext,
        Stage trainNext,
        Stage validationIterator,
        Stage validationHasNext,
        Stage validationNext) {
    TrainingReportRuntimeInputProfile {
        trainIterator = trainIterator == null ? Stage.empty() : trainIterator;
        trainHasNext = trainHasNext == null ? Stage.empty() : trainHasNext;
        trainNext = trainNext == null ? Stage.empty() : trainNext;
        validationIterator = validationIterator == null ? Stage.empty() : validationIterator;
        validationHasNext = validationHasNext == null ? Stage.empty() : validationHasNext;
        validationNext = validationNext == null ? Stage.empty() : validationNext;
    }

    static TrainingReportRuntimeInputProfile fromMetadata(Map<String, ?> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return empty();
        }
        return new TrainingReportRuntimeInputProfile(
                Stage.fromMetadata(metadata, "runtimeProfile.input.train.iterator"),
                Stage.fromMetadata(metadata, "runtimeProfile.input.train.hasNext"),
                Stage.fromMetadata(metadata, "runtimeProfile.input.train.next"),
                Stage.fromMetadata(metadata, "runtimeProfile.input.validation.iterator"),
                Stage.fromMetadata(metadata, "runtimeProfile.input.validation.hasNext"),
                Stage.fromMetadata(metadata, "runtimeProfile.input.validation.next"));
    }

    static TrainingReportRuntimeInputProfile empty() {
        return new TrainingReportRuntimeInputProfile(
                Stage.empty(),
                Stage.empty(),
                Stage.empty(),
                Stage.empty(),
                Stage.empty(),
                Stage.empty());
    }

    boolean available() {
        return trainIterator.available()
                || trainHasNext.available()
                || trainNext.available()
                || validationIterator.available()
                || validationHasNext.available()
                || validationNext.available();
    }

    Map<String, Object> toEvidenceMap() {
        if (!available()) {
            return Map.of();
        }
        Map<String, Object> evidence = new LinkedHashMap<>();
        putStage(evidence, "trainIterator", trainIterator);
        putStage(evidence, "trainHasNext", trainHasNext);
        putStage(evidence, "trainNext", trainNext);
        putStage(evidence, "validationIterator", validationIterator);
        putStage(evidence, "validationHasNext", validationHasNext);
        putStage(evidence, "validationNext", validationNext);
        putTotal(evidence, "trainInputTotalMillis", trainIterator, trainHasNext, trainNext);
        putTotal(evidence, "validationInputTotalMillis", validationIterator, validationHasNext, validationNext);
        putSummaryEvidence(evidence, ScopeSummary.of("train", trainIterator, trainHasNext, trainNext),
                ScopeSummary.of("validation", validationIterator, validationHasNext, validationNext));
        return Map.copyOf(evidence);
    }

    private static void putSummaryEvidence(Map<String, Object> evidence, ScopeSummary train, ScopeSummary validation) {
        double total = train.totalMillis() + validation.totalMillis();
        if (train.hasTotal() || validation.hasTotal()) {
            evidence.put("inputTotalMillis", total);
        }
        ScopeSummary dominant = train.totalMillis() >= validation.totalMillis() ? train : validation;
        if (dominant.hasTotal()) {
            evidence.put("dominantInputScope", dominant.name());
            evidence.put("dominantInputScopeTotalMillis", dominant.totalMillis());
            if (total > 0.0) {
                evidence.put("dominantInputScopePercent", dominant.totalMillis() * 100.0 / total);
            }
            dominant.dominantStage().ifPresent(stage -> {
                evidence.put("dominantInputStage", stage.name());
                evidence.put("dominantInputStageTotalMillis", stage.totalMillis());
                if (dominant.totalMillis() > 0.0) {
                    evidence.put("dominantInputStagePercent", stage.totalMillis() * 100.0 / dominant.totalMillis());
                }
            });
        }
        if (train.hasTotal() && validation.hasTotal() && validation.totalMillis() > 0.0) {
            evidence.put("trainToValidationInputTotalRatio", train.totalMillis() / validation.totalMillis());
        }
    }

    Map<String, Object> toMap() {
        Map<String, Object> view = new LinkedHashMap<>();
        view.put("available", available());
        ScopeSummary train = ScopeSummary.of("train", trainIterator, trainHasNext, trainNext);
        ScopeSummary validation = ScopeSummary.of("validation", validationIterator, validationHasNext, validationNext);
        view.put("train", train.toMap());
        view.put("validation", validation.toMap());
        putOverallSummary(view, train, validation);
        return Map.copyOf(view);
    }

    java.util.Optional<BottleneckSummary> bottleneckSummary() {
        ScopeSummary train = ScopeSummary.of("train", trainIterator, trainHasNext, trainNext);
        ScopeSummary validation = ScopeSummary.of("validation", validationIterator, validationHasNext, validationNext);
        double total = train.totalMillis() + validation.totalMillis();
        ScopeSummary dominantScope = train.totalMillis() >= validation.totalMillis() ? train : validation;
        if (!dominantScope.hasTotal()) {
            return java.util.Optional.empty();
        }
        java.util.Optional<StageTotal> dominantStage = dominantScope.dominantStage();
        if (dominantStage.isEmpty()) {
            return java.util.Optional.empty();
        }
        StageTotal stage = dominantStage.orElseThrow();
        return java.util.Optional.of(new BottleneckSummary(
                dominantScope.name(),
                dominantScope.totalMillis(),
                total > 0.0 ? java.util.OptionalDouble.of(dominantScope.totalMillis() * 100.0 / total)
                        : java.util.OptionalDouble.empty(),
                stage.name(),
                stage.totalMillis(),
                dominantScope.totalMillis() > 0.0
                        ? java.util.OptionalDouble.of(stage.totalMillis() * 100.0 / dominantScope.totalMillis())
                        : java.util.OptionalDouble.empty()));
    }

    private static void putOverallSummary(
            Map<String, Object> view,
            ScopeSummary train,
            ScopeSummary validation) {
        double total = train.totalMillis() + validation.totalMillis();
        if (train.hasTotal() || validation.hasTotal()) {
            view.put("totalMillis", total);
        }
        ScopeSummary dominant = train.totalMillis() >= validation.totalMillis() ? train : validation;
        if (dominant.hasTotal()) {
            view.put("dominantScope", dominant.name());
            view.put("dominantScopeTotalMillis", dominant.totalMillis());
            if (total > 0.0) {
                view.put("dominantScopePercent", dominant.totalMillis() * 100.0 / total);
            }
        }
        if (train.hasTotal() && validation.hasTotal() && validation.totalMillis() > 0.0) {
            view.put("trainToValidationTotalRatio", train.totalMillis() / validation.totalMillis());
        }
    }

    private static void putStage(Map<String, Object> evidence, String prefix, Stage stage) {
        if (!stage.available()) {
            return;
        }
        stage.count().ifPresent(value -> evidence.put(prefix + "Count", value));
        stage.totalMillis().ifPresent(value -> evidence.put(prefix + "TotalMillis", value));
        stage.averageMillis().ifPresent(value -> evidence.put(prefix + "AverageMillis", value));
        stage.maxMillis().ifPresent(value -> evidence.put(prefix + "MaxMillis", value));
    }

    private static void putTotal(Map<String, Object> evidence, String key, Stage first, Stage second, Stage third) {
        double total = 0.0;
        boolean present = false;
        for (Stage stage : java.util.List.of(first, second, third)) {
            if (stage.totalMillis().isPresent()) {
                total += stage.totalMillis().orElseThrow();
                present = true;
            }
        }
        if (present) {
            evidence.put(key, total);
        }
    }

    record Stage(
            OptionalLong count,
            OptionalDouble totalMillis,
            OptionalDouble averageMillis,
            OptionalDouble maxMillis) {
        Stage {
            count = count == null ? OptionalLong.empty() : count;
            totalMillis = totalMillis == null ? OptionalDouble.empty() : totalMillis;
            averageMillis = averageMillis == null ? OptionalDouble.empty() : averageMillis;
            maxMillis = maxMillis == null ? OptionalDouble.empty() : maxMillis;
        }

        static Stage fromMetadata(Map<String, ?> metadata, String prefix) {
            return new Stage(
                    optionalLong(metadata.get(prefix + ".count")),
                    optionalDouble(metadata.get(prefix + ".totalMillis")),
                    optionalDouble(metadata.get(prefix + ".averageMillis")),
                    optionalDouble(metadata.get(prefix + ".maxMillis")));
        }

        static Stage empty() {
            return new Stage(OptionalLong.empty(), OptionalDouble.empty(), OptionalDouble.empty(), OptionalDouble.empty());
        }

        boolean available() {
            return count.isPresent()
                    || totalMillis.isPresent()
                    || averageMillis.isPresent()
                    || maxMillis.isPresent();
        }

        Map<String, Object> toMap() {
            Map<String, Object> view = new LinkedHashMap<>();
            view.put("available", available());
            count.ifPresent(value -> view.put("count", value));
            totalMillis.ifPresent(value -> view.put("totalMillis", value));
            averageMillis.ifPresent(value -> view.put("averageMillis", value));
            maxMillis.ifPresent(value -> view.put("maxMillis", value));
            return Map.copyOf(view);
        }
    }

    private record ScopeSummary(
            String name,
            Stage iterator,
            Stage hasNext,
            Stage next,
            boolean hasTotal,
            double totalMillis) {
        static ScopeSummary of(String name, Stage iterator, Stage hasNext, Stage next) {
            double total = 0.0;
            boolean present = false;
            for (Stage stage : java.util.List.of(iterator, hasNext, next)) {
                if (stage.totalMillis().isPresent()) {
                    total += stage.totalMillis().orElseThrow();
                    present = true;
                }
            }
            return new ScopeSummary(name, iterator, hasNext, next, present, total);
        }

        Map<String, Object> toMap() {
            Map<String, Object> scope = new LinkedHashMap<>();
            scope.put("iterator", iterator.toMap());
            scope.put("hasNext", hasNext.toMap());
            scope.put("next", next.toMap());
            if (hasTotal) {
                scope.put("totalMillis", totalMillis);
                putDominantStage(scope);
            }
            return Map.copyOf(scope);
        }

        private void putDominantStage(Map<String, Object> scope) {
            java.util.Optional<StageTotal> dominant = dominantStage();
            if (dominant.isEmpty()) {
                return;
            }
            StageTotal stage = dominant.orElseThrow();
            scope.put("dominantStage", stage.name());
            scope.put("dominantStageTotalMillis", stage.totalMillis());
            if (totalMillis > 0.0) {
                scope.put("dominantStagePercent", stage.totalMillis() * 100.0 / totalMillis);
            }
        }

        java.util.Optional<StageTotal> dominantStage() {
            StageTotal dominant = StageTotal.dominant(
                    new StageTotal("iterator", iterator),
                    new StageTotal("hasNext", hasNext),
                    new StageTotal("next", next));
            return dominant.available() ? java.util.Optional.of(dominant) : java.util.Optional.empty();
        }
    }

    private record StageTotal(String name, Stage stage) {
        static StageTotal dominant(StageTotal first, StageTotal second, StageTotal third) {
            StageTotal dominant = first;
            if (second.totalMillis() > dominant.totalMillis()) {
                dominant = second;
            }
            if (third.totalMillis() > dominant.totalMillis()) {
                dominant = third;
            }
            return dominant;
        }

        boolean available() {
            return stage.totalMillis().isPresent();
        }

        double totalMillis() {
            return stage.totalMillis().orElse(0.0);
        }
    }

    record BottleneckSummary(
            String scope,
            double scopeTotalMillis,
            java.util.OptionalDouble scopePercent,
            String stage,
            double stageTotalMillis,
            java.util.OptionalDouble stagePercent) {
        BottleneckSummary {
            scope = scope == null ? "" : scope;
            stage = stage == null ? "" : stage;
            scopePercent = scopePercent == null ? java.util.OptionalDouble.empty() : scopePercent;
            stagePercent = stagePercent == null ? java.util.OptionalDouble.empty() : stagePercent;
        }
    }
}
