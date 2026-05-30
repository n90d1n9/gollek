package tech.kayys.gollek.ml.train;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Metrics for next-token language-modeling batches shaped [batch, sequence, vocabulary]. */
final class CausalLanguageModelingTrainingMetrics {
    static final float DEFAULT_IGNORE_INDEX = -100.0f;

    private CausalLanguageModelingTrainingMetrics() {
    }

    static Supplier<TrainingMetric> causalLanguageModelingTokenAccuracy() {
        return causalLanguageModelingTokenAccuracy(DEFAULT_IGNORE_INDEX);
    }

    static Supplier<TrainingMetric> causalLanguageModelingTokenAccuracy(float ignoreIndex) {
        float checkedIgnoreIndex = requireFiniteIgnoreIndex(ignoreIndex);
        return () -> new TokenAccuracyMetric(checkedIgnoreIndex);
    }

    static Supplier<TrainingMetric> causalLanguageModelingLogLoss() {
        return causalLanguageModelingLogLoss(DEFAULT_IGNORE_INDEX);
    }

    static Supplier<TrainingMetric> causalLanguageModelingLogLoss(float ignoreIndex) {
        float checkedIgnoreIndex = requireFiniteIgnoreIndex(ignoreIndex);
        return () -> new LogLossMetric(checkedIgnoreIndex);
    }

    static Supplier<TrainingMetric> causalLanguageModelingPerplexity() {
        return causalLanguageModelingPerplexity(DEFAULT_IGNORE_INDEX);
    }

    static Supplier<TrainingMetric> causalLanguageModelingPerplexity(float ignoreIndex) {
        float checkedIgnoreIndex = requireFiniteIgnoreIndex(ignoreIndex);
        return () -> new PerplexityMetric(checkedIgnoreIndex);
    }

    private abstract static class BaseMetric implements DetailedTrainingMetric {
        final float ignoreIndex;
        int vocabularySize = -1;
        long totalPositions;
        long validTokens;
        long ignoredTokens;

        BaseMetric(float ignoreIndex) {
            this.ignoreIndex = requireFiniteIgnoreIndex(ignoreIndex);
        }

        @Override
        public void reset() {
            vocabularySize = -1;
            totalPositions = 0L;
            validTokens = 0L;
            ignoredTokens = 0L;
            resetMetricState();
        }

        @Override
        public final void update(GradTensor predictions, GradTensor targets) {
            long[] predictionShape = predictions.shape();
            long[] targetShape = targets.shape();
            if (predictionShape.length != 3) {
                throw new IllegalArgumentException(
                        name() + " expects predictions shaped [batch, sequence, vocabulary], got "
                                + Arrays.toString(predictionShape));
            }
            int batch = Math.toIntExact(predictionShape[0]);
            int sequenceLength = Math.toIntExact(predictionShape[1]);
            int currentVocabularySize = Math.toIntExact(predictionShape[2]);
            if (batch < 0 || sequenceLength < 0 || currentVocabularySize <= 0) {
                throw new IllegalArgumentException(
                        name() + " expects non-negative batch/sequence and positive vocabulary size, got "
                                + Arrays.toString(predictionShape));
            }
            if (targetShape.length != 2 || targetShape[0] != batch || targetShape[1] != sequenceLength) {
                throw new IllegalArgumentException(
                        name() + " expects targets shaped [batch, sequence], got "
                                + Arrays.toString(targetShape) + " for predictions "
                                + Arrays.toString(predictionShape));
            }
            ensureVocabularySize(currentVocabularySize);

            float[] predictionData = predictions.data();
            float[] targetData = targets.data();
            for (int row = 0; row < batch; row++) {
                for (int position = 0; position < sequenceLength; position++) {
                    totalPositions++;
                    float target = targetData[row * sequenceLength + position];
                    if (Float.compare(target, ignoreIndex) == 0) {
                        ignoredTokens++;
                        continue;
                    }
                    int targetClass = targetClass(target, currentVocabularySize, row, position);
                    int offset = (row * sequenceLength + position) * currentVocabularySize;
                    validTokens++;
                    updateToken(predictionData, offset, currentVocabularySize, targetClass, row, position);
                }
            }
        }

        protected void resetMetricState() {
        }

        abstract void updateToken(
                float[] predictionData,
                int offset,
                int vocabularySize,
                int targetClass,
                int row,
                int position);

        final Map<String, Object> baseDetails(String type) {
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("type", type);
            details.put("task", "causal_language_modeling");
            details.put("vocabularySize", Math.max(0, vocabularySize));
            details.put("totalPositions", totalPositions);
            details.put("validTokens", validTokens);
            details.put("ignoredTokens", ignoredTokens);
            details.put("ignoreIndex", ignoreIndex);
            details.put("input", "logits");
            details.put("targetEncoding", "token_ids");
            return details;
        }

        private void ensureVocabularySize(int currentVocabularySize) {
            if (vocabularySize < 0) {
                vocabularySize = currentVocabularySize;
                return;
            }
            if (vocabularySize != currentVocabularySize) {
                throw new IllegalArgumentException(
                        name() + " expected vocabulary size " + vocabularySize
                                + " but got " + currentVocabularySize);
            }
        }

        private int targetClass(float target, int vocabularySize, int row, int position) {
            if (!Float.isFinite(target)) {
                throw new IllegalArgumentException(
                        name() + " expects finite token ids or ignoreIndex, got " + target
                                + " at batch " + row + ", position " + position);
            }
            if (target != Math.rint(target)) {
                throw new IllegalArgumentException(
                        name() + " expects integer token ids, got " + target
                                + " at batch " + row + ", position " + position);
            }
            int targetClass = Math.toIntExact((long) target);
            if (targetClass < 0 || targetClass >= vocabularySize) {
                throw new IllegalArgumentException(
                        name() + " target token " + targetClass + " out of range [0, "
                                + (vocabularySize - 1) + "] at batch " + row
                                + ", position " + position);
            }
            return targetClass;
        }
    }

    private static final class TokenAccuracyMetric extends BaseMetric {
        private long correctTokens;

        TokenAccuracyMetric(float ignoreIndex) {
            super(ignoreIndex);
        }

        @Override
        public String name() {
            return "causal_lm_token_accuracy";
        }

        @Override
        protected void resetMetricState() {
            correctTokens = 0L;
        }

        @Override
        void updateToken(
                float[] predictionData,
                int offset,
                int vocabularySize,
                int targetClass,
                int row,
                int position) {
            int bestClass = 0;
            float bestLogit = requireFiniteLogit(predictionData[offset], row, position, 0, name());
            for (int classIndex = 1; classIndex < vocabularySize; classIndex++) {
                float logit = requireFiniteLogit(
                        predictionData[offset + classIndex],
                        row,
                        position,
                        classIndex,
                        name());
                if (logit > bestLogit) {
                    bestClass = classIndex;
                    bestLogit = logit;
                }
            }
            if (bestClass == targetClass) {
                correctTokens++;
            }
        }

        @Override
        public double value() {
            return validTokens == 0L ? Double.NaN : correctTokens / (double) validTokens;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = baseDetails("causal_lm_token_accuracy");
            details.put("correctTokens", correctTokens);
            details.put("accuracy", validTokens == 0L ? null : value());
            return Collections.unmodifiableMap(details);
        }
    }

    private abstract static class NllMetric extends BaseMetric {
        double totalNegativeLogLikelihood;
        double correctTokenProbabilitySum;
        double maximumTokenLogLoss;

        NllMetric(float ignoreIndex) {
            super(ignoreIndex);
        }

        @Override
        protected void resetMetricState() {
            totalNegativeLogLikelihood = 0.0;
            correctTokenProbabilitySum = 0.0;
            maximumTokenLogLoss = 0.0;
        }

        @Override
        void updateToken(
                float[] predictionData,
                int offset,
                int vocabularySize,
                int targetClass,
                int row,
                int position) {
            double maxLogit = Double.NEGATIVE_INFINITY;
            for (int classIndex = 0; classIndex < vocabularySize; classIndex++) {
                float logit = requireFiniteLogit(
                        predictionData[offset + classIndex],
                        row,
                        position,
                        classIndex,
                        name());
                maxLogit = Math.max(maxLogit, logit);
            }

            double sumExp = 0.0;
            for (int classIndex = 0; classIndex < vocabularySize; classIndex++) {
                sumExp += Math.exp(predictionData[offset + classIndex] - maxLogit);
            }
            double logSumExp = maxLogit + Math.log(sumExp);
            double tokenLogLoss = logSumExp - predictionData[offset + targetClass];
            totalNegativeLogLikelihood += tokenLogLoss;
            correctTokenProbabilitySum += Math.exp(-tokenLogLoss);
            maximumTokenLogLoss = Math.max(maximumTokenLogLoss, tokenLogLoss);
        }

        final double logLoss() {
            return validTokens == 0L ? Double.NaN : totalNegativeLogLikelihood / validTokens;
        }

        final Map<String, Object> nllDetails(String type) {
            double logLoss = logLoss();
            Map<String, Object> details = baseDetails(type);
            details.put("logLoss", validTokens == 0L ? null : logLoss);
            details.put("crossEntropy", validTokens == 0L ? null : logLoss);
            details.put("negativeLogLikelihood", validTokens == 0L ? null : logLoss);
            details.put("perplexity", validTokens == 0L ? null : finiteExpOrNull(logLoss));
            details.put(
                    "meanCorrectTokenProbability",
                    validTokens == 0L ? null : correctTokenProbabilitySum / validTokens);
            details.put("maximumTokenLogLoss", validTokens == 0L ? null : maximumTokenLogLoss);
            return details;
        }
    }

    private static final class LogLossMetric extends NllMetric {
        LogLossMetric(float ignoreIndex) {
            super(ignoreIndex);
        }

        @Override
        public String name() {
            return "causal_lm_log_loss";
        }

        @Override
        public double value() {
            return logLoss();
        }

        @Override
        public Map<String, Object> details() {
            return Collections.unmodifiableMap(nllDetails("causal_lm_log_loss"));
        }
    }

    private static final class PerplexityMetric extends NllMetric {
        PerplexityMetric(float ignoreIndex) {
            super(ignoreIndex);
        }

        @Override
        public String name() {
            return "causal_lm_perplexity";
        }

        @Override
        public double value() {
            double logLoss = logLoss();
            return Double.isNaN(logLoss) ? Double.NaN : Math.exp(logLoss);
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> details = nllDetails("causal_lm_perplexity");
            details.put("valueMeaning", "exp(logLoss)");
            return Collections.unmodifiableMap(details);
        }
    }

    private static float requireFiniteLogit(
            float logit,
            int row,
            int position,
            int classIndex,
            String metricName) {
        if (!Float.isFinite(logit)) {
            throw new IllegalArgumentException(
                    metricName + " expects finite logits, got " + logit
                            + " at batch " + row + ", position " + position
                            + ", class " + classIndex);
        }
        return logit;
    }

    private static float requireFiniteIgnoreIndex(float ignoreIndex) {
        if (!Float.isFinite(ignoreIndex)) {
            throw new IllegalArgumentException("ignoreIndex must be finite, got: " + ignoreIndex);
        }
        return ignoreIndex;
    }

    private static Double finiteExpOrNull(double value) {
        double result = Math.exp(value);
        return Double.isFinite(result) ? result : null;
    }
}
