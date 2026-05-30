package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;

/**
 * Applies derived diagnostics to epoch-history rows.
 */
final class TrainerEpochHistoryDiagnostics {
    private TrainerEpochHistoryDiagnostics() {
    }

    static int epochOf(Map<String, Object> row) {
        Object epoch = row.get("epoch");
        return epoch instanceof Number number ? number.intValue() : Integer.MAX_VALUE;
    }

    static void putTrain(Map<String, Object> row, List<Map<String, Object>> rows, int epoch) {
        Map<String, Object> previousRow = previousRow(rows, epoch);
        putTrainLoss(row, rows, previousRow);
        TrainerGeneralizationMetadata.putEpoch(row, previousRow);
    }

    static void putValidation(Map<String, Object> row, List<Map<String, Object>> rows, int epoch) {
        Map<String, Object> previousRow = previousRow(rows, epoch);
        putValidationLoss(row, rows, previousRow);
        TrainerGeneralizationMetadata.putEpoch(row, previousRow);
    }

    static void recompute(List<Map<String, Object>> rows) {
        for (Map<String, Object> row : rows) {
            Object rowEpoch = row.get("epoch");
            if (!(rowEpoch instanceof Number number)) {
                continue;
            }
            Map<String, Object> previousRow = previousRow(rows, number.intValue());
            if (row.containsKey("trainLoss")) {
                putTrainLoss(row, rows, previousRow);
            }
            if (row.containsKey("validationLoss")) {
                putValidationLoss(row, rows, previousRow);
            }
            TrainerGeneralizationMetadata.putEpoch(row, previousRow);
        }
    }

    private static void putTrainLoss(
            Map<String, Object> row,
            List<Map<String, Object>> rows,
            Map<String, Object> previousRow) {
        TrainerLossTrendMetadata.putEpoch(row, previousRow, "trainLoss", "trainLossDelta", "trainLossImproved");
        TrainerLossImprovementMetadata.putEpoch(row, rows, "trainLoss");
        TrainerLossSlopeMetadata.putEpoch(row, rows, "trainLoss");
        TrainerLossWindowStatsMetadata.putEpoch(row, rows, "trainLoss");
    }

    private static void putValidationLoss(
            Map<String, Object> row,
            List<Map<String, Object>> rows,
            Map<String, Object> previousRow) {
        TrainerLossTrendMetadata.putEpoch(
                row,
                previousRow,
                "validationLoss",
                "validationLossDelta",
                "validationLossImproved");
        TrainerLossImprovementMetadata.putEpoch(row, rows, "validationLoss");
        TrainerLossSlopeMetadata.putEpoch(row, rows, "validationLoss");
        TrainerLossWindowStatsMetadata.putEpoch(row, rows, "validationLoss");
        TrainerValidationProgressMetadata.putEpoch(row, rows);
        TrainerBestModelMonitorProgressMetadata.putEpoch(row, rows);
    }

    private static Map<String, Object> previousRow(List<Map<String, Object>> rows, int epoch) {
        Map<String, Object> previous = null;
        int previousEpoch = Integer.MIN_VALUE;
        for (Map<String, Object> row : rows) {
            Object rowEpoch = row.get("epoch");
            if (rowEpoch instanceof Number number) {
                int candidateEpoch = number.intValue();
                if (candidateEpoch < epoch && candidateEpoch > previousEpoch) {
                    previous = row;
                    previousEpoch = candidateEpoch;
                }
            }
        }
        return previous;
    }
}
