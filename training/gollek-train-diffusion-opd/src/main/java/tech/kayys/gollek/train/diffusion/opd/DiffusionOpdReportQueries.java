package tech.kayys.gollek.train.diffusion.opd;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdGroupedRoundHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdPairedRoundHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistoryRow;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRoundHistorySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSectionView;

/**
 * Typed query helpers for normalized DiffusionOPD reports.
 */
public final class DiffusionOpdReportQueries {

    private DiffusionOpdReportQueries() {
    }

    public static List<DiffusionOpdRoundHistoryRow> roundHistoryRows(DiffusionOpdReport report) {
        Objects.requireNonNull(report, "report must not be null");
        return report.roundHistory().stream()
                .map(DiffusionOpdRoundHistoryRow::fromMap)
                .toList();
    }

    public static List<DiffusionOpdRoundHistoryRow> roundHistoryForTask(
            DiffusionOpdReport report,
            String taskId) {
        return roundHistoryRows(report).stream()
                .filter(row -> taskId.equals(row.taskId()))
                .toList();
    }

    public static List<DiffusionOpdRoundHistoryRow> roundHistoryForTeacher(
            DiffusionOpdReport report,
            String teacherKey) {
        return roundHistoryRows(report).stream()
                .filter(row -> teacherKey.equals(row.teacherKey()))
                .toList();
    }

    public static List<DiffusionOpdRoundHistoryRow> roundHistoryForStage(
            DiffusionOpdReport report,
            String stageName) {
        return roundHistoryRows(report).stream()
                .filter(row -> stageName.equals(row.stageName()))
                .toList();
    }

    public static DiffusionOpdRoundHistorySummary summarizeTaskHistory(
            DiffusionOpdReport report,
            String taskId) {
        return summarizeRows(roundHistoryForTask(report, taskId));
    }

    public static DiffusionOpdRoundHistorySummary summarizeTeacherHistory(
            DiffusionOpdReport report,
            String teacherKey) {
        return summarizeRows(roundHistoryForTeacher(report, teacherKey));
    }

    public static DiffusionOpdRoundHistorySummary summarizeStageHistory(
            DiffusionOpdReport report,
            String stageName) {
        return summarizeRows(roundHistoryForStage(report, stageName));
    }

    public static Double meanLossForTask(DiffusionOpdReport report, String taskId) {
        return summarizeTaskHistory(report, taskId).meanLoss();
    }

    public static Double meanLossForTeacher(DiffusionOpdReport report, String teacherKey) {
        return summarizeTeacherHistory(report, teacherKey).meanLoss();
    }

    public static Double meanLossForStage(DiffusionOpdReport report, String stageName) {
        return summarizeStageHistory(report, stageName).meanLoss();
    }

    public static Double maxLossForTask(DiffusionOpdReport report, String taskId) {
        return maxLoss(roundHistoryForTask(report, taskId));
    }

    public static Double maxLossForTeacher(DiffusionOpdReport report, String teacherKey) {
        return maxLoss(roundHistoryForTeacher(report, teacherKey));
    }

    public static Double maxLossForStage(DiffusionOpdReport report, String stageName) {
        return maxLoss(roundHistoryForStage(report, stageName));
    }

    public static Integer lastRoundForTask(DiffusionOpdReport report, String taskId) {
        return lastRound(roundHistoryForTask(report, taskId));
    }

    public static Integer lastRoundForTeacher(DiffusionOpdReport report, String teacherKey) {
        return lastRound(roundHistoryForTeacher(report, teacherKey));
    }

    public static Integer lastRoundForStage(DiffusionOpdReport report, String stageName) {
        return lastRound(roundHistoryForStage(report, stageName));
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> taskSummaries(
            DiffusionOpdReport report) {
        return groupedSummaries(roundHistoryRows(report), "taskId");
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> teacherSummaries(
            DiffusionOpdReport report) {
        return groupedSummaries(roundHistoryRows(report), "teacherKey");
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> stageSummaries(
            DiffusionOpdReport report) {
        return groupedSummaries(roundHistoryRows(report), "stageName");
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> topTaskSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topGroupedSummariesByMeanLoss(taskSummaries(report), limit);
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> taskSummariesForStage(
            DiffusionOpdReport report,
            String stageName) {
        return groupedSummaries(roundHistoryForStage(report, stageName), "taskId");
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> taskSummariesForTeacher(
            DiffusionOpdReport report,
            String teacherKey) {
        return groupedSummaries(roundHistoryForTeacher(report, teacherKey), "taskId");
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> topTaskSummariesForStageByMeanLoss(
            DiffusionOpdReport report,
            String stageName,
            int limit) {
        return topGroupedSummariesByMeanLoss(taskSummariesForStage(report, stageName), limit);
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> topTaskSummariesForTeacherByMeanLoss(
            DiffusionOpdReport report,
            String teacherKey,
            int limit) {
        return topGroupedSummariesByMeanLoss(taskSummariesForTeacher(report, teacherKey), limit);
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> topTeacherSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topGroupedSummariesByMeanLoss(teacherSummaries(report), limit);
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> teacherSummariesForTask(
            DiffusionOpdReport report,
            String taskId) {
        return groupedSummaries(roundHistoryForTask(report, taskId), "teacherKey");
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> teacherSummariesForStage(
            DiffusionOpdReport report,
            String stageName) {
        return groupedSummaries(roundHistoryForStage(report, stageName), "teacherKey");
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> topTeacherSummariesForTaskByMeanLoss(
            DiffusionOpdReport report,
            String taskId,
            int limit) {
        return topGroupedSummariesByMeanLoss(teacherSummariesForTask(report, taskId), limit);
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> topTeacherSummariesForStageByMeanLoss(
            DiffusionOpdReport report,
            String stageName,
            int limit) {
        return topGroupedSummariesByMeanLoss(teacherSummariesForStage(report, stageName), limit);
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> topStageSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topGroupedSummariesByMeanLoss(stageSummaries(report), limit);
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> stageSummariesForTask(
            DiffusionOpdReport report,
            String taskId) {
        return groupedSummaries(roundHistoryForTask(report, taskId), "stageName");
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> stageSummariesForTeacher(
            DiffusionOpdReport report,
            String teacherKey) {
        return groupedSummaries(roundHistoryForTeacher(report, teacherKey), "stageName");
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> topStageSummariesForTaskByMeanLoss(
            DiffusionOpdReport report,
            String taskId,
            int limit) {
        return topGroupedSummariesByMeanLoss(stageSummariesForTask(report, taskId), limit);
    }

    public static List<DiffusionOpdGroupedRoundHistorySummary> topStageSummariesForTeacherByMeanLoss(
            DiffusionOpdReport report,
            String teacherKey,
            int limit) {
        return topGroupedSummariesByMeanLoss(stageSummariesForTeacher(report, teacherKey), limit);
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> taskTeacherSummaries(
            DiffusionOpdReport report) {
        return pairedSummaries(roundHistoryRows(report), "taskId", "teacherKey");
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> taskStageSummaries(
            DiffusionOpdReport report) {
        return pairedSummaries(roundHistoryRows(report), "taskId", "stageName");
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> teacherStageSummaries(
            DiffusionOpdReport report) {
        return pairedSummaries(roundHistoryRows(report), "teacherKey", "stageName");
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> topTaskTeacherSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topPairedSummariesByMeanLoss(taskTeacherSummaries(report), limit);
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> taskTeacherSummariesForStage(
            DiffusionOpdReport report,
            String stageName) {
        return pairedSummaries(roundHistoryForStage(report, stageName), "taskId", "teacherKey");
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> topTaskTeacherSummariesForStageByMeanLoss(
            DiffusionOpdReport report,
            String stageName,
            int limit) {
        return topPairedSummariesByMeanLoss(taskTeacherSummariesForStage(report, stageName), limit);
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> topTaskStageSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topPairedSummariesByMeanLoss(taskStageSummaries(report), limit);
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> taskStageSummariesForTeacher(
            DiffusionOpdReport report,
            String teacherKey) {
        return pairedSummaries(roundHistoryForTeacher(report, teacherKey), "taskId", "stageName");
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> topTaskStageSummariesForTeacherByMeanLoss(
            DiffusionOpdReport report,
            String teacherKey,
            int limit) {
        return topPairedSummariesByMeanLoss(taskStageSummariesForTeacher(report, teacherKey), limit);
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> topTeacherStageSummariesByMeanLoss(
            DiffusionOpdReport report,
            int limit) {
        return topPairedSummariesByMeanLoss(teacherStageSummaries(report), limit);
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> teacherStageSummariesForTask(
            DiffusionOpdReport report,
            String taskId) {
        return pairedSummaries(roundHistoryForTask(report, taskId), "teacherKey", "stageName");
    }

    public static List<DiffusionOpdPairedRoundHistorySummary> topTeacherStageSummariesForTaskByMeanLoss(
            DiffusionOpdReport report,
            String taskId,
            int limit) {
        return topPairedSummariesByMeanLoss(teacherStageSummariesForTask(report, taskId), limit);
    }

    public static DiffusionOpdSectionView teachers(DiffusionOpdReport report) {
        return section("teachers", report.teachers());
    }

    public static DiffusionOpdSectionView stages(DiffusionOpdReport report) {
        return section("stages", report.stages());
    }

    public static DiffusionOpdSectionView tasks(DiffusionOpdReport report) {
        return section("tasks", report.tasks());
    }

    public static DiffusionOpdSectionView conditioning(DiffusionOpdReport report) {
        return section("conditioning", report.conditioning());
    }

    public static DiffusionOpdSectionView adaptive(DiffusionOpdReport report) {
        return section("adaptive", report.adaptive());
    }

    public static DiffusionOpdSectionView bindings(DiffusionOpdReport report) {
        return section("bindings", report.bindings());
    }

    private static DiffusionOpdSectionView section(String name, java.util.Map<String, Object> values) {
        return new DiffusionOpdSectionView(name, values);
    }

    private static Double maxLoss(List<DiffusionOpdRoundHistoryRow> rows) {
        return rows.stream()
                .map(DiffusionOpdRoundHistoryRow::averageLoss)
                .filter(Objects::nonNull)
                .max(Double::compareTo)
                .orElse(null);
    }

    private static Integer lastRound(List<DiffusionOpdRoundHistoryRow> rows) {
        return rows.stream()
                .map(DiffusionOpdRoundHistoryRow::round)
                .max(Integer::compareTo)
                .orElse(null);
    }

    private static List<DiffusionOpdGroupedRoundHistorySummary> topGroupedSummariesByMeanLoss(
            List<DiffusionOpdGroupedRoundHistorySummary> summaries,
            int limit) {
        return summaries.stream()
                .sorted(Comparator.comparing(
                        summary -> summary.summary().meanLoss(),
                        Comparator.nullsLast(Comparator.reverseOrder())))
                .limit(normalizedLimit(limit))
                .toList();
    }

    private static List<DiffusionOpdPairedRoundHistorySummary> topPairedSummariesByMeanLoss(
            List<DiffusionOpdPairedRoundHistorySummary> summaries,
            int limit) {
        return summaries.stream()
                .sorted(Comparator.comparing(
                        summary -> summary.summary().meanLoss(),
                        Comparator.nullsLast(Comparator.reverseOrder())))
                .limit(normalizedLimit(limit))
                .toList();
    }

    private static int normalizedLimit(int limit) {
        if (limit < 0) {
            throw new IllegalArgumentException("limit must be >= 0 but was " + limit);
        }
        return limit;
    }

    private static List<DiffusionOpdGroupedRoundHistorySummary> groupedSummaries(
            List<DiffusionOpdRoundHistoryRow> rows,
            String field) {
        LinkedHashMap<String, List<DiffusionOpdRoundHistoryRow>> grouped = new LinkedHashMap<>();
        for (DiffusionOpdRoundHistoryRow row : rows) {
            String value = switch (field) {
                case "taskId" -> row.taskId();
                case "teacherKey" -> row.teacherKey();
                case "stageName" -> row.stageName();
                default -> throw new IllegalArgumentException("Unsupported grouped field '" + field + "'.");
            };
            if (value == null || value.isBlank()) {
                continue;
            }
            grouped.computeIfAbsent(value, ignored -> new java.util.ArrayList<>()).add(row);
        }
        return grouped.entrySet().stream()
                .map(entry -> new DiffusionOpdGroupedRoundHistorySummary(
                        field,
                        entry.getKey(),
                        summarizeRows(entry.getValue())))
                .toList();
    }

    private static List<DiffusionOpdPairedRoundHistorySummary> pairedSummaries(
            List<DiffusionOpdRoundHistoryRow> rows,
            String firstField,
            String secondField) {
        LinkedHashMap<String, List<DiffusionOpdRoundHistoryRow>> grouped = new LinkedHashMap<>();
        Map<String, String[]> values = new LinkedHashMap<>();
        for (DiffusionOpdRoundHistoryRow row : rows) {
            String firstValue = fieldValue(row, firstField);
            String secondValue = fieldValue(row, secondField);
            if (firstValue == null || firstValue.isBlank() || secondValue == null || secondValue.isBlank()) {
                continue;
            }
            String pair = firstValue + "," + secondValue;
            grouped.computeIfAbsent(pair, ignored -> new java.util.ArrayList<>()).add(row);
            values.putIfAbsent(pair, new String[] {firstValue, secondValue});
        }
        return grouped.entrySet().stream()
                .map(entry -> {
                    String pair = entry.getKey();
                    String[] pairValues = values.get(pair);
                    return new DiffusionOpdPairedRoundHistorySummary(
                            firstField,
                            pairValues[0],
                            secondField,
                            pairValues[1],
                            pair,
                            summarizeRows(entry.getValue()));
                })
                .toList();
    }

    private static String fieldValue(DiffusionOpdRoundHistoryRow row, String field) {
        return switch (field) {
            case "taskId" -> row.taskId();
            case "teacherKey" -> row.teacherKey();
            case "stageName" -> row.stageName();
            default -> throw new IllegalArgumentException("Unsupported paired field '" + field + "'.");
        };
    }

    private static DiffusionOpdRoundHistorySummary summarizeRows(
            List<DiffusionOpdRoundHistoryRow> rows) {
        OptionalDouble meanLoss = rows.stream()
                .map(DiffusionOpdRoundHistoryRow::averageLoss)
                .filter(Objects::nonNull)
                .mapToDouble(Double::doubleValue)
                .average();
        DiffusionOpdRoundHistoryRow last = rows.stream()
                .max(Comparator.comparingInt(DiffusionOpdRoundHistoryRow::round))
                .orElse(null);
        List<DiffusionOpdRoundHistoryRow> topLosses = rows.stream()
                .sorted(Comparator.comparing(
                        DiffusionOpdRoundHistoryRow::averageLoss,
                        Comparator.nullsLast(Comparator.reverseOrder())))
                .limit(3)
                .toList();
        List<DiffusionOpdRoundHistoryRow> firstRounds = rows.stream()
                .sorted(Comparator.comparingInt(DiffusionOpdRoundHistoryRow::round))
                .limit(3)
                .toList();
        return new DiffusionOpdRoundHistorySummary(
                rows.size(),
                meanLoss.isPresent() ? meanLoss.getAsDouble() : null,
                last,
                topLosses,
                firstRounds);
    }
}
