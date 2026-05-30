package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A typed, actionable recommendation derived from a training report diagnostic.
 */
public record TrainingReportRecommendation(
        Priority priority,
        Category category,
        TrainingReportDiagnostics.Severity diagnosticSeverity,
        String diagnosticCode,
        String title,
        String rationale,
        List<String> actions,
        Map<String, Object> evidence) {
    public enum Priority {
        LOW,
        MEDIUM,
        HIGH,
        BLOCKER
    }

    public enum Category {
        RUN_HEALTH,
        DATA_HEALTH,
        REPORTING,
        VALIDATION,
        GENERALIZATION,
        TRAINING_DYNAMICS,
        LEARNING_RATE,
        OPTIMIZATION
    }

    public TrainingReportRecommendation {
        priority = Objects.requireNonNull(priority, "priority must not be null");
        category = Objects.requireNonNull(category, "category must not be null");
        diagnosticSeverity = Objects.requireNonNull(diagnosticSeverity, "diagnosticSeverity must not be null");
        diagnosticCode = normalizeRequired(diagnosticCode, "diagnosticCode");
        title = normalizeRequired(title, "title");
        rationale = normalizeRequired(rationale, "rationale");
        actions = immutableActions(actions);
        evidence = immutableEvidence(evidence);
    }

    public boolean blocksPromotion() {
        return priority == Priority.BLOCKER;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("priority", priority.name());
        map.put("category", category.name());
        map.put("diagnosticSeverity", diagnosticSeverity.name());
        map.put("diagnosticCode", diagnosticCode);
        map.put("title", title);
        map.put("rationale", rationale);
        map.put("actions", actions);
        map.put("evidence", evidence);
        return Map.copyOf(map);
    }

    static TrainingReportRecommendation fromFinding(TrainingReportDiagnostics.Finding finding) {
        Objects.requireNonNull(finding, "finding must not be null");
        Template template = template(finding.code());
        return new TrainingReportRecommendation(
                template.priority(),
                template.category(),
                finding.severity(),
                finding.code(),
                template.title(),
                finding.message(),
                template.actions(),
                finding.evidence());
    }

    private static Template template(String diagnosticCode) {
        String code = diagnosticCode == null ? "" : diagnosticCode.trim();
        return switch (code) {
            case "run_health.gate_failed" -> new Template(
                    Priority.BLOCKER,
                    Category.RUN_HEALTH,
                    "Fix the blocked training run before promotion",
                    List.of(
                            "Inspect the run health evidence and the primary blocking issue.",
                            "Apply the run health recommended action, then rerun training from a clean or validated checkpoint.",
                            "Do not promote this report until the run health gate passes."));
            case "run_health.issue_detected" -> new Template(
                    Priority.HIGH,
                    Category.RUN_HEALTH,
                    "Review training run health warnings",
                    List.of(
                            "Inspect the run health issue list before comparing this model against baselines.",
                            "Apply the recommended action if the issue affects model quality or reproducibility.",
                            "Rerun the job once the warning is understood or intentionally accepted."));
            case "data_health.gate_failed" -> new Template(
                    Priority.BLOCKER,
                    Category.DATA_HEALTH,
                    "Fix blocked trainer data health before promotion",
                    List.of(
                            "Inspect the data-health evidence and primary data issue.",
                            "Repair the loader plan, sample coverage, label balance, or dataset split reported by the issue.",
                            "Do not promote this report until the data-health gate passes."));
            case "data_health.issue_detected" -> new Template(
                    Priority.HIGH,
                    Category.DATA_HEALTH,
                    "Review trainer data-health warnings",
                    List.of(
                            "Inspect loader-plan and distribution health before comparing this model against baselines.",
                            "Apply the data-health recommended action if it affects sample coverage, label balance, or reproducibility.",
                            "Rerun the job once the warning is understood or intentionally accepted."));
            case "history.missing" -> new Template(
                    Priority.LOW,
                    Category.REPORTING,
                    "Capture epoch history before judging training quality",
                    List.of(
                            "Enable canonical trainer history persistence for every epoch.",
                            "Rerun at least a short smoke training job so diagnostics have loss and metric context."));
            case "validation.missing" -> new Template(
                    Priority.MEDIUM,
                    Category.VALIDATION,
                    "Add validation data before trusting the run",
                    List.of(
                            "Provide a validation split or validation DataLoader for the trainer.",
                            "Track validation loss and task metrics so promotion gates can catch overfitting."));
            case "validation.loss_worsened" -> new Template(
                    Priority.HIGH,
                    Category.GENERALIZATION,
                    "Investigate worsening validation loss",
                    List.of(
                            "Try a lower learning rate or a scheduler with warmup/decay.",
                            "Enable early stopping and restore-best-model checkpoints.",
                            "Inspect data leakage, label noise, and train/validation distribution shift."));
            case "generalization.overfit_risk" -> new Template(
                    Priority.HIGH,
                    Category.GENERALIZATION,
                    "Reduce overfitting pressure",
                    List.of(
                            "Increase regularization such as weight decay, dropout, or data augmentation.",
                            "Prefer the best validation checkpoint instead of the final epoch.",
                            "Consider collecting more validation-like training data before promotion."));
            case "train.loss_plateau" -> new Template(
                    Priority.MEDIUM,
                    Category.TRAINING_DYNAMICS,
                    "Unblock stalled training loss",
                    List.of(
                            "Check that gradients flow through all expected parameters.",
                            "Try a learning-rate range test or a more adaptive optimizer.",
                            "Review model capacity and feature/label preprocessing."));
            case "learning_rate.too_small" -> new Template(
                    Priority.HIGH,
                    Category.LEARNING_RATE,
                    "Restore a useful learning rate",
                    List.of(
                            "Raise the optimizer learning rate or review scheduler decay limits.",
                            "Check resume state to make sure a previous run did not restore a near-zero rate.",
                            "Use a minimum learning-rate floor for long schedules."));
            case "learning_rate.spiked" -> new Template(
                    Priority.HIGH,
                    Category.LEARNING_RATE,
                    "Smooth learning-rate jumps",
                    List.of(
                            "Inspect scheduler state and warmup configuration around the reported epochs.",
                            "Lower the maximum learning rate or increase warmup length.",
                            "Verify checkpoint resume did not restore stale scheduler state."));
            case "learning_rate.flat_with_train_plateau" -> new Template(
                    Priority.MEDIUM,
                    Category.LEARNING_RATE,
                    "Adapt the learning rate during plateaus",
                    List.of(
                            "Add a scheduler such as cosine decay or reduce-on-plateau.",
                            "Run a short learning-rate sweep to find a productive range.",
                            "Consider switching optimizer or unfreezing parameters if loss remains flat."));
            case "optimization.non_finite_values" -> new Template(
                    Priority.BLOCKER,
                    Category.OPTIMIZATION,
                    "Stop and fix non-finite optimizer values",
                    List.of(
                            "Lower the learning rate and enable gradient clipping.",
                            "If mixed precision is enabled, verify loss scaling and overflow handling.",
                            "Normalize inputs and inspect batches around the reported epoch."));
            case "optimization.zero_gradient" -> new Template(
                    Priority.HIGH,
                    Category.OPTIMIZATION,
                    "Restore useful gradient flow",
                    List.of(
                            "Check frozen parameters, detached tensors, and loss graph connectivity.",
                            "Verify the loss function receives predictions and targets with compatible shapes.",
                            "Inspect optimizer parameter groups for missing trainable parameters."));
            case "optimization.no_parameter_update" -> new Template(
                    Priority.HIGH,
                    Category.OPTIMIZATION,
                    "Make optimizer steps update parameters",
                    List.of(
                            "Verify the optimizer step executes after backpropagation.",
                            "Check for zero learning rate, empty parameter groups, or disabled updates.",
                            "Confirm gradient accumulation calls step at the intended interval."));
            case "optimization.update_too_large" -> new Template(
                    Priority.HIGH,
                    Category.OPTIMIZATION,
                    "Tame oversized parameter updates",
                    List.of(
                            "Lower the learning rate or add warmup before full-rate training.",
                            "Enable norm or value gradient clipping.",
                            "Inspect optimizer hyperparameters such as momentum, betas, and weight decay."));
            default -> new Template(
                    priorityForUnknown(code),
                    Category.REPORTING,
                    "Review diagnostic " + (code.isBlank() ? "unknown" : code),
                    List.of(
                            "Inspect the diagnostic evidence and trainer configuration.",
                            "Add a targeted regression test once the root cause is understood."));
        };
    }

    private static Priority priorityForUnknown(String code) {
        if (code != null && code.startsWith("run_health.")) {
            return Priority.BLOCKER;
        }
        if (code != null && code.startsWith("data_health.")) {
            return Priority.HIGH;
        }
        if (code != null && code.startsWith("optimization.")) {
            return Priority.HIGH;
        }
        if (code != null && code.startsWith("comparison.")) {
            return Priority.HIGH;
        }
        return Priority.MEDIUM;
    }

    private static String normalizeRequired(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value.trim();
    }

    private static List<String> immutableActions(List<String> actions) {
        if (actions == null || actions.isEmpty()) {
            return List.of();
        }
        List<String> normalized = new ArrayList<>(actions.size());
        for (String action : actions) {
            if (action != null && !action.isBlank()) {
                normalized.add(action.trim());
            }
        }
        return List.copyOf(normalized);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableEvidence(Map<String, ?> evidence) {
        if (evidence == null || evidence.isEmpty()) {
            return Map.of();
        }
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(evidence);
        if (snapshot instanceof Map<?, ?> map) {
            return (Map<String, Object>) map;
        }
        return Map.of();
    }

    private record Template(
            Priority priority,
            Category category,
            String title,
            List<String> actions) {
    }
}
