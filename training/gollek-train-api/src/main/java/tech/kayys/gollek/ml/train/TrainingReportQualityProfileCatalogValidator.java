package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Preflight validator for custom trainer quality-profile catalogs before CI gates consume them.
 */
public final class TrainingReportQualityProfileCatalogValidator {
    private static final Set<String> CATALOG_KEYS = Set.of("format", "profileCount", "profiles");
    private static final Set<String> PROFILE_KEYS = Set.of(
            "id",
            "displayName",
            "description",
            "validationPolicy",
            "performancePolicy",
            "promotionPolicy");
    private static final Set<String> VALIDATION_POLICY_KEYS = Set.of(
            "maxDiagnosticSeverity",
            "requireRunHealthGate",
            "requireDataHealthGate",
            "requireDataHealthAvailable",
            "requireFreshDiagnostics",
            "requireValidation",
            "requireCheckpointIntegrity");
    private static final Set<String> PERFORMANCE_POLICY_KEYS = Set.of(
            "failOnAcceleratorFallback",
            "minTrainSamplesPerSecond",
            "maxValidationToTrainAverageBatchMillisRatio");
    private static final Set<String> PROMOTION_POLICY_KEYS = Set.of(
            "maxCandidateDiagnosticSeverity",
            "maxComparisonFindingSeverity",
            "minimumValidationImprovement",
            "requireTrackedMetricImprovement",
            "requireCandidateDataHealthAvailable",
            "requireCandidateDataHealthGate",
            "requireCandidateDataHealthClean");

    private TrainingReportQualityProfileCatalogValidator() {
    }

    public record Issue(String severity, String code, String path, String message, String action) {
        public Issue {
            severity = severity == null || severity.isBlank() ? "error" : severity.trim();
            code = code == null || code.isBlank() ? "catalog.invalid" : code.trim();
            path = path == null || path.isBlank() ? "$" : path.trim();
            message = message == null ? "" : message.trim();
            action = action == null ? "" : action.trim();
        }

        public boolean error() {
            return "error".equalsIgnoreCase(severity);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("severity", severity);
            map.put("code", code);
            map.put("path", path);
            map.put("message", message);
            map.put("action", action);
            return Map.copyOf(map);
        }
    }

    public record Result(
            boolean validJson,
            int profileCount,
            List<String> profileIds,
            List<Issue> issues) {
        public Result {
            profileIds = profileIds == null ? List.of() : List.copyOf(profileIds);
            issues = issues == null ? List.of() : List.copyOf(issues);
        }

        public boolean passed() {
            return validJson && issues.stream().noneMatch(Issue::error);
        }

        public List<Issue> errors() {
            return issues.stream().filter(Issue::error).toList();
        }

        public List<Issue> warnings() {
            return issues.stream().filter(issue -> !issue.error()).toList();
        }

        public String message() {
            if (passed()) {
                return "Training report quality profile catalog passed validation.";
            }
            return "Training report quality profile catalog has " + errors().size()
                    + " error(s) and " + warnings().size() + " warning(s).";
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("validJson", validJson);
            map.put("passed", passed());
            map.put("message", message());
            map.put("profileCount", profileCount);
            map.put("profileIds", profileIds);
            map.put("errorCount", errors().size());
            map.put("warningCount", warnings().size());
            map.put("issues", issues.stream().map(Issue::toMap).toList());
            return Map.copyOf(map);
        }

        public String markdown() {
            return TrainingReportQualityProfileCatalogValidatorMarkdown.render(this);
        }

        public String junitXml() {
            return TrainingReportQualityProfileCatalogValidatorJUnitXml.render(this);
        }

        public static Result fromMap(Map<String, ?> map) {
            Objects.requireNonNull(map, "map must not be null");
            List<String> ids = new ArrayList<>();
            Object idsValue = map.get("profileIds");
            if (idsValue instanceof Iterable<?> iterable) {
                for (Object item : iterable) {
                    if (item != null) {
                        ids.add(String.valueOf(item));
                    }
                }
            }
            List<Issue> parsedIssues = new ArrayList<>();
            Object issuesValue = map.get("issues");
            if (issuesValue instanceof Iterable<?> iterable) {
                for (Object item : iterable) {
                    if (item instanceof Map<?, ?> issueMap) {
                        parsedIssues.add(issueFromMap(TrainingReportMapValues.immutableMap(issueMap)));
                    }
                }
            }
            return new Result(
                    TrainingReportValues.booleanValue(map.get("validJson")),
                    TrainingReportValues.intValue(map.get("profileCount"), ids.size()),
                    ids,
                    parsedIssues);
        }
    }

    public static Result validate(TrainingReportQualityProfileCatalog catalog) {
        TrainingReportQualityProfileCatalog resolvedCatalog = catalog == null
                ? TrainingReportQualityProfileCatalog.defaults()
                : catalog;
        return validate(resolvedCatalog.toMap());
    }

    public static Result validate(Map<String, ?> catalog) {
        List<Issue> issues = new ArrayList<>();
        if (catalog == null) {
            issues.add(error(
                    "$",
                    "catalog.missing",
                    "Quality profile catalog map is null.",
                    "Pass a catalog object with format and profiles fields."));
            return new Result(false, 0, List.of(), issues);
        }

        warnUnknownKeys(issues, "$", catalog, CATALOG_KEYS);
        validateFormat(issues, catalog.get("format"));
        List<?> profiles = profiles(issues, catalog.get("profiles"));
        validateProfileCount(issues, catalog.get("profileCount"), profiles);

        List<String> profileIds = new ArrayList<>();
        Set<String> seenIds = new LinkedHashSet<>();
        for (int index = 0; index < profiles.size(); index++) {
            Object item = profiles.get(index);
            String path = "$.profiles[" + index + "]";
            if (!(item instanceof Map<?, ?> profileMap)) {
                issues.add(error(
                        path,
                        "profile.not_object",
                        "Quality profile entry must be an object.",
                        "Replace this entry with a profile object containing id and policy fields."));
                continue;
            }
            Map<String, Object> profile = TrainingReportMapValues.immutableMap(profileMap);
            validateProfile(issues, profile, path, profileIds, seenIds);
        }

        return new Result(true, profileIds.size(), profileIds, issues);
    }

    public static Result validateJson(Path file) throws IOException {
        Path resolvedFile = Objects.requireNonNull(file, "file must not be null")
                .toAbsolutePath()
                .normalize();
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(Files.readString(resolvedFile, StandardCharsets.UTF_8));
        } catch (RuntimeException error) {
            return new Result(
                    false,
                    0,
                    List.of(),
                    List.of(error(
                            "$",
                            "catalog.json_invalid",
                            "Quality profile catalog JSON could not be parsed: " + error.getMessage(),
                            "Fix the JSON syntax before loading the catalog.")));
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            return new Result(
                    true,
                    0,
                    List.of(),
                    List.of(error(
                            "$",
                            "catalog.not_object",
                            "Quality profile catalog JSON must contain a top-level object.",
                            "Wrap the catalog fields in a JSON object.")));
        }
        return validate(TrainingReportMapValues.immutableMap(map));
    }

    private static void validateFormat(List<Issue> issues, Object format) {
        String text = String.valueOf(format);
        if (!TrainingReportQualityProfileCatalog.FORMAT.equals(text)) {
            issues.add(error(
                    "$.format",
                    "catalog.format_unsupported",
                    "Unsupported quality profile catalog format: " + text + ".",
                    "Use format `" + TrainingReportQualityProfileCatalog.FORMAT + "`."));
        }
    }

    private static List<?> profiles(List<Issue> issues, Object profilesValue) {
        if (!(profilesValue instanceof Iterable<?> iterable)) {
            issues.add(error(
                    "$.profiles",
                    "catalog.profiles_missing",
                    "Quality profile catalog must include a profiles array.",
                    "Add a non-empty profiles array."));
            return List.of();
        }
        List<Object> profiles = new ArrayList<>();
        for (Object item : iterable) {
            profiles.add(item);
        }
        if (profiles.isEmpty()) {
            issues.add(error(
                    "$.profiles",
                    "catalog.profiles_empty",
                    "Quality profile catalog must include at least one profile.",
                    "Add one or more quality profiles."));
        }
        return List.copyOf(profiles);
    }

    private static void validateProfileCount(List<Issue> issues, Object expectedCount, List<?> profiles) {
        if (expectedCount == null) {
            issues.add(warning(
                    "$.profileCount",
                    "catalog.profile_count_missing",
                    "Quality profile catalog does not declare profileCount.",
                    "Write profileCount so artifact diffs catch truncated catalogs."));
            return;
        }
        if (!(expectedCount instanceof Number number)) {
            issues.add(error(
                    "$.profileCount",
                    "catalog.profile_count_invalid",
                    "Quality profile catalog profileCount must be numeric.",
                    "Set profileCount to the profiles array length."));
            return;
        }
        if (number.intValue() != profiles.size()) {
            issues.add(error(
                    "$.profileCount",
                    "catalog.profile_count_mismatch",
                    "Quality profile catalog profileCount does not match profiles array length.",
                    "Update profileCount to " + profiles.size() + "."));
        }
    }

    private static void validateProfile(
            List<Issue> issues,
            Map<String, Object> profile,
            String path,
            List<String> profileIds,
            Set<String> seenIds) {
        warnUnknownKeys(issues, path, profile, PROFILE_KEYS);
        String id = TrainingReportValues.stringValue(profile.get("id"), "");
        if (id.isBlank()) {
            issues.add(error(
                    path + ".id",
                    "profile.id_missing",
                    "Quality profile id must not be blank.",
                    "Set a stable profile id such as strict-gpu-ci."));
        } else {
            String normalized = TrainingReportQualityProfile.normalizeId(id);
            profileIds.add(normalized);
            if (!seenIds.add(normalized)) {
                issues.add(error(
                        path + ".id",
                        "profile.id_duplicate",
                        "Quality profile id duplicates another profile after normalization: " + normalized + ".",
                        "Rename one profile so normalized ids are unique."));
            }
        }
        if (TrainingReportValues.stringValue(profile.get("description"), "").isBlank()) {
            issues.add(warning(
                    path + ".description",
                    "profile.description_missing",
                    "Quality profile description is blank.",
                    "Describe the workflow this profile is designed to guard."));
        }

        validatePolicyObject(issues, profile, path, "validationPolicy", VALIDATION_POLICY_KEYS);
        validatePolicyObject(issues, profile, path, "performancePolicy", PERFORMANCE_POLICY_KEYS);
        validatePolicyObject(issues, profile, path, "promotionPolicy", PROMOTION_POLICY_KEYS);

        try {
            TrainingReportQualityProfile.fromMap(profile);
        } catch (RuntimeException error) {
            issues.add(error(
                    path,
                    "profile.parse_failed",
                    "Quality profile cannot be parsed: " + error.getMessage(),
                    "Fix invalid policy values before using the catalog in CI gates."));
        }
    }

    private static void validatePolicyObject(
            List<Issue> issues,
            Map<String, Object> profile,
            String profilePath,
            String key,
            Set<String> knownKeys) {
        Object policy = profile.get(key);
        String path = profilePath + "." + key;
        if (policy == null) {
            issues.add(warning(
                    path,
                    "profile.policy_missing",
                    key + " is not declared and will use Gollek defaults.",
                    "Declare " + key + " explicitly so CI behavior is visible in review."));
            return;
        }
        if (!(policy instanceof Map<?, ?> policyMap)) {
            issues.add(error(
                    path,
                    "profile.policy_not_object",
                    key + " must be an object.",
                    "Replace " + key + " with an object containing policy fields."));
            return;
        }
        warnUnknownKeys(issues, path, TrainingReportMapValues.immutableMap(policyMap), knownKeys);
    }

    private static void warnUnknownKeys(
            List<Issue> issues,
            String path,
            Map<String, ?> map,
            Set<String> knownKeys) {
        for (String key : map.keySet()) {
            if (!knownKeys.contains(key)) {
                issues.add(warning(
                        path + "." + key,
                        "catalog.unknown_key",
                        "Unknown quality profile catalog key: " + key + ".",
                        "Remove the key or add parser support before relying on it."));
            }
        }
    }

    private static Issue error(String path, String code, String message, String action) {
        return new Issue("error", code, path, message, action);
    }

    private static Issue warning(String path, String code, String message, String action) {
        return new Issue("warning", code, path, message, action);
    }

    private static Issue issueFromMap(Map<String, ?> map) {
        return new Issue(
                TrainingReportValues.stringValue(map.get("severity"), "error"),
                TrainingReportValues.stringValue(map.get("code"), "catalog.invalid"),
                TrainingReportValues.stringValue(map.get("path"), "$"),
                TrainingReportValues.stringValue(map.get("message"), ""),
                TrainingReportValues.stringValue(map.get("action"), ""));
    }
}
