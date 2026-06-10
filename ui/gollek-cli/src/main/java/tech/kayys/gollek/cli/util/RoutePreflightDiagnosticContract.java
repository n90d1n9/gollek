/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;

import tech.kayys.gollek.cli.util.RoutePreflightDiagnosticFields.Action;
import tech.kayys.gollek.cli.util.RoutePreflightDiagnosticFields.Problem;
import tech.kayys.gollek.cli.util.RoutePreflightDiagnosticFields.Schema;
import tech.kayys.gollek.cli.util.RoutePreflightDiagnosticFields.Validation;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight validator for route preflight problem and action entries.
 */
public final class RoutePreflightDiagnosticContract {
    private RoutePreflightDiagnosticContract() {
    }

    public static Map<String, Object> schema() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put(Schema.CONTRACT_ID, RoutePreflightDiagnosticFields.CONTRACT_ID);
        schema.put(Schema.SCHEMA_VERSION, RoutePreflightDiagnosticFields.SCHEMA_VERSION);
        schema.put(Schema.SCHEMA_FINGERPRINT, RoutePreflightDiagnosticFields.schemaFingerprint());
        schema.put(Schema.VALIDATION_ROOT, RoutePreflightDiagnosticFields.VALIDATION_ROOT);
        schema.put(Schema.PROBLEM_FIELDS, RoutePreflightDiagnosticFields.problemFields());
        schema.put(Schema.REQUIRED_PROBLEM_FIELDS, RoutePreflightDiagnosticFields.requiredProblemFields());
        schema.put(Schema.PROBLEM_DETAIL_FIELDS, RoutePreflightDiagnosticFields.problemDetailFields());
        schema.put(Schema.MISSING_RUNTIME_CAPABILITIES, RoutePreflightDiagnosticFields.missingRuntimeCapabilities());
        schema.put(Schema.ACTION_FIELDS, RoutePreflightDiagnosticFields.actionFields());
        schema.put(Schema.REQUIRED_ACTION_FIELDS, RoutePreflightDiagnosticFields.requiredActionFields());
        schema.put(Schema.ACTION_DETAIL_FIELDS, RoutePreflightDiagnosticFields.actionDetailFields());
        schema.put(Schema.VALIDATION_FIELDS, RoutePreflightDiagnosticFields.validationFields());
        schema.put(Schema.PROBLEM_CODES, RoutePreflightDiagnosticFields.problemCodes());
        schema.put(Schema.ACTION_KINDS, RoutePreflightDiagnosticFields.actionKinds());
        return schema;
    }

    public static List<String> validateSchema(Map<?, ?> schema) {
        if (schema == null) {
            return List.of("route preflight diagnostic schema is missing");
        }
        List<String> problems = new ArrayList<>();
        requireValue(problems, "schema." + Schema.CONTRACT_ID,
                schema.get(Schema.CONTRACT_ID), RoutePreflightDiagnosticFields.CONTRACT_ID);
        requireValue(problems, "schema." + Schema.SCHEMA_VERSION,
                schema.get(Schema.SCHEMA_VERSION), RoutePreflightDiagnosticFields.SCHEMA_VERSION);
        requireValue(problems, "schema." + Schema.SCHEMA_FINGERPRINT,
                schema.get(Schema.SCHEMA_FINGERPRINT), RoutePreflightDiagnosticFields.schemaFingerprint());
        requireValue(problems, "schema." + Schema.VALIDATION_ROOT,
                schema.get(Schema.VALIDATION_ROOT), RoutePreflightDiagnosticFields.VALIDATION_ROOT);
        requireList(problems, Schema.PROBLEM_FIELDS,
                schema.get(Schema.PROBLEM_FIELDS), RoutePreflightDiagnosticFields.problemFields());
        requireList(problems, Schema.REQUIRED_PROBLEM_FIELDS,
                schema.get(Schema.REQUIRED_PROBLEM_FIELDS), RoutePreflightDiagnosticFields.requiredProblemFields());
        requireList(problems, Schema.PROBLEM_DETAIL_FIELDS,
                schema.get(Schema.PROBLEM_DETAIL_FIELDS), RoutePreflightDiagnosticFields.problemDetailFields());
        requireList(problems, Schema.MISSING_RUNTIME_CAPABILITIES,
                schema.get(Schema.MISSING_RUNTIME_CAPABILITIES),
                RoutePreflightDiagnosticFields.missingRuntimeCapabilities());
        requireList(problems, Schema.ACTION_FIELDS,
                schema.get(Schema.ACTION_FIELDS), RoutePreflightDiagnosticFields.actionFields());
        requireList(problems, Schema.REQUIRED_ACTION_FIELDS,
                schema.get(Schema.REQUIRED_ACTION_FIELDS), RoutePreflightDiagnosticFields.requiredActionFields());
        requireList(problems, Schema.ACTION_DETAIL_FIELDS,
                schema.get(Schema.ACTION_DETAIL_FIELDS), RoutePreflightDiagnosticFields.actionDetailFields());
        requireList(problems, Schema.VALIDATION_FIELDS,
                schema.get(Schema.VALIDATION_FIELDS), RoutePreflightDiagnosticFields.validationFields());
        requireList(problems, Schema.PROBLEM_CODES,
                schema.get(Schema.PROBLEM_CODES), RoutePreflightDiagnosticFields.problemCodes());
        requireList(problems, Schema.ACTION_KINDS,
                schema.get(Schema.ACTION_KINDS), RoutePreflightDiagnosticFields.actionKinds());
        return List.copyOf(problems);
    }

    public static Map<String, Object> schemaValidationReport(Map<?, ?> schema) {
        return validationReport(validateSchema(schema));
    }

    public static Map<String, Object> diagnosticsValidationReport(Object problemsValue, Object actionsValue) {
        List<String> problems = new ArrayList<>();
        problems.addAll(validateProblems(problemsValue));
        problems.addAll(validateActions(actionsValue));
        return validationReport(problems);
    }

    public static List<String> validateValidationReport(Object value) {
        if (!(value instanceof Map<?, ?> validation)) {
            return List.of(RoutePreflightDiagnosticFields.VALIDATION_ROOT + " must be an object");
        }
        List<String> problems = new ArrayList<>();
        for (Object rawKey : validation.keySet()) {
            String key = String.valueOf(rawKey);
            if (!RoutePreflightDiagnosticFields.validationFields().contains(key)) {
                problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT
                        + " contains unknown field: " + key);
            }
        }
        requireValue(problems, RoutePreflightDiagnosticFields.VALIDATION_ROOT + "."
                        + Validation.CONTRACT_ID,
                validation.get(Validation.CONTRACT_ID), RoutePreflightDiagnosticFields.CONTRACT_ID);
        requireValue(problems, RoutePreflightDiagnosticFields.VALIDATION_ROOT + "."
                        + Validation.SCHEMA_VERSION,
                validation.get(Validation.SCHEMA_VERSION), RoutePreflightDiagnosticFields.SCHEMA_VERSION);
        requireValue(problems, RoutePreflightDiagnosticFields.VALIDATION_ROOT + "."
                        + Validation.SCHEMA_FINGERPRINT,
                validation.get(Validation.SCHEMA_FINGERPRINT), RoutePreflightDiagnosticFields.schemaFingerprint());
        requireBoolean(problems, Validation.PASSED, validation.get(Validation.PASSED));
        requireBoolean(problems, Validation.FAILED, validation.get(Validation.FAILED));
        requireInteger(problems, Validation.PROBLEM_COUNT, validation.get(Validation.PROBLEM_COUNT));
        requireListValue(problems, Validation.PROBLEMS, validation.get(Validation.PROBLEMS));
        requireValidationReportConsistency(
                problems,
                validation.get(Validation.PASSED),
                validation.get(Validation.FAILED),
                validation.get(Validation.PROBLEM_COUNT),
                validation.get(Validation.PROBLEMS));
        return List.copyOf(problems);
    }

    public static List<String> validateProblems(Object value) {
        List<String> problems = new ArrayList<>(validateEntryList(
                "preflight_problems",
                value,
                RoutePreflightDiagnosticFields.problemFields(),
                RoutePreflightDiagnosticFields.requiredProblemFields(),
                Problem.CODE,
                RoutePreflightDiagnosticFields.problemCodes()));
        if (!(value instanceof List<?> entries)) {
            return problems;
        }
        for (int index = 0; index < entries.size(); index++) {
            Object entry = entries.get(index);
            if (entry instanceof Map<?, ?> entryMap && entryMap.containsKey(Problem.DETAILS)) {
                validateDiagnosticDetails(
                        problems,
                        "preflight_problems[" + index + "]." + Problem.DETAILS,
                        entryMap.get(Problem.DETAILS),
                        RoutePreflightDiagnosticFields.problemDetailFields());
            }
        }
        return List.copyOf(problems);
    }

    public static List<String> validateActions(Object value) {
        List<String> problems = new ArrayList<>(validateEntryList(
                "next_actions",
                value,
                RoutePreflightDiagnosticFields.actionFields(),
                RoutePreflightDiagnosticFields.requiredActionFields(),
                Action.KIND,
                RoutePreflightDiagnosticFields.actionKinds()));
        if (!(value instanceof List<?> actions)) {
            return problems;
        }
        for (int index = 0; index < actions.size(); index++) {
            Object action = actions.get(index);
            if (action instanceof Map<?, ?> actionMap
                    && !(actionMap.get(Action.ARGV) instanceof List<?>)) {
                problems.add("next_actions[" + index + "]." + Action.ARGV + " must be a list");
            }
            if (action instanceof Map<?, ?> actionMap && actionMap.containsKey(Action.DETAILS)) {
                validateDiagnosticDetails(
                        problems,
                        "next_actions[" + index + "]." + Action.DETAILS,
                        actionMap.get(Action.DETAILS),
                        RoutePreflightDiagnosticFields.actionDetailFields());
            }
        }
        return List.copyOf(problems);
    }

    private static List<String> validateEntryList(
            String name,
            Object value,
            List<String> allowedFields,
            List<String> requiredFields,
            String discriminatorField,
            List<String> allowedDiscriminators) {
        if (!(value instanceof List<?> entries)) {
            return List.of(name + " must be a list");
        }
        List<String> problems = new ArrayList<>();
        for (int index = 0; index < entries.size(); index++) {
            Object entry = entries.get(index);
            String path = name + "[" + index + "]";
            if (!(entry instanceof Map<?, ?> entryMap)) {
                problems.add(path + " must be an object");
                continue;
            }
            validateEntry(problems, path, entryMap, allowedFields, requiredFields,
                    discriminatorField, allowedDiscriminators);
        }
        return List.copyOf(problems);
    }

    private static void validateEntry(
            List<String> problems,
            String path,
            Map<?, ?> entry,
            List<String> allowedFields,
            List<String> requiredFields,
            String discriminatorField,
            List<String> allowedDiscriminators) {
        for (Object rawKey : entry.keySet()) {
            String key = String.valueOf(rawKey);
            if (!allowedFields.contains(key)) {
                problems.add(path + " contains unknown field: " + key);
            }
        }
        for (String required : requiredFields) {
            if (!entry.containsKey(required)) {
                problems.add(path + " missing required field: " + required);
            }
        }
        Object discriminator = entry.get(discriminatorField);
        if (discriminator != null && !allowedDiscriminators.contains(String.valueOf(discriminator))) {
            problems.add(path + "." + discriminatorField + " has unknown value: " + discriminator);
        }
    }

    private static void validateDiagnosticDetails(
            List<String> problems,
            String path,
            Object value,
            List<String> allowedFields) {
        if (!(value instanceof Map<?, ?> details)) {
            problems.add(path + " must be an object");
            return;
        }
        for (Object rawKey : details.keySet()) {
            String key = String.valueOf(rawKey);
            if (!allowedFields.contains(key)) {
                problems.add(path + " contains unknown field: " + key);
            }
        }
        Object missingCapability = details.get(
                RoutePreflightDiagnosticFields.ProblemDetail.MISSING_RUNTIME_CAPABILITY);
        if (missingCapability != null
                && !RoutePreflightDiagnosticFields.missingRuntimeCapabilities()
                        .contains(String.valueOf(missingCapability))) {
            problems.add(path + "."
                    + RoutePreflightDiagnosticFields.ProblemDetail.MISSING_RUNTIME_CAPABILITY
                    + " has unknown value: " + missingCapability);
        }
    }

    private static Map<String, Object> validationReport(List<String> problems) {
        Map<String, Object> validation = new LinkedHashMap<>();
        validation.put(Validation.CONTRACT_ID, RoutePreflightDiagnosticFields.CONTRACT_ID);
        validation.put(Validation.SCHEMA_VERSION, RoutePreflightDiagnosticFields.SCHEMA_VERSION);
        validation.put(Validation.SCHEMA_FINGERPRINT, RoutePreflightDiagnosticFields.schemaFingerprint());
        validation.put(Validation.PASSED, problems.isEmpty());
        validation.put(Validation.FAILED, !problems.isEmpty());
        validation.put(Validation.PROBLEM_COUNT, problems.size());
        validation.put(Validation.PROBLEMS, problems);
        return validation;
    }

    private static void requireValue(
            List<String> problems,
            String path,
            Object actual,
            Object expected) {
        if (!Objects.equals(expected, actual)) {
            problems.add(path + " expected " + expected + " but was " + actual);
        }
    }

    private static void requireList(
            List<String> problems,
            String field,
            Object actual,
            List<String> expected) {
        String path = "schema." + field;
        if (!(actual instanceof List<?> values)) {
            problems.add(path + " must be a list");
            return;
        }
        List<String> normalized = values.stream()
                .map(String::valueOf)
                .toList();
        if (!Objects.equals(expected, normalized)) {
            problems.add(path + " expected " + expected + " but was " + normalized);
        }
    }

    private static void requireBoolean(List<String> problems, String field, Object actual) {
        if (!(actual instanceof Boolean)) {
            problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT + "." + field + " must be a boolean");
        }
    }

    private static void requireInteger(List<String> problems, String field, Object actual) {
        if (!(actual instanceof Number number) || number.intValue() != number.doubleValue()) {
            problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT + "." + field + " must be an integer");
        }
    }

    private static void requireListValue(List<String> problems, String field, Object actual) {
        if (!(actual instanceof List<?>)) {
            problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT + "." + field + " must be a list");
        }
    }

    private static void requireValidationReportConsistency(
            List<String> problems,
            Object passedValue,
            Object failedValue,
            Object countValue,
            Object problemsValue) {
        if (!(problemsValue instanceof List<?> validationProblems)) {
            return;
        }
        Integer count = validationInteger(countValue);
        int expectedCount = validationProblems.size();
        if (count != null && count != expectedCount) {
            problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT + "." + Validation.PROBLEM_COUNT
                    + " must match problems size: expected " + expectedCount + " but was " + count);
        }
        boolean expectedPassed = expectedCount == 0;
        requireValidationBooleanMatches(
                problems,
                Validation.PASSED,
                passedValue,
                expectedPassed);
        requireValidationBooleanMatches(
                problems,
                Validation.FAILED,
                failedValue,
                !expectedPassed);
    }

    private static void requireValidationBooleanMatches(
            List<String> problems,
            String field,
            Object actual,
            boolean expected) {
        if (actual instanceof Boolean value && value != expected) {
            problems.add(RoutePreflightDiagnosticFields.VALIDATION_ROOT + "." + field
                    + " must match problems emptiness: expected " + expected + " but was " + value);
        }
    }

    private static Integer validationInteger(Object actual) {
        if (!(actual instanceof Number number) || number.intValue() != number.doubleValue()) {
            return null;
        }
        return number.intValue();
    }
}
