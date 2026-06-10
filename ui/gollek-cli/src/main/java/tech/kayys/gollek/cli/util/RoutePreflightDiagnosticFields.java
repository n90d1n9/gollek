/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;

/**
 * Stable JSON field names for route preflight problems and next actions.
 */
public final class RoutePreflightDiagnosticFields {
    public static final String CONTRACT_ID = "gollek.route-preflight.diagnostics";
    public static final int SCHEMA_VERSION = 7;
    public static final String VALIDATION_ROOT = "route_preflight_diagnostics_validation";

    private RoutePreflightDiagnosticFields() {
    }

    public static final class Schema {
        public static final String CONTRACT_ID = "contractId";
        public static final String SCHEMA_VERSION = "schemaVersion";
        public static final String SCHEMA_FINGERPRINT = "schemaFingerprint";
        public static final String VALIDATION_ROOT = "validationRoot";
        public static final String PROBLEM_FIELDS = "problemFields";
        public static final String REQUIRED_PROBLEM_FIELDS = "requiredProblemFields";
        public static final String PROBLEM_DETAIL_FIELDS = "problemDetailFields";
        public static final String MISSING_RUNTIME_CAPABILITIES = "missingRuntimeCapabilities";
        public static final String ACTION_FIELDS = "actionFields";
        public static final String REQUIRED_ACTION_FIELDS = "requiredActionFields";
        public static final String ACTION_DETAIL_FIELDS = "actionDetailFields";
        public static final String VALIDATION_FIELDS = "validationFields";
        public static final String PROBLEM_CODES = "problemCodes";
        public static final String ACTION_KINDS = "actionKinds";

        private Schema() {
        }
    }

    public static final class Validation {
        public static final String CONTRACT_ID = Schema.CONTRACT_ID;
        public static final String SCHEMA_VERSION = Schema.SCHEMA_VERSION;
        public static final String SCHEMA_FINGERPRINT = Schema.SCHEMA_FINGERPRINT;
        public static final String PASSED = "passed";
        public static final String FAILED = "failed";
        public static final String PROBLEM_COUNT = "problemCount";
        public static final String PROBLEMS = "problems";

        private Validation() {
        }
    }

    public static final class Problem {
        public static final String CODE = "code";
        public static final String SEVERITY = "severity";
        public static final String MESSAGE = "message";
        public static final String DETAILS = "details";

        private Problem() {
        }
    }

    /**
     * Stable optional fields for structured route preflight problem/action details.
     */
    public static final class ProblemDetail {
        public static final String MODEL_FAMILY = "modelFamily";
        public static final String RUNTIME_ROUTE = "runtimeRoute";
        public static final String CHECKPOINT_PROFILE = "checkpointProfile";
        public static final String REQUEST_INPUT_MODE = "requestInputMode";
        public static final String MISSING_RUNTIME_CAPABILITY = "missingRuntimeCapability";

        private ProblemDetail() {
        }
    }

    /**
     * Stable identifiers for runtime capabilities required to clear a preflight blocker.
     */
    public static final class MissingRuntimeCapability {
        public static final String GEMMA4_TEXT_SAFETENSOR_HEADERS = "gemma4_text_safetensor_headers";
        public static final String GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER = "gemma4_unified_multimodal_embedder";
        public static final String GEMMA4_PACKED_MOE_ROUTER = "gemma4_packed_moe_router";
        public static final String GEMMA4_MOBILE_QAT_LOADER = "gemma4_mobile_qat_loader";

        private MissingRuntimeCapability() {
        }
    }

    public static final class Action {
        public static final String KIND = "kind";
        public static final String REASON = "reason";
        public static final String DESCRIPTION = "description";
        public static final String ARGV = "argv";
        public static final String DETAILS = "details";

        private Action() {
        }
    }

    public static final class ProblemCode {
        public static final String MODEL_NOT_LOCAL = "model_not_local";
        public static final String PROVIDER_NOT_RESOLVED = "provider_not_resolved";
        public static final String FORMAT_NOT_RESOLVED = "format_not_resolved";
        public static final String DIRECT_ROUTE_VALIDATION_FAILED = "direct_route_validation_failed";
        public static final String GEMMA4_MULTIMODAL_RUNTIME_MISSING = "gemma4_multimodal_runtime_missing";
        public static final String GEMMA4_PACKED_MOE_RUNTIME_MISSING = "gemma4_packed_moe_runtime_missing";
        public static final String GEMMA4_TEXT_HEADER_MISMATCH = "gemma4_text_header_mismatch";
        public static final String GEMMA4_MOBILE_QAT_LOADER_MISSING = "gemma4_mobile_qat_loader_missing";

        private ProblemCode() {
        }
    }

    public static final class ActionKind {
        public static final String PULL_MODEL = "pull_model";
        public static final String ALLOW_PULL_RESOLUTION = "allow_pull_resolution";
        public static final String INSPECT_MODULES = "inspect_modules";

        private ActionKind() {
        }
    }

    public static List<String> problemFields() {
        return List.of(Problem.CODE, Problem.SEVERITY, Problem.MESSAGE, Problem.DETAILS);
    }

    public static List<String> requiredProblemFields() {
        return List.of(Problem.CODE, Problem.SEVERITY, Problem.MESSAGE);
    }

    public static List<String> problemDetailFields() {
        return List.of(
                ProblemDetail.MODEL_FAMILY,
                ProblemDetail.RUNTIME_ROUTE,
                ProblemDetail.CHECKPOINT_PROFILE,
                ProblemDetail.REQUEST_INPUT_MODE,
                ProblemDetail.MISSING_RUNTIME_CAPABILITY);
    }

    public static List<String> missingRuntimeCapabilities() {
        return List.of(
                MissingRuntimeCapability.GEMMA4_TEXT_SAFETENSOR_HEADERS,
                MissingRuntimeCapability.GEMMA4_UNIFIED_MULTIMODAL_EMBEDDER,
                MissingRuntimeCapability.GEMMA4_PACKED_MOE_ROUTER,
                MissingRuntimeCapability.GEMMA4_MOBILE_QAT_LOADER);
    }

    public static List<String> actionFields() {
        return List.of(Action.KIND, Action.REASON, Action.DESCRIPTION, Action.ARGV, Action.DETAILS);
    }

    public static List<String> requiredActionFields() {
        return List.of(Action.KIND, Action.REASON, Action.DESCRIPTION, Action.ARGV);
    }

    public static List<String> actionDetailFields() {
        return problemDetailFields();
    }

    public static List<String> validationFields() {
        return List.of(
                Validation.CONTRACT_ID,
                Validation.SCHEMA_VERSION,
                Validation.SCHEMA_FINGERPRINT,
                Validation.PASSED,
                Validation.FAILED,
                Validation.PROBLEM_COUNT,
                Validation.PROBLEMS);
    }

    public static List<String> problemCodes() {
        return List.of(
                ProblemCode.MODEL_NOT_LOCAL,
                ProblemCode.PROVIDER_NOT_RESOLVED,
                ProblemCode.FORMAT_NOT_RESOLVED,
                ProblemCode.DIRECT_ROUTE_VALIDATION_FAILED,
                ProblemCode.GEMMA4_MULTIMODAL_RUNTIME_MISSING,
                ProblemCode.GEMMA4_PACKED_MOE_RUNTIME_MISSING,
                ProblemCode.GEMMA4_TEXT_HEADER_MISMATCH,
                ProblemCode.GEMMA4_MOBILE_QAT_LOADER_MISSING);
    }

    public static List<String> actionKinds() {
        return List.of(
                ActionKind.PULL_MODEL,
                ActionKind.ALLOW_PULL_RESOLUTION,
                ActionKind.INSPECT_MODULES);
    }

    public static String schemaFingerprint() {
        String payload = String.join("\n",
                "contractId=" + CONTRACT_ID,
                "schemaVersion=" + SCHEMA_VERSION,
                "validationRoot=" + VALIDATION_ROOT,
                "problemFields=" + String.join(",", problemFields()),
                "requiredProblemFields=" + String.join(",", requiredProblemFields()),
                "problemDetailFields=" + String.join(",", problemDetailFields()),
                "missingRuntimeCapabilities=" + String.join(",", missingRuntimeCapabilities()),
                "actionFields=" + String.join(",", actionFields()),
                "requiredActionFields=" + String.join(",", requiredActionFields()),
                "actionDetailFields=" + String.join(",", actionDetailFields()),
                "validationFields=" + String.join(",", validationFields()),
                "problemCodes=" + String.join(",", problemCodes()),
                "actionKinds=" + String.join(",", actionKinds()));
        return "sha256:" + sha256(payload);
    }

    private static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder(digest.length * 2);
            for (byte raw : digest) {
                int valueByte = raw & 0xff;
                if (valueByte < 16) {
                    hex.append('0');
                }
                hex.append(Integer.toHexString(valueByte));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is required for route preflight schema fingerprints.", error);
        }
    }
}
