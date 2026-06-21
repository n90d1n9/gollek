/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.gollek.cli.util.ModelFamilyBundleGateReportFields.Gate;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Builds the reusable model-family bundle gate section for CI reports.
 */
final class ModelFamilyBundleGateReports {
    private ModelFamilyBundleGateReports() {
    }

    static Map<String, Object> gate(ModelFamilyBundleGate gate) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put(Gate.PASSED, gate.passed());
        report.put(Gate.FAILED, gate.failed());
        report.put(Gate.STATUS, gate.status());
        report.put(Gate.VIOLATION_COUNT, gate.violationCount());
        report.put(Gate.VIOLATIONS, gate.violations());
        report.put(Gate.CONTRACT_CATEGORY_COUNTS, gate.contractCategoryCounts());
        report.put(Gate.CONTRACT_REMEDIATION_HINTS, gate.contractRemediationHints());
        return report;
    }
}
