package tech.kayys.gollek.cli.commands;

import tech.kayys.gollek.cli.util.RunnerRoutePolicyFields;

import java.util.Map;

/**
 * Command-package facade for the shared runner-selection policy contract.
 */
final class RunnerRoutePolicyContract {
    static final String CONTRACT_ID = RunnerRoutePolicyFields.CONTRACT_ID;
    static final int SCHEMA_VERSION = RunnerRoutePolicyFields.SCHEMA_VERSION;

    private RunnerRoutePolicyContract() {
    }

    static Map<String, Object> report() {
        return RunnerRoutePolicyFields.report();
    }

    static String schemaFingerprint() {
        return RunnerRoutePolicyFields.schemaFingerprint();
    }
}
