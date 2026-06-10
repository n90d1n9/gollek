package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class TrainerParameterUpdateDiagnosticsPolicyTest {
    @Test
    void disabledPolicyNeverCaptures() {
        TrainerParameterUpdateDiagnosticsPolicy policy = TrainerParameterUpdateDiagnosticsPolicy.disabled();

        assertFalse(policy.shouldCapture(1));
        assertFalse(policy.shouldCapture(100));
    }

    @Test
    void intervalPolicyCapturesOnlyMatchingPositiveSteps() {
        TrainerParameterUpdateDiagnosticsPolicy policy =
                TrainerParameterUpdateDiagnosticsPolicy.of(true, 3);

        assertFalse(policy.shouldCapture(0));
        assertFalse(policy.shouldCapture(1));
        assertFalse(policy.shouldCapture(2));
        assertTrue(policy.shouldCapture(3));
        assertFalse(policy.shouldCapture(4));
        assertTrue(policy.shouldCapture(6));
    }

    @Test
    void intervalIsClampedToEveryStep() {
        TrainerParameterUpdateDiagnosticsPolicy policy =
                TrainerParameterUpdateDiagnosticsPolicy.of(true, 0);

        assertTrue(policy.shouldCapture(1));
        assertTrue(policy.shouldCapture(2));
    }
}
