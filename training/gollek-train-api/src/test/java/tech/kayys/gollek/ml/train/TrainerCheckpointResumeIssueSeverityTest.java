package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class TrainerCheckpointResumeIssueSeverityTest {

    @Test
    void exposesStablePublishedValuesAndBlockingSemantics() {
        assertEquals("warning", TrainerCheckpointResumeIssueSeverity.WARNING.value());
        assertFalse(TrainerCheckpointResumeIssueSeverity.WARNING.blocking());

        assertEquals("error", TrainerCheckpointResumeIssueSeverity.ERROR.value());
        assertTrue(TrainerCheckpointResumeIssueSeverity.ERROR.blocking());
    }
}
