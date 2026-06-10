package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainingReportEvidenceSummaryTest {

    @Test
    void compactSortsKeysFiltersNullsAndLimitsFields() {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("zeta", 9);
        evidence.put("alpha", true);
        evidence.put("ignored", null);
        evidence.put("batchSize", 32);
        evidence.put("loader", "train");

        assertEquals(
                "alpha=true, batchSize=32, loader=train",
                TrainingReportEvidenceSummary.compact(evidence, 3));
    }

    @Test
    void compactReturnsBlankForMissingOrNonMapEvidence() {
        assertEquals("", TrainingReportEvidenceSummary.compact(null));
        assertEquals("", TrainingReportEvidenceSummary.compact("not-a-map"));
        assertEquals("", TrainingReportEvidenceSummary.compact(Map.of(), 3));
        assertEquals("", TrainingReportEvidenceSummary.compact(Map.of("a", 1), 0));
    }
}
