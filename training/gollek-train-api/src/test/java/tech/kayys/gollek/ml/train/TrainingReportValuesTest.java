package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainingReportValuesTest {
    @Test
    void coercesReportScalarsSafely() {
        assertEquals(0.25, TrainingReportValues.optionalDouble("0.25").orElseThrow(), 1e-12);
        assertEquals(7, TrainingReportValues.optionalInt("7").orElseThrow());
        assertEquals(11L, TrainingReportValues.optionalLong("11").orElseThrow());
        assertTrue(TrainingReportValues.booleanValue("true"));
        assertEquals("stable", TrainingReportValues.stringValue(" stable ", "fallback"));
        assertEquals("unknown", TrainingReportValues.normalizedString(" ", "unknown"));

        assertTrue(TrainingReportValues.optionalDouble(Double.NaN).isEmpty());
        assertTrue(TrainingReportValues.optionalDouble(Double.POSITIVE_INFINITY).isEmpty());
        assertTrue(TrainingReportValues.optionalDouble("not-a-number").isEmpty());
        assertTrue(TrainingReportValues.optionalInt("3.14").isEmpty());
        assertEquals(42, TrainingReportValues.intValue("bad", 42));
        assertEquals(99L, TrainingReportValues.longValue("bad", 99L));
        assertFalse(TrainingReportValues.booleanValue("not-true"));
    }

    @Test
    void exposesImmutableMapSections() {
        Map<String, Object> section = TrainingReportValues.mapValue(
                Map.of("optimization", Map.of("available", true, "count", 2)),
                "optimization");

        assertEquals(Boolean.TRUE, section.get("available"));
        assertEquals(2, section.get("count"));
        assertThrows(UnsupportedOperationException.class, () -> section.put("count", 3));
        assertEquals(Map.of(), TrainingReportValues.mapValue(Map.of(), "missing"));
    }
}
