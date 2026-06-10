package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class TrainingReportPromotionGateMapHelpersTest {
    @Test
    void mapValuesExposeTypedImmutableSnapshots() {
        Map<String, Object> nested = new LinkedHashMap<>();
        nested.put("name", "artifact");
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("text", "present");
        values.put("flag", "true");
        values.put("items", List.of("one", "two"));
        values.put("nested", nested);

        assertEquals("present", TrainingReportPromotionGateMapValues.stringValue(values, "text").orElseThrow());
        assertTrue(TrainingReportPromotionGateMapValues.booleanValue(values, "flag").orElseThrow());
        assertEquals(List.of("one", "two"),
                TrainingReportPromotionGateMapValues.iterableValue(values, "items").orElseThrow());

        Map<String, Object> snapshot =
                TrainingReportPromotionGateMapValues.objectValue(values, "nested").orElseThrow();
        assertEquals("artifact", snapshot.get("name"));
        assertThrows(UnsupportedOperationException.class, () -> snapshot.put("name", "mutated"));
    }

    @Test
    void pathValueNormalizesAndRejectsInvalidPaths() {
        Map<String, Object> values = Map.of(
                "relative", "build/../build/report.json",
                "invalid", "\0");

        Path path = TrainingReportPromotionGateMapValues.pathValue(values, "relative").orElseThrow();
        assertTrue(path.isAbsolute());
        assertTrue(path.endsWith(Path.of("build", "report.json")));
        assertTrue(TrainingReportPromotionGateMapValues.pathValue(values, "invalid").isEmpty());
        assertTrue(TrainingReportPromotionGateMapValues.pathValue(values, "missing").isEmpty());
    }

    @Test
    void requirementsRecordPreciseMissingFieldMessages() {
        Map<String, Object> values = Map.of(
                "name", "package",
                "flag", Boolean.TRUE,
                "items", List.of("a"),
                "object", Map.of("nested", "value"));
        List<String> failures = new ArrayList<>();

        TrainingReportPromotionGateMapRequirements.requireString(values, "missingText", "owner", failures);
        TrainingReportPromotionGateMapRequirements.requireBoolean(values, "missingFlag", "owner", failures);
        TrainingReportPromotionGateMapRequirements.requireIterable(values, "missingItems", "owner", failures);
        TrainingReportPromotionGateMapRequirements.requireObject(values, "missingObject", "owner", failures);
        TrainingReportPromotionGateMapRequirements.requireString(values, "name", "owner", failures);
        TrainingReportPromotionGateMapRequirements.requireBoolean(values, "flag", "owner", failures);
        TrainingReportPromotionGateMapRequirements.requireIterable(values, "items", "owner", failures);
        TrainingReportPromotionGateMapRequirements.requireObject(values, "object", "owner", failures);

        assertEquals(List.of(
                "owner is missing string field missingText",
                "owner is missing boolean field missingFlag",
                "owner is missing array field missingItems",
                "owner is missing object field missingObject"), failures);
    }

    @Test
    void booleanValueRejectsNonBooleanText() {
        Map<String, Object> values = Map.of(
                "yes", "true",
                "no", "false",
                "invalid", "not-bool");

        assertTrue(TrainingReportPromotionGateMapValues.booleanValue(values, "yes").orElseThrow());
        assertFalse(TrainingReportPromotionGateMapValues.booleanValue(values, "no").orElseThrow());
        assertTrue(TrainingReportPromotionGateMapValues.booleanValue(values, "invalid").isEmpty());
    }
}
