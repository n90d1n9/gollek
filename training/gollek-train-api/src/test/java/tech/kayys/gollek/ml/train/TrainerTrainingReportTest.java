package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainerTrainingReportTest {

    @Test
    void payloadPreservesCanonicalSchemaAndSummaryValues() {
        Map<String, Object> metadata = Map.of(
                "device", "metal",
                "metrics", List.of("accuracy", "f1"));
        TrainingSummary summary = new TrainingSummary(
                4,
                0.125,
                3,
                0.25,
                0.15,
                1234L,
                metadata);

        Map<String, Object> payload = TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-18T02:03:04Z"));

        assertEquals(TrainerTrainingReport.SCHEMA, payload.get("schema"));
        assertEquals("2026-05-18T02:03:04Z", payload.get("generatedAt"));
        assertEquals(4, payload.get("epochCount"));
        assertEquals(0.125, payload.get("bestValidationLoss"));
        assertEquals(3, payload.get("bestValidationEpoch"));
        assertEquals(0.25, payload.get("latestTrainLoss"));
        assertEquals(0.15, payload.get("latestValidationLoss"));
        assertEquals(1234L, payload.get("durationMs"));
        assertSame(metadata, payload.get("metadata"));
    }

    @Test
    void payloadSerializesWithStableSchemaField() {
        TrainingSummary summary = new TrainingSummary(
                1,
                Double.NaN,
                -1,
                0.5,
                null,
                10L,
                Map.of("checkpointManifestSaved", true));

        String json = TrainerJson.toJson(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-18T05:06:07Z")));

        assertEquals(
                "{\"bestValidationEpoch\":-1,\"bestValidationLoss\":null,\"durationMs\":10,"
                        + "\"epochCount\":1,\"generatedAt\":\"2026-05-18T05:06:07Z\","
                        + "\"latestTrainLoss\":0.5,\"latestValidationLoss\":null,"
                        + "\"metadata\":{\"checkpointManifestSaved\":true},"
                        + "\"schema\":\"gollek.canonical-trainer.report.v1\"}",
                json);
    }
}
