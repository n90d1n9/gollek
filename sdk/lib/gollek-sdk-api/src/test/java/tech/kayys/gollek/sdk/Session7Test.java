package tech.kayys.gollek.sdk;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Linear;
import tech.kayys.gollek.ml.nn.ReLU;
import tech.kayys.gollek.ml.nn.Sequential;
import tech.kayys.gollek.ml.optimize.GradCheckpoint;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Session 7 tests: GollekClient unified API and GradCheckpoint.
 */
class Session7Test {

    // ── GollekClient ──────────────────────────────────────────────────────

    @Test
    void clientBuildsWithModel() {
        try (GollekClient client = GollekClient.builder()
                .model("model.gguf").build()) {
            assertNotNull(client);
            assertEquals("gguf", client.modelInfo().architecture());
        }
    }

    @Test
    void clientDetectsOnnxBackend() {
        try (GollekClient client = GollekClient.builder()
                .model("model.onnx").build()) {
            assertEquals("onnx", client.modelInfo().architecture());
        }
    }

    @Test
    void clientGenerateReturnsResult() {
        try (GollekClient client = GollekClient.builder()
                .model("model.gguf").build()) {
            GollekClient.GenerationResult r = client.generate("Hello");
            assertNotNull(r.text());
            assertTrue(r.tokenCount() > 0);
        }
    }

    @Test
    void clientGenerateBatch() {
        try (GollekClient client = GollekClient.builder()
                .model("model.gguf").build()) {
            List<GollekClient.GenerationResult> results =
                client.generateBatch(List.of("Hello", "World", "Test"));
            assertEquals(3, results.size());
        }
    }

    @Test
    void clientEmbedReturns768Dims() {
        try (GollekClient client = GollekClient.builder()
                .model("model.safetensors").build()) {
            float[] emb = client.embed("Hello world");
            assertEquals(768, emb.length);
        }
    }

    @Test
    void clientStreamingFuture() throws Exception {
        try (GollekClient client = GollekClient.builder()
                .model("model.gguf").build()) {
            CompletableFuture<GollekClient.GenerationResult> future =
                client.generateStream("Test prompt").toFuture();
            GollekClient.GenerationResult r = future.get(5, TimeUnit.SECONDS);
            assertNotNull(r);
        }
    }

    @Test
    void clientSupportsFeatures() {
        try (GollekClient client = GollekClient.builder()
                .model("model.gguf").build()) {
            assertTrue(client.supports(GollekClient.Feature.STREAMING));
            assertTrue(client.supports(GollekClient.Feature.BATCH_INFERENCE));
            assertTrue(client.supports(GollekClient.Feature.KV_CACHE));
        }
    }

    @Test
    void generationRequestDefaults() {
        var req = GollekClient.GenerationRequest.of("Hello");
        assertEquals("Hello", req.prompt());
        assertEquals(512, req.maxTokens());
        assertEquals(0.7f, req.temperature(), 1e-5f);
    }

    @Test
    void tokensPerSecondCalculation() {
        var r = new GollekClient.GenerationResult("text", 100, 10, 1000L);
        assertEquals(100f, r.tokensPerSecond(), 1e-3f);
    }

    // ── GradCheckpoint ────────────────────────────────────────────────────

    @Test
    void checkpointOutputShape() {
        Linear layer = new Linear(4, 4);
        GradTensor x = GradTensor.randn(2, 4).requiresGrad(true);
        GradTensor out = GradCheckpoint.checkpoint(layer, x);
        assertArrayEquals(new long[]{2, 4}, out.shape());
    }

    @Test
    void checkpointHasGradFn() {
        Linear layer = new Linear(4, 4);
        GradTensor x = GradTensor.randn(2, 4).requiresGrad(true);
        GradTensor out = GradCheckpoint.checkpoint(layer, x);
        assertTrue(out.requiresGrad(), "Checkpointed output must require grad");
        assertNotNull(out.gradFn(), "Checkpointed output must have grad_fn");
    }

    @Test
    void sequentialCheckpointOutputShape() {
        List<tech.kayys.gollek.ml.nn.Module> layers = List.of(
            new Linear(8, 8), new ReLU(),
            new Linear(8, 4), new ReLU()
        );
        GradTensor x = GradTensor.randn(2, 8).requiresGrad(true);
        GradTensor out = GradCheckpoint.sequentialCheckpoint(layers, x);
        assertArrayEquals(new long[]{2, 4}, out.shape());
    }

    @Test
    void checkpointSupplierForm() {
        GradTensor x = GradTensor.randn(3, 3).requiresGrad(true);
        GradTensor out = GradCheckpoint.checkpoint(() -> x.relu().mul(2f));
        assertArrayEquals(new long[]{3, 3}, out.shape());
        assertTrue(out.requiresGrad());
    }
}
