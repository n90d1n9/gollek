package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.metrics.NLPMetrics;
import tech.kayys.gollek.ml.nn.optim.Adam;
import tech.kayys.gollek.ml.nn.optim.Lookahead;
import tech.kayys.gollek.ml.wayang.WayangIntegration;
import tech.kayys.gollek.sdk.GollekClient;
import tech.kayys.gollek.sdk.session.InferenceSession;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Session 15 tests: ConvTranspose2d, AdaptiveAvgPool2d, Lookahead,
 * NLPMetrics, WayangIntegration, InferenceSession.
 */
class Session15Test {

    // ── ConvTranspose2d ───────────────────────────────────────────────────

    @Test
    void convTranspose2dUpsamplesBy2() {
        ConvTranspose2d deconv = new ConvTranspose2d(4, 2, 4, 2, 1); // 2× upsample
        GradTensor x = GradTensor.randn(1, 4, 4, 4);
        GradTensor out = deconv.forward(x);
        // H_out = (4-1)*2 - 2*1 + 4 = 8
        assertArrayEquals(new long[]{1, 2, 8, 8}, out.shape());
    }

    @Test
    void convTranspose2dNoStridePreservesSize() {
        ConvTranspose2d deconv = new ConvTranspose2d(4, 4, 3, 1, 1); // stride=1, pad=1
        GradTensor x = GradTensor.randn(2, 4, 8, 8);
        // H_out = (8-1)*1 - 2*1 + 3 = 8
        assertArrayEquals(new long[]{2, 4, 8, 8}, deconv.forward(x).shape());
    }

    // ── AdaptiveAvgPool2d ─────────────────────────────────────────────────

    @Test
    void adaptiveAvgPool2dGlobalPool() {
        AdaptiveAvgPool2d pool = new AdaptiveAvgPool2d(1, 1);
        GradTensor x = GradTensor.randn(2, 8, 14, 14);
        assertArrayEquals(new long[]{2, 8, 1, 1}, pool.forward(x).shape());
    }

    @Test
    void adaptiveAvgPool2dFixedOutput() {
        AdaptiveAvgPool2d pool = new AdaptiveAvgPool2d(7, 7);
        GradTensor x = GradTensor.randn(1, 4, 28, 28);
        assertArrayEquals(new long[]{1, 4, 7, 7}, pool.forward(x).shape());
    }

    @Test
    void adaptiveAvgPool2dAveragesCorrectly() {
        AdaptiveAvgPool2d pool = new AdaptiveAvgPool2d(1, 1);
        // 1×1×2×2 input: [1,2,3,4] → mean = 2.5
        GradTensor x = GradTensor.of(new float[]{1f, 2f, 3f, 4f}, 1, 1, 2, 2);
        assertEquals(2.5f, pool.forward(x).item(0), 1e-5f);
    }

    // ── Lookahead ─────────────────────────────────────────────────────────

    @Test
    void lookaheadReducesLoss() {
        Linear model = new Linear(4, 2);
        Adam base = new Adam(model.parameters(), 0.01f);
        Lookahead opt = new Lookahead(base, 5, 0.5f);
        GradTensor x = GradTensor.randn(8, 4);
        GradTensor y = GradTensor.randn(8, 2);

        float first = 0f, last = 0f;
        for (int i = 0; i < 30; i++) {
            model.zeroGrad();
            GradTensor loss = model.forward(x).sub(y).pow(2f).mean();
            loss.backward();
            opt.step();
            if (i == 0)  first = loss.item();
            if (i == 29) last  = loss.item();
        }
        assertTrue(last < first, "Lookahead should reduce loss");
    }

    @Test
    void lookaheadInnerOptimizerAccessible() {
        Adam base = new Adam(new Linear(2, 2).parameters(), 0.001f);
        Lookahead opt = new Lookahead(base);
        assertSame(base, opt.inner());
    }

    // ── NLPMetrics ────────────────────────────────────────────────────────

    @Test
    void bleuPerfectMatch() {
        List<String> hyp = List.of("the", "cat", "sat");
        List<List<String>> refs = List.of(List.of("the", "cat", "sat"));
        float bleu = NLPMetrics.bleu(hyp, refs, 2);
        assertTrue(bleu > 0.9f, "Perfect match should have high BLEU");
    }

    @Test
    void bleuNoMatch() {
        List<String> hyp = List.of("a", "b", "c");
        List<List<String>> refs = List.of(List.of("x", "y", "z"));
        assertEquals(0f, NLPMetrics.bleu(hyp, refs, 1), 1e-5f);
    }

    @Test
    void perplexityFromLoss() {
        // loss=0 → PPL=1 (perfect), loss=1 → PPL=e≈2.718
        assertEquals(1f, NLPMetrics.perplexity(0f), 1e-5f);
        assertEquals((float) Math.E, NLPMetrics.perplexity(1f), 1e-4f);
    }

    @Test
    void rougeN1PerfectMatch() {
        List<String> hyp = List.of("the", "cat", "sat");
        List<String> ref = List.of("the", "cat", "sat");
        assertEquals(1f, NLPMetrics.rouge(hyp, ref, 1), 1e-5f);
    }

    @Test
    void rougeN1NoMatch() {
        List<String> hyp = List.of("a", "b");
        List<String> ref = List.of("x", "y");
        assertEquals(0f, NLPMetrics.rouge(hyp, ref, 1), 1e-5f);
    }

    // ── WayangIntegration ─────────────────────────────────────────────────

    @Test
    void wayangSkillInvoke() {
        GollekClient client = GollekClient.builder().model("model.gguf").build();
        WayangIntegration integration = new WayangIntegration(client);
        WayangIntegration.Skill skill = integration.asSkill("echo", s -> "echo:" + s);
        assertEquals("echo:hello", skill.invoke("hello"));
    }

    @Test
    void wayangEmbedderReturnsVector() {
        GollekClient client = GollekClient.builder().model("model.gguf").build();
        WayangIntegration integration = new WayangIntegration(client);
        WayangIntegration.Embedder embedder = integration.asEmbedder();
        float[] vec = embedder.embed("test");
        assertEquals(768, vec.length);
    }

    @Test
    void wayangWorkflowNodeExecutes() {
        GollekClient client = GollekClient.builder().model("model.gguf").build();
        WayangIntegration integration = new WayangIntegration(client);
        WayangIntegration.WorkflowNode node = integration.asWorkflowNode("inference");
        Map<String, Object> ctx = new java.util.HashMap<>();
        ctx.put("prompt", "Hello");
        Map<String, Object> result = node.execute(ctx);
        assertTrue(result.containsKey("output"));
    }

    // ── InferenceSession ──────────────────────────────────────────────────

    @Test
    void inferenceSessionTracksMetrics() {
        GollekClient client = GollekClient.builder().model("model.gguf").build();
        try (InferenceSession session = InferenceSession.create(client)) {
            session.run("Hello");
            session.run("World");
            InferenceSession.SessionMetrics m = session.metrics();
            assertEquals(2, m.totalRequests());
            assertTrue(m.totalTokens() > 0);
        }
    }

    @Test
    void inferenceSessionBatch() {
        GollekClient client = GollekClient.builder().model("model.gguf").build();
        try (InferenceSession session = InferenceSession.create(client)) {
            List<GollekClient.GenerationResult> results =
                session.runBatch(List.of("A", "B", "C"));
            assertEquals(3, results.size());
        }
    }
}
