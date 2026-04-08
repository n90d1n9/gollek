package tech.kayys.gollek.ml.autograd;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.nn.Module;
import tech.kayys.gollek.ml.nn.Linear;
import tech.kayys.gollek.ml.nn.Sequential;
import tech.kayys.gollek.ml.nn.ReLU;
import tech.kayys.gollek.ml.nn.StateDict;
import tech.kayys.gollek.ml.nn.loss.FocalLoss;
import tech.kayys.gollek.ml.nn.optim.RMSprop;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class Phase1AdvancedTest {

    // ── TensorOps: slice ─────────────────────────────────────────────────

    @Test
    void sliceAlongDim0() {
        GradTensor t = GradTensor.of(new float[]{1,2,3,4,5,6}, 3, 2);
        GradTensor s = TensorOps.slice(t, 0, 1, 3);
        assertArrayEquals(new long[]{2, 2}, s.shape());
        assertArrayEquals(new float[]{3,4,5,6}, s.data(), 1e-5f);
    }

    @Test
    void sliceAlongDim1() {
        GradTensor t = GradTensor.of(new float[]{1,2,3,4,5,6}, 2, 3);
        GradTensor s = TensorOps.slice(t, 1, 0, 2);
        assertArrayEquals(new long[]{2, 2}, s.shape());
        assertArrayEquals(new float[]{1,2,4,5}, s.data(), 1e-5f);
    }

    // ── TensorOps: cat ───────────────────────────────────────────────────

    @Test
    void catDim0() {
        GradTensor a = GradTensor.of(new float[]{1,2,3,4}, 2, 2);
        GradTensor b = GradTensor.of(new float[]{5,6,7,8}, 2, 2);
        GradTensor c = TensorOps.cat(List.of(a, b), 0);
        assertArrayEquals(new long[]{4, 2}, c.shape());
        assertArrayEquals(new float[]{1,2,3,4,5,6,7,8}, c.data(), 1e-5f);
    }

    @Test
    void catDim1() {
        GradTensor a = GradTensor.of(new float[]{1,2,3,4}, 2, 2);
        GradTensor b = GradTensor.of(new float[]{5,6,7,8}, 2, 2);
        GradTensor c = TensorOps.cat(List.of(a, b), 1);
        assertArrayEquals(new long[]{2, 4}, c.shape());
        assertArrayEquals(new float[]{1,2,5,6,3,4,7,8}, c.data(), 1e-5f);
    }

    // ── TensorOps: stack ─────────────────────────────────────────────────

    @Test
    void stackDim0() {
        GradTensor a = GradTensor.of(new float[]{1,2,3}, 3);
        GradTensor b = GradTensor.of(new float[]{4,5,6}, 3);
        GradTensor s = TensorOps.stack(List.of(a, b), 0);
        assertArrayEquals(new long[]{2, 3}, s.shape());
        assertArrayEquals(new float[]{1,2,3,4,5,6}, s.data(), 1e-5f);
    }

    // ── TensorOps: gather ────────────────────────────────────────────────

    @Test
    void gatherDim1() {
        // input [2,3], index [2,2]
        GradTensor input = GradTensor.of(new float[]{1,2,3, 4,5,6}, 2, 3);
        GradTensor index = GradTensor.of(new float[]{0,2, 1,0}, 2, 2);
        GradTensor out   = TensorOps.gather(input, 1, index);
        assertArrayEquals(new long[]{2, 2}, out.shape());
        assertArrayEquals(new float[]{1,3, 5,4}, out.data(), 1e-5f);
    }

    // ── TensorOps: einsum ────────────────────────────────────────────────

    @Test
    void einsumBatchedMatmul() {
        GradTensor a = GradTensor.randn(2, 3, 4);
        GradTensor b = GradTensor.randn(2, 4, 5);
        GradTensor c = TensorOps.einsum("bij,bjk->bik", a, b);
        assertArrayEquals(new long[]{2, 3, 5}, c.shape());
    }

    @Test
    void einsumAttentionScores() {
        // Q[b,h,i,d] x K[b,h,j,d] -> [b,h,i,j]
        GradTensor q = GradTensor.randn(2, 4, 8, 16); // [B,H,I,D]
        GradTensor k = GradTensor.randn(2, 4, 8, 16); // [B,H,J,D]
        GradTensor s = TensorOps.einsum("bhid,bhjd->bhij", q, k);
        assertArrayEquals(new long[]{2, 4, 8, 8}, s.shape());
    }

    // ── StateDict save/load ──────────────────────────────────────────────

    @Test
    void stateDictRoundTrip() throws IOException {
        Sequential model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
        Map<String, GradTensor> before = model.stateDict();

        Path tmp = Files.createTempFile("gollek-test-", ".safetensors");
        try {
            StateDict.save(before, tmp);
            Map<String, GradTensor> loaded = StateDict.load(tmp);

            assertEquals(before.keySet(), loaded.keySet());
            for (String key : before.keySet()) {
                assertArrayEquals(before.get(key).data(), loaded.get(key).data(), 1e-5f,
                    "Mismatch for key: " + key);
            }
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    @Test
    void loadStateDictStrict() throws IOException {
        Sequential model1 = new Sequential(new Linear(4, 8), new Linear(8, 2));
        Sequential model2 = new Sequential(new Linear(4, 8), new Linear(8, 2));

        Path tmp = Files.createTempFile("gollek-test-", ".safetensors");
        try {
            StateDict.save(model1.stateDict(), tmp);
            model2.loadStateDict(StateDict.load(tmp));

            // Weights should now be identical
            for (String key : model1.stateDict().keySet()) {
                assertArrayEquals(
                    model1.stateDict().get(key).data(),
                    model2.stateDict().get(key).data(), 1e-5f);
            }
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    // ── FocalLoss ────────────────────────────────────────────────────────

    @Test
    void focalLossIsScalar() {
        FocalLoss loss = new FocalLoss();
        GradTensor logits  = GradTensor.randn(4, 10);
        GradTensor targets = GradTensor.of(new float[]{0, 3, 7, 9}, 4);
        GradTensor out = loss.forward(logits, targets);
        assertEquals(0, out.ndim());
        assertTrue(out.item() > 0f);
    }

    @Test
    void focalLossLowerThanCrossEntropy() {
        // Focal loss should be <= cross-entropy (gamma=0 reduces to CE)
        FocalLoss focal = new FocalLoss(2.0f, 1.0f);
        FocalLoss ce    = new FocalLoss(0.0f, 1.0f);
        GradTensor logits  = GradTensor.randn(8, 5);
        GradTensor targets = GradTensor.of(new float[]{0,1,2,3,4,0,1,2}, 8);
        assertTrue(focal.forward(logits, targets).item()
                <= ce.forward(logits, targets).item() + 1e-4f);
    }

    // ── RMSprop ──────────────────────────────────────────────────────────

    @Test
    void rmspropReducesLoss() {
        Linear model = new Linear(2, 1);
        RMSprop opt  = new RMSprop(model.parameters(), 0.01f);

        GradTensor x = GradTensor.of(new float[]{1,2, 3,4, 5,6}, 3, 2);
        GradTensor y = GradTensor.of(new float[]{1, 2, 3}, 3, 1);

        float firstLoss = 0, lastLoss = 0;
        for (int i = 0; i < 50; i++) {
            model.zeroGrad();
            GradTensor pred = model.forward(x);
            GradTensor loss = pred.sub(y).pow(2f).mean();
            loss.backward();
            opt.step();
            if (i == 0)  firstLoss = loss.item();
            if (i == 49) lastLoss  = loss.item();
        }
        assertTrue(lastLoss < firstLoss, "RMSprop should reduce loss");
    }
}
