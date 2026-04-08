package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.models.LLaMA;
import tech.kayys.gollek.ml.nn.optim.AdamW;

import java.io.IOException;
import java.nio.file.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Session 14 tests: LoRALinear, LLaMA, TextDataset, ImageDataset, GradTensor.silu().
 */
class Session14Test {

    // ── GradTensor.silu ───────────────────────────────────────────────────

    @Test
    void siluPositiveInput() {
        GradTensor x = GradTensor.of(new float[]{1f, 2f, 3f}, 3);
        GradTensor out = x.silu();
        // silu(x) = x * sigmoid(x) > 0 for x > 0
        for (float v : out.data()) assertTrue(v > 0f);
    }

    @Test
    void siluZeroInput() {
        GradTensor x = GradTensor.of(new float[]{0f}, 1);
        assertEquals(0f, x.silu().item(), 1e-6f);
    }

    @Test
    void siluShape() {
        GradTensor x = GradTensor.randn(3, 4);
        assertArrayEquals(new long[]{3, 4}, x.silu().shape());
    }

    // ── LoRALinear ────────────────────────────────────────────────────────

    @Test
    void loraOutputShape() {
        Linear base = new Linear(8, 16);
        LoRALinear lora = new LoRALinear(base, 8, 16, 4, 8f);
        GradTensor x = GradTensor.randn(2, 8);
        assertArrayEquals(new long[]{2, 16}, lora.forward(x).shape());
    }

    @Test
    void loraParametersOnlyAB() {
        Linear base = new Linear(8, 16);
        LoRALinear lora = new LoRALinear(base, 8, 16, 4, 8f);
        List<Parameter> loraParams = lora.loraParameters();
        assertEquals(2, loraParams.size()); // only A and B
    }

    @Test
    void loraBaseParamsFrozen() {
        Linear base = new Linear(8, 16);
        LoRALinear lora = new LoRALinear(base, 8, 16, 4, 8f);
        // Base parameters should not require grad
        for (Parameter p : base.parameters())
            assertFalse(p.data().requiresGrad(), "Base params should be frozen");
    }

    @Test
    void loraReducesLoss() {
        Linear base = new Linear(4, 2);
        LoRALinear lora = new LoRALinear(base, 4, 2, 2, 4f);
        AdamW opt = new AdamW(lora.loraParameters(), 0.01f);
        GradTensor x = GradTensor.randn(8, 4);
        GradTensor y = GradTensor.randn(8, 2);

        float first = 0f, last = 0f;
        for (int i = 0; i < 20; i++) {
            lora.zeroGrad();
            GradTensor loss = lora.forward(x).sub(y).pow(2f).mean();
            loss.backward();
            opt.step();
            if (i == 0)  first = loss.item();
            if (i == 19) last  = loss.item();
        }
        assertTrue(last < first, "LoRA should reduce loss");
    }

    @Test
    void loraMergedWeightShape() {
        Linear base = new Linear(4, 8);
        LoRALinear lora = new LoRALinear(base, 4, 8, 2, 4f);
        GradTensor merged = lora.mergedWeight();
        assertArrayEquals(new long[]{8, 4}, merged.shape());
    }

    // ── LLaMA ─────────────────────────────────────────────────────────────

    @Test
    void llamaTinyOutputShape() {
        NNModule llama = LLaMA.llamaTiny(100);
        GradTensor x = GradTensor.randn(2, 8, 256); // [B, T, dModel]
        GradTensor out = llama.forward(x);
        assertArrayEquals(new long[]{2, 8, 100}, out.shape());
    }

    @Test
    void llamaTinyHasParameters() {
        NNModule llama = LLaMA.llamaTiny(100);
        assertTrue(llama.parameterCount() > 10_000L);
    }

    @Test
    void llamaRMSNormOutputShape() {
        LLaMA.RMSNorm norm = new LLaMA.RMSNorm(32);
        GradTensor x = GradTensor.randn(2, 8, 32);
        assertArrayEquals(new long[]{2, 8, 32}, norm.forward(x).shape());
    }

    @Test
    void llamaRMSNormNormalizes() {
        LLaMA.RMSNorm norm = new LLaMA.RMSNorm(4);
        // All same value → RMS = value → output = 1.0 (weight=1)
        float[] data = new float[]{2f, 2f, 2f, 2f};
        GradTensor x = GradTensor.of(data, 1, 4);
        GradTensor out = norm.forward(x);
        for (float v : out.data()) assertEquals(1f, v, 1e-4f);
    }

    // ── TextDataset ───────────────────────────────────────────────────────

    @Test
    void textDatasetFromDirectory(@TempDir Path tmpDir) throws IOException {
        // Create class directories with text files
        Path pos = tmpDir.resolve("pos"); Files.createDirectory(pos);
        Path neg = tmpDir.resolve("neg"); Files.createDirectory(neg);
        Files.writeString(pos.resolve("a.txt"), "great movie");
        Files.writeString(pos.resolve("b.txt"), "loved it");
        Files.writeString(neg.resolve("c.txt"), "terrible film");

        // Use a simple mock tokenizer
        tech.kayys.gollek.ml.nlp.Tokenizer tok = new tech.kayys.gollek.ml.nlp.BpeTokenizer(
            java.util.Map.of("great",1,"movie",2,"loved",3,"it",4,"terrible",5,"film",6),
            java.util.Map.of(), 0, 1, 2, 0);

        var dataset = tech.kayys.gollek.ml.data.TextDataset.fromDirectory(tmpDir, tok, 8);
        assertEquals(3, dataset.size());
        assertNotNull(dataset.get(0).input());
        assertNotNull(dataset.get(0).label());
    }

    // ── ImageDataset ──────────────────────────────────────────────────────

    @Test
    void imageDatasetFromDirectory(@TempDir Path tmpDir) throws IOException {
        // Create class directories with tiny PNG images
        Path cat = tmpDir.resolve("cat"); Files.createDirectory(cat);
        Path dog = tmpDir.resolve("dog"); Files.createDirectory(dog);

        // Write minimal 1x1 PNG files
        writeTinyPng(cat.resolve("img1.png"));
        writeTinyPng(dog.resolve("img2.png"));

        var dataset = new tech.kayys.gollek.ml.data.ImageDataset(tmpDir);
        assertEquals(2, dataset.size());
        assertEquals(2, dataset.numClasses());
        assertTrue(dataset.classNames().contains("cat"));
        assertTrue(dataset.classNames().contains("dog"));
    }

    private static void writeTinyPng(Path path) throws IOException {
        // Minimal 1x1 white PNG
        java.awt.image.BufferedImage img = new java.awt.image.BufferedImage(
            1, 1, java.awt.image.BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, 0xFFFFFF);
        javax.imageio.ImageIO.write(img, "png", path.toFile());
    }
}
