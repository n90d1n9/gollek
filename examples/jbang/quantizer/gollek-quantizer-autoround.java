///usr/bin/env jbang "$0" "$@" ; exit $?
// Gollek AutoRound Quantization Example
//
// Demonstrates AutoRound quantization with optimization-based rounding.
// AutoRound learns both rounding decisions AND scale factors via SignSGD.
//
// Prerequisites:
//   - Java 25+
//   - JBang: curl -Ls https://sh.jbang.dev | bash -s -
//   - AutoRound module built: cd gollek/core/quantizer/gollek-quantizer-autoround && mvn clean install
//
// Usage:
//   jbang gollek-quantizer-autoround.java --help
//   jbang gollek-quantizer-autoround.java --model /path/to/model.safetensors --output /path/to/quantized
//   jbang gollek-quantizer-autoround.java --demo  # Run with synthetic data
//
// DEPS tech.kayys.gollek:gollek-quantizer-autoround:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-safetensor-loader:0.1.0-SNAPSHOT
// DEPS info.picocli:picocli:4.7.5
// JAVA 25+

import tech.kayys.gollek.quantizer.autoround.AutoRoundConfig;
import tech.kayys.gollek.quantizer.autoround.AutoRoundConfig.ScaleDtype;
import tech.kayys.gollek.quantizer.autoround.AutoRoundConfig.PackFormat;
import tech.kayys.gollek.quantizer.autoround.AutoRoundLoader;
import tech.kayys.gollek.quantizer.autoround.AutoRoundDequantizer;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.nio.file.Path;
import java.util.concurrent.Callable;

@Command(name = "autoround-quantizer", mixinStandardHelpOptions = true, version = "autoround-quantizer 0.1.0",
        description = "AutoRound Quantization Example - Optimization-based rounding with SignSGD")
public class gollek_quantizer_autoround implements Callable<Integer> {

    @Parameters(index = "0", description = "Path to AutoRound-quantized model directory", arity = "0..1")
    private Path modelPath;

    @Option(names = {"--demo"}, description = "Run demo mode with synthetic data")
    private boolean demoMode;

    @Option(names = {"--bits"}, description = "Quantization bits (2, 3, 4, 8)", defaultValue = "4")
    private int bits;

    @Option(names = {"--group-size"}, description = "Group size", defaultValue = "128")
    private int groupSize;

    @Option(names = {"--scale-dtype"}, description = "Scale tensor dtype (FLOAT32, FLOAT16)", defaultValue = "FLOAT32")
    private String scaleDtype;

    @Option(names = {"--pack-format"}, description = "Packing format (AUTOROUND_NATIVE, GPTQ_COMPAT, IPEX_WEIGHT_ONLY)", defaultValue = "AUTOROUND_NATIVE")
    private String packFormat;

    public static void main(String[] args) {
        int exitCode = new CommandLine(new gollek_quantizer_autoround()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║        Gollek AutoRound Quantization Example             ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        if (demoMode || modelPath == null) {
            runDemoMode();
            return 0;
        }

        // Build AutoRound configuration
        ScaleDtype scaleDtypeEnum = ScaleDtype.valueOf(this.scaleDtype.toUpperCase());
        PackFormat packFormatEnum = PackFormat.valueOf(this.packFormat.toUpperCase());

        AutoRoundConfig config = new AutoRoundConfig(
                bits,
                groupSize,
                true,            // hasZeroPoint
                scaleDtypeEnum,
                packFormatEnum,
                "float32",       // targetDtype
                200,             // numIterations
                0.001,           // lr
                false,           // descAct
                128,             // numSamples
                2048,            // seqLen
                "exllamav2",     // backend
                null             // calibrationDataPath
        );

        System.out.println("Configuration:");
        System.out.println("  Model:         " + modelPath);
        System.out.println("  Bits:          " + config.bits());
        System.out.println("  Group Size:    " + config.groupSize());
        System.out.println("  Scale Dtype:   " + config.scaleDtype());
        System.out.println("  Pack Format:   " + config.packFormat());
        System.out.println("  Has Zero Point:" + config.hasZeroPoint());
        System.out.println("  Backend:       " + config.backend());
        System.out.println();

        // Load AutoRound model
        System.out.println("Loading AutoRound model...");
        try (AutoRoundLoader loader = new AutoRoundLoader(modelPath, config)) {
            loader.load();

            System.out.println("✓ Model loaded successfully!");
            System.out.println();

            // Show layer information
            var layers = loader.getLayers();
            System.out.println("Model Layers: " + layers.size());
            System.out.println();

            for (var entry : layers.entrySet()) {
                String name = entry.getKey();
                var layer = entry.getValue();

                if (layer.isComplete()) {
                    System.out.printf("  %-50s %d×%d (groups=%d, hasBias=%b)%n",
                            name,
                            layer.getInFeatures(),
                            layer.getOutFeatures(),
                            layer.numGroups(),
                            layer.hasBias());
                }
            }

            System.out.println();
            System.out.println("AutoRound Model Statistics:");
            System.out.println("  Total Layers:    " + layers.size());
            System.out.println("  Complete Layers: " + layers.values().stream().filter(l -> l.isComplete()).count());
            System.out.println("  Pack Factor:     " + config.packFactor());
            System.out.println("  Quant Mask:      0x" + Integer.toHexString(config.quantMask()));
            System.out.println("  Implicit Zero:   " + config.implicitZero());
            System.out.println();

            // Example dequantization
            System.out.println("To dequantize a layer:");
            System.out.println("  AutoRoundDequantizer dequantizer = new AutoRoundDequantizer(config);");
            System.out.println("  float[] output = new float[inFeatures * outFeatures];");
            System.out.println("  dequantizer.dequantizeNative(qweight, scale, zp, inFeatures, outFeatures, output);");

        } catch (Exception e) {
            System.err.println("❌ Failed to load AutoRound model: " + e.getMessage());
            e.printStackTrace();
            return 1;
        }

        return 0;
    }

    /**
     * Run demo mode showing AutoRound algorithm and configuration options.
     */
    private void runDemoMode() {
        System.out.println("Running in demo mode (no real model required)");
        System.out.println();

        System.out.println("AutoRound Algorithm Overview:");
        System.out.println();
        System.out.println("  AutoRound optimizes BOTH rounding decisions AND scale factors:");
        System.out.println();
        System.out.println("  For each transformer block B:");
        System.out.println("  1. Collect input activations X from calibration data");
        System.out.println("  2. Initialize s, z from min/max of each weight group");
        System.out.println("  3. For T iterations (default 200):");
        System.out.println("     a. q(W) = clamp(round(W/s + 0.5*v) − z, 0, 2^b − 1)");
        System.out.println("     b. W̃ = (q(W) + z) × s");
        System.out.println("     c. L = ||BX − B̃X||² (block reconstruction loss)");
        System.out.println("     d. Gradient step on s, z via Adam");
        System.out.println("     e. v update via SignSGD: v ← v − η × sign(∂L/∂v)");
        System.out.println();

        System.out.println("AutoRound Configuration Examples:");
        System.out.println();

        // Example 1: Standard 4-bit
        AutoRoundConfig config1 = AutoRoundConfig.autoRound4bit();
        System.out.println("1. Standard 4-bit AutoRound:");
        System.out.println("   Bits:          " + config1.bits());
        System.out.println("   Group Size:    " + config1.groupSize());
        System.out.println("   Has Zero Point:" + config1.hasZeroPoint());
        System.out.println("   Scale Dtype:   " + config1.scaleDtype());
        System.out.println("   Pack Format:   " + config1.packFormat());
        System.out.println("   Backend:       " + config1.backend());
        System.out.println();

        // Example 2: Symmetric (no zero-point)
        AutoRoundConfig config2 = AutoRoundConfig.autoRound4bitSymmetric();
        System.out.println("2. Symmetric 4-bit AutoRound:");
        System.out.println("   Bits:          " + config2.bits());
        System.out.println("   Has Zero Point:" + config2.hasZeroPoint());
        System.out.println("   Simpler:       No zero-point optimization");
        System.out.println();

        // Example 3: Group size 32 for higher quality
        AutoRoundConfig config3 = AutoRoundConfig.autoRound4bitGroup32();
        System.out.println("3. 4-bit AutoRound with group size 32 (highest quality):");
        System.out.println("   Bits:          " + config3.bits());
        System.out.println("   Group Size:    " + config3.groupSize());
        System.out.println("   Quality gain:  ~10-15% vs group size 128");
        System.out.println("   Memory cost:   ~30% more for scales/zeros");
        System.out.println();

        // Example 4: 2-bit for maximum compression
        AutoRoundConfig config4 = AutoRoundConfig.autoRound2bit();
        System.out.println("4. 2-bit AutoRound (maximum compression):");
        System.out.println("   Bits:          " + config4.bits());
        System.out.println("   Compression:   16x vs FP32");
        System.out.println("   Quality loss:  ~5-10% perplexity increase");
        System.out.println();

        // Example 5: 8-bit for minimal quality loss
        AutoRoundConfig config5 = AutoRoundConfig.autoRound8bit();
        System.out.println("5. 8-bit AutoRound (minimal quality loss):");
        System.out.println("   Bits:          " + config5.bits());
        System.out.println("   Compression:   4x vs FP32");
        System.out.println("   Quality loss:  <0.5% perplexity increase");
        System.out.println();

        // Example 6: FP16 scales for memory efficiency
        AutoRoundConfig config6 = AutoRoundConfig.autoRound4bitFP16();
        System.out.println("6. 4-bit AutoRound with FP16 scales:");
        System.out.println("   Bits:          " + config6.bits());
        System.out.println("   Scale Dtype:   " + config6.scaleDtype());
        System.out.println("   Memory saving: 50% for scale tensors");
        System.out.println();

        // Example 7: Marlin backend for GPU optimization
        AutoRoundConfig config7 = AutoRoundConfig.autoRound4bitMarlin();
        System.out.println("7. 4-bit AutoRound Marlin backend (GPU optimized):");
        System.out.println("   Bits:          " + config7.bits());
        System.out.println("   Backend:       " + config7.backend());
        System.out.println("   Best for:      NVIDIA GPU inference");
        System.out.println();

        System.out.println("AutoRound vs GPTQ vs AWQ Comparison:");
        System.out.println("  ┌─────────────┬──────────┬──────────┬──────────┐");
        System.out.println("  │ Metric      │ GPTQ     │ AWQ      │ AutoRnd  │");
        System.out.println("  ├─────────────┼──────────┼──────────┼──────────┤");
        System.out.println("  │ Quality     │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐   │ ⭐⭐⭐⭐   │");
        System.out.println("  │ Speed       │ ⭐⭐      │ ⭐⭐⭐⭐   │ ⭐⭐⭐    │");
        System.out.println("  │ Memory      │ ⭐⭐⭐⭐   │ ⭐⭐⭐⭐   │ ⭐⭐⭐⭐   │");
        System.out.println("  │ Calibration │ Required │ Required │ Required │");
        System.out.println("  │ Algorithm   │ Hessian  │ Scaling  │ SignSGD  │");
        System.out.println("  │ Best For    │ Quality  │ Speed    │ Balance  │");
        System.out.println("  └─────────────┴──────────┴──────────┴──────────┘");
        System.out.println();

        System.out.println("To load a real AutoRound model:");
        System.out.println("  jbang gollek-quantizer-autoround.java \\");
        System.out.println("    --model /path/to/autoround-model.safetensors \\");
        System.out.println("    --bits 4 \\");
        System.out.println("    --group-size 128 \\");
        System.out.println("    --scale-dtype FLOAT32 \\");
        System.out.println("    --pack-format AUTOROUND_NATIVE");
        System.out.println();

        // Simulate loading progress
        System.out.println("Simulating AutoRound model loading...");
        String[] steps = {"Discovering shards", "Parsing headers", "Loading layer 1/32",
                "Optimizing scales (iteration 100/200)", "Optimizing scales (iteration 200/200)",
                "Loading layer 32/32", "Building layer graph"};

        for (int i = 0; i < steps.length; i++) {
            System.out.printf("  [%3d%%] %s%n", (i + 1) * 100 / steps.length, steps[i]);
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        System.out.println();
        System.out.println("✓ Demo complete!");
        System.out.println();
        System.out.println("Expected Results for LLaMA-7B AutoRound:");
        System.out.println("  Original Size:     13.5 GB (FP16)");
        System.out.println("  Quantized Size:    3.8 GB (4-bit)");
        System.out.println("  Compression:       3.5x");
        System.out.println("  Quantization Time: ~15 minutes");
        System.out.println("  Quality Loss:      ~0.9% perplexity increase");
    }
}
