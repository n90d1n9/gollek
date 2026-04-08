///usr/bin/env jbang "$0" "$@" ; exit $?
// Gollek AWQ Quantization Example
//
// Demonstrates AWQ (Activation-Aware Weight Quantization) for fast 4-bit quantization.
// AWQ protects salient input channels via per-channel scaling before quantization.
//
// Prerequisites:
//   - Java 25+
//   - JBang: curl -Ls https://sh.jbang.dev | bash -s -
//   - AWQ module built: cd gollek/core/quantizer/gollek-quantizer-awq && mvn clean install
//
// Usage:
//   jbang gollek-quantizer-awq.java --help
//   jbang gollek-quantizer-awq.java --model /path/to/model.safetensors --output /path/to/quantized
//   jbang gollek-quantizer-awq.java --demo  # Run with synthetic data
//
// DEPS tech.kayys.gollek:gollek-quantizer-awq:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-safetensor-loader:0.1.0-SNAPSHOT
// DEPS info.picocli:picocli:4.7.5
// JAVA 25+

import tech.kayys.gollek.quantizer.awq.AWQConfig;
import tech.kayys.gollek.quantizer.awq.AWQConfig.KernelFormat;
import tech.kayys.gollek.quantizer.awq.AWQLoader;
import tech.kayys.gollek.quantizer.awq.AWQDequantizer;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.nio.file.Path;
import java.util.concurrent.Callable;

@Command(name = "awq-quantizer", mixinStandardHelpOptions = true, version = "awq-quantizer 0.1.0",
        description = "AWQ Activation-Aware Quantization Example - Fast 4-bit with channel protection")
public class gollek_quantizer_awq implements Callable<Integer> {

    @Parameters(index = "0", description = "Path to AWQ-quantized model directory", arity = "0..1")
    private Path modelPath;

    @Option(names = {"--demo"}, description = "Run demo mode with synthetic data")
    private boolean demoMode;

    @Option(names = {"--bits"}, description = "Quantization bits", defaultValue = "4")
    private int bits;

    @Option(names = {"--group-size"}, description = "Group size", defaultValue = "128")
    private int groupSize;

    @Option(names = {"--format"}, description = "Kernel format (GEMM, GEMV, MARLIN)", defaultValue = "GEMM")
    private String format;

    @Option(names = {"--symmetric"}, description = "Use symmetric quantization (no zero-points)")
    private boolean symmetric;

    public static void main(String[] args) {
        int exitCode = new CommandLine(new gollek_quantizer_awq()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║          Gollek AWQ Quantization Example                 ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        if (demoMode || modelPath == null) {
            runDemoMode();
            return 0;
        }

        // Build AWQ configuration
        KernelFormat kernelFormat = KernelFormat.valueOf(format.toUpperCase());

        AWQConfig config = new AWQConfig(
                bits,
                groupSize,
                kernelFormat,
                !symmetric,  // hasZeros = !symmetric
                "float32",   // dequantDtype
                false,       // exllamaV2
                128,         // numSamples
                2048,        // seqLen
                true,        // activationAware
                null         // calibrationDataPath
        );

        System.out.println("Configuration:");
        System.out.println("  Model:         " + modelPath);
        System.out.println("  Bits:          " + bits);
        System.out.println("  Group Size:    " + groupSize);
        System.out.println("  Kernel Format: " + kernelFormat);
        System.out.println("  Has Zeros:     " + config.hasZeros());
        System.out.println("  Activation-Aware: " + config.activationAware());
        System.out.println();

        // Load AWQ model
        System.out.println("Loading AWQ model...");
        try (AWQLoader loader = new AWQLoader(modelPath, config)) {
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
                    System.out.printf("  %-50s %d×%d (groups=%d)%n",
                            name,
                            layer.getInFeatures(),
                            layer.getOutFeatures(),
                            layer.numGroups());
                }
            }

            System.out.println();
            System.out.println("AWQ Model Statistics:");
            System.out.println("  Total Layers:    " + layers.size());
            System.out.println("  Complete Layers: " + layers.values().stream().filter(l -> l.isComplete()).count());
            System.out.println("  Pack Factor:     " + config.packFactor());
            System.out.println("  Quant Mask:      0x" + Integer.toHexString(config.quantMask()));
            System.out.println();

            // Example dequantization
            System.out.println("To dequantize a layer:");
            System.out.println("  AWQDequantizer dequantizer = new AWQDequantizer(config);");
            System.out.println("  float[] output = new float[inFeatures * outFeatures];");
            System.out.println("  dequantizer.dequantize(qweight, qzeros, scales, inFeatures, outFeatures, output);");

        } catch (Exception e) {
            System.err.println("❌ Failed to load AWQ model: " + e.getMessage());
            e.printStackTrace();
            return 1;
        }

        return 0;
    }

    /**
     * Run demo mode showing AWQ configuration options.
     */
    private void runDemoMode() {
        System.out.println("Running in demo mode (no real model required)");
        System.out.println();

        System.out.println("AWQ Configuration Examples:");
        System.out.println();

        // Example 1: Standard 4-bit GEMM
        AWQConfig config1 = AWQConfig.awq4bit();
        System.out.println("1. Standard 4-bit AWQ (GEMM format):");
        System.out.println("   Bits:          " + config1.bits());
        System.out.println("   Group Size:    " + config1.groupSize());
        System.out.println("   Kernel Format: " + config1.kernelFormat());
        System.out.println("   Has Zeros:     " + config1.hasZeros());
        System.out.println("   Pack Factor:   " + config1.packFactor());
        System.out.println();

        // Example 2: Symmetric (no zero-points)
        AWQConfig config2 = AWQConfig.awq4bitSymmetric();
        System.out.println("2. Symmetric 4-bit AWQ (implicit zero=8):");
        System.out.println("   Bits:          " + config2.bits());
        System.out.println("   Has Zeros:     " + config2.hasZeros());
        System.out.println("   Implicit Zero: " + config2.implicitZero());
        System.out.println("   Max Quant:     " + config2.maxQuantValue());
        System.out.println();

        // Example 3: GEMV format for single-token inference
        AWQConfig config3 = AWQConfig.awq4bitGemv();
        System.out.println("3. 4-bit AWQ GEMV (single-token inference):");
        System.out.println("   Bits:          " + config3.bits());
        System.out.println("   Kernel Format: " + config3.kernelFormat());
        System.out.println("   Best for:      Small batch sizes, generation");
        System.out.println();

        // Example 4: Group size 64 for higher quality
        AWQConfig config4 = AWQConfig.awq4bitGroup64();
        System.out.println("4. 4-bit AWQ with group size 64 (higher quality):");
        System.out.println("   Bits:          " + config4.bits());
        System.out.println("   Group Size:    " + config4.groupSize());
        System.out.println("   Quality gain:  ~5-10% vs group size 128");
        System.out.println("   Memory cost:   ~10% more for scales");
        System.out.println();

        // Example 5: FP16 output for memory efficiency
        AWQConfig config5 = AWQConfig.awq4bitFP16();
        System.out.println("5. 4-bit AWQ with FP16 output:");
        System.out.println("   Bits:          " + config5.bits());
        System.out.println("   Output Dtype:  " + config5.dequantDtype());
        System.out.println("   Memory saving: 50% vs FP32 output");
        System.out.println();

        // Example 6: Marlin format for GPU optimization
        AWQConfig config6 = AWQConfig.awq4bitMarlin();
        System.out.println("6. 4-bit AWQ Marlin format (GPU optimized):");
        System.out.println("   Bits:          " + config6.bits());
        System.out.println("   Kernel Format: " + config6.kernelFormat());
        System.out.println("   Best for:      NVIDIA GPU inference");
        System.out.println();

        System.out.println("AWQ vs GPTQ Comparison:");
        System.out.println("  ┌─────────────┬──────────┬──────────┬──────────┐");
        System.out.println("  │ Metric      │ GPTQ     │ AWQ      │ Winner   │");
        System.out.println("  ├─────────────┼──────────┼──────────┼──────────┤");
        System.out.println("  │ Quality     │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐   │ GPTQ     │");
        System.out.println("  │ Speed       │ ⭐⭐      │ ⭐⭐⭐⭐   │ AWQ      │");
        System.out.println("  │ Memory      │ ⭐⭐⭐⭐   │ ⭐⭐⭐⭐   │ Tie      │");
        System.out.println("  │ Calibration │ Required │ Required │ Tie      │");
        System.out.println("  │ Best For    │ Quality  │ Speed    │ -        │");
        System.out.println("  └─────────────┴──────────┴──────────┴──────────┘");
        System.out.println();

        System.out.println("To load a real AWQ model:");
        System.out.println("  jbang gollek-quantizer-awq.java \\");
        System.out.println("    --model /path/to/awq-model.safetensors \\");
        System.out.println("    --bits 4 \\");
        System.out.println("    --group-size 128 \\");
        System.out.println("    --format GEMM");
        System.out.println();

        // Simulate loading progress
        System.out.println("Simulating AWQ model loading...");
        String[] steps = {"Discovering shards", "Parsing headers", "Loading layer 1/32",
                "Loading layer 16/32", "Loading layer 32/32", "Building layer graph"};

        for (int i = 0; i < steps.length; i++) {
            System.out.printf("  [%3d%%] %s%n", (i + 1) * 100 / steps.length, steps[i]);
            try {
                Thread.sleep(150);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        System.out.println();
        System.out.println("✓ Demo complete!");
        System.out.println();
        System.out.println("Expected Results for LLaMA-7B AWQ:");
        System.out.println("  Original Size:     13.5 GB (FP16)");
        System.out.println("  Quantized Size:    3.8 GB (4-bit)");
        System.out.println("  Compression:       3.5x");
        System.out.println("  Quantization Time: ~5 minutes (3-5x faster than GPTQ)");
        System.out.println("  Quality Loss:      ~1-2% perplexity increase");
    }
}
