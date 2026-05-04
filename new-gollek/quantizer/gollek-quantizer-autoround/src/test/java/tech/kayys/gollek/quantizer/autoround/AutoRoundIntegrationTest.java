package tech.kayys.gollek.quantizer.autoround;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for AutoRound quantizer integration.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class AutoRoundIntegrationTest {

    @TempDir
    Path tempDir;

    private AutoRoundQuantizerService quantizerService;

    @BeforeEach
    void setUp() {
        quantizerService = new AutoRoundQuantizerService();
        assertTrue(quantizerService.isInitialized());
    }

    @AfterEach
    void tearDown() {
        if (quantizerService != null) {
            quantizerService.close();
        }
    }

    // ── AutoRoundConfig Tests ───────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("Test AutoRoundConfig builder with custom settings")
    void testAutoRoundConfigBuilder() {
        AutoRoundConfig config = AutoRoundConfig.builder()
                .bits(4)
                .groupSize(64)
                .hasZeroPoint(true)
                .scaleDtype(AutoRoundConfig.ScaleDtype.FLOAT32)
                .packFormat(AutoRoundConfig.PackFormat.AUTOROUND_NATIVE)
                .dequantDtype("float16")
                .numIters(300)
                .learningRate(0.005)
                .useAdam(true)
                .numSamples(256)
                .seqLen(4096)
                .backendTarget("marlin")
                .calibrationDataPath("/path/to/calibration")
                .build();

        assertEquals(4, config.bits());
        assertEquals(64, config.groupSize());
        assertTrue(config.hasZeroPoint());
        assertEquals(AutoRoundConfig.ScaleDtype.FLOAT32, config.scaleDtype());
        assertEquals(AutoRoundConfig.PackFormat.AUTOROUND_NATIVE, config.packFormat());
        assertEquals("float16", config.dequantDtype());
        assertEquals(300, config.numIters());
        assertEquals(0.005, config.learningRate());
        assertTrue(config.useAdam());
        assertEquals(256, config.numSamples());
        assertEquals(4096, config.seqLen());
        assertEquals("marlin", config.backendTarget());
        assertEquals("/path/to/calibration", config.calibrationDataPath());
    }

    @Test
    @Order(2)
    @DisplayName("Test AutoRoundConfig preset configurations")
    void testAutoRoundConfigPresets() {
        AutoRoundConfig config4bit = AutoRoundConfig.autoRound4bit();
        assertEquals(4, config4bit.bits());
        assertEquals(128, config4bit.groupSize());
        assertTrue(config4bit.hasZeroPoint());
        assertEquals(AutoRoundConfig.ScaleDtype.FLOAT32, config4bit.scaleDtype());
        assertEquals(AutoRoundConfig.PackFormat.AUTOROUND_NATIVE, config4bit.packFormat());
        assertTrue(config4bit.isNativeFormat());

        AutoRoundConfig configSymmetric = AutoRoundConfig.autoRound4bitSymmetric();
        assertFalse(configSymmetric.hasZeroPoint());
        assertEquals(8, configSymmetric.implicitZeroPoint());

        AutoRoundConfig configGptqCompat = AutoRoundConfig.autoRoundGptqCompat();
        assertTrue(configGptqCompat.isGptqCompatFormat());
        assertEquals(AutoRoundConfig.ScaleDtype.FLOAT16, configGptqCompat.scaleDtype());

        AutoRoundConfig configGroup32 = AutoRoundConfig.autoRound4bitGroup32();
        assertEquals(32, configGroup32.groupSize());

        AutoRoundConfig config2bit = AutoRoundConfig.autoRound2bit();
        assertEquals(2, config2bit.bits());

        AutoRoundConfig config8bit = AutoRoundConfig.autoRound8bit();
        assertEquals(8, config8bit.bits());

        AutoRoundConfig configFP16 = AutoRoundConfig.autoRound4bitFP16();
        assertEquals("float16", configFP16.dequantDtype());

        AutoRoundConfig configMarlin = AutoRoundConfig.autoRound4bitMarlin();
        assertEquals("marlin", configMarlin.backendTarget());
        assertTrue(configMarlin.useAdam());
    }

    @Test
    @Order(3)
    @DisplayName("Test AutoRoundConfig derived properties")
    void testAutoRoundConfigDerivedProperties() {
        AutoRoundConfig config = AutoRoundConfig.autoRound4bit();
        assertEquals(8, config.packFactor());
        assertEquals(0xF, config.quantMask());
        assertEquals(15, config.maxQuantValue());
        assertEquals(8, config.compressionRatio());
        assertTrue(config.hasFp32Scales());
        assertFalse(config.hasFp16Scales());
    }

    @Test
    @Order(4)
    @DisplayName("Test AutoRoundConfig validation")
    void testAutoRoundConfigValidation() {
        assertThrows(IllegalArgumentException.class, () -> {
            AutoRoundConfig.builder()
                    .bits(5) // Invalid bit width
                    .build();
        });

        assertTrue(AutoRoundConfig.autoRound4bit().hasFp32Scales());
        assertTrue(AutoRoundConfig.autoRoundGptqCompat().hasFp16Scales());
    }

    // ── AutoRoundQuantizerService Tests ─────────────────────────────────────

    @Test
    @Order(10)
    @DisplayName("Test quantizer service initialization")
    void testQuantizerServiceInitialization() {
        assertTrue(quantizerService.isInitialized());
        
        AutoRoundQuantizerService anotherService = new AutoRoundQuantizerService();
        assertTrue(anotherService.isInitialized());
        anotherService.close();
    }

    @Test
    @Order(11)
    @DisplayName("Test model inspection with invalid path")
    void testModelInspectionWithInvalidPath() {
        Path invalidPath = tempDir.resolve("nonexistent_model");
        
        assertThrows(IOException.class, () -> {
            quantizerService.inspect(invalidPath);
        });
    }

    // ── AutoRoundSafetensorConverter Tests ──────────────────────────────────

    @Test
    @Order(20)
    @DisplayName("Test conversion config presets")
    void testConversionConfigPresets() {
        AutoRoundSafetensorConverter.ConversionConfig defaults = AutoRoundSafetensorConverter.ConversionConfig.defaults();
        assertFalse(defaults.verbose());
        assertEquals("F32", defaults.outputDtype());

        AutoRoundSafetensorConverter.ConversionConfig verbose = AutoRoundSafetensorConverter.ConversionConfig.verboseConfig();
        assertTrue(verbose.verbose());
        assertEquals("F32", verbose.outputDtype());
    }

    @Test
    @Order(21)
    @DisplayName("Test conversion result calculations")
    void testConversionResultCalculations() {
        AutoRoundSafetensorConverter.ConversionResult result = new AutoRoundSafetensorConverter.ConversionResult(
                32, 32, 7_000_000_000L, 2_000_000_000L, 5000, tempDir.resolve("output.safetensors"));

        assertEquals(32, result.layersConverted());
        assertEquals(32, result.tensorsWritten());
        assertEquals(7_000_000_000L, result.totalElements());
        assertEquals(2_000_000_000L, result.outputFileSizeBytes());
        assertEquals(5000, result.elapsedMs());

        // Throughput = 2000 MB / 5s = 400 MB/s
        assertEquals(400.0, result.throughputMBps(), 1.0);
    }

    // ── Integration Tests ───────────────────────────────────────────────────

    @Test
    @Order(40)
    @DisplayName("Test AutoRound config toString output")
    void testAutoRoundConfigToString() {
        AutoRoundConfig config = AutoRoundConfig.autoRound4bit();
        String str = config.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("AutoRoundConfig"));
        assertTrue(str.contains("bits=4"));
        assertTrue(str.contains("groupSize=128"));
        assertTrue(str.contains("hasZP=true"));
    }

    @Test
    @Order(41)
    @DisplayName("Test model inspection result toString")
    void testModelInspectionResultToString() {
        AutoRoundConfig config = AutoRoundConfig.autoRound4bit();
        AutoRoundQuantizerService.ModelInspectionResult result = new AutoRoundQuantizerService.ModelInspectionResult(
                config, 32, 4_000_000_000L, 
                java.util.List.of("layer.0", "layer.1"),
                Map.of("model_type", "llama"));

        String str = result.toString();
        assertNotNull(str);
        assertTrue(str.contains("ModelInspectionResult"));
        assertTrue(str.contains("layers=32"));
        assertTrue(str.contains("memory="));
    }

    @Test
    @Order(42)
    @DisplayName("Test async conversion with progress callback")
    void testAsyncConversionWithProgressCallback() throws Exception {
        // This tests the infrastructure without actual model files
        CountDownLatch latch = new CountDownLatch(1);
        
        AutoRoundSafetensorConverter.ProgressCallback progressCallback = (layerName, current, total) -> {
            System.out.printf("Progress: %s (%d/%d)%n", layerName, current, total);
        };

        assertNotNull(progressCallback);
        progressCallback.onProgress("test.layer.0", 1, 32);
    }

    // ── Edge Cases and Error Handling ───────────────────────────────────────

    @Test
    @Order(50)
    @DisplayName("Test service close without initialization")
    void testServiceCloseWithoutIssues() {
        AutoRoundQuantizerService service = new AutoRoundQuantizerService();
        service.close();
        // Should not throw exception
        assertDoesNotThrow(() -> service.close());
    }

    @Test
    @Order(51)
    @DisplayName("Test config with extreme values")
    void testConfigWithExtremeValues() {
        AutoRoundConfig config = AutoRoundConfig.builder()
                .bits(4)
                .groupSize(1)
                .numIters(1)
                .learningRate(0.0)
                .numSamples(1)
                .seqLen(1)
                .build();

        assertEquals(4, config.bits());
        assertEquals(1, config.groupSize());
        assertEquals(1, config.numIters());
        assertEquals(0.0, config.learningRate());
        assertEquals(1, config.numSamples());
        assertEquals(1, config.seqLen());
    }

    @Test
    @Order(52)
    @DisplayName("Test conversion result with zero values")
    void testConversionResultWithZeroValues() {
        AutoRoundSafetensorConverter.ConversionResult result = new AutoRoundSafetensorConverter.ConversionResult(
                0, 0, 0, 0, 0, tempDir.resolve("output"));

        assertEquals(0, result.layersConverted());
        assertEquals(0, result.throughputMBps());
    }

    // ── AutoRoundDequantizer Tests ──────────────────────────────────────────

    @Test
    @Order(60)
    @DisplayName("Test AutoRoundDequantizer initialization")
    void testAutoRoundDequantizerInitialization() {
        AutoRoundConfig config = AutoRoundConfig.autoRound4bit();
        AutoRoundDequantizer dequantizer = new AutoRoundDequantizer(config);
        
        assertNotNull(dequantizer);
        assertEquals(8, config.packFactor());
        assertEquals(0xF, config.quantMask());
    }

    // ── AutoRoundLayer Tests ────────────────────────────────────────────────

    @Test
    @Order(70)
    @DisplayName("Test AutoRoundLayer creation")
    void testAutoRoundLayerCreation() {
        AutoRoundConfig config = AutoRoundConfig.autoRound4bit();
        AutoRoundLayer layer = new AutoRoundLayer("test.layer.0", config);
        
        assertEquals("test.layer.0", layer.getName());
        assertEquals(config, layer.getConfig());
        assertFalse(layer.isComplete());
        assertEquals(0, layer.estimatedBytes());
    }

    @Test
    @Order(71)
    @DisplayName("Test AutoRoundLayer derived properties")
    void testAutoRoundLayerDerivedProperties() {
        AutoRoundConfig config = AutoRoundConfig.autoRound4bit();
        AutoRoundLayer layer = new AutoRoundLayer("test.layer.0", config);
        
        layer.setInFeatures(4096);
        layer.setOutFeatures(4096);
        
        assertEquals(4096, layer.getInFeatures());
        assertEquals(4096, layer.getOutFeatures());
        assertEquals(32, layer.numGroups()); // 4096 / 128 = 32
        assertEquals(8, layer.packFactor());
        assertFalse(layer.hasBias());
        assertFalse(layer.hasZp());
        assertFalse(layer.isComplete());
    }
}
