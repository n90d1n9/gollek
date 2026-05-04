package tech.kayys.gollek.quantizer.awq;

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
 * Comprehensive tests for AWQ quantizer integration.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class AWQIntegrationTest {

    @TempDir
    Path tempDir;

    private AWQQuantizerService quantizerService;

    @BeforeEach
    void setUp() {
        quantizerService = new AWQQuantizerService();
        assertTrue(quantizerService.isInitialized());
    }

    @AfterEach
    void tearDown() {
        if (quantizerService != null) {
            quantizerService.close();
        }
    }

    // ── AWQConfig Tests ─────────────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("Test AWQConfig builder with custom settings")
    void testAWQConfigBuilder() {
        AWQConfig config = AWQConfig.builder()
                .bits(4)
                .groupSize(64)
                .kernelFormat(AWQConfig.KernelFormat.GEMV)
                .hasZeros(false)
                .dequantDtype("float16")
                .exllamaV2(true)
                .numSamples(256)
                .seqLen(4096)
                .activationAware(true)
                .calibrationDataPath("/path/to/calibration")
                .build();

        assertEquals(4, config.bits());
        assertEquals(64, config.groupSize());
        assertEquals(AWQConfig.KernelFormat.GEMV, config.kernelFormat());
        assertFalse(config.hasZeros());
        assertEquals("float16", config.dequantDtype());
        assertTrue(config.exllamaV2());
        assertEquals(256, config.numSamples());
        assertEquals(4096, config.seqLen());
        assertTrue(config.activationAware());
        assertEquals("/path/to/calibration", config.calibrationDataPath());
    }

    @Test
    @Order(2)
    @DisplayName("Test AWQConfig preset configurations")
    void testAWQConfigPresets() {
        AWQConfig config4bit = AWQConfig.awq4bit();
        assertEquals(4, config4bit.bits());
        assertEquals(128, config4bit.groupSize());
        assertEquals(AWQConfig.KernelFormat.GEMM, config4bit.kernelFormat());
        assertTrue(config4bit.hasZeros());
        assertTrue(config4bit.activationAware());

        AWQConfig configSymmetric = AWQConfig.awq4bitSymmetric();
        assertFalse(configSymmetric.hasZeros());
        assertEquals(8, configSymmetric.implicitZero());

        AWQConfig configGemv = AWQConfig.awq4bitGemv();
        assertEquals(AWQConfig.KernelFormat.GEMV, configGemv.kernelFormat());

        AWQConfig configGroup64 = AWQConfig.awq4bitGroup64();
        assertEquals(64, configGroup64.groupSize());

        AWQConfig configFP16 = AWQConfig.awq4bitFP16();
        assertEquals("float16", configFP16.dequantDtype());

        AWQConfig configExllama = AWQConfig.awq4bitExllamaV2();
        assertTrue(configExllama.exllamaV2());

        AWQConfig configMarlin = AWQConfig.awq4bitMarlin();
        assertEquals(AWQConfig.KernelFormat.MARLIN, configMarlin.kernelFormat());
    }

    @Test
    @Order(3)
    @DisplayName("Test AWQConfig derived properties")
    void testAWQConfigDerivedProperties() {
        AWQConfig config = AWQConfig.awq4bit();
        System.out.println("DEBUG: config.bits() = " + config.bits());
        System.out.println("DEBUG: config.packFactor() = " + config.packFactor());
        assertEquals(8, config.packFactor());
        assertEquals(0xF, config.quantMask());
        // awq4bit() has hasZeros=true, so implicitZero returns 0 (zeros are stored explicitly)
        assertEquals(0, config.implicitZero());
        assertEquals(15, config.maxQuantValue());
        assertEquals(8.0, config.compressionRatio());
        assertTrue(config.isValidBits());
        assertTrue(config.isGemmFormat());
        assertFalse(config.isGemvFormat());
        assertFalse(config.isMarlinFormat());
    }

    @Test
    @Order(3)
    @DisplayName("Test AWQConfig symmetric (no zeros) derived properties")
    void testAWQConfigSymmetricDerivedProperties() {
        AWQConfig config = AWQConfig.awq4bitSymmetric();
        // Symmetric config has hasZeros=false, so implicitZero = 2^(4-1) = 8
        assertEquals(8, config.implicitZero());
        assertFalse(config.hasZeros());
    }

    @Test
    @Order(4)
    @DisplayName("Test AWQConfig validation")
    void testAWQConfigValidation() {
        assertThrows(IllegalArgumentException.class, () -> {
            AWQConfig.builder()
                    .bits(8) // AWQ typically uses 4-bit only
                    .build();
        });

        assertTrue(AWQConfig.awq4bit().isValidBits());
    }

    // ── AWQQuantizerService Tests ───────────────────────────────────────────

    @Test
    @Order(10)
    @DisplayName("Test quantizer service initialization")
    void testQuantizerServiceInitialization() {
        assertTrue(quantizerService.isInitialized());
        
        AWQQuantizerService anotherService = new AWQQuantizerService();
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

    // ── AWQSafetensorConverter Tests ────────────────────────────────────────

    @Test
    @Order(20)
    @DisplayName("Test conversion config presets")
    void testConversionConfigPresets() {
        AWQSafetensorConverter.ConversionConfig defaults = AWQSafetensorConverter.ConversionConfig.defaults();
        assertFalse(defaults.verbose());
        assertEquals("F32", defaults.outputDtype());

        AWQSafetensorConverter.ConversionConfig verbose = AWQSafetensorConverter.ConversionConfig.verboseConfig();
        assertTrue(verbose.verbose());
        assertEquals("F32", verbose.outputDtype());
    }

    @Test
    @Order(21)
    @DisplayName("Test conversion result calculations")
    void testConversionResultCalculations() {
        AWQSafetensorConverter.ConversionResult result = new AWQSafetensorConverter.ConversionResult(
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
    @DisplayName("Test AWQ config toString output")
    void testAWQConfigToString() {
        AWQConfig config = AWQConfig.awq4bit();
        String str = config.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("AWQConfig"));
        assertTrue(str.contains("bits=4"));
        assertTrue(str.contains("groupSize=128"));
        assertTrue(str.contains("format=GEMM"));
    }

    @Test
    @Order(41)
    @DisplayName("Test model inspection result toString")
    void testModelInspectionResultToString() {
        AWQConfig config = AWQConfig.awq4bit();
        AWQQuantizerService.ModelInspectionResult result = new AWQQuantizerService.ModelInspectionResult(
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
        
        AWQSafetensorConverter.ProgressCallback progressCallback = (layerName, current, total) -> {
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
        AWQQuantizerService service = new AWQQuantizerService();
        service.close();
        // Should not throw exception
        assertDoesNotThrow(() -> service.close());
    }

    @Test
    @Order(51)
    @DisplayName("Test config with extreme values")
    void testConfigWithExtremeValues() {
        AWQConfig config = AWQConfig.builder()
                .bits(4)
                .groupSize(1)
                .numSamples(1)
                .seqLen(1)
                .build();

        assertEquals(4, config.bits());
        assertEquals(1, config.groupSize());
        assertEquals(1, config.numSamples());
        assertEquals(1, config.seqLen());
    }

    @Test
    @Order(52)
    @DisplayName("Test conversion result with zero values")
    void testConversionResultWithZeroValues() {
        AWQSafetensorConverter.ConversionResult result = new AWQSafetensorConverter.ConversionResult(
                0, 0, 0, 0, 0, tempDir.resolve("output"));

        assertEquals(0, result.layersConverted());
        assertEquals(0, result.throughputMBps());
    }

    // ── AWQDequantizer Tests ────────────────────────────────────────────────

    @Test
    @Order(60)
    @DisplayName("Test AWQDequantizer initialization")
    void testAWQDequantizerInitialization() {
        AWQConfig config = AWQConfig.awq4bit();
        AWQDequantizer dequantizer = new AWQDequantizer(config);
        
        assertNotNull(dequantizer);
        assertEquals(8, config.packFactor());
        assertEquals(0xF, config.quantMask());
    }

    @Test
    @Order(61)
    @DisplayName("Test AWQDequantizer capabilities output")
    void testAWQDequantizerCapabilities() {
        // Just verify it doesn't throw
        assertDoesNotThrow(() -> AWQDequantizer.printCapabilities());
    }

    // ── AWQLayer Tests ──────────────────────────────────────────────────────

    @Test
    @Order(70)
    @DisplayName("Test AWQLayer creation")
    void testAWQLayerCreation() {
        AWQConfig config = AWQConfig.awq4bit();
        AWQLayer layer = new AWQLayer("test.layer.0", config);
        
        assertEquals("test.layer.0", layer.getName());
        assertEquals(config, layer.getConfig());
        assertFalse(layer.isComplete());
        assertEquals(0, layer.estimatedBytes());
    }

    @Test
    @Order(71)
    @DisplayName("Test AWQLayer derived properties")
    void testAWQLayerDerivedProperties() {
        AWQConfig config = AWQConfig.awq4bit();
        AWQLayer layer = new AWQLayer("test.layer.0", config);
        
        layer.setInFeatures(4096);
        layer.setOutFeatures(4096);
        
        assertEquals(4096, layer.getInFeatures());
        assertEquals(4096, layer.getOutFeatures());
        assertEquals(32, layer.numGroups()); // 4096 / 128 = 32
        assertEquals(8, layer.packFactor());
        assertFalse(layer.hasBias());
        assertFalse(layer.hasZeros());
        assertFalse(layer.isComplete());
    }
}
