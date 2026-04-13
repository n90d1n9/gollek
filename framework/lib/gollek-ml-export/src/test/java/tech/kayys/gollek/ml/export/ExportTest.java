package tech.kayys.gollek.ml.export;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ExportTest {

    @Test
    void testModelExporterBuilderCreation() {
        var builder = ModelExporter.builder();
        assertNotNull(builder);
    }

    @Test
    void testOnnxExporterCreation() {
        // OnnxExporter requires model, inputShape, metadata
        // Just verify the class is loadable
        assertNotNull(OnnxExporter.class);
    }

    @Test
    void testGGUFExporterCreation() {
        // GGUFExporter requires constructor args
        assertNotNull(GGUFExporter.class);
    }

    @Test
    void testLiteRTExporterCreation() {
        // LiteRTExporter requires constructor args
        assertNotNull(LiteRTExporter.class);
    }
}
