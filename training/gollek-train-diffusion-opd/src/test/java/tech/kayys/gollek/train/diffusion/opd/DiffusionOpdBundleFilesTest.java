package tech.kayys.gollek.train.diffusion.opd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiffusionOpdBundleFilesTest {

    @Test
    void listsTypedBundleFilesThroughPublicApi() {
        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.fromMap(Map.of(
                "bundleType", "standard",
                "generatedFiles", List.of(
                        Map.of("name", "overview.json", "section", "overview", "format", "json"),
                        Map.of("name", "tasks.csv", "section", "taskSummaries", "format", "csv"))));

        List<DiffusionOpdBundleGeneratedFile> files = DiffusionOpdBundleInspector.files(manifest);

        assertEquals(2, files.size());
        assertEquals("overview.json", files.getFirst().name());
        assertEquals("taskSummaries", files.get(1).section());
    }

    @Test
    void loadsTypedBundleFileThroughPublicApi() throws Exception {
        Path outputDir = Files.createTempDirectory("opd-bundle-file-load");
        Files.writeString(outputDir.resolve("overview.json"), "{\"status\":\"ok\"}");
        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.fromMap(Map.of(
                "bundleType", "standard",
                "outputDirectory", outputDir.toString(),
                "generatedFiles", List.of(
                        Map.of("name", "overview.json", "section", "overview", "format", "json"))));

        DiffusionOpdBundleLoadedFile loaded = DiffusionOpdBundleInspector.loadFile(manifest, "overview.json");

        assertTrue(loaded.found());
        assertEquals("overview.json", loaded.file().name());
        Map<?, ?> content = assertInstanceOf(Map.class, loaded.content());
        assertEquals("ok", content.get("status"));
    }

    @Test
    void reportsMissingTypedBundleFileThroughPublicApi() {
        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.fromMap(Map.of(
                "bundleType", "standard",
                "generatedFiles", List.of(
                        Map.of("name", "overview.json", "section", "overview", "format", "json"))));

        DiffusionOpdBundleLoadedFile loaded = DiffusionOpdBundleInspector.loadFile(manifest, "missing.json");

        assertFalse(loaded.found());
        assertEquals("missing.json", loaded.request());
        assertEquals(null, loaded.file());
        assertEquals(Map.of(), loaded.content());
    }
}
