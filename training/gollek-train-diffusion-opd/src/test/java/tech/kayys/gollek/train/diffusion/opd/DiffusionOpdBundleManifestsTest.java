package tech.kayys.gollek.train.diffusion.opd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiffusionOpdBundleManifestsTest {

    @Test
    void mapsTypedBundleManifestFromRawMap() {
        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.fromMap(Map.of(
                "bundleType", "standard",
                "sourceReportPath", "/tmp/report.json",
                "outputDirectory", "/tmp/out",
                "createdAt", "2026-05-18T00:00:00Z",
                "columnStrategy", "bundle-defaults",
                "generatedFiles", List.of(Map.of(
                        "name", "overview.json",
                        "section", "overview",
                        "format", "json",
                        "entryFormat", "json",
                        "columns", "",
                        "entryColumns", ""))));

        assertEquals("standard", manifest.bundleType());
        assertEquals("/tmp/report.json", manifest.sourceReportPath());
        assertEquals("/tmp/out", manifest.outputDirectory());
        assertEquals("2026-05-18T00:00:00Z", manifest.createdAt());
        assertEquals("bundle-defaults", manifest.columnStrategy());
        assertEquals(1, manifest.generatedFiles().size());
        DiffusionOpdBundleGeneratedFile file = manifest.generatedFiles().getFirst();
        assertEquals("overview.json", file.name());
        assertEquals("overview", file.section());
        assertEquals("json", file.format());
    }

    @Test
    void loadsTypedBundleManifestFromDisk() throws Exception {
        Path manifestFile = Files.createTempFile("diffusion-opd-bundle", ".json");
        Files.writeString(
                manifestFile,
                """
                {
                  "bundleType": "rollups",
                  "sourceReportPath": "/tmp/report.json",
                  "outputDirectory": "/tmp/out",
                  "createdAt": "2026-05-18T00:00:00Z",
                  "columnStrategy": "explicit",
                  "generatedFiles": [
                    {
                      "name": "taskSummaries.csv",
                      "section": "taskSummaries",
                      "format": "csv",
                      "entryFormat": "csv",
                      "columns": "compact",
                      "entryColumns": "compact"
                    }
                  ]
                }
                """);

        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.load(manifestFile);

        assertEquals("rollups", manifest.bundleType());
        assertEquals("explicit", manifest.columnStrategy());
        assertEquals(1, manifest.generatedFiles().size());
        assertEquals("taskSummaries.csv", manifest.generatedFiles().getFirst().name());
        assertFalse(manifest.toMap().isEmpty());
    }
}
