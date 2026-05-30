package tech.kayys.gollek.jupyter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;

class NotebookDependencyRendererTest {

    @Test
    void classpathPreviewFormatsEntriesAndEscapesHtml() {
        NotebookPreview preview = NotebookDependencyRenderer.classpathPreview(List.of(
                "/tmp/a.jar",
                "/tmp/<unsafe>.jar"
        ));

        assertTrue(preview.plain().contains("Classpath(entries=2)"));
        assertTrue(preview.plain().contains("/tmp/<unsafe>.jar"));
        assertTrue(preview.html().contains("<b>Classpath</b> entries=2"));
        assertTrue(preview.html().contains("/tmp/&lt;unsafe&gt;.jar"));
        assertFalse(preview.html().contains("/tmp/<unsafe>.jar"));
    }

    @Test
    void dependenciesPreviewKeepsEmptyPlainTextContract() {
        NotebookPreview empty = NotebookDependencyRenderer.dependenciesPreview(List.of());
        NotebookPreview loaded = NotebookDependencyRenderer.dependenciesPreview(List.of("/tmp/<dep>.jar"));

        assertEquals("Dependencies(dynamic=0)", empty.plain());
        assertTrue(empty.html().contains("No dynamic notebook dependencies loaded yet."));
        assertTrue(loaded.plain().contains("Dependencies(dynamic=1)"));
        assertTrue(loaded.html().contains("/tmp/&lt;dep&gt;.jar"));
    }

    @Test
    void jarAddedPreviewEscapesPath() {
        NotebookPreview preview = NotebookDependencyRenderer.jarAddedPreview("/tmp/<lib>.jar");

        assertEquals("Added jar to notebook classpath: /tmp/<lib>.jar", preview.plain());
        assertTrue(preview.html().contains("<b>Classpath Updated</b>"));
        assertTrue(preview.html().contains("/tmp/&lt;lib&gt;.jar"));
    }

    @Test
    void mavenAddedPreviewsKeepLocalAndRemoteWording() {
        NotebookPreview local = NotebookDependencyRenderer.localMavenArtifactAddedPreview(
                "dev.gollek:<kernel>:1.0.0",
                "/tmp/<kernel>.jar"
        );
        NotebookPreview remote = NotebookDependencyRenderer.remoteMavenArtifactAddedPreview(
                "dev.gollek:<kernel>:1.0.0",
                "/tmp/<kernel>.jar"
        );

        assertTrue(local.plain().contains("Added Maven artifact from local cache: dev.gollek:<kernel>:1.0.0"));
        assertTrue(local.html().contains("Local Maven Artifact Added"));
        assertTrue(remote.plain().contains("Fetched and added Maven artifact: dev.gollek:<kernel>:1.0.0"));
        assertTrue(remote.html().contains("Remote Maven Artifact Added"));
        assertTrue(remote.html().contains("dev.gollek:&lt;kernel&gt;:1.0.0"));
        assertTrue(remote.html().contains("/tmp/&lt;kernel&gt;.jar"));
    }

    @Test
    void missingMavenArtifactPreviewCanExplainLocalSearchOnly() {
        NotebookPreview preview = NotebookDependencyRenderer.missingMavenArtifactPreview(
                "dev.gollek:<missing>:1.0.0",
                new NotebookDependencies.MavenLookup(
                        null,
                        Path.of("/tmp/<m2>/dev/gollek/missing.jar"),
                        Path.of("/tmp/<gradle>/dev.gollek/missing/1.0.0")
                ),
                false,
                true
        );

        assertTrue(preview.plain().contains("Artifact not found in local Maven cache: dev.gollek:<missing>:1.0.0"));
        assertTrue(preview.plain().contains("searched:"));
        assertFalse(preview.plain().contains("remoteResolution=not-available-in-kernel"));
        assertTrue(preview.html().contains("dev.gollek:&lt;missing&gt;:1.0.0"));
        assertTrue(preview.html().contains("/tmp/&lt;m2&gt;/dev/gollek/missing.jar"));
    }

    @Test
    void missingMavenArtifactPreviewIncludesRemoteHintsWhenAllowed() {
        NotebookPreview preview = NotebookDependencyRenderer.missingMavenArtifactPreview(
                "dev.gollek:missing:1.0.0",
                new NotebookDependencies.MavenLookup(
                        null,
                        Path.of("/tmp/m2/missing.jar"),
                        Path.of("/tmp/gradle/missing")
                ),
                true,
                false
        );

        assertTrue(preview.plain().contains("remoteResolution=not-available-in-kernel"));
        assertTrue(preview.plain().contains("fetchHint=%maven --allow-remote --fetch dev.gollek:missing:1.0.0"));
        assertTrue(preview.plain().contains("nextStep=mvn dependency:get -Dartifact=dev.gollek:missing:1.0.0"));
        assertTrue(preview.html().contains("remote resolution is not performed by the kernel in this environment."));
        assertTrue(preview.html().contains("%maven --allow-remote --fetch dev.gollek:missing:1.0.0"));
    }
}
