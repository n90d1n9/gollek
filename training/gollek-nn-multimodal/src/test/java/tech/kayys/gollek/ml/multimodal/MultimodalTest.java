package tech.kayys.gollek.ml.multimodal;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MultimodalTest {

    @Test
    void testVisionBuilderCreation() {
        var builder = new VisionBuilder("test-model", null);
        assertNotNull(builder);
    }

    @Test
    void testAudioBuilderCreation() {
        var builder = new AudioBuilder("test-model", null);
        assertNotNull(builder);
    }

    @Test
    void testVideoBuilderCreation() {
        var builder = new VideoBuilder("test-model", null);
        assertNotNull(builder);
    }

    @Test
    void testMultimodalBuilderCreation() {
        var builder = new MultimodalBuilder("test-model", null);
        assertNotNull(builder);
    }

    @Test
    void testContentPartCreation() throws Exception {
        var text = ContentPart.text("Hello world");
        assertNotNull(text);

        var image = ContentPart.image(java.nio.file.Path.of("path/to/image.jpg"));
        assertNotNull(image);

        var audio = ContentPart.audio(java.nio.file.Path.of("path/to/audio.wav"));
        assertNotNull(audio);
    }
}
