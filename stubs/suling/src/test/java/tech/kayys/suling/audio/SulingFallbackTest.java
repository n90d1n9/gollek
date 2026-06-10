package tech.kayys.suling.audio;

import org.junit.jupiter.api.Test;
import tech.kayys.suling.FlacLibraryCheck;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SulingFallbackTest {
    @Test
    void fallbackReportsWavOnlyDiagnostics() {
        assertEquals(List.of("wav"), Suling.supportedAudioFormats());
        assertTrue(Suling.diagnostics().contains("pure Java WAV fallback"));
        assertFalse(FlacLibraryCheck.isAvailable());
        assertEquals("unavailable", FlacLibraryCheck.getVersion());
        assertFalse(FfmpegAudioEncoder.isMp3EncodingAvailable());
    }

    @Test
    void encodeWritesCanonicalWavContainerAndMetadata() {
        PcmAudio pcm = PcmAudio.fromChannelMajorFloat(
                new float[] {0.0f, 0.5f, -0.5f, 1.0f},
                2,
                2,
                16000,
                Map.of("voice", "test"));

        EncodedMedia encoded = Suling.encode(pcm, AudioEncodeOptions.builder()
                .format("wav")
                .metadata(Map.of("request", "unit-test"))
                .build());

        byte[] bytes = encoded.bytes();
        assertEquals("wav", encoded.format());
        assertEquals("audio/wav", encoded.mimeType());
        assertEquals("suling-java-wav-fallback", encoded.metadata().get("audio_encoder"));
        assertEquals("unit-test", encoded.metadata().get("request"));
        assertAscii(bytes, 0, "RIFF");
        assertAscii(bytes, 8, "WAVE");
        assertAscii(bytes, 12, "fmt ");
        assertAscii(bytes, 36, "data");
        assertEquals(44 + pcm.data().length, bytes.length);
        assertArrayEquals(pcm.data(), tail(bytes, 44));
    }

    @Test
    void encodeRejectsNonWavFormats() {
        PcmAudio pcm = PcmAudio.fromChannelMajorFloat(new float[] {0.1f}, 1, 1, 16000, Map.of());

        UnsupportedOperationException error = assertThrows(
                UnsupportedOperationException.class,
                () -> Suling.encode(pcm, AudioEncodeOptions.builder().format("mp3").build()));

        assertTrue(error.getMessage().contains("fallback only supports WAV"));
    }

    @Test
    void processingCanNormalizeAndTrimSilence() {
        PcmAudio pcm = PcmAudio.fromChannelMajorFloat(
                new float[] {0.0f, 0.0f, 0.2f, 0.0f, 0.0f},
                1,
                5,
                1000,
                Map.of());

        PcmAudio processed = Suling.process(pcm, AudioProcessingOptions.builder()
                .removeDcOffset(false)
                .peakNormalizeDbfs(-6.0)
                .trimSilence(true)
                .trimSilenceThresholdDbfs(-40.0)
                .trimSilencePaddingSeconds(0.0)
                .build());

        assertEquals(1, processed.samples());
        assertTrue(Math.abs(processed.channelMajorFloat()[0]) > 0.49f);
        assertEquals("4", processed.metadata().get("audio_processing.trimmed_samples"));
    }

    private static void assertAscii(byte[] bytes, int offset, String expected) {
        assertEquals(expected, new String(bytes, offset, expected.length(), StandardCharsets.US_ASCII));
    }

    private static byte[] tail(byte[] bytes, int offset) {
        byte[] tail = new byte[bytes.length - offset];
        System.arraycopy(bytes, offset, tail, 0, tail.length);
        return tail;
    }
}
