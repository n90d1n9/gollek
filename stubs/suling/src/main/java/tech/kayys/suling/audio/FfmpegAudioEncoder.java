package tech.kayys.suling.audio;

/**
 * Fallback FFmpeg probe used when native Suling bindings are absent.
 */
public final class FfmpegAudioEncoder {
    private FfmpegAudioEncoder() {
    }

    public static boolean isMp3EncodingAvailable() {
        return false;
    }
}
