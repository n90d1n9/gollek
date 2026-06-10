package tech.kayys.suling.audio;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

public final class PcmAudio {
    private final float[] channelMajorFloat;
    private final int channels;
    private final int samples;
    private final double sampleRate;
    private final Map<String, String> metadata;

    PcmAudio(
            float[] channelMajorFloat,
            int channels,
            int samples,
            double sampleRate,
            Map<String, String> metadata) {
        this.channelMajorFloat = Objects.requireNonNull(channelMajorFloat, "channelMajorFloat must not be null").clone();
        if (channels <= 0) {
            throw new IllegalArgumentException("channels must be positive");
        }
        if (samples < 0) {
            throw new IllegalArgumentException("samples must not be negative");
        }
        if (channelMajorFloat.length != channels * samples) {
            throw new IllegalArgumentException("PCM data length must equal channels * samples");
        }
        if (sampleRate <= 0.0) {
            throw new IllegalArgumentException("sampleRate must be positive");
        }
        this.channels = channels;
        this.samples = samples;
        this.sampleRate = sampleRate;
        this.metadata = Collections.unmodifiableMap(new LinkedHashMap<>(
                metadata == null ? Map.of() : metadata));
    }

    public static PcmAudio fromChannelMajorFloat(
            float[] data,
            int channels,
            int samples,
            double sampleRate,
            Map<String, String> metadata) {
        return new PcmAudio(data, channels, samples, sampleRate, metadata);
    }

    public byte[] data() {
        byte[] pcm = new byte[samples * channels * 2];
        int cursor = 0;
        for (int sample = 0; sample < samples; sample++) {
            for (int channel = 0; channel < channels; channel++) {
                float value = channelMajorFloat[channel * samples + sample];
                int pcm16 = Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, Math.round(value * 32767f)));
                pcm[cursor++] = (byte) (pcm16 & 0xff);
                pcm[cursor++] = (byte) ((pcm16 >>> 8) & 0xff);
            }
        }
        return pcm;
    }

    public float[] channelMajorFloat() {
        return channelMajorFloat.clone();
    }

    public int channels() {
        return channels;
    }

    public int samples() {
        return samples;
    }

    public double sampleRate() {
        return sampleRate;
    }

    public Map<String, String> metadata() {
        return metadata;
    }
}
