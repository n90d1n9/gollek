package tech.kayys.suling.audio;

import java.io.ByteArrayOutputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Pure-Java fallback for the optional native Suling audio module.
 */
public final class Suling {
    private Suling() {
    }

    public static List<String> supportedAudioFormats() {
        return List.of("wav");
    }

    public static String diagnostics() {
        return "Suling native module is not installed; pure Java WAV fallback is active. "
                + "FLAC and MP3 encoding require the external Suling module.";
    }

    public static PcmAudio process(PcmAudio pcm, AudioProcessingOptions options) {
        Objects.requireNonNull(pcm, "pcm must not be null");
        if (options == null || !options.enabled()) {
            return pcm;
        }
        float[] data = pcm.channelMajorFloat();
        if (options.removeDcOffset()) {
            removeDcOffset(data, pcm.channels(), pcm.samples());
        }
        applyGain(data, dbToLinear(options.gainDb()));
        if (options.peakNormalizeDbfs() != null) {
            peakNormalize(data, options.peakNormalizeDbfs(), options.maxNormalizeGainDb());
        }
        applyFade(data, pcm.channels(), pcm.samples(), pcm.sampleRate(), options.fadeInSeconds(), true);
        applyFade(data, pcm.channels(), pcm.samples(), pcm.sampleRate(), options.fadeOutSeconds(), false);

        PcmAudio processed = new PcmAudio(data, pcm.channels(), pcm.samples(), pcm.sampleRate(),
                withMetadata(pcm.metadata(), "audio_processing.fallback", "suling-java-wav"));
        return options.trimSilence()
                ? trimSilence(processed, options.trimSilenceThresholdDbfs(), options.trimSilencePaddingSeconds())
                : processed;
    }

    public static EncodedMedia encode(PcmAudio pcm, AudioEncodeOptions options) {
        Objects.requireNonNull(pcm, "pcm must not be null");
        AudioEncodeOptions effective = options == null ? AudioEncodeOptions.builder().build() : options;
        if (!"wav".equals(effective.format())) {
            throw new UnsupportedOperationException(
                    "Suling fallback only supports WAV. Install the external Suling module for "
                            + effective.format().toUpperCase(java.util.Locale.ROOT) + ".");
        }
        Map<String, String> metadata = new LinkedHashMap<>(effective.metadata());
        metadata.put("audio_encoder", "suling-java-wav-fallback");
        metadata.put("audio_format", "wav");
        metadata.put("audio_mime", "audio/wav");
        return new EncodedMedia(wavBytes(pcm), "wav", "audio/wav", metadata);
    }

    private static byte[] wavBytes(PcmAudio pcm) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(44 + pcm.samples() * pcm.channels() * 2);
        int sampleRate = (int) Math.round(pcm.sampleRate());
        int byteRate = sampleRate * pcm.channels() * 2;
        int dataBytes = pcm.samples() * pcm.channels() * 2;
        writeAscii(out, "RIFF");
        writeLe32(out, 36 + dataBytes);
        writeAscii(out, "WAVE");
        writeAscii(out, "fmt ");
        writeLe32(out, 16);
        writeLe16(out, 1);
        writeLe16(out, pcm.channels());
        writeLe32(out, sampleRate);
        writeLe32(out, byteRate);
        writeLe16(out, pcm.channels() * 2);
        writeLe16(out, 16);
        writeAscii(out, "data");
        writeLe32(out, dataBytes);
        float[] channelMajor = pcm.channelMajorFloat();
        for (int sample = 0; sample < pcm.samples(); sample++) {
            for (int channel = 0; channel < pcm.channels(); channel++) {
                float value = channelMajor[channel * pcm.samples() + sample];
                int pcm16 = Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, Math.round(value * 32767f)));
                writeLe16(out, pcm16);
            }
        }
        return out.toByteArray();
    }

    private static void removeDcOffset(float[] data, int channels, int samples) {
        if (samples == 0) {
            return;
        }
        for (int channel = 0; channel < channels; channel++) {
            int offset = channel * samples;
            double sum = 0.0;
            for (int i = 0; i < samples; i++) {
                sum += data[offset + i];
            }
            float mean = (float) (sum / samples);
            for (int i = 0; i < samples; i++) {
                data[offset + i] -= mean;
            }
        }
    }

    private static void applyGain(float[] data, double gain) {
        if (gain == 1.0) {
            return;
        }
        for (int i = 0; i < data.length; i++) {
            data[i] = clamp((float) (data[i] * gain));
        }
    }

    private static void peakNormalize(float[] data, double targetDbfs, double maxGainDb) {
        float peak = 0f;
        for (float value : data) {
            peak = Math.max(peak, Math.abs(value));
        }
        if (peak <= 1.0e-9f) {
            return;
        }
        double target = dbToLinear(targetDbfs);
        double gain = Math.min(target / peak, dbToLinear(maxGainDb));
        applyGain(data, gain);
    }

    private static PcmAudio trimSilence(PcmAudio pcm, double thresholdDbfs, double paddingSeconds) {
        float[] data = pcm.channelMajorFloat();
        int samples = pcm.samples();
        int channels = pcm.channels();
        if (samples == 0) {
            return pcm;
        }
        double threshold = dbToLinear(thresholdDbfs);
        int first = 0;
        int last = samples - 1;
        while (first < samples && framePeak(data, channels, samples, first) < threshold) {
            first++;
        }
        while (last > first && framePeak(data, channels, samples, last) < threshold) {
            last--;
        }
        int padding = (int) Math.round(Math.max(0.0, paddingSeconds) * pcm.sampleRate());
        first = Math.max(0, first - padding);
        last = Math.min(samples - 1, last + padding);
        int trimmedSamples = Math.max(0, last - first + 1);
        if (trimmedSamples == samples) {
            return pcm;
        }
        float[] trimmed = new float[channels * trimmedSamples];
        for (int channel = 0; channel < channels; channel++) {
            System.arraycopy(data, channel * samples + first, trimmed, channel * trimmedSamples, trimmedSamples);
        }
        return new PcmAudio(trimmed, channels, trimmedSamples, pcm.sampleRate(),
                withMetadata(pcm.metadata(), "audio_processing.trimmed_samples", String.valueOf(samples - trimmedSamples)));
    }

    private static double framePeak(float[] data, int channels, int samples, int sample) {
        double peak = 0.0;
        for (int channel = 0; channel < channels; channel++) {
            peak = Math.max(peak, Math.abs(data[channel * samples + sample]));
        }
        return peak;
    }

    private static void applyFade(
            float[] data,
            int channels,
            int samples,
            double sampleRate,
            double seconds,
            boolean fadeIn) {
        int fadeSamples = Math.min(samples, (int) Math.round(Math.max(0.0, seconds) * sampleRate));
        if (fadeSamples <= 1) {
            return;
        }
        for (int i = 0; i < fadeSamples; i++) {
            float factor = i / (float) (fadeSamples - 1);
            if (!fadeIn) {
                factor = 1.0f - factor;
            }
            int sample = fadeIn ? i : samples - fadeSamples + i;
            for (int channel = 0; channel < channels; channel++) {
                int index = channel * samples + sample;
                data[index] = clamp(data[index] * factor);
            }
        }
    }

    private static Map<String, String> withMetadata(Map<String, String> existing, String key, String value) {
        Map<String, String> metadata = new LinkedHashMap<>(existing == null ? Map.of() : existing);
        metadata.put(key, value);
        return metadata;
    }

    private static float clamp(float value) {
        return Math.max(-1f, Math.min(1f, value));
    }

    private static double dbToLinear(double db) {
        return Math.pow(10.0, db / 20.0);
    }

    private static void writeAscii(ByteArrayOutputStream out, String value) {
        for (int i = 0; i < value.length(); i++) {
            out.write(value.charAt(i));
        }
    }

    private static void writeLe16(ByteArrayOutputStream out, int value) {
        out.write(value & 0xff);
        out.write((value >>> 8) & 0xff);
    }

    private static void writeLe32(ByteArrayOutputStream out, int value) {
        out.write(value & 0xff);
        out.write((value >>> 8) & 0xff);
        out.write((value >>> 16) & 0xff);
        out.write((value >>> 24) & 0xff);
    }
}
