package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.train.data.Dataset;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Dataset for audio files paired with sidecar text files.
 *
 * <p>Each audio file is paired with a UTF-8 {@code .txt} file that has the same
 * base name, for example {@code clip.wav} and {@code clip.txt}. Unpaired files
 * are skipped. Paths are scanned deterministically so train/validation splits
 * and tests stay reproducible across platforms.</p>
 */
public class AudioTextDataset implements Dataset<AudioTextDataset.Sample> {
    public record Sample(byte[] audio, String text, String mimeType, Path audioPath) {
        public Sample {
            Objects.requireNonNull(audio, "audio must not be null");
            if (audio.length == 0) {
                throw new IllegalArgumentException("audio must not be empty");
            }
            text = Objects.requireNonNull(text, "text must not be null").trim();
            if (text.isEmpty()) {
                throw new IllegalArgumentException("text must not be blank");
            }
            mimeType = Objects.requireNonNull(mimeType, "mimeType must not be null").trim();
            if (mimeType.isEmpty()) {
                throw new IllegalArgumentException("mimeType must not be blank");
            }
            audioPath = Objects.requireNonNull(audioPath, "audioPath must not be null").toAbsolutePath().normalize();
            audio = audio.clone();
        }

        @Override
        public byte[] audio() {
            return audio.clone();
        }
    }

    private final List<Path> audioPaths;
    private final List<Path> textPaths;

    public AudioTextDataset(Path directory) throws IOException {
        List<Path> audio = new ArrayList<>();
        List<Path> text = new ArrayList<>();
        for (Path audioPath : MultimodalFileSupport.regularFilesWithExtensions(
                directory,
                MultimodalFileSupport.AUDIO_EXTENSIONS)) {
            Path textPath = MultimodalFileSupport.sidecarTextPath(audioPath);
            if (Files.isRegularFile(textPath)) {
                audio.add(audioPath);
                text.add(textPath);
            }
        }
        this.audioPaths = Collections.unmodifiableList(audio);
        this.textPaths = Collections.unmodifiableList(text);
    }

    @Override
    public Sample get(int index) {
        Path audioPath = audioPaths.get(index);
        Path textPath = textPaths.get(index);
        try {
            return new Sample(
                    Files.readAllBytes(audioPath),
                    Files.readString(textPath, StandardCharsets.UTF_8),
                    MultimodalFileSupport.audioMimeType(audioPath),
                    audioPath);
        } catch (IOException error) {
            throw new RuntimeException("Error loading audio-text sample at index " + index, error);
        }
    }

    @Override
    public int size() {
        return audioPaths.size();
    }

    public Path getAudioPath(int index) {
        return audioPaths.get(index);
    }

    public Path getTextPath(int index) {
        return textPaths.get(index);
    }

    public List<Path> audioPaths() {
        return audioPaths;
    }

    public List<Path> textPaths() {
        return textPaths;
    }
}
