package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/**
 * Collators for multimodal datasets and trainer loops.
 */
public final class MultimodalCollators {
    private MultimodalCollators() {
    }

    public static Function<List<List<MultimodalContent>>, MultimodalBatch> batch() {
        return MultimodalBatch::new;
    }

    public static Function<List<List<MultimodalContent>>, MultimodalBatch> requireModalities(
            ModalityType first,
            ModalityType... rest) {
        Objects.requireNonNull(first, "first modality must not be null");
        EnumSet<ModalityType> required = EnumSet.of(first, rest == null ? new ModalityType[0] : rest);
        return batch -> {
            MultimodalBatch multimodalBatch = new MultimodalBatch(batch);
            for (int sampleIndex = 0; sampleIndex < multimodalBatch.sampleCount(); sampleIndex++) {
                Set<ModalityType> present = multimodalBatch.sampleModalities(sampleIndex);
                if (!present.containsAll(required)) {
                    EnumSet<ModalityType> missing = EnumSet.copyOf(required);
                    missing.removeAll(present);
                    throw new IllegalArgumentException(
                            "batch sample " + sampleIndex + " is missing required modalities " + missing);
                }
            }
            return multimodalBatch;
        };
    }

    public static Function<List<List<MultimodalContent>>, TextAssetBatch> textAssetBatch(ModalityType assetModality) {
        Objects.requireNonNull(assetModality, "assetModality must not be null");
        if (assetModality == ModalityType.TEXT) {
            throw new IllegalArgumentException("assetModality must be a non-text modality");
        }
        return batch -> collateTextAsset(batch, assetModality);
    }

    public static Function<List<ImageTextDataset.Sample>, ImageTextBatch> imageTextBatch() {
        return MultimodalCollators::collateImageText;
    }

    public static Function<List<AudioTextDataset.Sample>, AudioTextBatch> audioTextBatch() {
        return MultimodalCollators::collateAudioText;
    }

    private static ImageTextBatch collateImageText(List<ImageTextDataset.Sample> batch) {
        Objects.requireNonNull(batch, "batch must not be null");
        if (batch.isEmpty()) {
            throw new IllegalArgumentException("batch must not be empty");
        }
        GradTensor[] images = new GradTensor[batch.size()];
        List<String> texts = new ArrayList<>(batch.size());
        for (int i = 0; i < batch.size(); i++) {
            ImageTextDataset.Sample sample = Objects.requireNonNull(batch.get(i), "batch sample must not be null");
            images[i] = Objects.requireNonNull(sample.image(), "sample image tensor must not be null");
            String text = Objects.requireNonNull(sample.text(), "sample text must not be null").trim();
            if (text.isEmpty()) {
                throw new IllegalArgumentException("sample text must not be blank");
            }
            texts.add(text);
        }
        return new ImageTextBatch(GradTensor.stack(0, images), texts);
    }

    private static AudioTextBatch collateAudioText(List<AudioTextDataset.Sample> batch) {
        Objects.requireNonNull(batch, "batch must not be null");
        if (batch.isEmpty()) {
            throw new IllegalArgumentException("batch must not be empty");
        }
        int totalBytes = 0;
        int[] offsets = new int[batch.size()];
        int[] lengths = new int[batch.size()];
        List<String> texts = new ArrayList<>(batch.size());
        List<String> mimeTypes = new ArrayList<>(batch.size());
        for (int i = 0; i < batch.size(); i++) {
            AudioTextDataset.Sample sample = Objects.requireNonNull(batch.get(i), "batch sample must not be null");
            byte[] audio = sample.audio();
            if (audio.length == 0) {
                throw new IllegalArgumentException("sample audio must not be empty");
            }
            offsets[i] = totalBytes;
            lengths[i] = audio.length;
            totalBytes = Math.addExact(totalBytes, audio.length);
            String text = Objects.requireNonNull(sample.text(), "sample text must not be null").trim();
            if (text.isEmpty()) {
                throw new IllegalArgumentException("sample text must not be blank");
            }
            texts.add(text);
            mimeTypes.add(Objects.requireNonNull(sample.mimeType(), "sample mimeType must not be null"));
        }
        byte[] audioBytes = new byte[totalBytes];
        for (int i = 0; i < batch.size(); i++) {
            byte[] audio = batch.get(i).audio();
            System.arraycopy(audio, 0, audioBytes, offsets[i], audio.length);
        }
        return new AudioTextBatch(audioBytes, offsets, lengths, texts, mimeTypes);
    }

    private static TextAssetBatch collateTextAsset(List<List<MultimodalContent>> batch, ModalityType assetModality) {
        MultimodalBatch multimodalBatch = requireModalities(ModalityType.TEXT, assetModality).apply(batch);
        List<String> texts = new ArrayList<>(multimodalBatch.sampleCount());
        List<MultimodalContent> assets = new ArrayList<>(multimodalBatch.sampleCount());
        for (int sampleIndex = 0; sampleIndex < multimodalBatch.sampleCount(); sampleIndex++) {
            MultimodalContent text = firstContent(multimodalBatch.samples().get(sampleIndex), ModalityType.TEXT);
            MultimodalContent asset = firstContent(multimodalBatch.samples().get(sampleIndex), assetModality);
            String value = text.getText() == null ? "" : text.getText().trim();
            if (value.isEmpty()) {
                throw new IllegalArgumentException("batch sample " + sampleIndex + " has blank text content");
            }
            texts.add(value);
            assets.add(asset);
        }
        return new TextAssetBatch(assetModality, texts, assets);
    }

    private static MultimodalContent firstContent(List<MultimodalContent> sample, ModalityType modality) {
        for (MultimodalContent content : sample) {
            if (content.getModality() == modality) {
                return content;
            }
        }
        throw new IllegalArgumentException("sample is missing modality " + modality);
    }

    public record ImageTextBatch(GradTensor images, List<String> texts) {
        public ImageTextBatch {
            images = Objects.requireNonNull(images, "images must not be null");
            Objects.requireNonNull(texts, "texts must not be null");
            if (texts.isEmpty()) {
                throw new IllegalArgumentException("texts must not be empty");
            }
            texts = Collections.unmodifiableList(new ArrayList<>(texts));
        }

        public int size() {
            return texts.size();
        }
    }

    public record AudioTextBatch(
            byte[] audioBytes,
            int[] offsets,
            int[] lengths,
            List<String> texts,
            List<String> mimeTypes) {
        public AudioTextBatch {
            Objects.requireNonNull(audioBytes, "audioBytes must not be null");
            Objects.requireNonNull(offsets, "offsets must not be null");
            Objects.requireNonNull(lengths, "lengths must not be null");
            Objects.requireNonNull(texts, "texts must not be null");
            Objects.requireNonNull(mimeTypes, "mimeTypes must not be null");
            if (offsets.length == 0) {
                throw new IllegalArgumentException("audio batch must not be empty");
            }
            if (offsets.length != lengths.length || offsets.length != texts.size() || offsets.length != mimeTypes.size()) {
                throw new IllegalArgumentException("audio batch arrays must have the same sample count");
            }
            for (int i = 0; i < lengths.length; i++) {
                if (lengths[i] <= 0) {
                    throw new IllegalArgumentException("audio length must be positive at sample " + i);
                }
                if (offsets[i] < 0 || offsets[i] + lengths[i] > audioBytes.length) {
                    throw new IllegalArgumentException("audio offset/length is outside concatenated buffer at sample " + i);
                }
            }
            audioBytes = audioBytes.clone();
            offsets = offsets.clone();
            lengths = lengths.clone();
            texts = Collections.unmodifiableList(new ArrayList<>(texts));
            mimeTypes = Collections.unmodifiableList(new ArrayList<>(mimeTypes));
        }

        @Override
        public byte[] audioBytes() {
            return audioBytes.clone();
        }

        @Override
        public int[] offsets() {
            return offsets.clone();
        }

        @Override
        public int[] lengths() {
            return lengths.clone();
        }

        public int size() {
            return lengths.length;
        }

        public int totalBytes() {
            return audioBytes.length;
        }

        public byte[] sampleAudio(int index) {
            byte[] sample = new byte[lengths[index]];
            System.arraycopy(audioBytes, offsets[index], sample, 0, lengths[index]);
            return sample;
        }
    }
}
