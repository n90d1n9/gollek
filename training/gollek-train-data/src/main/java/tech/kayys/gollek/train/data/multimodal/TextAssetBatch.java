package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A typed projection for common multimodal supervised tasks.
 *
 * <p>Many trainer loops need one text target or prompt paired with one non-text
 * asset per sample: image-caption, audio-transcription, document-QA, video-caption,
 * and similar datasets. This record keeps that shape explicit without forcing the
 * generic manifest dataset to know about every future task.</p>
 */
public record TextAssetBatch(
        ModalityType assetModality,
        List<String> texts,
        List<MultimodalContent> assets) {
    public TextAssetBatch {
        assetModality = Objects.requireNonNull(assetModality, "assetModality must not be null");
        if (assetModality == ModalityType.TEXT) {
            throw new IllegalArgumentException("assetModality must be a non-text modality");
        }
        Objects.requireNonNull(texts, "texts must not be null");
        Objects.requireNonNull(assets, "assets must not be null");
        if (texts.isEmpty()) {
            throw new IllegalArgumentException("texts must not be empty");
        }
        if (texts.size() != assets.size()) {
            throw new IllegalArgumentException("texts and assets must have the same sample count");
        }

        List<String> textCopy = new ArrayList<>(texts.size());
        for (int i = 0; i < texts.size(); i++) {
            String text = Objects.requireNonNull(texts.get(i), "text must not be null at sample " + i).trim();
            if (text.isEmpty()) {
                throw new IllegalArgumentException("text must not be blank at sample " + i);
            }
            textCopy.add(text);
        }

        List<MultimodalContent> assetCopy = new ArrayList<>(assets.size());
        for (int i = 0; i < assets.size(); i++) {
            MultimodalContent asset = Objects.requireNonNull(assets.get(i), "asset must not be null at sample " + i);
            if (asset.getModality() != assetModality) {
                throw new IllegalArgumentException("asset at sample " + i + " must be " + assetModality
                        + " but was " + asset.getModality());
            }
            assetCopy.add(asset);
        }

        texts = Collections.unmodifiableList(textCopy);
        assets = Collections.unmodifiableList(assetCopy);
    }

    public int size() {
        return texts.size();
    }

    public String text(int index) {
        return texts.get(index);
    }

    public MultimodalContent asset(int index) {
        return assets.get(index);
    }

    public boolean assetIsInlined(int index) {
        return assets.get(index).getBase64Data() != null && !assets.get(index).getBase64Data().isBlank();
    }

    public String assetUri(int index) {
        return assets.get(index).getUri();
    }

    public String assetMimeType(int index) {
        String mimeType = assets.get(index).getMimeType();
        return mimeType == null || mimeType.isBlank() ? "application/octet-stream" : mimeType;
    }

    public String assetSourcePath(int index) {
        Map<String, Object> metadata = assets.get(index).getMetadata();
        Object value = metadata == null ? null : metadata.get("sourcePath");
        return value == null ? null : value.toString();
    }

    public List<String> assetUris() {
        List<String> uris = new ArrayList<>(assets.size());
        for (int i = 0; i < assets.size(); i++) {
            uris.add(assetUri(i));
        }
        return Collections.unmodifiableList(uris);
    }

    public List<String> assetMimeTypes() {
        List<String> mimeTypes = new ArrayList<>(assets.size());
        for (int i = 0; i < assets.size(); i++) {
            mimeTypes.add(assetMimeType(i));
        }
        return Collections.unmodifiableList(mimeTypes);
    }

    public List<String> assetSourcePaths() {
        List<String> sourcePaths = new ArrayList<>(assets.size());
        for (int i = 0; i < assets.size(); i++) {
            sourcePaths.add(assetSourcePath(i));
        }
        return Collections.unmodifiableList(sourcePaths);
    }
}
