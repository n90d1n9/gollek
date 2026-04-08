package tech.kayys.gollek.spi.model;

/**
 * Enumeration of content modality types.
 */
public enum ModalityType {
    TEXT,
    IMAGE,
    AUDIO,
    VIDEO,
    DOCUMENT,
    EMBEDDING,
    TIME_SERIES;

    public boolean isBinary() {
        return this != TEXT;
    }
}
