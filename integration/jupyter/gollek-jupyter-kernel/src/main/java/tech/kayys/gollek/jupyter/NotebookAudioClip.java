package tech.kayys.gollek.jupyter;

/**
 * Minimal notebook-native WAV audio payload for inline Jupyter rendering.
 */
public final class NotebookAudioClip {
    private final String title;
    private final int sampleRate;
    private final int channels;
    private final byte[] wavBytes;

    private NotebookAudioClip(String title, int sampleRate, int channels, byte[] wavBytes) {
        this.title = title;
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.wavBytes = wavBytes;
    }

    public static NotebookAudioClip ofWav(String title, int sampleRate, int channels, byte[] wavBytes) {
        if (wavBytes == null || wavBytes.length == 0) {
            throw new IllegalArgumentException("wavBytes must not be empty");
        }
        return new NotebookAudioClip(title, sampleRate, channels, wavBytes.clone());
    }

    public String title() {
        return title;
    }

    public int sampleRate() {
        return sampleRate;
    }

    public int channels() {
        return channels;
    }

    public byte[] wavBytes() {
        return wavBytes.clone();
    }

    public int byteSize() {
        return wavBytes.length;
    }

    @Override
    public String toString() {
        return "NotebookAudioClip(title=" + title + ", sampleRate=" + sampleRate
                + ", channels=" + channels + ", bytes=" + wavBytes.length + ")";
    }
}
