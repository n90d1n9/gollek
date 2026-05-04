package tech.kayys.gollek.ml.audio;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import static org.junit.jupiter.api.Assertions.*;

class AudioTest {

    @Test
    void testSTFTDimensions() {
        int nFft = 512;
        int hop = 160;
        int win = 400;
        STFT stft = new STFT(nFft, hop, win);

        // Input: 1 second of audio at 16k
        GradTensor input = GradTensor.randn(1, 16000);
        GradTensor output = stft.forward(input);

        // Output shape: [batch=1, freqs=257, frames]
        // frames = (16000 - 400) / 160 + 1 = 15600 / 160 + 1 = 97.5 + 1 -> 98
        long[] shape = output.shape();
        assertEquals(1, shape[0]);
        assertEquals(257, shape[1]);
        assertTrue(shape[2] > 0);
    }

    @Test
    void testMelSpectrogram() {
        int sampleRate = 16000;
        int nFft = 400;
        // Whisper uses n_fft=400, hop=160, win=400 (or similar)
        // Wait, n_fft must be power of 2 for my Cooley-Tukey STFT
        MelSpectrogram mel = new MelSpectrogram(sampleRate, 512, 400, 160, 80);

        GradTensor input = GradTensor.randn(1, 1600); // 0.1s
        GradTensor output = mel.forward(input);

        // Output shape: [batch=1, n_mels=80, frames]
        long[] shape = output.shape();
        assertEquals(1, shape[0]);
        assertEquals(80, shape[1]);
        assertTrue(shape[2] > 0);
    }
}
