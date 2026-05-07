package tech.kayys.gollek.ml.models;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import static org.junit.jupiter.api.Assertions.*;

class WhisperTest {

    @Test
    void testWhisperForwardShapes() {
        Whisper.WhisperConfig config = Whisper.WhisperConfig.tiny();
        Whisper model = new Whisper(config);

        // Mel Spectrogram shape: [batch, dMels, timeFrames]
        // Whisper Tiny uses dMels=80, and typically max 3000 frames (30 seconds)
        GradTensor mel = GradTensor.randn(1, 80, 100);
        
        // 1. Encoder forward
        GradTensor audioFeatures = model.encoder.forward(mel);
        
        // Encoder strides by 2, so 100 frames -> 50 sequence length
        // Shape: [batch, seqLen, dModel]
        long[] encShape = audioFeatures.shape();
        assertEquals(1, encShape[0]);
        assertEquals(50, encShape[1]);
        assertEquals(config.dModel, encShape[2]);

        // 2. Decoder decode step
        // Tokens shape: [batch, seqLen]
        GradTensor tokens = GradTensor.of(new float[]{50257, 50259}, 1, 2); // BOS tokens
        GradTensor logits = model.decode(tokens, audioFeatures);
        
        // Output shape: [batch, seqLen, vocabSize]
        long[] decShape = logits.shape();
        assertEquals(1, decShape[0]);
        assertEquals(2, decShape[1]);
        assertEquals(config.vocabSize, decShape[2]);
    }
}
