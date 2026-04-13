package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.audio.AudioBuffer;
import tech.kayys.gollek.ml.audio.MelSpectrogram;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import java.util.ArrayList;
import java.util.List;

/**
 * High-level orchestration for Whisper STT (Speech-to-Text).
 * <p>
 * Handles the full pipeline:
 * 1. Audio Preprocessing (SIMD-accelerated via AudioBuffer/MelSpectrogram)
 * 2. Encoder Pass (Audio Feature Extraction)
 * 3. Decoder Loop (Greedy Decoding for Transcription)
 */
public class WhisperPipeline {

    private final Whisper model;
    private final MelSpectrogram processor;
    private final Whisper.WhisperConfig config;

    public WhisperPipeline(Whisper model, MelSpectrogram processor, Whisper.WhisperConfig config) {
        this.model = model;
        this.processor = processor;
        this.config = config;
    }

    /**
     * Transcribes raw audio from an AudioBuffer.
     * Uses Greedy Search to generate token IDs.
     * 
     * @param audio the input audio buffer (must be 16kHz for standard Whisper)
     * @return a list of generated token IDs
     */
    public List<Integer> transcribe(AudioBuffer audio) {
        // 1. Audio to Mel Spectrogram
        // Convert AudioBuffer to GradTensor [1, d_mels, time]
        float[] audioData = audio.toArray();
        GradTensor audioTensor = GradTensor.of(audioData, 1, audioData.length);
        GradTensor mel = processor.forward(audioTensor);

        // 2. Encoder Pass
        GradTensor audioFeatures = model.encoder.forward(mel);

        // 3. Greedy Decoding Loop
        List<Integer> tokens = new ArrayList<>();
        tokens.add(config.sot);
        tokens.add(config.langEn);
        tokens.add(config.transcribe);
        tokens.add(config.noTimestamps);

        int maxNewTokens = config.maxTargetPos;
        for (int i = 0; i < maxNewTokens; i++) {
            // Prepare decoder input: [batch=1, seq_len]
            float[] tokenData = new float[tokens.size()];
            for (int t = 0; t < tokens.size(); t++) tokenData[t] = tokens.get(t);
            GradTensor decoderIn = GradTensor.of(tokenData, 1, tokens.size());

            // Decoder forward: [1, seq_len, vocab_size]
            GradTensor logits = model.decoder.forward(decoderIn, audioFeatures);
            
            // Extract last token's logits: [1, 1, vocab_size]
            // We use indexSelect on dimension 1 (sequence dimension)
            GradTensor lastTokenLogits = logits.indexSelect(1, tokens.size() - 1);
            
            // Argmax to finding next token
            GradTensor nextTokenTensor = lastTokenLogits.argmax(-1);
            int nextTokenId = (int) nextTokenTensor.item();
            
            System.out.print("."); // Progress indicator
            System.out.flush();

            if (nextTokenId == config.eot) {
                break;
            }

            tokens.add(nextTokenId);
        }
        System.out.println(); // New line after dots

        return tokens;
    }
}
