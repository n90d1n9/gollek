package tech.kayys.gollek.ml.audio;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Computes a Mel Spectrogram from audio signals.
 * <p>
 * Input: [Batch, Time] or [Time] raw audio waveform.
 * Output: [Batch, n_mels, Time_frames] log-mel spectrogram.
 */
public class MelSpectrogram extends NNModule {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private final STFT stft;
    private final float[][] melFilters;
    private final int nMels;
    private final int numFreqs;

    public MelSpectrogram(int sampleRate, int nFft, int winLength, int hopLength, int nMels) {
        this.stft = new STFT(nFft, hopLength, winLength);
        this.nMels = nMels;
        this.numFreqs = nFft / 2 + 1;
        this.melFilters = computeMelFilterbank(sampleRate, nFft, nMels, 0f, sampleRate / 2f);
    }

    @Override
    public GradTensor forward(GradTensor input) {
        // 1. Get Linear Spectrogram Magnitudes -> [Batch, Freqs, Frames]
        GradTensor spec = stft.forward(input);
        
        long[] shape = spec.shape();
        int b = (int) shape[0];
        int f = (int) shape[1]; // should equal numFreqs
        int frames = (int) shape[2];

        float[] specData = spec.data();
        float[] melData = new float[b * nMels * frames];

        // 2. Matrix multiply with Mel Filterbank -> [Batch, n_mels, Frames]
        for (int batch = 0; batch < b; batch++) {
            int specBatchOffset = batch * f * frames;
            int melBatchOffset = batch * nMels * frames;

            for (int t = 0; t < frames; t++) {
                int[] indices = new int[SPECIES.length()];
                for (int l = 0; l < indices.length; l++) indices[l] = l * frames;

                for (int m = 0; m < nMels; m++) {
                    int i = 0;
                    int bound = SPECIES.loopBound(f);
                    FloatVector vSum = FloatVector.zero(SPECIES);
                    
                    for (; i < bound; i += SPECIES.length()) {
                        FloatVector vMag = FloatVector.fromArray(SPECIES, specData, specBatchOffset + i * frames + t, indices, 0);
                        FloatVector vFilter = FloatVector.fromArray(SPECIES, melFilters[m], i);
                        vSum = vSum.add(vMag.mul(vMag).mul(vFilter));
                    }
                    
                    float sum = vSum.reduceLanes(VectorOperators.ADD);
                    for (; i < f; i++) {
                        float magnitude = specData[specBatchOffset + i * frames + t];
                        sum += (magnitude * magnitude) * melFilters[m][i];
                    }
                    
                    float logMel = (float) Math.log(Math.max(sum, 1e-10f));
                    melData[melBatchOffset + m * frames + t] = logMel;
                }
            }
        }

        return GradTensor.of(melData, b, nMels, frames);
    }

    /**
     * Constructs a Mel filterbank matrix: [nMels, numFreqs]
     */
    private float[][] computeMelFilterbank(int sampleRate, int nFft, int nMels, float fMin, float fMax) {
        float[][] filters = new float[nMels][numFreqs];

        float minMel = hzToMel(fMin);
        float maxMel = hzToMel(fMax);
        float melStep = (maxMel - minMel) / (nMels + 1);

        float[] melPoints = new float[nMels + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = minMel + i * melStep;
        }

        float[] hzPoints = new float[melPoints.length];
        for (int i = 0; i < hzPoints.length; i++) {
            hzPoints[i] = melToHz(melPoints[i]);
        }

        int[] binPoints = new int[hzPoints.length];
        for (int i = 0; i < hzPoints.length; i++) {
            binPoints[i] = (int) Math.floor((nFft + 1) * hzPoints[i] / sampleRate);
        }

        for (int m = 1; m <= nMels; m++) {
            int left = binPoints[m - 1];
            int center = binPoints[m];
            int right = binPoints[m + 1];

            for (int k = left; k < center; k++) {
                filters[m - 1][k] = (k - left) / (float) (center - left);
            }
            for (int k = center; k < right; k++) {
                filters[m - 1][k] = (right - k) / (float) (right - center);
            }
        }

        return filters;
    }

    private float hzToMel(float hz) {
        return 2595f * (float) Math.log10(1 + hz / 700f);
    }

    private float melToHz(float mel) {
        return 700f * ((float) Math.pow(10, mel / 2595f) - 1f);
    }
}
