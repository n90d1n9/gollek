package tech.kayys.gollek.ml.audio;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.nio.ByteOrder;

/**
 * Short-Time Fourier Transform (STFT) for calculating complex spectrograms.
 * Computes frequency magnitudes across sliding windows.
 */
public class STFT extends NNModule {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private final int nFft;
    private final int hopLength;
    private final int winLength;
    private final float[] window;

    public STFT(int nFft, int hopLength, int winLength) {
        this.nFft = nFft;
        this.hopLength = hopLength;
        this.winLength = winLength;
        this.window = computeHannWindow(winLength);
        
        if ((nFft & (nFft - 1)) != 0) {
            throw new IllegalArgumentException("nFft must be a power of 2 for radix-2 FFT");
        }
    }

    private float[] computeHannWindow(int length) {
        float[] win = new float[length];
        for (int i = 0; i < length; i++) {
            win[i] = (float) (0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (length - 1))));
        }
        return win;
    }

    @Override
    public GradTensor forward(GradTensor input) {
        // Assume input is [batch, time] or [time]
        long[] shape = input.shape();
        boolean isBatched = shape.length == 2;
        int batchSize = isBatched ? (int) shape[0] : 1;
        int samples = (int) shape[shape.length - 1];

        int numFrames = (samples - winLength) / hopLength + 1;
        int numFreqs = nFft / 2 + 1;

        float[] inData = input.data();
        float[] outData = new float[batchSize * numFreqs * numFrames];

        for (int b = 0; b < batchSize; b++) {
            int batchOffset = b * samples;
            int outBatchOffset = b * numFrames * numFreqs;

            for (int f = 0; f < numFrames; f++) {
                int frameStart = batchOffset + f * hopLength;

                // 1. Copy and apply window (Vectorized)
                float[] bufferReal = new float[nFft];
                float[] bufferImag = new float[nFft];
                
                int i = 0;
                int windowBound = SPECIES.loopBound(winLength);
                for (; i < windowBound; i += SPECIES.length()) {
                    if (frameStart + i + SPECIES.length() <= samples) {
                        FloatVector vIn = FloatVector.fromArray(SPECIES, inData, frameStart + i);
                        FloatVector vWin = FloatVector.fromArray(SPECIES, window, i);
                        vIn.mul(vWin).intoArray(bufferReal, i);
                    } else {
                        break; // fallback to scalar for safe padding
                    }
                }
                
                // Scalar tail and padding
                for (; i < winLength; i++) {
                    if (frameStart + i < samples) {
                        bufferReal[i] = inData[frameStart + i] * window[i];
                    }
                }

                // 2. Perform FFT (in-place)
                fft(bufferReal, bufferImag);

                // 3. Calculate Magnitudes and populate output (Vectorized)
                int m = 0;
                int magBound = SPECIES.loopBound(numFreqs);
                for (; m < magBound; m += SPECIES.length()) {
                    FloatVector vReal = FloatVector.fromArray(SPECIES, bufferReal, m);
                    FloatVector vImag = FloatVector.fromArray(SPECIES, bufferImag, m);
                    // mag = sqrt(real^2 + imag^2)
                    FloatVector vMag = vReal.mul(vReal).add(vImag.mul(vImag)).sqrt();
                    
                    // We need to scatter these into outData[outBatchOffset + m * numFrames + f]
                    // This is still somewhat scalar because of the strided output format
                    for (int l = 0; l < SPECIES.length(); l++) {
                        outData[outBatchOffset + (m + l) * numFrames + f] = vMag.lane(l);
                    }
                }

                // Scalar tail
                for (; m < numFreqs; m++) {
                    float real = bufferReal[m];
                    float imag = bufferImag[m];
                    float mag = (float) Math.sqrt(real * real + imag * imag);
                    outData[outBatchOffset + m * numFrames + f] = mag;
                }
            }
        }

        return GradTensor.of(outData, batchSize, numFreqs, numFrames);
    }

    /**
     * Radix-2 Cooley-Tukey FFT algorithm
     */
    private void fft(float[] real, float[] imag) {
        int n = real.length;

        // Bit reversal permutation
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j >= bit; bit >>= 1) {
                j -= bit;
            }
            j += bit;
            if (i < j) {
                float tempR = real[i];
                real[i] = real[j];
                real[j] = tempR;

                float tempI = imag[i];
                imag[i] = imag[j];
                imag[j] = tempI;
            }
        }

        // Cooley-Tukey
        for (int len = 2; len <= n; len <<= 1) {
            float angle = (float) (-2 * Math.PI / len);
            float wlenReal = (float) Math.cos(angle);
            float wlenImag = (float) Math.sin(angle);

            for (int i = 0; i < n; i += len) {
                float wReal = 1;
                float wImag = 0;
                for (int j = 0; j < len / 2; j++) {
                    int u = i + j;
                    int v = i + j + len / 2;
                    float tx = real[v] * wReal - imag[v] * wImag;
                    float ty = real[v] * wImag + imag[v] * wReal;

                    real[v] = real[u] - tx;
                    imag[v] = imag[u] - ty;
                    real[u] += tx;
                    imag[u] += ty;

                    float nextWReal = wReal * wlenReal - wImag * wlenImag;
                    float nextWImag = wReal * wlenImag + wImag * wlenReal;
                    wReal = nextWReal;
                    wImag = nextWImag;
                }
            }
        }
    }
}
