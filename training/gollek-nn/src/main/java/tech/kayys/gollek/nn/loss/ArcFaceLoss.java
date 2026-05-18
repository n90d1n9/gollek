package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.VectorOps;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;

import java.util.Arrays;

/**
 * ArcFace Loss (Additive Angular Margin) — improves face recognition and
 * metric learning by adding an angular margin to the target class logit.
 *
 * <p>
 * Based on <em>"ArcFace: Additive Angular Margin Loss for Deep Face
 * Recognition"</em>
 * (Deng et al., 2019).
 *
 * <p>
 * Formula:
 * 
 * <pre>
 *   L = -log( exp(s·cos(θ_yi + m)) / (exp(s·cos(θ_yi + m)) + Σ_{j≠yi} exp(s·cos(θ_j))) )
 * </pre>
 * 
 * where θ is the angle between the feature and the weight vector,
 * m is the angular margin, and s is the feature scale.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var loss = new ArcFaceLoss(numClasses = 10575, margin = 0.5f, scale = 64f);
 * GradTensor l = loss.forward(features, labels); // features [N,D], labels [N]
 * }</pre>
 */
public final class ArcFaceLoss extends NNModule {

    private final float margin;
    private final float scale;
    private final float cosMargin; // cos(m)
    private final float sinMargin; // sin(m)
    private final Parameter weight; // [numClasses, featureDim]

    /**
     * Creates an ArcFace loss layer.
     *
     * @param numClasses number of identity classes
     * @param featureDim embedding dimension
     * @param margin     angular margin m in radians (default 0.5 ≈ 28.6°)
     * @param scale      feature scale s (default 64)
     */
    public ArcFaceLoss(int numClasses, int featureDim, float margin, float scale) {
        int validatedClasses = validatePositive(numClasses, "numClasses");
        int validatedFeatureDim = validatePositive(featureDim, "featureDim");
        this.margin = validateMargin(margin);
        this.scale = validateScale(scale);
        this.cosMargin = (float) Math.cos(this.margin);
        this.sinMargin = (float) Math.sin(this.margin);
        // Weight matrix: each row is a class center (normalized)
        this.weight = registerParameter("weight",
                GradTensor.randn(validatedClasses, validatedFeatureDim));
    }

    /** Creates ArcFace with default margin=0.5, scale=64. */
    public ArcFaceLoss(int numClasses, int featureDim) {
        this(numClasses, featureDim, 0.5f, 64f);
    }

    /**
     * Computes ArcFace loss.
     *
     * @param features L2-normalized embeddings {@code [N, featureDim]}
     * @param labels   class indices {@code [N]}
     * @return scalar cross-entropy loss with angular margin
     */
    public GradTensor forward(GradTensor features, GradTensor labels) {
        long[] featureShape = features.shape();
        if (featureShape.length != 2) {
            throw new IllegalArgumentException(
                    "features must be 2D [batch, featureDim], got shape: " + Arrays.toString(featureShape));
        }
        long[] weightShape = weight.data().shape();
        int N = (int) featureShape[0];
        int C = (int) weightShape[0];
        int D = (int) featureShape[1];
        if (D != (int) weightShape[1]) {
            throw new IllegalArgumentException(
                    "features featureDim " + D + " must match ArcFace weight featureDim " + weightShape[1]);
        }
        float[] lb = ClassIndexTargets.requireVectorData(labels, N, "labels");
        int[] targetClasses = new int[N];
        for (int n = 0; n < N; n++) {
            targetClasses[n] = ClassIndexTargets.require(lb[n], C, n);
        }

        // Normalize weight vectors
        GradTensor weightTensor = weight.data();
        float[] featureData = features.data();
        float[] weightData = weightTensor.data();
        float[] wn = normalizeRows(weightData, C, D);
        float[] fn = normalizeRows(featureData, N, D);

        // Cosine similarity: [N, C]
        float[] cosTheta = VectorOps.matmul(fn, transpose(wn, C, D), N, D, C);

        // Apply angular margin to target class
        float[] logits = cosTheta.clone();
        for (int n = 0; n < N; n++) {
            int cls = targetClasses[n];
            float cos = cosTheta[n * C + cls];
            float sin = (float) Math.sqrt(Math.max(0, 1 - cos * cos));
            // cos(θ + m) = cos·cos(m) - sin·sin(m)
            logits[n * C + cls] = cos * cosMargin - sin * sinMargin;
        }

        // Scale and cross-entropy
        float[] losses = new float[N];
        float[] softmax = new float[N * C];
        for (int n = 0; n < N; n++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < C; c++)
                max = Math.max(max, scale * logits[n * C + c]);
            float sumExp = 0;
            for (int c = 0; c < C; c++) {
                softmax[n * C + c] = (float) Math.exp(scale * logits[n * C + c] - max);
                sumExp += softmax[n * C + c];
            }
            for (int c = 0; c < C; c++) {
                softmax[n * C + c] /= sumExp;
            }
            int cls = targetClasses[n];
            losses[n] = -(scale * logits[n * C + cls] - max - (float) Math.log(sumExp));
        }
        GradTensor out = GradTensor.scalar(VectorOps.sum(losses) / N);
        if (features.requiresGrad() || weightTensor.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("ArcFaceLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float sampleScale = upstream.item() / N;
                    float[] gradCos = new float[N * C];

                    for (int n = 0; n < N; n++) {
                        int targetClass = targetClasses[n];
                        for (int c = 0; c < C; c++) {
                            float grad = softmax[n * C + c];
                            if (c == targetClass) {
                                grad -= 1.0f;
                            }
                            grad *= scale * sampleScale;

                            if (c == targetClass) {
                                float cos = cosTheta[n * C + c];
                                float sin = (float) Math.sqrt(Math.max(0.0f, 1.0f - cos * cos));
                                float safeSin = Math.max(sin, 1e-6f);
                                grad *= cosMargin + (cos * sinMargin / safeSin);
                            }
                            gradCos[n * C + c] = grad;
                        }
                    }

                    float[] gradFn = new float[N * D];
                    for (int n = 0; n < N; n++) {
                        for (int d = 0; d < D; d++) {
                            float grad = 0.0f;
                            for (int c = 0; c < C; c++) {
                                grad += gradCos[n * C + c] * wn[c * D + d];
                            }
                            gradFn[n * D + d] = grad;
                        }
                    }

                    float[] gradWn = new float[C * D];
                    for (int c = 0; c < C; c++) {
                        for (int d = 0; d < D; d++) {
                            float grad = 0.0f;
                            for (int n = 0; n < N; n++) {
                                grad += gradCos[n * C + c] * fn[n * D + d];
                            }
                            gradWn[c * D + d] = grad;
                        }
                    }

                    if (features.requiresGrad()) {
                        features.backward(GradTensor.of(denormalizeRowGradient(gradFn, featureData, N, D),
                                features.shape()));
                    }
                    if (weightTensor.requiresGrad()) {
                        weightTensor.backward(GradTensor.of(denormalizeRowGradient(gradWn, weightData, C, D),
                                weightTensor.shape()));
                    }
                }
            });
        }
        return out;
    }

    @Override
    public GradTensor forward(GradTensor x) {
        throw new UnsupportedOperationException("Use forward(features, labels)");
    }

    private static float[] normalizeRows(float[] m, int rows, int cols) {
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            float norm = 0;
            for (int c = 0; c < cols; c++)
                norm += m[r * cols + c] * m[r * cols + c];
            norm = (float) Math.sqrt(norm) + 1e-8f;
            for (int c = 0; c < cols; c++)
                out[r * cols + c] = m[r * cols + c] / norm;
        }
        return out;
    }

    private static float[] denormalizeRowGradient(float[] gradNormalized, float[] raw, int rows, int cols) {
        float[] gradRaw = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            int base = r * cols;
            float squaredNorm = 0.0f;
            float dot = 0.0f;
            for (int c = 0; c < cols; c++) {
                float value = raw[base + c];
                squaredNorm += value * value;
                dot += gradNormalized[base + c] * value;
            }
            float normWithoutEps = (float) Math.sqrt(squaredNorm);
            float norm = normWithoutEps + 1e-8f;
            for (int c = 0; c < cols; c++) {
                float grad = gradNormalized[base + c] / norm;
                if (normWithoutEps > 1e-12f) {
                    grad -= raw[base + c] * dot / (normWithoutEps * norm * norm);
                }
                gradRaw[base + c] = grad;
            }
        }
        return gradRaw;
    }

    private static float[] transpose(float[] m, int rows, int cols) {
        float[] t = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                t[c * rows + r] = m[r * cols + c];
        return t;
    }

    private static int validatePositive(int value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive, got: " + value);
        }
        return value;
    }

    private static float validateMargin(float margin) {
        if (!Float.isFinite(margin) || margin < 0.0f || margin >= Math.PI) {
            throw new IllegalArgumentException("margin must be finite and in [0, PI), got: " + margin);
        }
        return margin;
    }

    private static float validateScale(float scale) {
        if (!Float.isFinite(scale) || scale <= 0.0f) {
            throw new IllegalArgumentException("scale must be finite and positive, got: " + scale);
        }
        return scale;
    }
}
