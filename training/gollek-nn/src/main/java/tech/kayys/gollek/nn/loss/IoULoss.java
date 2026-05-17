package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Arrays;

/**
 * IoU Loss (Jaccard Loss) — 1 - IoU, for object detection bounding box
 * regression.
 *
 * <p>
 * Directly optimizes the IoU metric rather than a surrogate loss.
 * More robust to scale than MSE/SmoothL1 for bounding boxes.
 *
 * <p>
 * Expects boxes in {@code [x1, y1, x2, y2]} format.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var loss = new IoULoss();
 * // pred/target: [N, 4] in (x1,y1,x2,y2) format
 * GradTensor l = loss.forward(predBoxes, targetBoxes);
 * }</pre>
 */
public final class IoULoss {
    private static final float EPS = 1e-7f;

    /**
     * Computes mean IoU loss over a batch of bounding boxes.
     *
     * @param pred   predicted boxes {@code [N, 4]} in (x1,y1,x2,y2) format
     * @param target ground-truth boxes {@code [N, 4]}
     * @return scalar mean IoU loss (1 - mean_IoU)
     */
    public GradTensor forward(GradTensor pred, GradTensor target) {
        long[] predShape = pred.shape();
        long[] targetShape = target.shape();
        if (!Arrays.equals(predShape, targetShape)) {
            throw new IllegalArgumentException(
                    "pred and target shapes must match, got: "
                            + Arrays.toString(predShape) + " vs " + Arrays.toString(targetShape));
        }
        if (predShape.length != 2 || predShape[1] != 4) {
            throw new IllegalArgumentException(
                    "pred and target must be 2D bounding box tensors [batch, 4], got: "
                            + Arrays.toString(predShape));
        }
        int N = (int) predShape[0];
        if (N <= 0) {
            throw new IllegalArgumentException("pred must contain at least one bounding box");
        }
        float[] p = pred.data(), t = target.data();
        float totalIoU = 0.0f;
        float[] gradData = new float[p.length];

        for (int n = 0; n < N; n++) {
            int offset = n * 4;
            float px1 = p[offset], py1 = p[offset + 1], px2 = p[offset + 2], py2 = p[offset + 3];
            float tx1 = t[offset], ty1 = t[offset + 1], tx2 = t[offset + 2], ty2 = t[offset + 3];
            requireValidBox(px1, py1, px2, py2, "pred", n);
            requireValidBox(tx1, ty1, tx2, ty2, "target", n);

            float interX1 = Math.max(px1, tx1);
            float interY1 = Math.max(py1, ty1);
            float interX2 = Math.min(px2, tx2);
            float interY2 = Math.min(py2, ty2);
            float interW = Math.max(0.0f, interX2 - interX1);
            float interH = Math.max(0.0f, interY2 - interY1);
            float interArea = interW * interH;

            float predW = px2 - px1;
            float predH = py2 - py1;
            float predArea = predW * predH;
            float targetArea = (tx2 - tx1) * (ty2 - ty1);
            float unionArea = predArea + targetArea - interArea + EPS;

            totalIoU += interArea / unionArea;
            fillPredGradient(gradData, offset, px1, py1, px2, py2, tx1, ty1, tx2, ty2,
                    interW, interH, interArea, predW, predH, unionArea, N);
        }
        GradTensor out = GradTensor.scalar(1.0f - totalIoU / N);
        if (pred.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("IoULoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float upstreamScale = upstream.item();
                    float[] grad = new float[gradData.length];
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] = gradData[i] * upstreamScale;
                    }
                    pred.backward(GradTensor.of(grad, pred.shape()));
                }
            });
        }
        return out;
    }

    private static void fillPredGradient(
            float[] gradData,
            int offset,
            float px1,
            float py1,
            float px2,
            float py2,
            float tx1,
            float ty1,
            float tx2,
            float ty2,
            float interW,
            float interH,
            float interArea,
            float predW,
            float predH,
            float unionArea,
            int batch) {
        float dPredAreaX1 = -predH;
        float dPredAreaY1 = -predW;
        float dPredAreaX2 = predH;
        float dPredAreaY2 = predW;

        float dInterX1 = 0.0f;
        float dInterY1 = 0.0f;
        float dInterX2 = 0.0f;
        float dInterY2 = 0.0f;
        if (interW > 0.0f && interH > 0.0f) {
            dInterX1 = px1 > tx1 ? -interH : 0.0f;
            dInterX2 = px2 < tx2 ? interH : 0.0f;
            dInterY1 = py1 > ty1 ? -interW : 0.0f;
            dInterY2 = py2 < ty2 ? interW : 0.0f;
        }

        float unionSquared = unionArea * unionArea;
        gradData[offset] = lossGradient(dInterX1, dPredAreaX1, interArea, unionArea, unionSquared, batch);
        gradData[offset + 1] = lossGradient(dInterY1, dPredAreaY1, interArea, unionArea, unionSquared, batch);
        gradData[offset + 2] = lossGradient(dInterX2, dPredAreaX2, interArea, unionArea, unionSquared, batch);
        gradData[offset + 3] = lossGradient(dInterY2, dPredAreaY2, interArea, unionArea, unionSquared, batch);
    }

    private static float lossGradient(
            float dIntersection,
            float dPredArea,
            float intersection,
            float union,
            float unionSquared,
            int batch) {
        float dUnion = dPredArea - dIntersection;
        float dIoU = (dIntersection * union - intersection * dUnion) / unionSquared;
        return -dIoU / batch;
    }

    private static void requireValidBox(float x1, float y1, float x2, float y2, String name, int index) {
        if (!Float.isFinite(x1) || !Float.isFinite(y1) || !Float.isFinite(x2) || !Float.isFinite(y2)) {
            throw new IllegalArgumentException(name + " box " + index + " must contain only finite coordinates");
        }
        if (x2 <= x1 || y2 <= y1) {
            throw new IllegalArgumentException(
                    name + " box " + index + " must satisfy x2 > x1 and y2 > y1, got: ["
                            + x1 + ", " + y1 + ", " + x2 + ", " + y2 + "]");
        }
    }
}
