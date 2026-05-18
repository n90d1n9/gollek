package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ArcFaceLossTest {

    @Test
    void arcFaceLossBackpropagatesFeatureAndClassCenterGradients() {
        float[] featureData = new float[] {
                0.7f, -0.2f,
                -0.3f, 0.9f
        };
        float[] weightData = new float[] {
                0.8f, 0.1f,
                -0.4f, 1.1f,
                0.3f, -0.7f
        };
        GradTensor labels = GradTensor.of(new float[] {0.0f, 2.0f}, 2);
        ArcFaceLoss loss = arcFace(weightData);
        GradTensor features = GradTensor.of(featureData, 2, 2).requiresGrad(true);

        loss.forward(features, labels).backward();

        assertArrayEquals(
                finiteDifferenceFeatureGradient(featureData, weightData, labels),
                features.grad().data(),
                5e-3f);
        assertArrayEquals(
                finiteDifferenceWeightGradient(featureData, weightData, labels),
                loss.namedParameters().get("weight").grad().data(),
                5e-3f);
    }

    private static float[] finiteDifferenceFeatureGradient(float[] featureData, float[] weightData, GradTensor labels) {
        float[] grad = new float[featureData.length];
        float eps = 1e-3f;
        for (int i = 0; i < featureData.length; i++) {
            float[] plus = featureData.clone();
            float[] minus = featureData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (arcFace(weightData).forward(GradTensor.of(plus, 2, 2), labels).item()
                    - arcFace(weightData).forward(GradTensor.of(minus, 2, 2), labels).item())
                    / (2.0f * eps);
        }
        return grad;
    }

    private static float[] finiteDifferenceWeightGradient(float[] featureData, float[] weightData, GradTensor labels) {
        float[] grad = new float[weightData.length];
        float eps = 1e-3f;
        for (int i = 0; i < weightData.length; i++) {
            float[] plus = weightData.clone();
            float[] minus = weightData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (arcFace(plus).forward(GradTensor.of(featureData, 2, 2), labels).item()
                    - arcFace(minus).forward(GradTensor.of(featureData, 2, 2), labels).item())
                    / (2.0f * eps);
        }
        return grad;
    }

    private static ArcFaceLoss arcFace(float[] weightData) {
        ArcFaceLoss loss = new ArcFaceLoss(3, 2, 0.2f, 3.0f);
        Parameter weight = loss.namedParameters().get("weight");
        weight.setData(GradTensor.of(weightData.clone(), 3, 2));
        return loss;
    }
}
