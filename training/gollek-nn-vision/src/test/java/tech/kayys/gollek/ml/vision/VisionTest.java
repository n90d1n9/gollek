package tech.kayys.gollek.ml.vision;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.vision.transforms.VisionTransforms;
import tech.kayys.gollek.ml.vision.models.ResNet;

import static org.junit.jupiter.api.Assertions.*;

class VisionTest {

    @Test
    void testVisionTransformsCompose() {
        var pipeline = new VisionTransforms.Compose(
            new VisionTransforms.ToTensor()
        );
        assertNotNull(pipeline);
    }

    @Test
    void testToTensor() {
        var t = new VisionTransforms.ToTensor();
        assertNotNull(t);
    }

    @Test
    void testNormalize() {
        var t = new VisionTransforms.Normalize(
            new float[]{0.5f, 0.5f, 0.5f},
            new float[]{0.5f, 0.5f, 0.5f}
        );
        assertNotNull(t);
    }

    @Test
    void testElementWiseOps() {
        var ops = new tech.kayys.gollek.ml.vision.ops.ElementWiseOps();
        assertNotNull(ops);
    }

    @Test
    void testResNetCreation() {
        var model = ResNet.resnet18(10);
        assertNotNull(model);
    }
}
