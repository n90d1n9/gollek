package tech.kayys.gollek.ml.nn.loss;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.assertThrows;

class ClassIndexLossValidationTest {

    @Test
    void crossEntropyRejectsInvalidClassTargets() {
        CrossEntropyLoss loss = new CrossEntropyLoss();
        GradTensor logits = GradTensor.zeros(2, 3);

        assertInvalidClassTargets(targets -> loss.compute(logits, targets));
    }

    @Test
    void focalLossRejectsInvalidClassTargets() {
        FocalLoss loss = new FocalLoss();
        GradTensor logits = GradTensor.zeros(2, 3);

        assertInvalidClassTargets(targets -> loss.compute(logits, targets));
    }

    @Test
    void labelSmoothingRejectsInvalidConfigurationAndTargets() {
        assertThrows(IllegalArgumentException.class, () -> new LabelSmoothingLoss(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new LabelSmoothingLoss(1.0f));
        assertThrows(IllegalArgumentException.class, () -> new LabelSmoothingLoss(-0.01f));

        LabelSmoothingLoss loss = new LabelSmoothingLoss(0.1f);
        GradTensor logits = GradTensor.zeros(2, 3);

        assertInvalidClassTargets(targets -> loss.forward(logits, targets));
    }

    @Test
    void arcFaceRejectsInvalidConfigurationAndTargets() {
        assertThrows(IllegalArgumentException.class, () -> new ArcFaceLoss(0, 2));
        assertThrows(IllegalArgumentException.class, () -> new ArcFaceLoss(3, 0));
        assertThrows(IllegalArgumentException.class, () -> new ArcFaceLoss(3, 2, Float.NaN, 64.0f));
        assertThrows(IllegalArgumentException.class, () -> new ArcFaceLoss(3, 2, 0.5f, 0.0f));

        ArcFaceLoss loss = new ArcFaceLoss(3, 2);
        GradTensor features = GradTensor.ones(2, 2);

        assertInvalidClassTargets(labels -> loss.forward(features, labels));
    }

    @Test
    void ctcLossRejectsInvalidClassTargets() {
        CTCLoss loss = new CTCLoss();
        GradTensor logProbs = GradTensor.zeros(3, 2, 3);
        int[] inputLengths = new int[] {3, 3};
        int[] targetLengths = new int[] {2, 2};

        assertThrows(IllegalArgumentException.class, () ->
                loss.forward(logProbs, GradTensor.of(new float[] {1f, 2f, 1f, 1.5f}, 2, 2),
                        inputLengths, targetLengths));
        assertThrows(IllegalArgumentException.class, () ->
                loss.forward(logProbs, GradTensor.of(new float[] {1f, 2f, 1f, Float.NaN}, 2, 2),
                        inputLengths, targetLengths));
        assertThrows(IndexOutOfBoundsException.class, () ->
                loss.forward(logProbs, GradTensor.of(new float[] {1f, 2f, 1f, 3f}, 2, 2),
                        inputLengths, targetLengths));
    }

    @Test
    void ctcLossRejectsInvalidSequenceShapesAndLengths() {
        CTCLoss loss = new CTCLoss();
        GradTensor logProbs = GradTensor.zeros(3, 2, 3);
        GradTensor targets = GradTensor.of(new float[] {1f, 2f, 1f, 2f}, 2, 2);

        assertThrows(IllegalArgumentException.class,
                () -> new CTCLoss(-1));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.zeros(2, 3), targets, new int[] {2, 2}, new int[] {1, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(logProbs, GradTensor.of(new float[] {1f, 2f}, 2),
                        new int[] {2, 2}, new int[] {1, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(logProbs, targets, new int[] {0, 2}, new int[] {1, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(logProbs, targets, new int[] {2, 2}, new int[] {3, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> new CTCLoss(3).forward(logProbs, targets, new int[] {2, 2}, new int[] {1, 1}));
    }

    private static void assertInvalidClassTargets(TargetedLoss loss) {
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {0f, 1f}, 2, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {0f, 1.5f}, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> loss.forward(GradTensor.of(new float[] {0f, Float.NaN}, 2)));
        assertThrows(IndexOutOfBoundsException.class,
                () -> loss.forward(GradTensor.of(new float[] {0f, 3f}, 2)));
    }

    @FunctionalInterface
    private interface TargetedLoss {
        void forward(GradTensor targets);
    }
}
