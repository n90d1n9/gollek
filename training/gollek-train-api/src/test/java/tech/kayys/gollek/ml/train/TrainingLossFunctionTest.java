package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.optim.SGD;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.trainer.api.TrainingSummary;

@SuppressWarnings("deprecation")
class TrainingLossFunctionTest {

    @Test
    void canonicalTrainerAcceptsTopLevelTrainingLossFunction() {
        MSELoss mseLoss = new MSELoss();
        TrainingLossFunction loss = mseLoss::compute;
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {1.0f, 3.0f}, 2, 1),
                GradTensor.of(new float[] {2.0f, 1.0f}, 2, 1)));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(loss)
                .epochs(1)
                .build()) {
            trainer.fit(train, null);

            TrainingSummary summary = trainer.summary();
            assertNotNull(summary.latestTrainLoss());
            assertEquals(2.5, summary.latestTrainLoss(), 1e-6);
        }
    }

    @Test
    void legacyNestedLossFunctionRemainsAssignableToTopLevelContract() {
        CanonicalTrainer.LossFunction legacyLoss = (predictions, targets) -> GradTensor.scalar(0.25f);

        TrainingLossFunction loss = legacyLoss;

        assertEquals(0.25f, loss.compute(
                GradTensor.of(new float[] {1.0f}, 1),
                GradTensor.of(new float[] {0.0f}, 1)).data()[0], 1e-6);
    }

    private static final class IdentityModel extends NNModule {
        @Override
        public GradTensor forward(GradTensor input) {
            return input;
        }
    }
}
