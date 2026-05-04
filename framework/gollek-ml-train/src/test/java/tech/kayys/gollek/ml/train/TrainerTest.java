package tech.kayys.gollek.ml.train;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.ml.nn.layer.Sequential;
import tech.kayys.gollek.ml.nn.layer.ReLU;
import tech.kayys.gollek.ml.optim.Adam;

import static org.junit.jupiter.api.Assertions.*;

class TrainerTest {

    @Test
    void testTrainerBuilder() {
        var model = new Sequential(
            new Linear(4, 8),
            new ReLU(),
            new Linear(8, 2)
        );

        var trainer = Trainer.builder()
            .model(model::forward)
            .optimizer(Adam.builder(model.parameters(), 0.01f).build())
            .loss((preds, targets) -> {
                var diff = preds.sub(targets);
                return diff.mul(diff).mean();
            })
            .epochs(2)
            .build();

        assertNotNull(trainer);
        assertEquals(2, trainer.getConfig().epochs());
    }

    @Test
    void testTrainerFitSimple() {
        var model = new Sequential(
            new Linear(4, 8),
            new ReLU(),
            new Linear(8, 2)
        );

        var trainer = Trainer.builder()
            .model(model::forward)
            .optimizer(Adam.builder(model.parameters(), 0.01f).build())
            .loss((preds, targets) -> {
                var diff = preds.sub(targets);
                return diff.mul(diff).mean();
            })
            .epochs(1)
            .build();

        // Create dummy data
        var inputs = GradTensor.randn(10, 4);
        var targets = GradTensor.randn(10, 2);
        var dataset = new tech.kayys.gollek.ml.data.DataLoader.TensorDataset(inputs, targets);
        var loader = tech.kayys.gollek.ml.data.DataLoader.tensorBuilder(dataset)
            .batchSize(5)
            .build();

        // Should not throw
        assertDoesNotThrow(() -> trainer.fit(loader));
    }

    @Test
    void testModelCheckpoint() {
        var checkpoint = ModelCheckpoint.builder()
            .dirPath(java.nio.file.Path.of("/tmp/test-checkpoint"))
            .build();
        assertNotNull(checkpoint);
    }

    @Test
    void testEarlyStopping() {
        var es = EarlyStopping.patience(3);
        assertNotNull(es);
    }

    @Test
    void testTrainingMetrics() {
        var metrics = new TrainingMetrics();
        metrics.updateTrainLoss(0, 0.5);
        metrics.updateValLoss(0, 0.6);
        metrics.updateBatchLoss(0, 0.4);
        assertNotNull(metrics);
    }

    @Test
    void testCallback() {
        var callback = ConsoleLogger.create();
        assertNotNull(callback);
    }
}
