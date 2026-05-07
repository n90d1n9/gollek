package tech.kayys.gollek.train;

import tech.kayys.gollek.data.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.core.tensor.Tensor;
import java.util.Map;

public final class TrainingLoop {
    private final Trainer trainer;
    private final CheckpointManager checkpoint;

    public TrainingLoop(Trainer trainer,
            CheckpointManager checkpoint) {
        this.trainer = trainer;
        this.checkpoint = checkpoint;
    }

    public void train(int epochs,
            DataLoader loader,
            GGraph model,
            GValueId loss,
            Map<String, Tensor> params) throws Exception {
        for (int e = 0; e < epochs; e++) {
            for (Batch batch : loader.batches()) {
                trainer.trainStep(
                        model,
                        loss,
                        params,
                        Map.of(
                                "tokens", batch.tokens,
                                "targets", batch.targets));
            }

            checkpoint.save("checkpoint_" + e + ".bin", params);
        }
    }
}