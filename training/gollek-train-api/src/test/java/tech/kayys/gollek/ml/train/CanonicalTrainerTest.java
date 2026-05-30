package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.GollekML;
import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.loss.HuberLoss;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.optim.Adam;
import tech.kayys.gollek.ml.optim.AdamW;
import tech.kayys.gollek.ml.optim.ExponentialLR;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.optim.OneCycleLR;
import tech.kayys.gollek.ml.optim.RMSprop;
import tech.kayys.gollek.ml.optim.SGD;
import tech.kayys.gollek.ml.optim.SequentialLR;
import tech.kayys.gollek.ml.optim.StepLR;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.trainer.api.TrainerSession;
import tech.kayys.gollek.trainer.api.TrainingListener;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class CanonicalTrainerTest {

    @Test
    void canonicalTrainerRunsRealBatchTrainingLoop() {
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        MSELoss mseLoss = new MSELoss();

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(2)
                .gradientClip(1.0)
                .build();

        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> validation = List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));

        trainer.fit(train, validation);

        TrainingSummary summary = trainer.summary();
        assertEquals(2, summary.epochCount());
        assertNotNull(summary.latestTrainLoss());
        assertTrue(Double.isFinite(summary.latestTrainLoss()));
        assertNotNull(summary.latestValidationLoss());
        assertTrue(Double.isFinite(summary.latestValidationLoss()));
        assertEquals(4, trainer.session().globalStep());
        assertEquals(Boolean.TRUE, summary.metadata().get("trainBatchLossHook"));
        assertEquals(Boolean.FALSE, summary.metadata().get("trainSyntheticFallbackUsed"));
        assertEquals(Boolean.FALSE, summary.metadata().get("modelCheckpointEnabled"));
        assertEquals(Boolean.TRUE, summary.metadata().get("dataLoaderPlanMetadataCaptured"));
        assertEquals(Boolean.TRUE, summary.metadata().get("trainLoaderPlan.available"));
        assertEquals("batch-collection", summary.metadata().get("trainLoaderPlan.kind"));
        assertEquals(2, summary.metadata().get("trainLoaderPlan.batchCount"));
        assertEquals(4, summary.metadata().get("trainLoaderPlan.sampleCount"));
        assertEquals(Boolean.TRUE, summary.metadata().get("trainLoaderPlan.derivedFromBatchCollection"));
        assertEquals(Boolean.TRUE, summary.metadata().get("validationLoaderPlan.available"));
        assertEquals("batch-collection", summary.metadata().get("validationLoaderPlan.kind"));
        assertEquals(1, summary.metadata().get("validationLoaderPlan.batchCount"));
        assertEquals(2, summary.metadata().get("validationLoaderPlan.sampleCount"));
        assertEquals("healthy", summary.metadata().get("dataLoaderPlanHealthStatus"));
        assertEquals(Boolean.TRUE, summary.metadata().get("dataLoaderPlanHealthGatePassed"));
        assertEquals(0, summary.metadata().get("dataLoaderPlanHealthIssueCount"));
    }

    @Test
    void canonicalTrainerPublishesOptInDataDistributionDiagnosticsForTensorLoaders() {
        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        DataLoader.TensorDataLoader train = DataLoader.tensors(
                GradTensor.of(new float[] {0f, 1f, 2f, 3f}, 4, 1),
                GradTensor.of(new float[] {0f, 0f, 1f, 1f}, 4, 1),
                2);
        DataLoader.TensorDataLoader validation = DataLoader.tensors(
                GradTensor.of(new float[] {4f, 5f}, 2, 1),
                GradTensor.of(new float[] {0f, 1f}, 2, 1),
                2);

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .dataDistributionDiagnostics()
                .build()) {
            trainer.fit(train, validation);

            Map<String, Object> metadata = trainer.summary().metadata();
            assertEquals(Boolean.TRUE, metadata.get("dataDistributionDiagnosticsEnabled"));
            assertEquals(Boolean.TRUE, metadata.get("dataLoaderPlanMetadataCaptured"));
            assertEquals(Boolean.TRUE, metadata.get("trainLoaderPlan.available"));
            assertEquals("tensor", metadata.get("trainLoaderPlan.kind"));
            assertEquals(2, metadata.get("trainLoaderPlan.batchCount"));
            assertEquals(4, metadata.get("trainLoaderPlan.sampleCount"));
            assertEquals(2, metadata.get("trainLoaderPlan.batchSize"));
            assertEquals(Boolean.TRUE, metadata.get("validationLoaderPlan.available"));
            assertEquals("tensor", metadata.get("validationLoaderPlan.kind"));
            assertEquals(1, metadata.get("validationLoaderPlan.batchCount"));
            assertEquals("healthy", metadata.get("dataLoaderPlanHealthStatus"));
            assertEquals(Boolean.TRUE, metadata.get("dataLoaderPlanHealthGatePassed"));
            assertEquals(0, metadata.get("dataLoaderPlanHealthIssueCount"));
            assertEquals(Boolean.TRUE, metadata.get("trainDataDistribution.available"));
            assertEquals("classification", metadata.get("trainDataDistribution.kind"));
            assertEquals(4, metadata.get("trainDataDistribution.sampleCount"));
            assertEquals(List.of(2, 2), metadata.get("trainDataDistribution.classCounts"));
            assertEquals(0L, metadata.get("trainDataDistribution.diagnosticEpoch"));
            assertEquals(Boolean.TRUE, metadata.get("trainDataDistribution.diagnosticEpochViewUsed"));
            assertEquals(1.0, (double) metadata.get("trainDataDistribution.imbalanceRatio"), 1e-6);
            assertEquals(1.0, (double) metadata.get("trainDataDistribution.normalizedEntropy"), 1e-6);
            assertEquals(List.of(1.0f, 1.0f), metadata.get("trainDataDistribution.balancedClassWeights"));
            assertEquals(Boolean.TRUE, metadata.get("validationDataDistribution.available"));
            assertEquals(List.of(1, 1), metadata.get("validationDataDistribution.classCounts"));
            assertEquals(0L, metadata.get("validationDataDistribution.diagnosticEpoch"));
            assertEquals(Boolean.TRUE, metadata.get("validationDataDistribution.diagnosticEpochViewUsed"));
            assertEquals(Boolean.TRUE, metadata.get("dataDistributionDrift.available"));
            assertEquals("classification", metadata.get("dataDistributionDrift.kind"));
            assertEquals(0.0, (double) metadata.get("dataDistributionDrift.totalVariationDistance"), 1e-6);
            assertEquals(0.0, (double) metadata.get("dataDistributionDrift.maxAbsoluteFractionDelta"), 1e-6);
            assertEquals("healthy", metadata.get("dataDistributionHealthStatus"));
            assertEquals(Boolean.TRUE, metadata.get("dataDistributionHealthGatePassed"));
            assertEquals(0, metadata.get("dataDistributionHealthIssueCount"));
        }
    }

    @Test
    void dataDistributionDiagnosticsDoNotAdvanceReshufflingTensorLoaderEpoch() {
        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {0f, 1f, 2f, 3f, 4f, 5f}, 6, 1),
                GradTensor.of(new float[] {0f, 1f, 0f, 1f, 0f, 1f}, 6, 1));
        DataLoader.TensorDataLoader train = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(123L)
                .reshuffleEachEpoch()
                .build();
        DataLoader.TensorDataLoader expected = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(123L)
                .reshuffleEachEpoch()
                .build();

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .dataDistributionDiagnostics()
                .build()) {
            trainer.fit(train, null);

            Map<String, Object> metadata = trainer.summary().metadata();
            assertEquals(Boolean.TRUE, metadata.get("trainDataDistribution.diagnosticEpochViewUsed"));
            assertEquals(0L, metadata.get("trainDataDistribution.diagnosticEpoch"));
        }

        assertEquals(flattenInputs(expected.epoch(0L)), flattenInputs(train));
    }

    @Test
    void canonicalTrainerUpdatesLinearWeightsThroughAutogradGraph() {
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 0f;
        SGD optimizer = SGD.builder(model.parameters(), 0.1f).build();
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(1)
                .build()) {
            trainer.fit(train, null);
        }

        assertEquals(1.0f, model.parameters().get(0).data().data()[0], 1e-6f);
    }

    @Test
    void gradTensorMatmulFallsBackToCpuWhenNoNativeAcceleratorIsSelected() {
        GradTensor left = GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2);
        GradTensor right = GradTensor.of(new float[] {5f, 6f, 7f, 8f}, 2, 2);

        try (var ignored = tech.kayys.gollek.ml.autograd.Acceleration.prefer("cpu")) {
            GradTensor result = left.matmul(right);

            assertArrayEquals(new float[] {19f, 22f, 43f, 50f}, result.data(), 1e-6f);
            assertEquals("cpu", Gollek.DL.accelerationStatus("cpu").id());
            assertEquals(Boolean.FALSE, Gollek.DL.accelerationStatus("cpu").accelerated());
        }
    }

    @Test
    void canonicalTrainerReportsAccelerationMetadataForExplicitCpu() {
        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .device("cpu")
                .build()) {
            trainer.fit(train, null);

            TrainingSummary summary = trainer.summary();
            assertEquals("cpu", summary.metadata().get("requestedDevice"));
            assertEquals("cpu", summary.metadata().get("executionBackend"));
            assertEquals(Boolean.FALSE, summary.metadata().get("executionAccelerated"));
            assertEquals(Boolean.TRUE, summary.metadata().get("requestedDeviceAvailable"));
            assertEquals(Boolean.FALSE, summary.metadata().get("executionFallback"));
            assertEquals("cpu", summary.metadata().get("executionBackendAtStart"));
            assertEquals(Boolean.FALSE, summary.metadata().get("executionAcceleratedAtStart"));
            assertEquals(Boolean.TRUE, summary.metadata().get("requestedDeviceAvailableAtStart"));
            assertEquals(Boolean.FALSE, summary.metadata().get("acceleratedMatmulUsed"));
            assertEquals(Boolean.FALSE, summary.metadata().get("executionBackendChanged"));
            assertTrue(summary.metadata().get("acceleratedMatmulCalls") instanceof Number);
            assertTrue(summary.metadata().get("acceleratedMatmulCallsDelta") instanceof Number);
        }
    }

    @Test
    void canonicalTrainerAllowsExplicitAcceleratorFallbackByDefault() {
        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .device("npu")
                .build()) {
            trainer.fit(train, null);

            TrainingSummary summary = trainer.summary();
            assertEquals("npu", summary.metadata().get("requestedDevice"));
            assertEquals("cpu", summary.metadata().get("executionBackend"));
            assertEquals(Boolean.FALSE, summary.metadata().get("requestedDeviceAvailable"));
            assertEquals(Boolean.TRUE, summary.metadata().get("executionFallback"));
        }
    }

    @Test
    void canonicalTrainerCanFailFastOnExplicitAcceleratorFallback() {
        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .device("npu")
                .requireAccelerator()
                .build()) {
            IllegalStateException error = assertThrows(IllegalStateException.class, () ->
                    trainer.fit(train, null));

            assertTrue(error.getMessage().contains("Requested accelerator 'npu' is unavailable"));
        }
    }

    @Test
    void gollekDlFacadeExposesCanonicalTrainerBuilder() {
        assertNotNull(Gollek.DL.trainer());
    }

    @Test
    void gollekDlConvenienceFitSupportsTrainingOptionsDevice() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .gradientClip(1.0)
                        .gradientClipByValue(0.05)
                        .gradientAccumulationSteps(1)
                        .build());

        assertEquals("cpu", summary.metadata().get("requestedDevice"));
        assertEquals("cpu", summary.metadata().get("executionBackend"));
        assertEquals(Boolean.FALSE, summary.metadata().get("executionAccelerated"));
        assertEquals(1, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals(Boolean.TRUE, summary.metadata().get("gradientClipValueEnabled"));
        assertEquals(0.05, summary.metadata().get("gradientClipValueThreshold"));
    }

    @Test
    void gollekDlTrainingOptionsEnableDataDistributionDiagnostics() {
        Linear model = new Linear(1, 1);
        DataLoader.TensorDataLoader train = DataLoader.tensors(
                GradTensor.of(new float[] {0f, 1f, 2f, 3f}, 4, 1),
                GradTensor.of(new float[] {0f, 0f, 1f, 1f}, 4, 1),
                2);

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .dataDistributionDiagnostics()
                        .saveBestModelCheckpoint(false)
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("dataDistributionDiagnosticsEnabled"));
        assertEquals(Boolean.TRUE, summary.metadata().get("trainDataDistribution.available"));
        assertEquals("classification", summary.metadata().get("trainDataDistribution.kind"));
        assertEquals(List.of(2, 2), summary.metadata().get("trainDataDistribution.classCounts"));
        assertEquals("healthy", summary.metadata().get("dataDistributionHealthStatus"));
        assertEquals(Boolean.TRUE, summary.metadata().get("dataDistributionHealthGatePassed"));
        assertEquals(0, summary.metadata().get("dataDistributionHealthIssueCount"));
    }

    @Test
    void gollekDlTrainingOptionsCanRequireAccelerator() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        IllegalStateException error = assertThrows(IllegalStateException.class, () ->
                Gollek.DL.fit(
                        model,
                        train,
                        List.of(),
                        1,
                        0.01f,
                        Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                        Gollek.DL.trainingOptions()
                                .device("npu")
                                .requireAccelerator()
                                .build()));

        assertTrue(error.getMessage().contains("Requested accelerator 'npu' is unavailable"));
    }

    @Test
    void gollekDlDataLoaderFeedsCanonicalTrainer() {
        Linear model = new Linear(1, 1);
        var train = Gollek.DL.dataLoader(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f}, 4, 1),
                2);

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions().device("cpu").build());

        assertEquals(2, train.numBatches());
        assertEquals(2, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals(Boolean.TRUE, summary.metadata().get("trainBatchLossHook"));
    }

    @Test
    void gollekDlSplitFeedsTrainAndValidationLoaders() {
        Linear model = new Linear(1, 1);
        var split = Gollek.DL.trainValidationSplit(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f, 10f, 12f}, 6, 1),
                0.67,
                23L);
        var train = split.trainLoader(2, true, 77L);
        var validation = split.validationLoader(2);

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                validation,
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions().device("cpu").build());

        assertEquals(4, train.size());
        assertEquals(2, validation.size());
        assertEquals(2, train.numBatches());
        assertEquals(1, validation.numBatches());
        assertNotNull(summary.latestValidationLoss());
        assertEquals(Boolean.TRUE, summary.metadata().get("trainBatchLossHook"));
    }

    @Test
    void gollekDlFitAcceptsTensorDatasetSplitDirectly() {
        Linear model = new Linear(1, 1);
        var split = Gollek.DL.trainValidationSplit(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f, 10f, 12f}, 6, 1),
                0.67,
                5L);

        TrainingSummary summary = Gollek.DL.fit(
                model,
                split,
                2,
                true,
                17L,
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions().device("cpu").build());

        assertEquals(1, summary.epochCount());
        assertNotNull(summary.latestValidationLoss());
        assertEquals(2, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals("cpu", summary.metadata().get("executionBackend"));
    }

    @Test
    void gollekDlEvaluateReportsLossMetricsAndRestoresMode() {
        IdentityModel model = new IdentityModel();
        MSELoss mseLoss = new MSELoss();
        List<Batch> loader = List.of(
                batch(new float[] {1f, 3f}, new float[] {2f, 1f}, 2),
                batch(new float[] {2f}, new float[] {5f}, 1));

        assertTrue(model.isTraining());
        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                model,
                loader,
                mseLoss::compute,
                "cpu",
                Gollek.DL.meanAbsoluteErrorMetric(),
                Gollek.DL.meanSquaredErrorMetric(),
                Gollek.DL.rmseMetric(),
                Gollek.DL.medaeMetric(),
                Gollek.DL.maxErrorMetric(),
                Gollek.DL.pinballLossMetric(0.5),
                Gollek.DL.pinballLossMetric(0.9),
                Gollek.DL.r2Metric(),
                Gollek.DL.mapeMetric(),
                Gollek.DL.smapeMetric(),
                Gollek.DL.mbeMetric(),
                Gollek.DL.explainedVarianceMetric());

        assertTrue(model.isTraining());
        assertEquals(14.0 / 3.0, summary.loss(), 1e-6);
        assertEquals(2, summary.batchCount());
        assertEquals(3, summary.sampleCount());
        assertEquals(2.0, summary.metric("mae"), 1e-6);
        assertEquals(14.0 / 3.0, summary.metric("mse"), 1e-6);
        assertEquals(Math.sqrt(14.0 / 3.0), summary.metric("rmse"), 1e-6);
        assertEquals(2.0, summary.metric("median_absolute_error"), 1e-6);
        assertEquals(3.0, summary.metric("max_error"), 1e-6);
        assertEquals(1.0, summary.metric("pinball_loss_q50"), 1e-6);
        assertEquals(3.8 / 3.0, summary.metric("pinball_loss_q90"), 1e-6);
        assertEquals(-8.0 / 13.0, summary.metric("r2"), 1e-6);
        assertEquals(31.0 / 30.0, summary.metric("mape"), 1e-6);
        assertEquals(53.0 / 63.0, summary.metric("smape"), 1e-6);
        assertEquals(-2.0 / 3.0, summary.metric("mbe"), 1e-6);
        assertEquals(-6.0 / 13.0, summary.metric("explained_variance"), 1e-6);
        assertEquals("cpu", summary.metadata().get("requestedDevice"));
        assertEquals("cpu", summary.metadata().get("executionBackend"));
        assertEquals(Boolean.FALSE, summary.metadata().get("executionAccelerated"));
        assertEquals(Boolean.FALSE, summary.metadata().get("executionFallback"));
        assertEquals(Boolean.FALSE, summary.metadata().get("failOnAcceleratorFallback"));
        assertTrue(summary.metadata().get("acceleratedMatmulCallsDelta") instanceof Number);
        assertTrue(summary.metadata().get("acceleratedMatmulUsed") instanceof Boolean);
    }

    @Test
    void gollekDlEvaluateSupportsEvaluationOptionsDeviceAliases() {
        IdentityModel model = new IdentityModel();
        List<Batch> loader = List.of(batch(new float[] {1f, 3f}, new float[] {2f, 1f}, 2));

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                model,
                loader,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.evaluationOptions()
                        .device("off")
                        .build(),
                Gollek.DL.meanSquaredErrorMetric());

        assertEquals("cpu", summary.metadata().get("requestedDevice"));
        assertEquals("cpu", summary.metadata().get("executionBackend"));
        assertEquals(Boolean.FALSE, summary.metadata().get("executionFallback"));
        assertEquals(2.5, summary.loss(), 1e-6);
        assertEquals(2.5, summary.metric("mse"), 1e-6);
    }

    @Test
    void gollekDlEvaluateCanRequireAccelerator() {
        IdentityModel model = new IdentityModel();
        List<Batch> loader = List.of(batch(new float[] {1f, 3f}, new float[] {2f, 1f}, 2));

        IllegalStateException error = assertThrows(IllegalStateException.class, () ->
                Gollek.DL.evaluate(
                        model,
                        loader,
                        Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                        Gollek.DL.evaluationOptions()
                                .device("npu")
                                .requireAccelerator()
                                .build()));

        assertTrue(error.getMessage().contains("Requested accelerator 'npu' is unavailable"));
    }

    @Test
    void gollekDlEvaluateReportsPredictionIntervalMetrics() {
        List<Batch> loader = List.of(new Batch(
                GradTensor.of(new float[] {
                        -1.0f, 1.0f,
                        1.5f, 2.5f,
                        4.0f, 6.0f,
                        12.0f, 8.0f
                }, 4, 2),
                GradTensor.of(new float[] {0.0f, 1.0f, 5.0f, 10.0f}, 4)));

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                (predictions, targets) -> GradTensor.scalar(0.0f),
                "cpu",
                Gollek.DL.predictionIntervalCoverageMetric(),
                Gollek.DL.predictionIntervalMeanWidthMetric(),
                Gollek.DL.predictionIntervalNormalizedMeanWidthMetric());

        assertEquals(0.0, summary.loss(), 1e-6);
        assertEquals(0.75, summary.metric("prediction_interval_coverage"), 1e-6);
        assertEquals(2.25, summary.metric("prediction_interval_mean_width"), 1e-6);
        assertEquals(0.225, summary.metric("prediction_interval_normalized_mean_width"), 1e-6);
        Map<String, Object> details = metricDetails(summary.metricDetails(), "prediction_interval_coverage");
        assertEquals(1L, details.get("crossedIntervals"));
        assertEquals(Boolean.TRUE, details.get("boundsReorderedForEvaluation"));
    }

    @Test
    void gollekDlEvaluateSupportsPresetLoss() {
        IdentityModel model = new IdentityModel();
        List<Batch> loader = List.of(batch(new float[] {1f, 3f}, new float[] {2f, 1f}, 2));

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                model,
                loader,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                "cpu",
                Gollek.DL.meanSquaredErrorMetric(),
                Gollek.DL.rootMeanSquaredErrorMetric(),
                Gollek.DL.r2ScoreMetric());

        assertEquals(2.5, summary.loss(), 1e-6);
        assertEquals(2.5, summary.metric("mse"), 1e-6);
        assertEquals(Math.sqrt(2.5), summary.metric("rmse"), 1e-6);
        assertEquals(-9.0, summary.metric("r2"), 1e-6);
        assertEquals(2, summary.sampleCount());
    }

    @Test
    void huberLossBackpropagatesRobustRegressionGradient() {
        GradTensor predictions = GradTensor.of(new float[] {0f, 2f, 5f}, 3, 1).requiresGrad(true);
        GradTensor targets = GradTensor.of(new float[] {0f, 0f, 1f}, 3, 1);

        HuberLoss huber = Gollek.DL.huberLoss(1.5f);
        GradTensor loss = huber.compute(predictions, targets);
        loss.backward();

        assertEquals(2.25, loss.item(), 1e-6);
        assertArrayEquals(new float[] {0.0f, 0.5f, 0.5f}, predictions.grad().data(), 1e-6f);
    }

    @Test
    void gollekDlClassificationLoaderFeedsCrossEntropyEvaluation() {
        var loader = Gollek.DL.classificationDataLoader(
                GradTensor.of(new float[] {
                        3f, 1f, 0f,
                        0f, 1f, 4f,
                        0f, 5f, 1f
                }, 3, 3),
                new int[] {0, 1, 1},
                3);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_SGD,
                "cpu",
                Gollek.DL.accuracyMetric(),
                Gollek.DL.topKAccuracyMetric(2),
                Gollek.DL.precisionMetric(),
                Gollek.DL.recallMetric(),
                Gollek.DL.f1Metric(),
                Gollek.DL.classificationBalancedAccuracyMetric(),
                Gollek.DL.classificationMccMetric(),
                Gollek.DL.classificationWeightedPrecisionMetric(),
                Gollek.DL.classificationWeightedRecallMetric(),
                Gollek.DL.classificationWeightedF1Metric(),
                Gollek.DL.classificationKappaMetric(),
                Gollek.DL.classificationMacroRocAucMetric(),
                Gollek.DL.classificationMacroAveragePrecisionMetric());

        assertEquals(3, summary.sampleCount());
        assertEquals(1, summary.batchCount());
        assertEquals(2.0 / 3.0, summary.metric("accuracy"), 1e-6);
        assertEquals(1.0, summary.metric("top2_accuracy"), 1e-6);
        assertEquals(2.0 / 3.0, summary.metric("precision"), 1e-6);
        assertEquals(0.5, summary.metric("recall"), 1e-6);
        assertEquals(5.0 / 9.0, summary.metric("f1"), 1e-6);
        assertEquals(0.75, summary.metric("classification_balanced_accuracy"), 1e-6);
        assertEquals(3.0 / Math.sqrt(24.0), summary.metric("classification_matthews_correlation_coefficient"), 1e-6);
        assertEquals(1.0, summary.metric("classification_weighted_precision"), 1e-6);
        assertEquals(2.0 / 3.0, summary.metric("classification_weighted_recall"), 1e-6);
        assertEquals(7.0 / 9.0, summary.metric("classification_weighted_f1"), 1e-6);
        assertEquals(0.5, summary.metric("classification_cohens_kappa"), 1e-6);
        assertEquals(0.875, summary.metric("classification_macro_roc_auc"), 1e-6);
        assertEquals(11.0 / 12.0, summary.metric("classification_macro_average_precision"), 1e-6);
        assertTrue(Double.isFinite(summary.loss()));
    }

    @Test
    void gollekDlClassificationRankingMetricsAcceptOneHotTargets() {
        var loader = Gollek.DL.dataLoader(
                GradTensor.of(new float[] {
                        4f, 0f, 0f,
                        0f, 4f, 0f,
                        0f, 0f, 4f,
                        2f, 3f, 1f
                }, 4, 3),
                GradTensor.of(new float[] {
                        1f, 0f, 0f,
                        0f, 1f, 0f,
                        0f, 0f, 1f,
                        1f, 0f, 0f
                }, 4, 3),
                4);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                (predictions, targets) -> GradTensor.scalar(0f),
                "cpu",
                Gollek.DL.classificationMacroAurocMetric(),
                Gollek.DL.classificationMacroAveragePrecisionMetric());

        assertEquals(1.0, summary.metric("classification_macro_roc_auc"), 1e-6);
        assertEquals(1.0, summary.metric("classification_macro_average_precision"), 1e-6);
    }

    @Test
    void gollekDlClassificationCalibrationMetricsReportTopLabelReliability() {
        var loader = Gollek.DL.classificationDataLoader(
                GradTensor.of(new float[] {
                        logProb(0.8), logProb(0.1), logProb(0.1),
                        logProb(0.4), logProb(0.5), logProb(0.1),
                        logProb(0.2), logProb(0.3), logProb(0.5)
                }, 3, 3),
                new int[] {0, 0, 2},
                3);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_SGD,
                "cpu",
                Gollek.DL.classificationBrierScoreMetric(),
                Gollek.DL.classificationExpectedCalibrationErrorMetric(5));

        assertEquals(1.06 / 3.0, summary.metric("classification_brier_score"), 1e-6);
        assertEquals(1.0 / 15.0, summary.metric("classification_expected_calibration_error"), 1e-6);
        Map<String, Object> details = metricDetails(
                summary.metricDetails(),
                "classification_expected_calibration_error");
        assertEquals("classification_calibration", details.get("type"));
        assertEquals("top_label", details.get("mode"));
        assertEquals(5, details.get("bins"));
        assertEquals(List.of(0L, 0L, 2L, 0L, 1L), details.get("binCount"));
        assertEquals(
                details,
                metricDetails(summary.metadata(), "metricDetails", "classification_expected_calibration_error"));
    }

    @Test
    void crossEntropyLossSupportsClassWeights() {
        GradTensor logits = GradTensor.of(new float[] {
                0f, 0f, 0f,
                0f, 0f, 0f,
                0f, 0f, 0f
        }, 3, 3).requiresGrad(true);
        GradTensor targets = Gollek.DL.classLabels(0, 2, 2);

        GradTensor loss = Gollek.DL.crossEntropy(new float[] {1.0f, 1.0f, 4.0f}).compute(logits, targets);
        loss.backward();

        assertEquals(Math.log(3.0), loss.item(), 1e-6);
        assertArrayEquals(new float[] {
                -2.0f / 27.0f, 1.0f / 27.0f, 1.0f / 27.0f,
                4.0f / 27.0f, 4.0f / 27.0f, -8.0f / 27.0f,
                4.0f / 27.0f, 4.0f / 27.0f, -8.0f / 27.0f
        }, logits.grad().data(), 1e-6f);
        assertArrayEquals(
                new float[] {2.0f / 3.0f, 2.0f},
                Gollek.DL.classWeights(0, 0, 0, 1),
                1e-6f);
    }

    @Test
    void gollekDlFitUsesCrossEntropyClassWeightsFromTrainingOptions() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {
                        0f, 0f, 0f,
                        0f, 0f, 0f,
                        0f, 0f, 0f
                }, 3, 3),
                Gollek.DL.classLabels(0, 2, 2)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .crossEntropyClassWeights(1.0f, 1.0f, 4.0f)
                        .build());

        assertEquals(Math.log(3.0), summary.latestTrainLoss(), 1e-6);
        assertEquals(1, summary.epochCount());
        assertEquals("cpu", summary.metadata().get("executionBackend"));
    }

    @Test
    void focalLossWithGammaZeroMatchesCrossEntropyGradient() {
        GradTensor logits = GradTensor.of(new float[] {
                0f, 0f, 0f,
                0f, 0f, 0f
        }, 2, 3).requiresGrad(true);
        GradTensor targets = Gollek.DL.classLabels(0, 2);

        GradTensor loss = Gollek.DL.focalLoss(0.0f, 1.0f).compute(logits, targets);
        loss.backward();

        assertEquals(Math.log(3.0), loss.item(), 1e-6);
        assertArrayEquals(new float[] {
                -1.0f / 3.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                1.0f / 6.0f, 1.0f / 6.0f, -1.0f / 3.0f
        }, logits.grad().data(), 1e-6f);
    }

    @Test
    void focalLossSupportsPerClassAlphaWeights() {
        GradTensor logits = GradTensor.of(new float[] {
                0f, 0f, 0f,
                0f, 0f, 0f
        }, 2, 3);
        GradTensor targets = Gollek.DL.classLabels(0, 2);

        GradTensor loss = Gollek.DL.focalLoss(2.0f, new float[] {1.0f, 1.0f, 3.0f}).compute(logits, targets);

        assertEquals((8.0 / 9.0) * Math.log(3.0), loss.item(), 1e-6);
    }

    @Test
    void gollekDlFitUsesFocalPresetAndClassWeightsFromTrainingOptions() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {
                        0f, 0f, 0f,
                        0f, 0f, 0f
                }, 2, 3),
                Gollek.DL.classLabels(0, 2)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.CLASSIFICATION_FOCAL_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .focalGamma(2.0f)
                        .focalClassWeights(1.0f, 1.0f, 3.0f)
                        .build());

        assertEquals((8.0 / 9.0) * Math.log(3.0), summary.latestTrainLoss(), 1e-6);
        assertEquals(1, summary.epochCount());
        assertEquals("cpu", summary.metadata().get("executionBackend"));
    }

    @Test
    void gollekDlClassificationSplitFeedsOneCallFit() {
        Linear model = new Linear(3, 2);
        var split = Gollek.DL.classificationStratifiedTrainValidationSplit(
                GradTensor.of(new float[] {
                        1f, 0f, 0f,
                        0f, 1f, 0f,
                        0f, 0f, 1f,
                        1f, 1f, 0f,
                        1f, 0f, 1f,
                        0f, 1f, 1f,
                        0.8f, 0.1f, 0.1f,
                        0.1f, 0.8f, 0.1f
                }, 8, 3),
                new int[] {0, 1, 1, 0, 0, 1, 0, 1},
                0.75,
                13L);

        assertLabelCounts(split.train(), 3, 3);
        assertLabelCounts(split.validation(), 1, 1);

        TrainingSummary summary = Gollek.DL.fit(
                model,
                split,
                2,
                false,
                0L,
                1,
                0.01f,
                Gollek.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .classificationMetrics()
                        .classificationRankingMetrics()
                        .classificationImbalanceMetrics()
                        .classificationAgreementMetrics()
                        .classificationCalibrationMetrics()
                        .topKAccuracyMetric(2)
                        .build());

        assertEquals(1, summary.epochCount());
        assertNotNull(summary.latestValidationLoss());
        assertEquals(Boolean.TRUE, summary.metadata().get("metricsEnabled"));
        assertTrue(summary.metadata().get("trainMetric.accuracy") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.precision") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.recall") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.f1") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.classification_macro_roc_auc") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.classification_macro_average_precision") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.classification_balanced_accuracy") instanceof Number);
        assertTrue(
                summary.metadata().get("trainMetric.classification_matthews_correlation_coefficient")
                        instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.classification_weighted_precision") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.classification_weighted_recall") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.classification_weighted_f1") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.classification_cohens_kappa") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.classification_brier_score") instanceof Number);
        assertTrue(
                summary.metadata().get("trainMetric.classification_expected_calibration_error") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.top2_accuracy") instanceof Number);
    }

    @Test
    void gollekDlBinaryLoaderFeedsBceEvaluationMetrics() {
        var loader = Gollek.DL.binaryDataLoader(
                GradTensor.of(new float[] {2f, -1f, 0.5f, -2f}, 4, 1),
                new int[] {1, 0, 0, 1},
                4);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.binaryAccuracyMetric(),
                Gollek.DL.binaryPrecisionMetric(),
                Gollek.DL.binaryRecallMetric(),
                Gollek.DL.binaryF1Metric(),
                Gollek.DL.binaryBalancedAccuracyMetric(),
                Gollek.DL.binaryMccMetric(),
                Gollek.DL.binaryKappaMetric(),
                Gollek.DL.binaryRocAucMetric(),
                Gollek.DL.binaryAveragePrecisionMetric(),
                Gollek.DL.binaryBestF1ThresholdMetric());

        assertEquals(4, summary.sampleCount());
        assertEquals(1, summary.batchCount());
        assertEquals(0.5, summary.metric("binary_accuracy"), 1e-6);
        assertEquals(0.5, summary.metric("binary_precision"), 1e-6);
        assertEquals(0.5, summary.metric("binary_recall"), 1e-6);
        assertEquals(0.5, summary.metric("binary_f1"), 1e-6);
        assertEquals(0.5, summary.metric("binary_balanced_accuracy"), 1e-6);
        assertEquals(0.0, summary.metric("binary_matthews_correlation_coefficient"), 1e-6);
        assertEquals(0.0, summary.metric("binary_cohens_kappa"), 1e-6);
        assertEquals(0.5, summary.metric("binary_roc_auc"), 1e-6);
        assertEquals(0.75, summary.metric("binary_average_precision"), 1e-6);
        assertEquals(2.0 / 3.0, summary.metric("binary_best_f1"), 1e-6);
        assertEquals(2.0f, (Float) metricDetails(summary.metricDetails(), "binary_best_f1").get("threshold"), 1e-6f);
        assertTrue(Double.isFinite(summary.loss()));
    }

    @Test
    void gollekDlBinaryCalibrationMetricsReportProbabilityQuality() {
        var loader = Gollek.DL.binaryDataLoader(
                GradTensor.of(new float[] {
                        logit(0.9),
                        logit(0.8),
                        logit(0.35),
                        logit(0.1)
                }, 4, 1),
                new int[] {1, 1, 0, 0},
                4);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.binaryBrierScoreMetric(),
                Gollek.DL.binaryExpectedCalibrationErrorMetric(5));

        assertEquals(0.045625, summary.metric("binary_brier_score"), 1e-6);
        assertEquals(0.1875, summary.metric("binary_expected_calibration_error"), 1e-6);
        Map<String, Object> details = metricDetails(
                summary.metricDetails(),
                "binary_expected_calibration_error");
        assertEquals("binary_calibration", details.get("type"));
        assertEquals(5, details.get("bins"));
        assertEquals(List.of(1L, 1L, 0L, 0L, 2L), details.get("binCount"));
        assertEquals(details, metricDetails(summary.metadata(), "metricDetails", "binary_expected_calibration_error"));
    }

    @Test
    void gollekDlBinaryMetricsSupportCustomLogitThreshold() {
        var loader = Gollek.DL.binaryDataLoader(
                GradTensor.of(new float[] {2f, -1f, 0.5f, -2f}, 4, 1),
                new int[] {1, 0, 0, 1},
                4);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.binaryAccuracyMetric(0.75f),
                Gollek.DL.binaryPrecisionMetric(0.75f),
                Gollek.DL.binaryRecallMetric(0.75f),
                Gollek.DL.binaryF1Metric(0.75f));

        assertEquals(0.75, summary.metric("binary_accuracy"), 1e-6);
        assertEquals(1.0, summary.metric("binary_precision"), 1e-6);
        assertEquals(0.5, summary.metric("binary_recall"), 1e-6);
        assertEquals(2.0 / 3.0, summary.metric("binary_f1"), 1e-6);
        assertThrows(IllegalArgumentException.class, () -> Gollek.DL.binaryF1Metric(Float.NaN));
    }

    @Test
    void bceWithLogitsLossSupportsPositiveClassWeight() {
        GradTensor logits = GradTensor.of(new float[] {0f, 0f}, 2, 1).requiresGrad(true);
        GradTensor targets = Gollek.DL.binaryLabels(1, 0);

        GradTensor loss = Gollek.DL.bceWithLogitsLoss(3.0f).compute(logits, targets);
        loss.backward();

        assertEquals(2.0 * Math.log(2.0), loss.item(), 1e-6);
        assertArrayEquals(new float[] {-0.75f, 0.25f}, logits.grad().data(), 1e-6f);
    }

    @Test
    void bceWithLogitsLossSupportsPerLabelPositiveWeights() {
        GradTensor logits = GradTensor.of(new float[] {
                0f, 0f, 0f,
                0f, 0f, 0f
        }, 2, 3);
        GradTensor targets = Gollek.DL.multiLabelBinaryLabels(new int[][] {
                {1, 0, 1},
                {0, 1, 0}
        });

        GradTensor loss = Gollek.DL.bceWithLogitsLoss(new float[] {2.0f, 1.0f, 3.0f}).compute(logits, targets);

        assertEquals(1.5 * Math.log(2.0), loss.item(), 1e-6);
        assertArrayEquals(
                new float[] {1.0f, 1.0f, 1.0f},
                Gollek.DL.multiLabelPositiveWeights(new int[][] {
                        {1, 0, 1},
                        {0, 1, 0}
                }),
                1e-6f);
        assertArrayEquals(
                new float[] {3.0f, 3.0f, 1.0f},
                Gollek.DL.multiLabelPositiveWeights(new int[][] {
                        {1, 0, 1},
                        {0, 0, 0},
                        {0, 1, 0},
                        {0, 0, 1}
                }),
                1e-6f);
    }

    @Test
    void binaryFocalWithLogitsLossGammaZeroMatchesBceScale() {
        GradTensor logits = GradTensor.of(new float[] {0f, 0f}, 2, 1).requiresGrad(true);
        GradTensor targets = Gollek.DL.binaryLabels(1, 0);

        GradTensor loss = Gollek.DL.binaryFocalWithLogitsLoss(0.0f, 0.5f).compute(logits, targets);
        loss.backward();

        assertEquals(0.5 * Math.log(2.0), loss.item(), 1e-6);
        assertArrayEquals(new float[] {-0.125f, 0.125f}, logits.grad().data(), 1e-6f);
    }

    @Test
    void binaryFocalWithLogitsLossSupportsPositiveClassWeight() {
        GradTensor logits = GradTensor.of(new float[] {0f, 0f}, 2, 1).requiresGrad(true);
        GradTensor targets = Gollek.DL.binaryLabels(1, 0);

        GradTensor loss = Gollek.DL.binaryFocalWithLogitsLoss(0.0f, 0.5f, 3.0f).compute(logits, targets);
        loss.backward();

        assertEquals(Math.log(2.0), loss.item(), 1e-6);
        assertArrayEquals(new float[] {-0.375f, 0.125f}, logits.grad().data(), 1e-6f);
    }

    @Test
    void binaryFocalWithLogitsLossSupportsPerLabelPositiveWeights() {
        GradTensor logits = GradTensor.of(new float[] {
                0f, 0f, 0f,
                0f, 0f, 0f
        }, 2, 3);
        GradTensor targets = Gollek.DL.multiLabelBinaryLabels(new int[][] {
                {1, 0, 1},
                {0, 1, 0}
        });

        GradTensor loss = Gollek.DL.binaryFocalWithLogitsLoss(
                0.0f,
                0.5f,
                new float[] {2.0f, 1.0f, 3.0f}).compute(logits, targets);

        assertEquals(0.75 * Math.log(2.0), loss.item(), 1e-6);
    }

    @Test
    void gollekDlFitUsesBinaryFocalPresetAndPositiveWeightFromTrainingOptions() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {0f, 0f}, 2, 1),
                Gollek.DL.binaryLabels(1, 0)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.BINARY_FOCAL_WITH_LOGITS_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .focalGamma(0.0f)
                        .focalAlpha(0.5f)
                        .bcePositiveWeight(3.0f)
                        .build());

        assertEquals(Math.log(2.0), summary.latestTrainLoss(), 1e-6);
        assertEquals(1, summary.epochCount());
        assertEquals("cpu", summary.metadata().get("executionBackend"));
    }

    @Test
    void gollekDlFitUsesBcePositiveWeightFromTrainingOptions() {
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {0f, 0f}, 2, 1),
                Gollek.DL.binaryLabels(1, 0)));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .bcePositiveWeight(3.0f)
                        .build());

        assertEquals(2.0 * Math.log(2.0), summary.latestTrainLoss(), 1e-6);
        assertEquals(1, summary.epochCount());
        assertEquals("cpu", summary.metadata().get("executionBackend"));
    }

    @Test
    void gollekDlMultiLabelBinaryLoaderFeedsBceEvaluationMetrics() {
        var loader = Gollek.DL.multiLabelBinaryDataLoader(
                GradTensor.of(new float[] {
                        2f, -1f, 0.5f,
                        -2f, 3f, -0.5f
                }, 2, 3),
                new int[][] {
                        {1, 0, 1},
                        {0, 1, 0}
                },
                2);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.binaryAccuracyMetric(),
                Gollek.DL.binaryPrecisionMetric(),
                Gollek.DL.binaryRecallMetric(),
                Gollek.DL.binaryF1Metric(),
                Gollek.DL.multiLabelMacroRocAucMetric(),
                Gollek.DL.multiLabelMacroAveragePrecisionMetric());

        assertEquals(2, summary.sampleCount());
        assertEquals(1, summary.batchCount());
        assertEquals(1.0, summary.metric("binary_accuracy"), 1e-6);
        assertEquals(1.0, summary.metric("binary_precision"), 1e-6);
        assertEquals(1.0, summary.metric("binary_recall"), 1e-6);
        assertEquals(1.0, summary.metric("binary_f1"), 1e-6);
        assertEquals(1.0, summary.metric("multilabel_macro_roc_auc"), 1e-6);
        assertEquals(1.0, summary.metric("multilabel_macro_average_precision"), 1e-6);
        assertTrue(Double.isFinite(summary.loss()));
    }

    @Test
    void gollekDlMultiLabelMetricsReportExactHammingAndMacroScores() {
        var loader = Gollek.DL.multiLabelBinaryDataLoader(
                GradTensor.of(new float[] {
                        2f, -1f, 0.5f,
                        -2f, 3f, -0.5f,
                        1f, 2f, -3f
                }, 3, 3),
                new int[][] {
                        {1, 0, 1},
                        {0, 1, 1},
                        {1, 0, 0}
                },
                3);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.multiLabelExactMatchMetric(),
                Gollek.DL.multiLabelHammingLossMetric(),
                Gollek.DL.multiLabelSamplePrecisionMetric(),
                Gollek.DL.multiLabelSampleRecallMetric(),
                Gollek.DL.multiLabelSampleF1Metric(),
                Gollek.DL.multiLabelSampleJaccardMetric(),
                Gollek.DL.multiLabelMacroPrecisionMetric(),
                Gollek.DL.multiLabelMacroRecallMetric(),
                Gollek.DL.multiLabelMacroF1Metric(),
                Gollek.DL.multiLabelMacroRocAucMetric(),
                Gollek.DL.multiLabelMacroAveragePrecisionMetric());

        assertEquals(3, summary.sampleCount());
        assertEquals(1.0 / 3.0, summary.metric("multilabel_exact_match"), 1e-6);
        assertEquals(2.0 / 9.0, summary.metric("multilabel_hamming_loss"), 1e-6);
        assertEquals(5.0 / 6.0, summary.metric("multilabel_sample_precision"), 1e-6);
        assertEquals(5.0 / 6.0, summary.metric("multilabel_sample_recall"), 1e-6);
        assertEquals(7.0 / 9.0, summary.metric("multilabel_sample_f1"), 1e-6);
        assertEquals(2.0 / 3.0, summary.metric("multilabel_sample_jaccard"), 1e-6);
        assertEquals(5.0 / 6.0, summary.metric("multilabel_macro_precision"), 1e-6);
        assertEquals(5.0 / 6.0, summary.metric("multilabel_macro_recall"), 1e-6);
        assertEquals(7.0 / 9.0, summary.metric("multilabel_macro_f1"), 1e-6);
        assertEquals(1.0, summary.metric("multilabel_macro_roc_auc"), 1e-6);
        assertEquals(1.0, summary.metric("multilabel_macro_average_precision"), 1e-6);
    }

    @Test
    void gollekDlMultiLabelMetricsSupportCustomLogitThreshold() {
        var loader = Gollek.DL.multiLabelBinaryDataLoader(
                GradTensor.of(new float[] {
                        0.2f, -0.3f, 1.1f,
                        0.8f, -0.8f, 0.4f
                }, 2, 3),
                new int[][] {
                        {1, 0, 1},
                        {0, 0, 1}
                },
                2);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.multiLabelExactMatchMetric(0.5f),
                Gollek.DL.multiLabelHammingLossMetric(0.5f),
                Gollek.DL.multiLabelSamplePrecisionMetric(0.5f),
                Gollek.DL.multiLabelSampleRecallMetric(0.5f),
                Gollek.DL.multiLabelSampleF1Metric(0.5f),
                Gollek.DL.multiLabelSampleJaccardMetric(0.5f),
                Gollek.DL.multiLabelMacroPrecisionMetric(0.5f),
                Gollek.DL.multiLabelMacroRecallMetric(0.5f),
                Gollek.DL.multiLabelMacroF1Metric(0.5f));

        assertEquals(0.0, summary.metric("multilabel_exact_match"), 1e-6);
        assertEquals(0.5, summary.metric("multilabel_hamming_loss"), 1e-6);
        assertEquals(0.5, summary.metric("multilabel_sample_precision"), 1e-6);
        assertEquals(0.25, summary.metric("multilabel_sample_recall"), 1e-6);
        assertEquals(1.0 / 3.0, summary.metric("multilabel_sample_f1"), 1e-6);
        assertEquals(0.25, summary.metric("multilabel_sample_jaccard"), 1e-6);
        assertEquals(1.0 / 3.0, summary.metric("multilabel_macro_precision"), 1e-6);
        assertEquals(1.0 / 6.0, summary.metric("multilabel_macro_recall"), 1e-6);
        assertEquals(2.0 / 9.0, summary.metric("multilabel_macro_f1"), 1e-6);
        assertThrows(IllegalArgumentException.class, () -> Gollek.DL.multiLabelMacroF1Metric(Float.POSITIVE_INFINITY));
    }

    @Test
    void gollekDlMultiLabelRankingMetricsReportMacroScores() {
        var loader = Gollek.DL.multiLabelBinaryDataLoader(
                GradTensor.of(new float[] {
                        0.2f, 0.3f,
                        0.1f, 0.2f,
                       -0.1f, 0.1f
                }, 3, 2),
                new int[][] {
                        {1, 0},
                        {0, 1},
                        {1, 0}
                },
                3);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.multiLabelMacroRocAucMetric(),
                Gollek.DL.multiLabelMacroAveragePrecisionMetric(),
                Gollek.DL.multiLabelLrapMetric(),
                Gollek.DL.multiLabelRankingLossMetric(),
                Gollek.DL.multiLabelCoverageErrorMetric());

        assertEquals(0.5, summary.metric("multilabel_macro_roc_auc"), 1e-6);
        assertEquals(2.0 / 3.0, summary.metric("multilabel_macro_average_precision"), 1e-6);
        assertEquals(2.0 / 3.0, summary.metric("multilabel_label_ranking_average_precision"), 1e-6);
        assertEquals(2.0 / 3.0, summary.metric("multilabel_ranking_loss"), 1e-6);
        assertEquals(5.0 / 3.0, summary.metric("multilabel_coverage_error"), 1e-6);
    }

    @Test
    void gollekDlMultiLabelThresholdTuningReportsPerLabelBestF1() {
        var loader = Gollek.DL.multiLabelBinaryDataLoader(
                GradTensor.of(new float[] {
                        0.9f, 0.1f, 0.2f,
                        0.8f, 0.7f, 0.6f,
                        0.2f, 0.6f, 0.4f
                }, 3, 3),
                new int[][] {
                        {1, 0, 0},
                        {1, 0, 1},
                        {0, 1, 0}
                },
                3);

        Gollek.DL.EvaluationSummary summary = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                "cpu",
                Gollek.DL.multiLabelBestF1ThresholdsMetric());

        assertEquals(8.0 / 9.0, summary.metric("multilabel_macro_best_f1"), 1e-6);
        Map<String, Object> details = metricDetails(summary.metricDetails(), "multilabel_macro_best_f1");
        assertEquals(List.of(0.8f, 0.6f, 0.6f), details.get("perLabelThreshold"));
        assertEquals(List.of(1.0, 2.0 / 3.0, 1.0), details.get("perLabelF1"));
        assertEquals(List.of(2L, 1L, 1L), details.get("truePositive"));
        assertEquals(details, metricDetails(summary.metadata(), "metricDetails", "multilabel_macro_best_f1"));
    }

    @Test
    void gollekDlBinarySplitFeedsOneCallBceFit() {
        Linear model = new Linear(2, 1);
        var split = Gollek.DL.binaryStratifiedTrainValidationSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0f, 1f,
                        1f, 1f,
                        0f, 0f
                }, 4, 2),
                new int[] {1, 0, 1, 0},
                0.75,
                17L);

        assertLabelCounts(split.train(), 1, 1);
        assertLabelCounts(split.validation(), 1, 1);

        TrainingSummary summary = Gollek.DL.fit(
                model,
                split,
                2,
                false,
                0L,
                1,
                0.05f,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .binaryClassificationMetrics()
                        .binaryImbalanceMetrics()
                        .binaryAgreementMetrics()
                        .binaryRankingMetrics()
                        .binaryThresholdTuningMetrics()
                        .binaryCalibrationMetrics()
                        .multiLabelBinaryMetrics()
                        .multiLabelThresholdTuningMetrics()
                        .build());

        assertEquals(1, summary.epochCount());
        assertNotNull(summary.latestValidationLoss());
        assertEquals(Boolean.TRUE, summary.metadata().get("metricsEnabled"));
        assertTrue(summary.metadata().get("trainMetric.binary_accuracy") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_precision") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_recall") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_f1") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_balanced_accuracy") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_matthews_correlation_coefficient") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_cohens_kappa") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_roc_auc") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_average_precision") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_best_f1") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_brier_score") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_expected_calibration_error") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_exact_match") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_hamming_loss") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_sample_precision") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_sample_recall") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_sample_f1") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_sample_jaccard") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_macro_precision") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_macro_recall") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_macro_f1") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_macro_best_f1") instanceof Number);
    }

    @Test
    void gollekDlMultiLabelBinarySplitFeedsOneCallBceFit() {
        Linear model = new Linear(2, 3);
        var split = Gollek.DL.multiLabelBinaryStratifiedTrainValidationSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0f, 1f,
                        0.1f, 0.9f,
                        0.5f, 0.5f,
                        0.6f, 0.4f,
                        1f, 1f,
                        0f, 0f
                }, 8, 2),
                new int[][] {
                        {1, 0, 1},
                        {1, 0, 1},
                        {0, 1, 0},
                        {0, 1, 0},
                        {1, 1, 0},
                        {1, 1, 0},
                        {0, 0, 1},
                        {0, 0, 1}
                },
                0.5,
                19L);

        assertPositiveCounts(split.train(), new int[] {2, 2, 2});
        assertPositiveCounts(split.validation(), new int[] {2, 2, 2});

        TrainingSummary summary = Gollek.DL.fit(
                model,
                split,
                2,
                false,
                0L,
                1,
                0.05f,
                Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .binaryClassificationMetrics()
                        .multiLabelRankingMetrics()
                        .multiLabelThresholdTuningMetrics()
                        .build());

        assertEquals(1, summary.epochCount());
        assertNotNull(summary.latestValidationLoss());
        assertEquals(Boolean.TRUE, summary.metadata().get("metricsEnabled"));
        assertTrue(summary.metadata().get("trainMetric.binary_accuracy") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_precision") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_recall") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.binary_f1") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_macro_roc_auc") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_macro_average_precision") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_label_ranking_average_precision") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_ranking_loss") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_coverage_error") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.multilabel_macro_best_f1") instanceof Number);
    }

    @Test
    void gollekDlFitTrainingOptionsExposeCanonicalMetrics() {
        List<Batch> train = List.of(
                batch(new float[] {1f, 3f}, new float[] {2f, 1f}, 2),
                batch(new float[] {2f}, new float[] {5f}, 1));
        List<Batch> validation = List.of(batch(new float[] {0f, 10f}, new float[] {1f, 7f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                validation,
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .regressionExtendedMetrics()
                        .regressionLogScaleMetrics()
                        .regressionQuantileMetrics()
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("metricsEnabled"));
        assertEquals(2.0, metric(summary, "trainMetric.mae"), 1e-6);
        assertEquals(14.0 / 3.0, metric(summary, "trainMetric.mse"), 1e-6);
        assertEquals(Math.sqrt(14.0 / 3.0), metric(summary, "trainMetric.rmse"), 1e-6);
        assertEquals(2.0, metric(summary, "trainMetric.median_absolute_error"), 1e-6);
        assertEquals(3.0, metric(summary, "trainMetric.max_error"), 1e-6);
        assertEquals(trainMsle(), metric(summary, "trainMetric.msle"), 1e-6);
        assertEquals(Math.sqrt(trainMsle()), metric(summary, "trainMetric.rmsle"), 1e-6);
        assertEquals(2.2 / 3.0, metric(summary, "trainMetric.pinball_loss_q10"), 1e-6);
        assertEquals(1.0, metric(summary, "trainMetric.pinball_loss_q50"), 1e-6);
        assertEquals(3.8 / 3.0, metric(summary, "trainMetric.pinball_loss_q90"), 1e-6);
        assertEquals(-8.0 / 13.0, metric(summary, "trainMetric.r2"), 1e-6);
        assertEquals(31.0 / 30.0, metric(summary, "trainMetric.mape"), 1e-6);
        assertEquals(53.0 / 63.0, metric(summary, "trainMetric.smape"), 1e-6);
        assertEquals(-2.0 / 3.0, metric(summary, "trainMetric.mbe"), 1e-6);
        assertEquals(-6.0 / 13.0, metric(summary, "trainMetric.explained_variance"), 1e-6);
        assertEquals(2.0, metric(summary, "validationMetric.mae"), 1e-6);
        assertEquals(5.0, metric(summary, "validationMetric.mse"), 1e-6);
        assertEquals(Math.sqrt(5.0), metric(summary, "validationMetric.rmse"), 1e-6);
        assertEquals(2.0, metric(summary, "validationMetric.median_absolute_error"), 1e-6);
        assertEquals(3.0, metric(summary, "validationMetric.max_error"), 1e-6);
        assertEquals(validationMsle(), metric(summary, "validationMetric.msle"), 1e-6);
        assertEquals(Math.sqrt(validationMsle()), metric(summary, "validationMetric.rmsle"), 1e-6);
        assertEquals(1.4, metric(summary, "validationMetric.pinball_loss_q10"), 1e-6);
        assertEquals(1.0, metric(summary, "validationMetric.pinball_loss_q50"), 1e-6);
        assertEquals(0.6, metric(summary, "validationMetric.pinball_loss_q90"), 1e-6);
        assertEquals(4.0 / 9.0, metric(summary, "validationMetric.r2"), 1e-6);
        assertEquals(5.0 / 7.0, metric(summary, "validationMetric.mape"), 1e-6);
        assertEquals(20.0 / 17.0, metric(summary, "validationMetric.smape"), 1e-6);
        assertEquals(1.0, metric(summary, "validationMetric.mbe"), 1e-6);
        assertEquals(5.0 / 9.0, metric(summary, "validationMetric.explained_variance"), 1e-6);
    }

    @Test
    void gollekDlFitSupportsHuberRegressionPreset() {
        List<Batch> train = List.of(
                batch(new float[] {0f, 1f, 2f}, new float[] {0f, 2f, 4f}, 3),
                batch(new float[] {3f, 4f}, new float[] {6f, 30f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                new Linear(1, 1),
                train,
                List.of(),
                2,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_HUBER_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .regressionMetrics()
                        .build());

        assertEquals(2, summary.epochCount());
        assertTrue(Double.isFinite(summary.latestTrainLoss()));
        assertEquals(Boolean.TRUE, summary.metadata().get("metricsEnabled"));
        assertTrue(summary.metadata().get("trainMetric.rmse") instanceof Number);
        assertTrue(summary.metadata().get("trainMetric.r2") instanceof Number);
        assertEquals("cpu", summary.metadata().get("executionBackend"));
    }

    @Test
    void gollekDlTrainingOptionsAttachStepLrSchedulerToOneCallFit() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                List.of(),
                1,
                0.1f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .stepLrBatches(1, 0.5f)
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("learningRateSchedulerEnabled"));
        assertEquals("BATCH", summary.metadata().get("learningRateSchedulerStepUnit"));
        assertEquals(2, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
        assertEquals(0.05f, ((Number) summary.metadata().get("learningRate")).floatValue(), 1e-6f);
    }

    @Test
    void gollekDlTrainingOptionsAttachCosineAnnealingSchedulerToOneCallFit() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                List.of(),
                1,
                0.1f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .cosineAnnealingLrBatches(1, 0.01f)
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("learningRateSchedulerEnabled"));
        assertEquals("BATCH", summary.metadata().get("learningRateSchedulerStepUnit"));
        assertEquals(2, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
        assertEquals(0.01f, ((Number) summary.metadata().get("learningRate")).floatValue(), 1e-6f);
    }

    @Test
    void gollekDlTrainingOptionsAttachCosineWarmRestartsSchedulerToOneCallFit() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                List.of(),
                1,
                0.1f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .cosineAnnealingWarmRestartsLrBatches(2, 2, 0.01f)
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("learningRateSchedulerEnabled"));
        assertEquals("CosineAnnealingWarmRestartsLR", summary.metadata().get("learningRateSchedulerType"));
        assertEquals("BATCH", summary.metadata().get("learningRateSchedulerStepUnit"));
        assertEquals(2, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
        assertEquals(0.01f, ((Number) summary.metadata().get("learningRate")).floatValue(), 1e-6f);
        assertEquals(1, ((Number) summary.metadata().get("learningRateSchedulerState.cycleIndex")).intValue());
        assertEquals(0, ((Number) summary.metadata().get("learningRateSchedulerState.cycleStep")).intValue());
        assertEquals(4, ((Number) summary.metadata().get("learningRateSchedulerState.cycleLength")).intValue());
    }

    @Test
    void gollekDlConvenienceFitRunsWithPreset() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> validation = List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                validation,
                2,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_ADAMW,
                1.0);

        assertEquals(2, summary.epochCount());
        assertNotNull(summary.latestTrainLoss());
        assertTrue(Double.isFinite(summary.latestTrainLoss()));
        assertNotNull(summary.latestValidationLoss());
        assertTrue(Double.isFinite(summary.latestValidationLoss()));
        assertEquals(Boolean.TRUE, summary.metadata().get("trainBatchLossHook"));
    }

    @Test
    void gollekDlConvenienceFitRunsClassificationPreset() {
        Linear model = new Linear(2, 3);
        List<Batch> train = List.of(
                new Batch(
                        GradTensor.of(new float[] {1f, 0f, 0f, 1f}, 2, 2),
                        GradTensor.of(new float[] {0f, 2f}, 2)));

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                1,
                0.01f,
                Gollek.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_ADAMW);

        assertEquals(1, summary.epochCount());
        assertNotNull(summary.latestTrainLoss());
        assertTrue(Double.isFinite(summary.latestTrainLoss()));
        assertEquals(Boolean.TRUE, summary.metadata().get("trainBatchLossHook"));
    }

    @Test
    void gollekDlConvenienceFitSupportsEarlyStopping() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> validation = List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                validation,
                10,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_ADAMW,
                1.0,
                1,
                1_000_000.0);

        assertEquals(2, summary.epochCount());
        assertEquals("early-stopping", summary.metadata().get("stopReason"));
        assertEquals(Boolean.TRUE, summary.metadata().get("earlyStoppingTriggered"));
        assertEquals(1, ((Number) summary.metadata().get("earlyStoppingPatience")).intValue());
    }

    @Test
    void gollekDlConvenienceFitSupportsCheckpointOptions() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-dl-fit-checkpoint");
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_ADAMW,
                1.0,
                0,
                0.0,
                checkpointDir,
                false,
                true);

        assertEquals(1, summary.epochCount());
        assertEquals(Boolean.TRUE, summary.metadata().get("modelCheckpointEnabled"));
        assertEquals(Boolean.TRUE, summary.metadata().get("modelCheckpointSaved"));
        assertEquals(Boolean.TRUE, summary.metadata().get("optimizerCheckpointSupported"));
        assertEquals(Boolean.TRUE, summary.metadata().get("optimizerCheckpointSaved"));
        assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistoryEnabled"));
        assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistorySaved"));
        assertEquals(Boolean.FALSE, summary.metadata().get("trainingHistorySaveFailed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("checkpointManifestSaved"));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-model.safetensors")));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-optimizer.state")));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-history.csv")));
        String manifest = Files.readString(checkpointDir.resolve("canonical-checkpoints.metadata"));
        assertTrue(manifest.contains("artifact.runtime.bytes="));
        assertTrue(manifest.contains("artifact.runtime.sha256="));
        assertTrue(manifest.contains("artifact.optimizer.bytes="));
        assertTrue(manifest.contains("artifact.optimizer.sha256="));
    }

    @Test
    void canonicalTrainerSavesAndRestoresBestValidationModelCheckpoint() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-best-checkpoint");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 2f;
        MSELoss mseLoss = new MSELoss();
        List<Batch> validation = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        TrainingListener corruptAfterFirstValidation = new TrainingListener() {
            @Override
            public void onValidationEnd(TrainerSession session, int epoch, double valLoss) {
                if (epoch == 0) {
                    model.parameters().get(0).data().data()[0] = -8f;
                }
            }
        };

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .restoreBestModelAtEnd()
                .listener(corruptAfterFirstValidation)
                .build()) {
            trainer.fit(List.of(), validation);

            TrainingSummary summary = trainer.summary();
            assertEquals(2, summary.epochCount());
            assertEquals(0, summary.bestValidationEpoch());
            assertEquals(0.0, summary.bestValidationLoss(), 1e-6);
            assertEquals(Boolean.TRUE, summary.metadata().get("bestModelCheckpointEnabled"));
            assertEquals(Boolean.TRUE, summary.metadata().get("bestModelCheckpointSaved"));
            assertEquals(Boolean.TRUE, summary.metadata().get("bestModelCheckpointPresent"));
            assertEquals(Boolean.TRUE, summary.metadata().get("bestModelCheckpointRestored"));
            assertEquals(0, ((Number) summary.metadata().get("bestModelCheckpointEpoch")).intValue());
            assertEquals(0.0, metric(summary, "bestModelCheckpointValidationLoss"), 1e-6);
        }

        assertEquals(2f, model.parameters().get(0).data().data()[0], 1e-6f);
        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-best-model.safetensors")));
    }

    @Test
    void canonicalTrainerCanSelectBestCheckpointByValidationMetric() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-best-metric");
        ScoredIdentityModel model = new ScoredIdentityModel();
        MSELoss mseLoss = new MSELoss();
        EpochBatches validation = new EpochBatches(List.of(
                List.of(batch(new float[] {0f}, new float[] {0f}, 1)),
                List.of(batch(new float[] {5f}, new float[] {0f}, 1))));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .metric(MeanPredictionMetric::new)
                .bestModelMonitorMetric("mean_prediction", CanonicalTrainer.BestModelMonitorMode.MAX)
                .restoreBestModelAtEnd()
                .build()) {
            trainer.fit(List.of(), validation);

            TrainingSummary summary = trainer.summary();
            assertEquals(2, summary.epochCount());
            assertEquals(0, summary.bestValidationEpoch());
            assertEquals(0.0, summary.bestValidationLoss(), 1e-6);
            assertEquals(Boolean.TRUE, summary.metadata().get("bestModelCheckpointSaved"));
            assertEquals(Boolean.TRUE, summary.metadata().get("bestModelCheckpointRestored"));
            assertEquals("validationMetric.mean_prediction", summary.metadata().get("bestModelCheckpointMonitor"));
            assertEquals("MAX", summary.metadata().get("bestModelCheckpointMonitorMode"));
            assertEquals(1, ((Number) summary.metadata().get("bestModelCheckpointEpoch")).intValue());
            assertEquals(25.0, metric(summary, "bestModelCheckpointValidationLoss"), 1e-6);
            assertEquals(5.0, metric(summary, "bestModelCheckpointMonitorValue"), 1e-6);
        }

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-best-model.safetensors")));
    }

    @Test
    void canonicalTrainerRejectsCorruptedBestModelCheckpointBeforeRestoreByDefault() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-best-integrity-mismatch");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 2f;
        MSELoss mseLoss = new MSELoss();
        List<Batch> validation = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        TrainingListener corruptBestAfterFinalValidation = new TrainingListener() {
            @Override
            public void onValidationEnd(TrainerSession session, int epoch, double valLoss) {
                if (epoch == 1) {
                    writeCorruptCheckpoint(checkpointDir.resolve("canonical-best-model.safetensors"));
                }
            }
        };

        IllegalStateException error = assertThrows(IllegalStateException.class, () -> {
            try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                    .model(model)
                    .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                    .loss(mseLoss::compute)
                    .epochs(2)
                    .checkpointDir(checkpointDir)
                    .restoreBestModelAtEnd()
                    .listener(corruptBestAfterFinalValidation)
                    .build()) {
                trainer.fit(List.of(), validation);
            }
        });
        assertTrue(error.getMessage().contains("Best model checkpoint integrity mismatch"));
        assertTrue(error.getMessage().contains("bestModel checkpoint size mismatch"));
    }

    @Test
    void canonicalTrainerReportsCorruptedBestModelCheckpointWhenRestoreGuardIsDisabled() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-best-integrity-lenient");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 2f;
        MSELoss mseLoss = new MSELoss();
        List<Batch> validation = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        TrainingListener corruptBestAfterFinalValidation = new TrainingListener() {
            @Override
            public void onValidationEnd(TrainerSession session, int epoch, double valLoss) {
                if (epoch == 1) {
                    writeCorruptCheckpoint(checkpointDir.resolve("canonical-best-model.safetensors"));
                }
            }
        };

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .restoreBestModelAtEnd()
                .failOnCheckpointLoadError(false)
                .listener(corruptBestAfterFinalValidation)
                .build()) {
            trainer.fit(List.of(), validation);

            Map<String, Object> metadata = trainer.summary().metadata();
            assertEquals(Boolean.TRUE, metadata.get("bestModelCheckpointLoadFailed"));
            assertEquals(Boolean.FALSE, metadata.get("bestModelCheckpointRestored"));
            assertEquals(Boolean.TRUE, metadata.get("checkpointManifestIntegrityMismatch"));
            assertEquals(Boolean.TRUE, metadata.get("checkpointCompatibilityMismatch"));
            assertTrue(String.valueOf(metadata.get("bestModelCheckpointLoadError"))
                    .contains("bestModel checkpoint size mismatch"));
            assertTrue(String.valueOf(metadata.get("checkpointCompatibilityMismatches"))
                    .contains("bestModel checkpoint size mismatch"));
        }
    }

    @Test
    void canonicalTrainerCanEarlyStopByValidationMetric() {
        ScoredIdentityModel model = new ScoredIdentityModel();
        MSELoss mseLoss = new MSELoss();
        EpochBatches validation = new EpochBatches(List.of(
                List.of(batch(new float[] {1f}, new float[] {10f}, 1)),
                List.of(batch(new float[] {2f}, new float[] {10f}, 1)),
                List.of(batch(new float[] {2f}, new float[] {9f}, 1)),
                List.of(batch(new float[] {2f}, new float[] {8f}, 1)),
                List.of(batch(new float[] {2f}, new float[] {7f}, 1))));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(8)
                .earlyStopping(2, 0.0)
                .metric(MeanPredictionMetric::new)
                .earlyStoppingMonitorMetric("mean_prediction", CanonicalTrainer.BestModelMonitorMode.MAX)
                .build()) {
            trainer.fit(List.of(), validation);

            TrainingSummary summary = trainer.summary();
            assertEquals(4, summary.epochCount());
            assertEquals(3, summary.bestValidationEpoch());
            assertEquals(36.0, summary.bestValidationLoss(), 1e-6);
            assertEquals(Boolean.TRUE, summary.metadata().get("earlyStoppingTriggered"));
            assertEquals("early-stopping", summary.metadata().get("stopReason"));
            assertEquals("validationMetric.mean_prediction", summary.metadata().get("earlyStoppingMonitor"));
            assertEquals("MAX", summary.metadata().get("earlyStoppingMonitorMode"));
            assertEquals(Boolean.TRUE, summary.metadata().get("earlyStoppingMonitorMetricDriven"));
            assertEquals(Boolean.TRUE, summary.metadata().get("earlyStoppingEnabled"));
            assertEquals(2, ((Number) summary.metadata().get("earlyStoppingPatience")).intValue());
            assertEquals(0.0, metric(summary, "earlyStoppingMinDelta"), 1e-6);
            assertEquals(1, ((Number) summary.metadata().get("earlyStoppingMonitorBestEpoch")).intValue());
            assertEquals(3, ((Number) summary.metadata().get("earlyStoppingEpoch")).intValue());
            assertEquals(2, ((Number) summary.metadata()
                    .get("earlyStoppingMonitorEpochsWithoutImprovement")).intValue());
            assertEquals(2.0, metric(summary, "earlyStoppingMonitorBestValue"), 1e-6);
            assertEquals(2.0, metric(summary, "earlyStoppingMonitorLatestValue"), 1e-6);
        }
    }

    @Test
    void canonicalTrainerRecordsPerEpochHistoryWithMetricsAndLearningRate() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-history");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(
                batch(new float[] {1f, 3f}, new float[] {2f, 1f}, 2),
                batch(new float[] {2f}, new float[] {5f}, 1));
        List<Batch> validation = List.of(batch(new float[] {0f, 10f}, new float[] {1f, 7f}, 2));
        SGD optimizer = SGD.builder(List.of(), 0.1f).build();

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(optimizer)
                .scheduler(new StepLR(optimizer, 1, 0.5f))
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .build()) {
            trainer.fit(train, validation);

            TrainingSummary summary = trainer.summary();
            List<Map<String, Object>> history = epochHistory(summary);
            assertEquals(2, history.size());
            assertEquals(2, ((Number) summary.metadata().get("epochHistorySize")).intValue());

            Map<String, Object> firstEpoch = history.get(0);
            assertEquals(0, ((Number) firstEpoch.get("epoch")).intValue());
            assertTrue(firstEpoch.get("trainLoss") instanceof Number);
            assertTrue(firstEpoch.get("validationLoss") instanceof Number);
            assertTrue(firstEpoch.get("learningRate") instanceof Number);
            assertTrue(firstEpoch.get("optimizerStepCount") instanceof Number);
            assertTrue(firstEpoch.get("schedulerStepCount") instanceof Number);
            assertEquals(2.0, ((Number) firstEpoch.get("trainMetric.mae")).doubleValue(), 1e-6);
            assertEquals(2.0, ((Number) firstEpoch.get("validationMetric.mae")).doubleValue(), 1e-6);
            assertEquals(2.0, metricMap(firstEpoch, "trainMetrics").get("mae"), 1e-6);
            assertEquals(2.0, metricMap(firstEpoch, "validationMetrics").get("mae"), 1e-6);
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistoryEnabled"));
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistorySaved"));
            assertEquals(Boolean.FALSE, summary.metadata().get("trainingHistorySaveFailed"));
            assertEquals(checkpointDir.resolve("canonical-history.csv").toString(),
                    summary.metadata().get("trainingHistoryFile"));
        }

        String csv = Files.readString(checkpointDir.resolve("canonical-history.csv"));
        assertTrue(csv.startsWith("epoch,trainLoss,validationLoss,learningRate,optimizerStepCount,schedulerStepCount"));
        assertTrue(csv.contains("trainMetric.mae"));
        assertTrue(csv.contains("validationMetric.mae"));
        assertEquals(3, csv.lines().count());
    }

    @Test
    void canonicalTrainerRecordsGradientAndParameterDiagnostics() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-diagnostics");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 0f;
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {10f, 20f}, new float[] {100f, 200f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(1)
                .gradientClip(0.1)
                .parameterUpdateDiagnostics()
                .checkpointDir(checkpointDir)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .build()) {
            trainer.fit(train);

            TrainingSummary summary = trainer.summary();
            assertTrue(metric(summary, "latestGradientL2NormBeforeClip") > 0.1);
            assertTrue(metric(summary, "latestGradientL2Norm") <= 0.1001);
            assertTrue(metric(summary, "latestGradientMaxAbsBeforeClip") > 0.1);
            assertTrue(metric(summary, "latestGradientMaxAbs") <= 0.1001);
            assertTrue(metric(summary, "latestGradientMeanAbsBeforeClip") > 0.1);
            assertTrue(metric(summary, "latestGradientMeanAbs") <= 0.1001);
            assertTrue(metric(summary, "latestGradientRmsBeforeClip") > 0.1);
            assertTrue(metric(summary, "latestGradientRms") <= 0.1001);
            assertEquals(Boolean.TRUE, summary.metadata().get("latestGradientClipped"));
            assertTrue(metric(summary, "latestGradientClipScale") > 0.0);
            assertTrue(metric(summary, "latestGradientClipScale") < 1.0);
            assertEquals(0L, ((Number) summary.metadata().get("latestGradientZeroCount")).longValue());
            assertEquals(0.0, metric(summary, "latestGradientZeroFraction"), 1e-9);
            assertEquals(Boolean.TRUE, summary.metadata().get("gradientClipEnabled"));
            assertEquals(0.1, metric(summary, "gradientClipThreshold"), 1e-6);
            assertEquals(1, ((Number) summary.metadata().get("latestGradientParameterCount")).intValue());
            assertEquals(1L, ((Number) summary.metadata().get("latestGradientValueCount")).longValue());
            assertEquals(1, ((Number) summary.metadata().get("latestParameterCount")).intValue());
            assertEquals(1L, ((Number) summary.metadata().get("latestParameterValueCount")).longValue());
            assertEquals(0L, ((Number) summary.metadata().get("latestParameterZeroCount")).longValue());
            assertEquals(0.0, metric(summary, "latestParameterZeroFraction"), 1e-9);
            assertTrue(metric(summary, "latestParameterL2Norm") > 0.0);
            assertTrue(metric(summary, "latestParameterMeanAbs") > 0.0);
            assertTrue(metric(summary, "latestParameterRms") > 0.0);
            assertTrue(metric(summary, "latestGradientToParameterL2Ratio") > 0.0);
            assertTrue(metric(summary, "latestGradientToParameterMaxAbsRatio") > 0.0);
            assertTrue(metric(summary, "latestGradientToParameterMeanAbsRatio") > 0.0);
            assertTrue(metric(summary, "latestGradientToParameterRmsRatio") > 0.0);
            assertEquals(Boolean.TRUE, summary.metadata().get("parameterUpdateDiagnosticsEnabled"));
            assertEquals(1, ((Number) summary.metadata().get("latestParameterUpdateCount")).intValue());
            assertEquals(1L, ((Number) summary.metadata().get("latestParameterUpdateValueCount")).longValue());
            assertEquals(0L, ((Number) summary.metadata().get("latestParameterUpdateZeroCount")).longValue());
            assertEquals(0.0, metric(summary, "latestParameterUpdateZeroFraction"), 1e-9);
            assertTrue(metric(summary, "latestParameterUpdateL2Norm") > 0.0);
            assertTrue(metric(summary, "latestParameterUpdateMaxAbs") > 0.0);
            assertTrue(metric(summary, "latestParameterUpdateMeanAbs") > 0.0);
            assertTrue(metric(summary, "latestParameterUpdateRms") > 0.0);
            assertTrue(metric(summary, "latestParameterUpdateToParameterL2Ratio") > 0.0);
            assertTrue(metric(summary, "latestParameterUpdateToParameterMaxAbsRatio") > 0.0);
            assertTrue(metric(summary, "latestParameterUpdateToParameterMeanAbsRatio") > 0.0);
            assertTrue(metric(summary, "latestParameterUpdateToParameterRmsRatio") > 0.0);

            Map<String, Object> firstEpoch = epochHistory(summary).get(0);
            assertEquals(Boolean.TRUE, firstEpoch.get("gradientClipped"));
            assertTrue(((Number) firstEpoch.get("gradientL2NormBeforeClip")).doubleValue() > 0.1);
            assertTrue(((Number) firstEpoch.get("gradientL2Norm")).doubleValue() <= 0.1001);
            assertTrue(((Number) firstEpoch.get("gradientMeanAbsBeforeClip")).doubleValue() > 0.1);
            assertTrue(((Number) firstEpoch.get("gradientMeanAbs")).doubleValue() <= 0.1001);
            assertTrue(((Number) firstEpoch.get("gradientRmsBeforeClip")).doubleValue() > 0.1);
            assertTrue(((Number) firstEpoch.get("gradientRms")).doubleValue() <= 0.1001);
            assertTrue(((Number) firstEpoch.get("gradientClipScale")).doubleValue() < 1.0);
            assertEquals(0L, ((Number) firstEpoch.get("gradientZeroCount")).longValue());
            assertEquals(0.0, ((Number) firstEpoch.get("gradientZeroFraction")).doubleValue(), 1e-9);
            assertTrue(((Number) firstEpoch.get("parameterL2Norm")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("parameterMeanAbs")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("parameterRms")).doubleValue() > 0.0);
            assertEquals(0L, ((Number) firstEpoch.get("parameterZeroCount")).longValue());
            assertEquals(0.0, ((Number) firstEpoch.get("parameterZeroFraction")).doubleValue(), 1e-9);
            assertTrue(((Number) firstEpoch.get("gradientToParameterL2Ratio")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("gradientToParameterMaxAbsRatio")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("gradientToParameterMeanAbsRatio")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("gradientToParameterRmsRatio")).doubleValue() > 0.0);
            assertEquals(Boolean.TRUE, firstEpoch.get("parameterUpdateDiagnosticsEnabled"));
            assertTrue(((Number) firstEpoch.get("parameterUpdateL2Norm")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("parameterUpdateMaxAbs")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("parameterUpdateMeanAbs")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("parameterUpdateRms")).doubleValue() > 0.0);
            assertEquals(1, ((Number) firstEpoch.get("parameterUpdateCount")).intValue());
            assertEquals(1L, ((Number) firstEpoch.get("parameterUpdateValueCount")).longValue());
            assertEquals(0L, ((Number) firstEpoch.get("parameterUpdateZeroCount")).longValue());
            assertEquals(0.0, ((Number) firstEpoch.get("parameterUpdateZeroFraction")).doubleValue(), 1e-9);
            assertTrue(((Number) firstEpoch.get("parameterUpdateToParameterL2Ratio")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("parameterUpdateToParameterMaxAbsRatio")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("parameterUpdateToParameterMeanAbsRatio")).doubleValue() > 0.0);
            assertTrue(((Number) firstEpoch.get("parameterUpdateToParameterRmsRatio")).doubleValue() > 0.0);
        }

        String csv = Files.readString(checkpointDir.resolve("canonical-history.csv"));
        assertTrue(csv.contains("gradientL2NormBeforeClip"));
        assertTrue(csv.contains("gradientMeanAbsBeforeClip"));
        assertTrue(csv.contains("gradientRmsBeforeClip"));
        assertTrue(csv.contains("gradientZeroFraction"));
        assertTrue(csv.contains("gradientClipScale"));
        assertTrue(csv.contains("gradientClipped"));
        assertTrue(csv.contains("parameterMeanAbs"));
        assertTrue(csv.contains("parameterRms"));
        assertTrue(csv.contains("parameterZeroFraction"));
        assertTrue(csv.contains("gradientToParameterL2Ratio"));
        assertTrue(csv.contains("gradientToParameterRmsRatio"));
        assertTrue(csv.contains("parameterUpdateL2Norm"));
        assertTrue(csv.contains("parameterUpdateToParameterL2Ratio"));
        assertTrue(csv.contains("parameterL2Norm"));
    }

    @Test
    void canonicalTrainerRejectsNonFiniteLossBeforeBackwardAndSkipsCheckpoint() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-nonfinite-loss");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 0.5f;
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {Float.MAX_VALUE}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.1f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, null));
        assertTrue(error.getMessage().contains("train loss must be finite"));
        assertEquals(0.5f, model.parameters().get(0).data().data()[0], 1e-6f);

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteGuardEnabled"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteDetected"));
        assertEquals("train", summary.metadata().get("nonFinitePhase"));
        assertEquals("loss", summary.metadata().get("nonFiniteKind"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteOptimizerStepSkipped"));
        assertEquals("non-finite-train-loss", summary.metadata().get("stopReason"));
        assertEquals(0, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals(0, ((Number) summary.metadata().get("pendingGradientAccumulationBatches")).intValue());

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsVectorTrainingLossBeforeBackwardAndSkipsCheckpoint() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-vector-loss");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 0.5f;
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {1f, 2f}, 2));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.1f).build())
                .loss((predictions, targets) -> predictions)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, null));
        assertTrue(error.getMessage().contains("train loss tensor must contain exactly one value"));
        assertEquals(0.5f, model.parameters().get(0).data().data()[0], 1e-6f);

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("lossShapeGuardEnabled"));
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidLossShapeDetected"));
        assertEquals("train", summary.metadata().get("invalidLossShapePhase"));
        assertEquals("[2, 1]", summary.metadata().get("invalidLossShape"));
        assertEquals(2L, ((Number) summary.metadata().get("invalidLossShapeElementCount")).longValue());
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidLossShapeOptimizerStepSkipped"));
        assertEquals("invalid-loss-shape-train", summary.metadata().get("stopReason"));
        assertEquals(0, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals(0, ((Number) summary.metadata().get("pendingGradientAccumulationBatches")).intValue());

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsNonFiniteTrainingInputBeforeForwardAndSkipsCheckpoint() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-nonfinite-input");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 0.5f;
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {Float.NaN}, new float[] {1f}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.1f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, null));
        assertTrue(error.getMessage().contains("train input must be finite"));
        assertEquals(0.5f, model.parameters().get(0).data().data()[0], 1e-6f);

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteDetected"));
        assertEquals("train", summary.metadata().get("nonFinitePhase"));
        assertEquals("input", summary.metadata().get("nonFiniteKind"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteOptimizerStepSkipped"));
        assertEquals(1L, ((Number) summary.metadata().get("nonFiniteTotalValueCount")).longValue());
        assertEquals(1L, ((Number) summary.metadata().get("nonFiniteValueCount")).longValue());
        assertEquals(1L, ((Number) summary.metadata().get("nonFiniteNanCount")).longValue());
        assertEquals(0L, ((Number) summary.metadata().get("nonFinitePositiveInfinityCount")).longValue());
        assertEquals(0L, ((Number) summary.metadata().get("nonFiniteNegativeInfinityCount")).longValue());
        assertEquals(1.0, (double) summary.metadata().get("nonFiniteFraction"), 1e-12);
        assertEquals("non-finite-train-input", summary.metadata().get("stopReason"));
        assertEquals(0, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals(0, ((Number) summary.metadata().get("pendingGradientAccumulationBatches")).intValue());

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsNonFiniteTrainingPredictionBeforeLossAndSkipsCheckpoint() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-nonfinite-prediction");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new NonFinitePredictionModel(Float.POSITIVE_INFINITY))
                .optimizer(SGD.builder(List.of(), 0.1f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, null));
        assertTrue(error.getMessage().contains("train prediction must be finite"));

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteDetected"));
        assertEquals("train", summary.metadata().get("nonFinitePhase"));
        assertEquals("prediction", summary.metadata().get("nonFiniteKind"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteOptimizerStepSkipped"));
        assertEquals("non-finite-train-prediction", summary.metadata().get("stopReason"));
        assertEquals(0, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals(0, ((Number) summary.metadata().get("pendingGradientAccumulationBatches")).intValue());

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsNonFiniteValidationLabelBeforeLoss() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-nonfinite-validation-label");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 0.5f;
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));
        List<Batch> validation = List.of(batch(new float[] {1f}, new float[] {Float.POSITIVE_INFINITY}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.1f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, validation));
        assertTrue(error.getMessage().contains("validation label must be finite"));

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteDetected"));
        assertEquals("validation", summary.metadata().get("nonFinitePhase"));
        assertEquals("label", summary.metadata().get("nonFiniteKind"));
        assertEquals(Boolean.FALSE, summary.metadata().get("nonFiniteOptimizerStepSkipped"));
        assertEquals("non-finite-validation-label", summary.metadata().get("stopReason"));

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
    }

    @Test
    void canonicalTrainerRejectsNonFiniteValidationPredictionBeforeLoss() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-validation-nonfinite-prediction");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));
        List<Batch> validation = List.of(batch(new float[] {99f}, new float[] {99f}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new SentinelNonFinitePredictionModel(99f, Float.NaN))
                .optimizer(SGD.builder(List.of(), 0.1f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, validation));
        assertTrue(error.getMessage().contains("validation prediction must be finite"));

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteDetected"));
        assertEquals("validation", summary.metadata().get("nonFinitePhase"));
        assertEquals("prediction", summary.metadata().get("nonFiniteKind"));
        assertEquals(Boolean.FALSE, summary.metadata().get("nonFiniteOptimizerStepSkipped"));
        assertEquals("non-finite-validation-prediction", summary.metadata().get("stopReason"));

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
    }

    @Test
    void canonicalTrainerRejectsVectorValidationLossBeforeMetrics() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-validation-vector-loss");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 0.5f;
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));
        List<Batch> validation = List.of(batch(new float[] {1f, 2f}, new float[] {1f, 2f}, 2));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.1f).build())
                .loss((predictions, targets) -> predictions.numel() == 1L ? predictions.mean() : predictions)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, validation));
        assertTrue(error.getMessage().contains("validation loss tensor must contain exactly one value"));

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidLossShapeDetected"));
        assertEquals("validation", summary.metadata().get("invalidLossShapePhase"));
        assertEquals("[2, 1]", summary.metadata().get("invalidLossShape"));
        assertEquals(2L, ((Number) summary.metadata().get("invalidLossShapeElementCount")).longValue());
        assertEquals(Boolean.FALSE, summary.metadata().get("invalidLossShapeOptimizerStepSkipped"));
        assertEquals("invalid-loss-shape-validation", summary.metadata().get("stopReason"));

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
    }

    @Test
    void canonicalTrainerRejectsNonFiniteTrainingMetricAndSkipsCheckpoint() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-nonfinite-train-metric");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.1f).build())
                .loss(mseLoss::compute)
                .metric(() -> new ConstantMetric("bad_metric", Double.POSITIVE_INFINITY))
                .epochs(2)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, null));
        assertTrue(error.getMessage().contains("train metric bad_metric must be finite"));

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("metricFiniteGuardEnabled"));
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidMetricDetected"));
        assertEquals("train", summary.metadata().get("invalidMetricPhase"));
        assertEquals("bad_metric", summary.metadata().get("invalidMetricName"));
        assertEquals(Double.POSITIVE_INFINITY,
                ((Number) summary.metadata().get("invalidMetricValue")).doubleValue());
        assertEquals(Boolean.FALSE, summary.metadata().get("invalidMetricOptimizerStepSkipped"));
        assertEquals("invalid-metric-train-bad_metric", summary.metadata().get("stopReason"));

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsDuplicateMetricNamesBeforeRun() {
        MSELoss mseLoss = new MSELoss();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () -> CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.1f).build())
                .loss(mseLoss::compute)
                .metric(() -> new ConstantMetric("duplicate_metric", 1.0))
                .metric(() -> new ConstantMetric("duplicate_metric", 2.0))
                .epochs(1)
                .build());

        assertTrue(error.getMessage().contains("train metric name must be unique, duplicate: duplicate_metric"));
    }

    @Test
    void canonicalTrainerRejectsDynamicDuplicateMetricNamesAndSkipsCheckpoint() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-dynamic-duplicate-metric");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.1f).build())
                .loss(mseLoss::compute)
                .metric(() -> new DynamicNameMetric("dynamic_metric_a", "duplicate_metric"))
                .metric(() -> new DynamicNameMetric("dynamic_metric_b", "duplicate_metric"))
                .epochs(2)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, null));
        assertTrue(error.getMessage().contains("train metric name must be unique, duplicate: duplicate_metric"));

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidMetricDetected"));
        assertEquals("train", summary.metadata().get("invalidMetricPhase"));
        assertEquals("duplicate_metric", summary.metadata().get("invalidMetricName"));
        assertEquals("name", summary.metadata().get("invalidMetricKind"));
        assertEquals("invalid-metric-train-duplicate_metric", summary.metadata().get("stopReason"));

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsMetricValueFailureAndSkipsCheckpoint() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-throwing-metric");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.1f).build())
                .loss(mseLoss::compute)
                .metric(ThrowingMetric::new)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, null));
        assertTrue(error.getMessage().contains("train metric throwing_metric failed to produce a value"));
        assertTrue(error.getCause() instanceof IllegalStateException);

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidMetricDetected"));
        assertEquals("train", summary.metadata().get("invalidMetricPhase"));
        assertEquals("throwing_metric", summary.metadata().get("invalidMetricName"));
        assertTrue(Double.isNaN(((Number) summary.metadata().get("invalidMetricValue")).doubleValue()));
        assertEquals("IllegalStateException", summary.metadata().get("invalidMetricErrorType"));
        assertEquals(Boolean.FALSE, summary.metadata().get("invalidMetricOptimizerStepSkipped"));
        assertEquals("invalid-metric-train-throwing_metric", summary.metadata().get("stopReason"));

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsNonFiniteMetricDetailsAndSkipsCheckpoint() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-nonfinite-metric-detail");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.1f).build())
                .loss(mseLoss::compute)
                .metric(NonFiniteDetailMetric::new)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, null));
        assertTrue(error.getMessage().contains("train metric detail_metric detail details.bad must be finite"));

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidMetricDetected"));
        assertEquals("train", summary.metadata().get("invalidMetricPhase"));
        assertEquals("detail_metric", summary.metadata().get("invalidMetricName"));
        assertEquals("detail", summary.metadata().get("invalidMetricKind"));
        assertEquals("details.bad", summary.metadata().get("invalidMetricDetailPath"));
        assertTrue(Double.isNaN(((Number) summary.metadata().get("invalidMetricValue")).doubleValue()));
        assertEquals("invalid-metric-train-detail_metric", summary.metadata().get("stopReason"));

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsMetricDetailsFailureAndKeepsCause() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-throwing-metric-detail");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.1f).build())
                .loss(mseLoss::compute)
                .metric(ThrowingDetailMetric::new)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, null));
        assertTrue(error.getMessage()
                .contains("train metric throwing_detail_metric detail details failed to produce a value"));
        assertTrue(error.getCause() instanceof IllegalStateException);

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidMetricDetected"));
        assertEquals("train", summary.metadata().get("invalidMetricPhase"));
        assertEquals("throwing_detail_metric", summary.metadata().get("invalidMetricName"));
        assertEquals("detail", summary.metadata().get("invalidMetricKind"));
        assertEquals("details", summary.metadata().get("invalidMetricDetailPath"));
        assertEquals("IllegalStateException", summary.metadata().get("invalidMetricErrorType"));
        assertEquals("invalid-metric-train-throwing_detail_metric", summary.metadata().get("stopReason"));

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsNonFiniteValidationMetricWithPhaseMetadata() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-nonfinite-validation-metric");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));
        List<Batch> validation = List.of(batch(new float[] {99f}, new float[] {99f}, 1));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.1f).build())
                .loss(mseLoss::compute)
                .metric(() -> new SentinelNonFiniteMetric(99f))
                .epochs(2)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, validation));
        assertTrue(error.getMessage().contains("validation metric sentinel_metric must be finite"));

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidMetricDetected"));
        assertEquals("validation", summary.metadata().get("invalidMetricPhase"));
        assertEquals("sentinel_metric", summary.metadata().get("invalidMetricName"));
        assertTrue(Double.isNaN(((Number) summary.metadata().get("invalidMetricValue")).doubleValue()));
        assertEquals(Boolean.FALSE, summary.metadata().get("invalidMetricOptimizerStepSkipped"));
        assertEquals("invalid-metric-validation-sentinel_metric", summary.metadata().get("stopReason"));

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
    }

    @Test
    void canonicalTrainerRejectsMismatchedTrainingBatchSamplesBeforeForward() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-invalid-batch");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 0.5f;
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {1f, 2f}, 2, 1),
                GradTensor.of(new float[] {1f}, 1, 1)));

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.1f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(train, null));
        assertTrue(error.getMessage().contains("train batch sample count mismatch"));
        assertEquals(0.5f, model.parameters().get(0).data().data()[0], 1e-6f);

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("batchDataGuardEnabled"));
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidBatchDetected"));
        assertEquals("train", summary.metadata().get("invalidBatchPhase"));
        assertEquals("sample-count-mismatch", summary.metadata().get("invalidBatchReason"));
        assertEquals(Boolean.TRUE, summary.metadata().get("invalidBatchOptimizerStepSkipped"));
        assertEquals("invalid-batch-train-sample-count-mismatch", summary.metadata().get("stopReason"));
        assertEquals(0, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals(0, ((Number) summary.metadata().get("pendingGradientAccumulationBatches")).intValue());

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsNonFiniteGradientsBeforeOptimizerStep() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-nonfinite-gradient");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 0.25f;
        TrainingLossFunction nanGradientLoss = (predictions, targets) -> {
            GradTensor out = GradTensor.scalar(1f).requiresGrad(true);
            out.setGradFn(new Function.Context("NaNGradientLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    predictions.backward(GradTensor.full(Float.NaN, predictions.shape()));
                }
            });
            return out;
        };

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.1f).build())
                .loss(nanGradientLoss)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(List.of(batch(new float[] {1f}, new float[] {0f}, 1)), null));
        assertTrue(error.getMessage().contains("train gradient must be finite"));
        assertEquals(0.25f, model.parameters().get(0).data().data()[0], 1e-6f);

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("failed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteDetected"));
        assertEquals("train", summary.metadata().get("nonFinitePhase"));
        assertEquals("gradient", summary.metadata().get("nonFiniteKind"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteOptimizerStepSkipped"));
        assertEquals("non-finite-train-gradient", summary.metadata().get("stopReason"));
        assertEquals(0, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals(0, ((Number) summary.metadata().get("pendingGradientAccumulationBatches")).intValue());

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRejectsNonFiniteGradientsFromEpochEndAccumulationFlush() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-nonfinite-accumulated-gradient");
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 0.75f;
        TrainingLossFunction nanGradientLoss = (predictions, targets) -> {
            GradTensor out = GradTensor.scalar(1f).requiresGrad(true);
            out.setGradFn(new Function.Context("NaNAccumulatedGradientLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    predictions.backward(GradTensor.full(Float.NaN, predictions.shape()));
                }
            });
            return out;
        };

        CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.1f).build())
                .loss(nanGradientLoss)
                .epochs(1)
                .gradientAccumulationSteps(2)
                .checkpointDir(checkpointDir)
                .build();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> trainer.fit(List.of(batch(new float[] {1f}, new float[] {0f}, 1)), null));
        assertTrue(error.getMessage().contains("train gradient must be finite"));
        assertEquals(0.75f, model.parameters().get(0).data().data()[0], 1e-6f);

        TrainingSummary summary = trainer.summary();
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteDetected"));
        assertEquals("gradient", summary.metadata().get("nonFiniteKind"));
        assertEquals(Boolean.TRUE, summary.metadata().get("nonFiniteOptimizerStepSkipped"));
        assertEquals("non-finite-train-gradient", summary.metadata().get("stopReason"));
        assertEquals(0, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals(0, ((Number) summary.metadata().get("pendingGradientAccumulationBatches")).intValue());

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-report.json")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-model.safetensors")));
        assertFalse(Files.exists(checkpointDir.resolve("canonical-optimizer.state")));
    }

    @Test
    void canonicalTrainerRecordsBatchAndThroughputDiagnostics() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-throughput");
        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f}, new float[] {6f}, 1));
        List<Batch> validation = List.of(batch(new float[] {4f, 5f}, new float[] {8f, 10f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .build()) {
            trainer.fit(train, validation);

            TrainingSummary summary = trainer.summary();
            assertEquals(4L, ((Number) summary.metadata().get("trainBatchCount")).longValue());
            assertEquals(2L, ((Number) summary.metadata().get("validationBatchCount")).longValue());
            assertEquals(6L, ((Number) summary.metadata().get("trainSampleCount")).longValue());
            assertEquals(4L, ((Number) summary.metadata().get("validationSampleCount")).longValue());
            assertEquals(6L, ((Number) summary.metadata().get("trainInputElementCount")).longValue());
            assertEquals(4L, ((Number) summary.metadata().get("validationInputElementCount")).longValue());
            assertEquals(6L, ((Number) summary.metadata().get("trainLabelElementCount")).longValue());
            assertEquals(4L, ((Number) summary.metadata().get("validationLabelElementCount")).longValue());
            assertTrue(metric(summary, "trainComputeMillis") >= 0.0);
            assertTrue(metric(summary, "validationComputeMillis") >= 0.0);
            assertTrue(metric(summary, "trainSamplesPerSecond") >= 0.0);
            assertTrue(metric(summary, "validationSamplesPerSecond") >= 0.0);

            Map<String, Object> firstEpoch = epochHistory(summary).get(0);
            assertEquals(2L, ((Number) firstEpoch.get("trainBatchCount")).longValue());
            assertEquals(1L, ((Number) firstEpoch.get("validationBatchCount")).longValue());
            assertEquals(3L, ((Number) firstEpoch.get("trainSampleCount")).longValue());
            assertEquals(2L, ((Number) firstEpoch.get("validationSampleCount")).longValue());
            assertTrue(((Number) firstEpoch.get("trainComputeMillis")).doubleValue() >= 0.0);
            assertTrue(((Number) firstEpoch.get("validationComputeMillis")).doubleValue() >= 0.0);
        }

        String csv = Files.readString(checkpointDir.resolve("canonical-history.csv"));
        assertTrue(csv.contains("trainBatchCount"));
        assertTrue(csv.contains("validationBatchCount"));
        assertTrue(csv.contains("trainSamplesPerSecond"));
        assertTrue(csv.contains("validationSamplesPerSecond"));
    }

    @Test
    void canonicalTrainerWritesStructuredTrainingReportJson() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-report");
        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f}, new float[] {6f}, 1));
        List<Batch> validation = List.of(batch(new float[] {4f, 5f}, new float[] {8f, 10f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .build()) {
            trainer.fit(train, validation);

            TrainingSummary summary = trainer.summary();
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingReportEnabled"));
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingReportSaved"));
            assertEquals(Boolean.FALSE, summary.metadata().get("trainingReportSaveFailed"));
            assertEquals(checkpointDir.resolve("canonical-report.json").toString(),
                    summary.metadata().get("trainingReportFile"));
        }

        Path reportFile = checkpointDir.resolve("canonical-report.json");
        assertTrue(Files.isRegularFile(reportFile));
        String reportJson = Files.readString(reportFile);
        assertTrue(reportJson.startsWith("{\"bestValidationEpoch\""));
        assertTrue(reportJson.contains("\"schema\":\"gollek.canonical-trainer.report.v1\""));
        assertTrue(reportJson.contains("\"epochCount\":2"));
        assertTrue(reportJson.contains("\"metadata\""));
        assertTrue(reportJson.contains("\"epochHistory\""));
        assertTrue(reportJson.contains("\"trainBatchCount\":4"));
        assertTrue(reportJson.contains("\"validationBatchCount\":2"));
        assertTrue(reportJson.contains("\"trainMetric.mae\""));
        assertTrue(reportJson.contains("\"trainingHistoryFile\""));
        assertTrue(reportJson.contains("\"trainingReportSaved\":true"));
    }

    @Test
    void canonicalTrainerPreservesHistoryCsvAcrossResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-history-resume");
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> validation = List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));

        TrainingListener stopAfterFirstEpoch = new TrainingListener() {
            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                if (epoch == 0) {
                    session.stop();
                }
            }
        };

        Linear firstModel = new Linear(1, 1);
        try (CanonicalTrainer firstRun = CanonicalTrainer.builder()
                .model(firstModel)
                .optimizer(SGD.builder(firstModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(3)
                .checkpointDir(checkpointDir)
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .metric(MatrixDetailMetric::new)
                .listener(stopAfterFirstEpoch)
                .build()) {
            firstRun.fit(train, validation);
            assertEquals(1, firstRun.summary().epochCount());
            String csv = Files.readString(checkpointDir.resolve("canonical-history.csv"));
            assertEquals(2, csv.lines().count());
            assertTrue(csv.contains("trainMetricDetails.matrix_detail"));
            assertTrue(csv.contains("\"\"matrix\"\":[[1,2],[3,4]]"));
            assertFalse(csv.contains("matrix=[["));
        }

        Linear resumedModel = new Linear(1, 1);
        try (CanonicalTrainer resumedRun = CanonicalTrainer.builder()
                .model(resumedModel)
                .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(3)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .metric(CanonicalTrainer.Metrics.meanAbsoluteError())
                .metric(MatrixDetailMetric::new)
                .build()) {
            resumedRun.fit(train, validation);

            TrainingSummary summary = resumedRun.summary();
            assertEquals(3, summary.epochCount());
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistoryLoaded"));
            assertEquals(Boolean.FALSE, summary.metadata().get("trainingHistoryLoadFailed"));
            assertEquals(Boolean.TRUE, summary.metadata().get("trainingHistorySaved"));

            List<Map<String, Object>> history = epochHistory(summary);
            assertEquals(3, history.size());
            assertEquals(0, ((Number) history.get(0).get("epoch")).intValue());
            assertEquals(1, ((Number) history.get(1).get("epoch")).intValue());
            assertEquals(2, ((Number) history.get(2).get("epoch")).intValue());
            assertTrue(history.get(0).get("trainMetric.mae") instanceof Number);
            assertNotNull(metricMap(history.get(0), "trainMetrics").get("mae"));
            assertEquals(
                    List.of(List.of(1, 2), List.of(3, 4)),
                    metricDetails(history.get(0), "trainMetricDetails", "matrix_detail").get("matrix"));
        }

        String csv = Files.readString(checkpointDir.resolve("canonical-history.csv"));
        assertEquals(4, csv.lines().count());
        assertTrue(csv.lines().anyMatch(line -> line.startsWith("0,")));
        assertTrue(csv.lines().anyMatch(line -> line.startsWith("1,")));
        assertTrue(csv.lines().anyMatch(line -> line.startsWith("2,")));
    }

    @Test
    void canonicalTrainerRejectsUnregisteredMonitorMetrics() {
        MSELoss mseLoss = new MSELoss();

        assertThrows(IllegalArgumentException.class, () -> CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .bestModelMonitorMetric("f1")
                .build());
        assertThrows(IllegalArgumentException.class, () -> CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .earlyStopping(1)
                .earlyStoppingMonitorMetric("f1")
                .build());
    }

    @Test
    void gollekDlConvenienceFitSupportsGradientAccumulation() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                1.0,
                0,
                0.0,
                2);

        assertEquals(2, ((Number) summary.metadata().get("gradientAccumulationSteps")).intValue());
        assertEquals(1, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        assertEquals(0, ((Number) summary.metadata().get("pendingGradientAccumulationBatches")).intValue());
    }

    @Test
    void gollekDlTrainingOptionsExposeMixedPrecision() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        TrainingSummary summary = Gollek.DL.fit(
                model,
                train,
                List.of(),
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .gradScaler(GradScaler.builder().initScale(8.0).growthInterval(1).build())
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("mixedPrecisionEnabled"));
        assertEquals(Boolean.FALSE, summary.metadata().get("mixedPrecisionOverflowDetected"));
        assertEquals(0, ((Number) summary.metadata().get("mixedPrecisionOverflowSkipCount")).intValue());
        assertEquals(16.0, ((Number) summary.metadata().get("mixedPrecisionLossScale")).doubleValue(), 1e-9);
        assertEquals(1, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
    }

    @Test
    void gollekMlCompatibilityFacadeDelegatesToCanonicalFit() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        TrainingSummary summary = GollekML.DL.fit(
                model,
                train,
                1,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD);

        assertEquals(1, summary.epochCount());
        assertNotNull(summary.latestTrainLoss());
        assertTrue(Double.isFinite(summary.latestTrainLoss()));
    }

    @Test
    void gollekMlCompatibilityFacadeSupportsEarlyStoppingOptions() {
        Linear model = new Linear(1, 1);
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> validation = List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));

        TrainingSummary summary = GollekML.DL.fit(
                model,
                train,
                validation,
                10,
                0.01f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_ADAMW,
                1.0,
                1,
                1_000_000.0);

        assertEquals(2, summary.epochCount());
        assertEquals("early-stopping", summary.metadata().get("stopReason"));
        assertEquals(Boolean.TRUE, summary.metadata().get("earlyStoppingTriggered"));
    }

    @Test
    void canonicalTrainerBuilderCanResumeFromCheckpointState() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-checkpoint");
        Linear firstModel = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> validation = List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));

        TrainingListener stopAfterFirstEpoch = new TrainingListener() {
            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                if (epoch == 0) {
                    session.stop();
                }
            }
        };

        try (CanonicalTrainer firstRun = CanonicalTrainer.builder()
                .model(firstModel)
                .optimizer(SGD.builder(firstModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(3)
                .checkpointDir(checkpointDir)
                .listener(stopAfterFirstEpoch)
                .build()) {
            firstRun.fit(train, validation);
            assertEquals(1, firstRun.summary().epochCount());
            assertEquals("manual-stop", firstRun.summary().metadata().get("stopReason"));
            assertEquals(Boolean.TRUE, firstRun.summary().metadata().get("modelCheckpointEnabled"));
            assertEquals(Boolean.TRUE, firstRun.summary().metadata().get("modelCheckpointSaved"));
            assertEquals(Boolean.TRUE, firstRun.summary().metadata().get("optimizerCheckpointSupported"));
            assertEquals(Boolean.TRUE, firstRun.summary().metadata().get("optimizerCheckpointSaved"));
        }

        float[] firstModelWeights = firstModel.parameters().get(0).data().data().clone();
        Linear resumedModel = new Linear(1, 1);
        float[] resumedModelInitialWeights = resumedModel.parameters().get(0).data().data().clone();

        try (CanonicalTrainer resumedRun = CanonicalTrainer.builder()
                .model(resumedModel)
                .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(3)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .build()) {
            resumedRun.fit(List.of(), List.of());
            assertEquals(3, resumedRun.summary().epochCount());
            assertEquals(Boolean.TRUE, resumedRun.summary().metadata().get("resumedFromCheckpoint"));
            assertEquals(Boolean.TRUE, resumedRun.summary().metadata().get("modelCheckpointLoaded"));
            assertEquals(Boolean.FALSE, resumedRun.summary().metadata().get("modelCheckpointLoadFailed"));
            assertEquals(Boolean.TRUE, resumedRun.summary().metadata().get("modelCheckpointSaved"));
            assertEquals(Boolean.TRUE, resumedRun.summary().metadata().get("optimizerCheckpointLoaded"));
            assertEquals(Boolean.FALSE, resumedRun.summary().metadata().get("optimizerCheckpointLoadFailed"));
            assertEquals(Boolean.TRUE, resumedRun.summary().metadata().get("optimizerCheckpointSaved"));
        }

        float[] resumedModelFinalWeights = resumedModel.parameters().get(0).data().data();
        assertTrue(!java.util.Arrays.equals(resumedModelInitialWeights, resumedModelFinalWeights));
        assertArrayEquals(firstModelWeights, resumedModelFinalWeights, 1e-4f);
    }

    @Test
    void canonicalTrainerBuilderExposesCheckpointLoadGuardControl() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-checkpoint-invalid");
        Files.writeString(checkpointDir.resolve("canonical-runtime.state"),
                String.join("\n",
                        "formatVersion=999",
                        "nextEpoch=1",
                        "completedEpochs=1",
                        "globalStep=1"));

        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        IllegalStateException strictError = assertThrows(IllegalStateException.class,
                () -> CanonicalTrainer.builder()
                        .model(model)
                        .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                        .loss(mseLoss::compute)
                        .epochs(1)
                        .checkpointDir(checkpointDir)
                        .resumeFromCheckpoint()
                        .build());
        assertTrue(strictError.getMessage().contains("unsupported-format-version"));

        try (CanonicalTrainer lenient = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .build()) {
            lenient.fit(train, null);
            assertEquals(Boolean.TRUE, lenient.summary().metadata().get("checkpointLoadFailed"));
            assertEquals(Boolean.FALSE, lenient.summary().metadata().get("resumedFromCheckpoint"));
        }
    }

    @Test
    void canonicalTrainerRejectsCorruptedRuntimeCheckpointByManifestBeforeResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-runtime-integrity-mismatch");
        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer initial = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build()) {
            initial.fit(train, null);
            assertEquals(Boolean.TRUE, initial.summary().metadata().get("checkpointManifestSaved"));
        }

        writeCorruptCheckpoint(checkpointDir.resolve("canonical-runtime.state"));

        Linear resumedModel = new Linear(1, 1);
        IllegalStateException strictError = assertThrows(IllegalStateException.class,
                () -> CanonicalTrainer.builder()
                        .model(resumedModel)
                        .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                        .loss(mseLoss::compute)
                        .epochs(2)
                        .checkpointDir(checkpointDir)
                        .resumeFromCheckpoint()
                        .build());
        assertTrue(strictError.getMessage().contains("Runtime checkpoint integrity mismatch"));
        assertTrue(strictError.getMessage().contains("runtime checkpoint size mismatch"));
    }

    @Test
    void canonicalTrainerSkipsCorruptedRuntimeCheckpointInLenientResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-runtime-integrity-lenient");
        Linear model = new Linear(1, 1);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer initial = CanonicalTrainer.builder()
                .model(model)
                .optimizer(SGD.builder(model.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build()) {
            initial.fit(train, null);
        }

        writeCorruptCheckpoint(checkpointDir.resolve("canonical-runtime.state"));

        Linear resumedModel = new Linear(1, 1);
        try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                .model(resumedModel)
                .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .build()) {
            resumed.fit(train, null);
            Map<String, Object> metadata = resumed.summary().metadata();
            assertEquals(Boolean.FALSE, metadata.get("resumedFromCheckpoint"));
            assertEquals(Boolean.TRUE, metadata.get("checkpointResumePartial"));
            assertEquals(Boolean.TRUE, metadata.get("runtimeCheckpointIntegrityMismatch"));
            assertEquals(Boolean.FALSE, metadata.get("runtimeCheckpointResumeAllowed"));
            assertEquals(Boolean.TRUE, metadata.get("runtimeCheckpointResumeSkipped"));
            assertEquals(Boolean.TRUE, metadata.get("runtimeCheckpointLoadFailed"));
            assertEquals(
                    "runtime-checkpoint-integrity-mismatch",
                    metadata.get("runtimeCheckpointResumeDecision"));
            assertEquals(
                    "skip runtime checkpoint and rebuild runtime state from trainer artifacts",
                    metadata.get("runtimeCheckpointRecommendedAction"));
            Map<?, ?> runtimePlan = (Map<?, ?>) metadata.get("runtimeCheckpointResumePlan");
            assertEquals(Boolean.FALSE, runtimePlan.get("resumeAllowed"));
            assertEquals(
                    "runtime-checkpoint-integrity-mismatch",
                    runtimePlan.get("decision"));
            assertEquals(Boolean.TRUE, metadata.get("checkpointManifestLoaded"));
            assertEquals(Boolean.TRUE, metadata.get("checkpointManifestIntegrityMismatch"));
            assertTrue(String.valueOf(metadata.get("runtimeCheckpointLoadError"))
                    .contains("runtime checkpoint size mismatch"));
            assertTrue(String.valueOf(metadata.get("checkpointCompatibilityMismatches"))
                    .contains("runtime checkpoint size mismatch"));
        }
    }

    @Test
    void canonicalTrainerRejectsModelCheckpointMetadataMismatchByDefault() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-model-metadata-mismatch");
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer initial = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build()) {
            initial.fit(train, null);
            Map<String, Object> metadata = initial.summary().metadata();
            assertEquals(Boolean.TRUE, metadata.get("modelCheckpointMetadataPresent"));
            assertEquals(Boolean.TRUE, metadata.get("modelCheckpointMetadataSaved"));
            String metadataFile = Files.readString(checkpointDir.resolve("canonical-model.metadata"));
            assertTrue(metadataFile.contains("modelCheckpointBytes="));
            assertTrue(metadataFile.contains("modelCheckpointSha256="));
        }

        Linear incompatibleModel = new Linear(2, 1);
        IllegalStateException strictError = assertThrows(IllegalStateException.class, () -> {
            try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                    .model(incompatibleModel)
                    .optimizer(SGD.builder(incompatibleModel.parameters(), 0.01f).build())
                    .loss(mseLoss::compute)
                    .epochs(2)
                    .checkpointDir(checkpointDir)
                    .resumeFromCheckpoint()
                    .build()) {
                resumed.fit(List.of(), null);
            }
        });
        assertTrue(strictError.getMessage().contains("Model checkpoint metadata mismatch"));
        assertTrue(strictError.getMessage().contains("model parameter signature mismatch"));
    }

    @Test
    void canonicalTrainerRejectsCorruptedModelCheckpointByDefault() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-model-integrity-mismatch");
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer initial = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build()) {
            initial.fit(train, null);
        }

        Files.write(checkpointDir.resolve("canonical-model.safetensors"), new byte[] {1, 2, 3});

        Linear resumedModel = new Linear(1, 1);
        IllegalStateException strictError = assertThrows(IllegalStateException.class, () -> {
            try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                    .model(resumedModel)
                    .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                    .loss(mseLoss::compute)
                    .epochs(2)
                    .checkpointDir(checkpointDir)
                    .resumeFromCheckpoint()
                    .build()) {
                resumed.fit(List.of(), null);
            }
        });
        assertTrue(strictError.getMessage().contains("Model checkpoint metadata mismatch"));
        assertTrue(strictError.getMessage().contains("model checkpoint size mismatch"));
    }

    @Test
    void canonicalTrainerReportsModelCheckpointMetadataMismatchWhenGuardIsDisabled() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-model-metadata-lenient");
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer initial = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build()) {
            initial.fit(train, null);
        }

        Linear incompatibleModel = new Linear(2, 1);
        try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                .model(incompatibleModel)
                .optimizer(SGD.builder(incompatibleModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .build()) {
            resumed.fit(List.of(), null);
            Map<String, Object> metadata = resumed.summary().metadata();
            assertEquals(Boolean.TRUE, metadata.get("checkpointResumePartial"));
            assertEquals(Boolean.TRUE, metadata.get("modelCheckpointCompatibilityMismatch"));
            assertEquals(Boolean.TRUE, metadata.get("modelCheckpointLoadFailed"));
            assertEquals(Boolean.FALSE, metadata.get("modelCheckpointLoaded"));
            assertEquals(Boolean.TRUE, metadata.get("modelCheckpointMetadataLoaded"));
            assertEquals(Boolean.FALSE, metadata.get("modelCheckpointMetadataMissingOnResume"));
            assertTrue(String.valueOf(metadata.get("modelCheckpointLoadError"))
                    .contains("model parameter signature mismatch"));
            assertTrue(String.valueOf(metadata.get("checkpointResumeCompatibilityMismatches"))
                    .contains("model parameter signature mismatch"));
        }
    }

    @Test
    void canonicalTrainerReportsCorruptedModelCheckpointWhenGuardIsDisabled() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-model-integrity-lenient");
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer initial = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build()) {
            initial.fit(train, null);
        }

        Files.write(checkpointDir.resolve("canonical-model.safetensors"), new byte[] {1, 2, 3});

        Linear resumedModel = new Linear(1, 1);
        try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                .model(resumedModel)
                .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .build()) {
            resumed.fit(List.of(), null);
            Map<String, Object> metadata = resumed.summary().metadata();
            assertEquals(Boolean.TRUE, metadata.get("checkpointResumePartial"));
            assertEquals(Boolean.TRUE, metadata.get("modelCheckpointCompatibilityMismatch"));
            assertEquals(Boolean.TRUE, metadata.get("modelCheckpointLoadFailed"));
            assertEquals(Boolean.FALSE, metadata.get("modelCheckpointLoaded"));
            assertTrue(String.valueOf(metadata.get("modelCheckpointLoadError"))
                    .contains("model checkpoint size mismatch"));
            assertTrue(String.valueOf(metadata.get("checkpointResumeCompatibilityMismatches"))
                    .contains("model checkpoint size mismatch"));
        }
    }

    @Test
    void canonicalTrainerRejectsCorruptedOptimizerCheckpointByManifestByDefault() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-optimizer-integrity-mismatch");
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer initial = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build()) {
            initial.fit(train, null);
            Map<String, Object> metadata = initial.summary().metadata();
            assertEquals(Boolean.TRUE, metadata.get("checkpointManifestSaved"));
            assertEquals(Boolean.TRUE, metadata.get("checkpointManifestPresent"));
        }

        Files.write(checkpointDir.resolve("canonical-optimizer.state"), new byte[] {9, 8, 7, 6});

        Linear resumedModel = new Linear(1, 1);
        IllegalStateException strictError = assertThrows(IllegalStateException.class, () -> {
            try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                    .model(resumedModel)
                    .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                    .loss(mseLoss::compute)
                    .epochs(2)
                    .checkpointDir(checkpointDir)
                    .resumeFromCheckpoint()
                    .build()) {
                resumed.fit(train, null);
            }
        });
        assertTrue(strictError.getMessage().contains("Optimizer checkpoint integrity mismatch"));
        assertTrue(strictError.getMessage().contains("optimizer checkpoint size mismatch"));
    }

    @Test
    void canonicalTrainerReportsCorruptedOptimizerCheckpointByManifestWhenGuardIsDisabled() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-optimizer-integrity-lenient");
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer initial = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build()) {
            initial.fit(train, null);
        }

        Files.write(checkpointDir.resolve("canonical-optimizer.state"), new byte[] {9, 8, 7, 6});

        Linear resumedModel = new Linear(1, 1);
        try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                .model(resumedModel)
                .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .build()) {
            resumed.fit(train, null);
            Map<String, Object> metadata = resumed.summary().metadata();
            assertEquals(Boolean.TRUE, metadata.get("checkpointResumePartial"));
            assertEquals(Boolean.TRUE, metadata.get("checkpointManifestLoaded"));
            assertEquals(Boolean.TRUE, metadata.get("checkpointManifestIntegrityMismatch"));
            assertEquals(Boolean.FALSE, metadata.get("optimizerCheckpointLoaded"));
            assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointLoadFailed"));
            assertTrue(String.valueOf(metadata.get("optimizerCheckpointLoadError"))
                    .contains("optimizer checkpoint size mismatch"));
            assertTrue(String.valueOf(metadata.get("checkpointResumeCompatibilityMismatches"))
                    .contains("optimizer checkpoint size mismatch"));
        }
    }

    @Test
    void canonicalTrainerReportsCheckpointArtifactsMissingAtResumeTime() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-partial-resume");
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.01f).build();
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer initial = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .scheduler(new StepLR(optimizer, 1, 0.5f))
                .gradScaler(GradScaler.builder().initScale(4.0).growthInterval(2).build())
                .build()) {
            initial.fit(train, null);
            Map<String, Object> metadata = initial.summary().metadata();
            assertEquals(Boolean.TRUE, metadata.get("modelCheckpointPresent"));
            assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointPresent"));
            assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointPresent"));
            assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointPresent"));
            assertEquals(Boolean.TRUE, metadata.get("trainingHistoryPresent"));
            assertEquals(Boolean.TRUE, metadata.get("trainingReportPresent"));
            assertEquals(Boolean.FALSE, metadata.get("optimizerCheckpointMissingOnResume"));
        }

        Files.deleteIfExists(checkpointDir.resolve("canonical-optimizer.state"));
        Files.deleteIfExists(checkpointDir.resolve("canonical-scheduler.state"));
        Files.deleteIfExists(checkpointDir.resolve("canonical-grad-scaler.state"));
        Files.deleteIfExists(checkpointDir.resolve("canonical-history.csv"));

        Linear resumedModel = new Linear(1, 1);
        SGD resumedOptimizer = SGD.builder(resumedModel.parameters(), 0.01f).build();
        IllegalStateException strictError = assertThrows(IllegalStateException.class, () -> {
            try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                    .model(resumedModel)
                    .optimizer(resumedOptimizer)
                    .loss(mseLoss::compute)
                    .epochs(2)
                    .checkpointDir(checkpointDir)
                    .resumeFromCheckpoint()
                    .scheduler(new StepLR(resumedOptimizer, 1, 0.5f))
                    .gradScaler(GradScaler.builder().initScale(4.0).growthInterval(2).build())
                    .build()) {
                resumed.fit(train, null);
            }
        });
        assertTrue(strictError.getMessage().contains("Missing optimizer checkpoint artifact"));

        Linear lenientModel = new Linear(1, 1);
        SGD lenientOptimizer = SGD.builder(lenientModel.parameters(), 0.01f).build();
        try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                .model(lenientModel)
                .optimizer(lenientOptimizer)
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .scheduler(new StepLR(lenientOptimizer, 1, 0.5f))
                .gradScaler(GradScaler.builder().initScale(4.0).growthInterval(2).build())
                .build()) {
            resumed.fit(train, null);
            Map<String, Object> metadata = resumed.summary().metadata();
            assertEquals(Boolean.TRUE, metadata.get("checkpointResumePartial"));
            assertEquals(
                    List.of("optimizer", "scheduler", "gradScaler", "history"),
                    metadata.get("checkpointResumeMissingArtifacts"));
            assertEquals(Boolean.TRUE, metadata.get("modelCheckpointLoaded"));
            assertEquals(Boolean.FALSE, metadata.get("modelCheckpointMissingOnResume"));
            assertEquals(Boolean.TRUE, metadata.get("modelCheckpointPresent"));
            assertEquals(Boolean.FALSE, metadata.get("optimizerCheckpointLoaded"));
            assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointMissingOnResume"));
            assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointPresent"));
            assertEquals(Boolean.FALSE, metadata.get("schedulerCheckpointLoaded"));
            assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointMissingOnResume"));
            assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointPresent"));
            assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointLoaded"));
            assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointMissingOnResume"));
            assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointPresent"));
            assertEquals(Boolean.FALSE, metadata.get("trainingHistoryLoaded"));
            assertEquals(Boolean.TRUE, metadata.get("trainingHistoryMissingOnResume"));
            assertEquals(Boolean.TRUE, metadata.get("trainingHistoryPresent"));
            assertEquals(Boolean.TRUE, metadata.get("trainingReportPresent"));
        }
    }

    @Test
    void canonicalTrainerResumeRestoresAdamwOptimizerStateContinuity() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-optimizer-resume");
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> validation = List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));
        MSELoss mseLoss = new MSELoss();

        Linear seed = new Linear(1, 1);
        Linear interruptedModel = new Linear(1, 1);
        Linear baselineModel = new Linear(1, 1);
        copyParameters(seed, interruptedModel);
        copyParameters(seed, baselineModel);

        TrainingListener stopAfterSecondEpoch = new TrainingListener() {
            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                if (epoch == 1) {
                    session.stop();
                }
            }
        };

        try (CanonicalTrainer interrupted = CanonicalTrainer.builder()
                .model(interruptedModel)
                .optimizer(AdamW.builder(interruptedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(4)
                .checkpointDir(checkpointDir)
                .listener(stopAfterSecondEpoch)
                .build()) {
            interrupted.fit(train, validation);
            assertEquals(2, interrupted.summary().epochCount());
            assertEquals(Boolean.TRUE, interrupted.summary().metadata().get("optimizerCheckpointSupported"));
            assertEquals(Boolean.TRUE, interrupted.summary().metadata().get("optimizerCheckpointSaved"));
            assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-optimizer.state")));
        }

        try (CanonicalTrainer baseline = CanonicalTrainer.builder()
                .model(baselineModel)
                .optimizer(AdamW.builder(baselineModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(4)
                .build()) {
            baseline.fit(train, validation);
            assertEquals(4, baseline.summary().epochCount());
        }

        Linear resumedModel = new Linear(1, 1);
        copyParameters(seed, resumedModel);
        try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                .model(resumedModel)
                .optimizer(AdamW.builder(resumedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(4)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .build()) {
            resumed.fit(train, validation);
            assertEquals(4, resumed.summary().epochCount());
            assertEquals(Boolean.TRUE, resumed.summary().metadata().get("resumedFromCheckpoint"));
            assertEquals(Boolean.TRUE, resumed.summary().metadata().get("optimizerCheckpointLoaded"));
            assertEquals(Boolean.FALSE, resumed.summary().metadata().get("optimizerCheckpointLoadFailed"));
        }

        assertParametersClose(baselineModel, resumedModel, 1e-5f);
    }

    @Test
    void canonicalTrainerResumeRestoresAdamOptimizerStateContinuity() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-adam-resume");
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> validation = List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));
        MSELoss mseLoss = new MSELoss();

        Linear seed = new Linear(1, 1);
        Linear interruptedModel = new Linear(1, 1);
        Linear baselineModel = new Linear(1, 1);
        copyParameters(seed, interruptedModel);
        copyParameters(seed, baselineModel);

        TrainingListener stopAfterSecondEpoch = new TrainingListener() {
            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                if (epoch == 1) {
                    session.stop();
                }
            }
        };

        try (CanonicalTrainer interrupted = CanonicalTrainer.builder()
                .model(interruptedModel)
                .optimizer(Adam.builder(interruptedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(4)
                .checkpointDir(checkpointDir)
                .listener(stopAfterSecondEpoch)
                .build()) {
            interrupted.fit(train, validation);
            assertEquals(2, interrupted.summary().epochCount());
            assertEquals(Boolean.TRUE, interrupted.summary().metadata().get("optimizerCheckpointSupported"));
            assertEquals(Boolean.TRUE, interrupted.summary().metadata().get("optimizerCheckpointSaved"));
            assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-optimizer.state")));
        }

        try (CanonicalTrainer baseline = CanonicalTrainer.builder()
                .model(baselineModel)
                .optimizer(Adam.builder(baselineModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(4)
                .build()) {
            baseline.fit(train, validation);
            assertEquals(4, baseline.summary().epochCount());
        }

        Linear resumedModel = new Linear(1, 1);
        copyParameters(seed, resumedModel);
        try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                .model(resumedModel)
                .optimizer(Adam.builder(resumedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(4)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .build()) {
            resumed.fit(train, validation);
            assertEquals(4, resumed.summary().epochCount());
            assertEquals(Boolean.TRUE, resumed.summary().metadata().get("resumedFromCheckpoint"));
            assertEquals(Boolean.TRUE, resumed.summary().metadata().get("optimizerCheckpointLoaded"));
            assertEquals(Boolean.FALSE, resumed.summary().metadata().get("optimizerCheckpointLoadFailed"));
        }

        assertParametersClose(baselineModel, resumedModel, 1e-5f);
    }

    @Test
    void canonicalTrainerResumeRestoresRmspropOptimizerStateContinuity() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-rmsprop-resume");
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> validation = List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));
        MSELoss mseLoss = new MSELoss();

        Linear seed = new Linear(1, 1);
        Linear interruptedModel = new Linear(1, 1);
        Linear baselineModel = new Linear(1, 1);
        copyParameters(seed, interruptedModel);
        copyParameters(seed, baselineModel);

        TrainingListener stopAfterSecondEpoch = new TrainingListener() {
            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                if (epoch == 1) {
                    session.stop();
                }
            }
        };

        try (CanonicalTrainer interrupted = CanonicalTrainer.builder()
                .model(interruptedModel)
                .optimizer(RMSprop.builder(interruptedModel.parameters(), 0.01f).momentum(0.9f).build())
                .loss(mseLoss::compute)
                .epochs(4)
                .checkpointDir(checkpointDir)
                .listener(stopAfterSecondEpoch)
                .build()) {
            interrupted.fit(train, validation);
            assertEquals(2, interrupted.summary().epochCount());
            assertEquals(Boolean.TRUE, interrupted.summary().metadata().get("optimizerCheckpointSupported"));
            assertEquals(Boolean.TRUE, interrupted.summary().metadata().get("optimizerCheckpointSaved"));
            assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-optimizer.state")));
        }

        try (CanonicalTrainer baseline = CanonicalTrainer.builder()
                .model(baselineModel)
                .optimizer(RMSprop.builder(baselineModel.parameters(), 0.01f).momentum(0.9f).build())
                .loss(mseLoss::compute)
                .epochs(4)
                .build()) {
            baseline.fit(train, validation);
            assertEquals(4, baseline.summary().epochCount());
        }

        Linear resumedModel = new Linear(1, 1);
        copyParameters(seed, resumedModel);
        try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                .model(resumedModel)
                .optimizer(RMSprop.builder(resumedModel.parameters(), 0.01f).momentum(0.9f).build())
                .loss(mseLoss::compute)
                .epochs(4)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .build()) {
            resumed.fit(train, validation);
            assertEquals(4, resumed.summary().epochCount());
            assertEquals(Boolean.TRUE, resumed.summary().metadata().get("resumedFromCheckpoint"));
            assertEquals(Boolean.TRUE, resumed.summary().metadata().get("optimizerCheckpointLoaded"));
            assertEquals(Boolean.FALSE, resumed.summary().metadata().get("optimizerCheckpointLoadFailed"));
        }

        assertParametersClose(baselineModel, resumedModel, 1e-5f);
    }

    @Test
    void canonicalTrainerResumeFailsOnOptimizerCheckpointHyperparameterMismatch() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-optimizer-mismatch");
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        MSELoss mseLoss = new MSELoss();

        Linear firstModel = new Linear(1, 1);
        try (CanonicalTrainer firstRun = CanonicalTrainer.builder()
                .model(firstModel)
                .optimizer(AdamW.builder(firstModel.parameters(), 0.01f).weightDecay(0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build()) {
            firstRun.fit(train, null);
            assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-optimizer.state")));
        }

        Linear resumedModel = new Linear(1, 1);
        IllegalStateException mismatchError = assertThrows(IllegalStateException.class, () -> {
            try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                    .model(resumedModel)
                    .optimizer(AdamW.builder(resumedModel.parameters(), 0.01f).weightDecay(0.0f).build())
                    .loss(mseLoss::compute)
                    .epochs(2)
                    .checkpointDir(checkpointDir)
                    .resumeFromCheckpoint()
                    .build()) {
                resumed.fit(train, null);
            }
        });

        assertTrue(mismatchError.getMessage().contains("optimizer checkpoint"));
    }

    @Test
    void canonicalTrainerStepsLearningRateSchedulerAndCheckpointsState() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-scheduler");
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.1f).build();
        StepLR scheduler = new StepLR(optimizer, 1, 0.5f);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .scheduler(scheduler)
                .loss(mseLoss::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .build()) {
            trainer.fit(train, null);
            TrainingSummary summary = trainer.summary();
            assertEquals(Boolean.TRUE, summary.metadata().get("learningRateSchedulerEnabled"));
            assertEquals("BATCH", summary.metadata().get("learningRateSchedulerStepUnit"));
            assertEquals(2, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
            assertEquals(Boolean.TRUE, summary.metadata().get("schedulerCheckpointSupported"));
            assertEquals(Boolean.TRUE, summary.metadata().get("schedulerCheckpointSaved"));
            assertEquals(0.05f, optimizer.learningRate(), 1e-6f);
        }

        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-scheduler.state")));
    }

    @Test
    void gollekTrainingOptionsWarmupCosineLrBatchesSchedulesFromZeroAndReportsState() throws Exception {
        SGD directOptimizer = SGD.builder(List.of(), 0.2f).build();
        var directScheduler = Gollek.DL.warmupCosineScheduler(directOptimizer, 2, 4, 0.2f, 0.02f);
        assertEquals(0.0f, directOptimizer.learningRate(), 1e-7f);
        directScheduler.step();
        assertEquals(0.1f, directOptimizer.learningRate(), 1e-6f);

        Path checkpointDir = Files.createTempDirectory("gollek-warmup-cosine-scheduler");
        List<Batch> train = List.of(
                batch(new float[] {1f}, new float[] {1f}, 1),
                batch(new float[] {2f}, new float[] {2f}, 1),
                batch(new float[] {3f}, new float[] {3f}, 1),
                batch(new float[] {4f}, new float[] {4f}, 1));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                null,
                1,
                0.1f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .checkpointDir(checkpointDir)
                        .warmupCosineLrBatches(2, 4, 0.1f, 0.01f)
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("learningRateSchedulerEnabled"));
        assertEquals("WarmupCosineScheduler", summary.metadata().get("learningRateSchedulerType"));
        assertEquals("BATCH", summary.metadata().get("learningRateSchedulerStepUnit"));
        assertEquals(4, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
        assertEquals(0.01, metric(summary, "learningRate"), 1e-6);
        assertEquals(2, ((Number) summary.metadata().get("learningRateSchedulerState.warmupSteps")).intValue());
        assertEquals(4, ((Number) summary.metadata().get("learningRateSchedulerState.totalSteps")).intValue());
        assertEquals(4, ((Number) summary.metadata().get("learningRateSchedulerState.currentStep")).intValue());
        assertEquals(0.01, metric(summary, "learningRateSchedulerState.currentLr"), 1e-6);
        assertEquals(Boolean.TRUE, summary.metadata().get("schedulerCheckpointSaved"));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-scheduler.state")));

        String reportJson = Files.readString(checkpointDir.resolve("canonical-report.json"));
        assertTrue(reportJson.contains("\"learningRateSchedulerType\":\"WarmupCosineScheduler\""));
        assertTrue(reportJson.contains("\"warmupSteps\":2"));
        assertTrue(reportJson.contains("\"currentStep\":4"));
    }

    @Test
    void gollekTrainingOptionsOneCycleLrBatchesSchedulesAndReportsState() throws Exception {
        SGD directOptimizer = SGD.builder(List.of(), 0.1f).build();
        var directScheduler = Gollek.DL.oneCycleScheduler(
                directOptimizer,
                4,
                0.2f,
                0.5f,
                10.0f,
                100.0f,
                OneCycleLR.AnnealStrategy.COSINE);
        assertEquals(0.02f, directOptimizer.learningRate(), 1e-6f);
        directScheduler.step();
        assertEquals(0.11f, directOptimizer.learningRate(), 1e-6f);

        Path checkpointDir = Files.createTempDirectory("gollek-one-cycle-scheduler");
        List<Batch> train = List.of(
                batch(new float[] {1f}, new float[] {1f}, 1),
                batch(new float[] {2f}, new float[] {2f}, 1),
                batch(new float[] {3f}, new float[] {3f}, 1),
                batch(new float[] {4f}, new float[] {4f}, 1));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                null,
                1,
                0.1f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .checkpointDir(checkpointDir)
                        .oneCycleLrBatches(
                                4,
                                0.2f,
                                0.5f,
                                10.0f,
                                100.0f,
                                OneCycleLR.AnnealStrategy.COSINE)
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("learningRateSchedulerEnabled"));
        assertEquals("OneCycleLR", summary.metadata().get("learningRateSchedulerType"));
        assertEquals("BATCH", summary.metadata().get("learningRateSchedulerStepUnit"));
        assertEquals(4, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
        assertEquals(0.0002, metric(summary, "learningRate"), 1e-6);
        assertEquals(4, ((Number) summary.metadata().get("learningRateSchedulerState.totalSteps")).intValue());
        assertEquals(2, ((Number) summary.metadata().get("learningRateSchedulerState.warmupSteps")).intValue());
        assertEquals(4, ((Number) summary.metadata().get("learningRateSchedulerState.currentStep")).intValue());
        assertEquals(0.2, metric(summary, "learningRateSchedulerState.maxLr"), 1e-6);
        assertEquals(0.02, metric(summary, "learningRateSchedulerState.initialLr"), 1e-6);
        assertEquals(0.0002, metric(summary, "learningRateSchedulerState.currentLr"), 1e-6);
        assertEquals("COSINE", summary.metadata().get("learningRateSchedulerState.annealStrategy"));
        assertEquals(Boolean.TRUE, summary.metadata().get("schedulerCheckpointSaved"));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-scheduler.state")));

        String reportJson = Files.readString(checkpointDir.resolve("canonical-report.json"));
        assertTrue(reportJson.contains("\"learningRateSchedulerType\":\"OneCycleLR\""));
        assertTrue(reportJson.contains("\"annealStrategy\":\"COSINE\""));
        assertTrue(reportJson.contains("\"currentStep\":4"));
    }

    @Test
    void gollekTrainingOptionsExponentialLrBatchesSchedulesAndReportsState() throws Exception {
        SGD directOptimizer = SGD.builder(List.of(), 0.2f).build();
        ExponentialLR directScheduler = Gollek.DL.exponentialScheduler(directOptimizer, 0.5f);
        directScheduler.step();
        assertEquals(0.1f, directOptimizer.learningRate(), 1e-6f);

        Path checkpointDir = Files.createTempDirectory("gollek-exponential-scheduler");
        List<Batch> train = List.of(
                batch(new float[] {1f}, new float[] {1f}, 1),
                batch(new float[] {2f}, new float[] {2f}, 1),
                batch(new float[] {3f}, new float[] {3f}, 1));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                null,
                1,
                0.1f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .checkpointDir(checkpointDir)
                        .exponentialLrBatches(0.5f)
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("learningRateSchedulerEnabled"));
        assertEquals("ExponentialLR", summary.metadata().get("learningRateSchedulerType"));
        assertEquals("BATCH", summary.metadata().get("learningRateSchedulerStepUnit"));
        assertEquals(3, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
        assertEquals(0.0125, metric(summary, "learningRate"), 1e-6);
        assertEquals(3, ((Number) summary.metadata().get("learningRateSchedulerState.currentStep")).intValue());
        assertEquals(0.5, metric(summary, "learningRateSchedulerState.gamma"), 1e-6);
        assertEquals(0.0125, metric(summary, "learningRateSchedulerState.currentLr"), 1e-6);
        assertEquals(Boolean.TRUE, summary.metadata().get("schedulerCheckpointSaved"));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-scheduler.state")));

        String reportJson = Files.readString(checkpointDir.resolve("canonical-report.json"));
        assertTrue(reportJson.contains("\"learningRateSchedulerType\":\"ExponentialLR\""));
        assertTrue(reportJson.contains("\"gamma\":0.5"));
        assertTrue(reportJson.contains("\"currentStep\":3"));
    }

    @Test
    void gollekTrainingOptionsSequentialLrBatchesComposesSchedulersAndReportsState() throws Exception {
        SGD directOptimizer = SGD.builder(List.of(), 0.1f).build();
        SequentialLR directScheduler = Gollek.DL.sequentialScheduler(
                directOptimizer,
                List.of(
                        Gollek.DL.exponentialScheduler(directOptimizer, 0.5f),
                        Gollek.DL.exponentialScheduler(directOptimizer, 0.1f)),
                2);
        directScheduler.step();
        assertEquals(0.05f, directOptimizer.learningRate(), 1e-6f);
        directScheduler.step();
        assertEquals(0.025f, directOptimizer.learningRate(), 1e-6f);
        directScheduler.step();
        assertEquals(0.01f, directOptimizer.learningRate(), 1e-6f);
        assertEquals(1, directScheduler.activeIndex());

        Path checkpointDir = Files.createTempDirectory("gollek-sequential-scheduler");
        List<Batch> train = List.of(
                batch(new float[] {1f}, new float[] {1f}, 1),
                batch(new float[] {2f}, new float[] {2f}, 1),
                batch(new float[] {3f}, new float[] {3f}, 1));
        List<Gollek.DL.SchedulerFactory> schedulers = List.of(
                optimizer -> Gollek.DL.exponentialScheduler(optimizer, 0.5f),
                optimizer -> Gollek.DL.exponentialScheduler(optimizer, 0.1f));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                null,
                1,
                0.1f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .checkpointDir(checkpointDir)
                        .sequentialLrBatches(schedulers, 2)
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("learningRateSchedulerEnabled"));
        assertEquals("SequentialLR", summary.metadata().get("learningRateSchedulerType"));
        assertEquals("BATCH", summary.metadata().get("learningRateSchedulerStepUnit"));
        assertEquals(3, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
        assertEquals(0.01, metric(summary, "learningRate"), 1e-6);
        assertEquals(3, ((Number) summary.metadata().get("learningRateSchedulerState.currentStep")).intValue());
        assertEquals(1, ((Number) summary.metadata().get("learningRateSchedulerState.activeIndex")).intValue());
        assertEquals(List.of(2), summary.metadata().get("learningRateSchedulerState.milestones"));
        assertEquals(0.01, metric(summary, "learningRateSchedulerState.currentLr"), 1e-6);
        assertEquals(Boolean.TRUE, summary.metadata().get("schedulerCheckpointSaved"));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-scheduler.state")));

        String reportJson = Files.readString(checkpointDir.resolve("canonical-report.json"));
        assertTrue(reportJson.contains("\"learningRateSchedulerType\":\"SequentialLR\""));
        assertTrue(reportJson.contains("\"milestones\":[2]"));
        assertTrue(reportJson.contains("\"scheduler\":\"ExponentialLR\""));
    }

    @Test
    void gollekTrainingOptionsReduceLrOnPlateauStepsAfterValidationAndReportsState() throws Exception {
        SGD directOptimizer = SGD.builder(List.of(), 0.1f).build();
        var directScheduler = Gollek.DL.reduceLrOnPlateauScheduler(
                directOptimizer,
                tech.kayys.gollek.ml.optim.ReduceLROnPlateau.Mode.MIN,
                0.5f,
                0,
                0.0,
                0,
                0.025f);
        directScheduler.step(1.0);
        directScheduler.step(1.0);
        assertEquals(0.05f, directOptimizer.learningRate(), 1e-6f);

        Path checkpointDir = Files.createTempDirectory("gollek-reduce-lr-plateau");
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {1f}, 1));
        List<Batch> validation = List.of(batch(new float[] {1f}, new float[] {2f}, 1));

        TrainingSummary summary = Gollek.DL.fit(
                new IdentityModel(),
                train,
                validation,
                3,
                0.1f,
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .device("cpu")
                        .checkpointDir(checkpointDir)
                        .reduceLrOnPlateauValidationLoss(0.5f, 0, 0.0, 0, 0.025f)
                        .build());

        assertEquals(Boolean.TRUE, summary.metadata().get("learningRateSchedulerEnabled"));
        assertEquals("ReduceLROnPlateau", summary.metadata().get("learningRateSchedulerType"));
        assertEquals("VALIDATION", summary.metadata().get("learningRateSchedulerStepUnit"));
        assertEquals("validation_loss", summary.metadata().get("learningRateSchedulerMonitor"));
        assertEquals(3, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
        assertEquals(0.025, metric(summary, "learningRate"), 1e-6);
        assertEquals(2, ((Number) summary.metadata().get("learningRateSchedulerState.reductionCount")).intValue());
        assertEquals(3, ((Number) summary.metadata().get("learningRateSchedulerState.stepCount")).intValue());
        assertEquals(0.025, metric(summary, "learningRateSchedulerState.currentLr"), 1e-6);
        assertEquals(Boolean.TRUE, summary.metadata().get("schedulerCheckpointSaved"));
        assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-scheduler.state")));

        String reportJson = Files.readString(checkpointDir.resolve("canonical-report.json"));
        assertTrue(reportJson.contains("\"learningRateSchedulerType\":\"ReduceLROnPlateau\""));
        assertTrue(reportJson.contains("\"learningRateSchedulerMonitor\":\"validation_loss\""));
        assertTrue(reportJson.contains("\"reductionCount\":2"));
    }

    @Test
    void canonicalTrainerResumeRestoresSchedulerProgressContinuity() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-scheduler-resume");
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> validation = List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));
        MSELoss mseLoss = new MSELoss();

        Linear seed = new Linear(1, 1);
        Linear interruptedModel = new Linear(1, 1);
        Linear baselineModel = new Linear(1, 1);
        copyParameters(seed, interruptedModel);
        copyParameters(seed, baselineModel);

        TrainingListener stopAfterSecondEpoch = new TrainingListener() {
            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                if (epoch == 1) {
                    session.stop();
                }
            }
        };

        SGD interruptedOptimizer = SGD.builder(interruptedModel.parameters(), 0.1f).momentum(0.9f).build();
        try (CanonicalTrainer interrupted = CanonicalTrainer.builder()
                .model(interruptedModel)
                .optimizer(interruptedOptimizer)
                .scheduler(new StepLR(interruptedOptimizer, 2, 0.5f))
                .loss(mseLoss::compute)
                .epochs(4)
                .checkpointDir(checkpointDir)
                .listener(stopAfterSecondEpoch)
                .build()) {
            interrupted.fit(train, validation);
            assertEquals(2, interrupted.summary().epochCount());
            assertEquals(Boolean.TRUE, interrupted.summary().metadata().get("schedulerCheckpointSaved"));
            assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-scheduler.state")));
        }

        SGD baselineOptimizer = SGD.builder(baselineModel.parameters(), 0.1f).momentum(0.9f).build();
        try (CanonicalTrainer baseline = CanonicalTrainer.builder()
                .model(baselineModel)
                .optimizer(baselineOptimizer)
                .scheduler(new StepLR(baselineOptimizer, 2, 0.5f))
                .loss(mseLoss::compute)
                .epochs(4)
                .build()) {
            baseline.fit(train, validation);
            assertEquals(4, baseline.summary().epochCount());
        }

        Linear resumedModel = new Linear(1, 1);
        copyParameters(seed, resumedModel);
        SGD resumedOptimizer = SGD.builder(resumedModel.parameters(), 0.1f).momentum(0.9f).build();
        try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                .model(resumedModel)
                .optimizer(resumedOptimizer)
                .scheduler(new StepLR(resumedOptimizer, 2, 0.5f))
                .loss(mseLoss::compute)
                .epochs(4)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .build()) {
            resumed.fit(train, validation);
            TrainingSummary summary = resumed.summary();
            assertEquals(4, summary.epochCount());
            assertEquals(Boolean.TRUE, summary.metadata().get("schedulerCheckpointLoaded"));
            assertEquals(Boolean.FALSE, summary.metadata().get("schedulerCheckpointLoadFailed"));
            assertEquals(8, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
        }

        assertEquals(baselineOptimizer.learningRate(), resumedOptimizer.learningRate(), 1e-6f);
        assertParametersClose(baselineModel, resumedModel, 1e-5f);
    }

    @Test
    void canonicalTrainerGradientAccumulationMatchesEquivalentFullBatch() {
        Linear seed = new Linear(1, 1);
        Linear accumulatedModel = new Linear(1, 1);
        Linear fullBatchModel = new Linear(1, 1);
        copyParameters(seed, accumulatedModel);
        copyParameters(seed, fullBatchModel);
        MSELoss mseLoss = new MSELoss();

        List<Batch> microBatches = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
        List<Batch> fullBatch = List.of(batch(new float[] {1f, 2f, 3f, 4f}, new float[] {2f, 4f, 6f, 8f}, 4));

        try (CanonicalTrainer accumulated = CanonicalTrainer.builder()
                .model(accumulatedModel)
                .optimizer(SGD.builder(accumulatedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .gradientAccumulationSteps(2)
                .build()) {
            accumulated.fit(microBatches, null);
            TrainingSummary summary = accumulated.summary();
            assertEquals(2, ((Number) summary.metadata().get("gradientAccumulationSteps")).intValue());
            assertEquals(1, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
            assertEquals(0, ((Number) summary.metadata().get("pendingGradientAccumulationBatches")).intValue());
        }

        try (CanonicalTrainer fullBatchTrainer = CanonicalTrainer.builder()
                .model(fullBatchModel)
                .optimizer(SGD.builder(fullBatchModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .build()) {
            fullBatchTrainer.fit(fullBatch, null);
            assertEquals(1, ((Number) fullBatchTrainer.summary().metadata().get("optimizerStepCount")).intValue());
        }

        assertParametersClose(fullBatchModel, accumulatedModel, 1e-5f);
    }

    @Test
    void canonicalTrainerGradientAccumulationStepsSchedulerOnlyOnOptimizerStep() {
        Linear model = new Linear(1, 1);
        SGD optimizer = SGD.builder(model.parameters(), 0.1f).build();
        StepLR scheduler = new StepLR(optimizer, 1, 0.5f);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(
                batch(new float[] {1f}, new float[] {2f}, 1),
                batch(new float[] {2f}, new float[] {4f}, 1),
                batch(new float[] {3f}, new float[] {6f}, 1),
                batch(new float[] {4f}, new float[] {8f}, 1));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .scheduler(scheduler)
                .loss(mseLoss::compute)
                .epochs(1)
                .gradientAccumulationSteps(2)
                .build()) {
            trainer.fit(train, null);
            TrainingSummary summary = trainer.summary();
            assertEquals(2, ((Number) summary.metadata().get("gradientAccumulationSteps")).intValue());
            assertEquals(2, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
            assertEquals(2, ((Number) summary.metadata().get("learningRateSchedulerStepCount")).intValue());
            assertEquals(0.05f, optimizer.learningRate(), 1e-6f);
        }
    }

    @Test
    void canonicalTrainerMixedPrecisionUsesGradScalerForEquivalentUpdate() {
        Linear seed = new Linear(1, 1);
        Linear fullPrecisionModel = new Linear(1, 1);
        Linear mixedPrecisionModel = new Linear(1, 1);
        copyParameters(seed, fullPrecisionModel);
        copyParameters(seed, mixedPrecisionModel);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        try (CanonicalTrainer fullPrecision = CanonicalTrainer.builder()
                .model(fullPrecisionModel)
                .optimizer(SGD.builder(fullPrecisionModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .build()) {
            fullPrecision.fit(train, null);
        }

        try (CanonicalTrainer mixedPrecision = CanonicalTrainer.builder()
                .model(mixedPrecisionModel)
                .optimizer(SGD.builder(mixedPrecisionModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .gradScaler(GradScaler.builder().initScale(8.0).growthInterval(1).build())
                .build()) {
            mixedPrecision.fit(train, null);
            TrainingSummary summary = mixedPrecision.summary();
            assertEquals(Boolean.TRUE, summary.metadata().get("mixedPrecisionEnabled"));
            assertEquals(Boolean.FALSE, summary.metadata().get("mixedPrecisionOverflowDetected"));
            assertEquals(0, ((Number) summary.metadata().get("mixedPrecisionOverflowSkipCount")).intValue());
            assertEquals(16.0, ((Number) summary.metadata().get("mixedPrecisionLossScale")).doubleValue(), 1e-9);
            assertEquals(1, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
        }

        assertParametersClose(fullPrecisionModel, mixedPrecisionModel, 1e-5f);
    }

    @Test
    void canonicalTrainerMixedPrecisionSkipsOverflowStepAndBacksOffScale() {
        Linear model = new Linear(1, 1, false);
        model.parameters().get(0).data().data()[0] = 1f;
        SGD optimizer = SGD.builder(model.parameters(), 0.1f).build();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {0f}, 1));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(model)
                .optimizer(optimizer)
                .loss((prediction, labels) -> prediction.mul(Float.MAX_VALUE / 2.0f).mean())
                .epochs(1)
                .gradScaler(GradScaler.builder().initScale(4.0).backoffFactor(0.5).build())
                .build()) {
            trainer.fit(train, null);
            TrainingSummary summary = trainer.summary();
            assertEquals(Boolean.TRUE, summary.metadata().get("mixedPrecisionEnabled"));
            assertEquals(Boolean.TRUE, summary.metadata().get("mixedPrecisionOverflowDetected"));
            assertEquals(1, ((Number) summary.metadata().get("mixedPrecisionOverflowSkipCount")).intValue());
            assertEquals(2.0, ((Number) summary.metadata().get("mixedPrecisionLossScale")).doubleValue(), 1e-9);
            assertEquals(0, ((Number) summary.metadata().get("optimizerStepCount")).intValue());
            assertEquals(1f, model.parameters().get(0).data().data()[0], 1e-6f);
        }
    }

    @Test
    void canonicalTrainerMixedPrecisionCheckpointsGradScalerState() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-grad-scaler");
        Linear seed = new Linear(1, 1);
        Linear interruptedModel = new Linear(1, 1);
        Linear baselineModel = new Linear(1, 1);
        copyParameters(seed, interruptedModel);
        copyParameters(seed, baselineModel);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        TrainingListener stopAfterFirstEpoch = new TrainingListener() {
            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                if (epoch == 0) {
                    session.stop();
                }
            }
        };

        try (CanonicalTrainer interrupted = CanonicalTrainer.builder()
                .model(interruptedModel)
                .optimizer(SGD.builder(interruptedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .gradScaler(GradScaler.builder().initScale(4.0).growthInterval(2).build())
                .listener(stopAfterFirstEpoch)
                .build()) {
            interrupted.fit(train, null);
            TrainingSummary summary = interrupted.summary();
            assertEquals(Boolean.TRUE, summary.metadata().get("gradScalerCheckpointSupported"));
            assertEquals(Boolean.TRUE, summary.metadata().get("gradScalerCheckpointSaved"));
            assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-grad-scaler.state")));
            assertEquals(4.0, ((Number) summary.metadata().get("mixedPrecisionLossScale")).doubleValue(), 1e-9);
        }

        try (CanonicalTrainer baseline = CanonicalTrainer.builder()
                .model(baselineModel)
                .optimizer(SGD.builder(baselineModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .gradScaler(GradScaler.builder().initScale(4.0).growthInterval(2).build())
                .build()) {
            baseline.fit(train, null);
            assertEquals(
                    8.0,
                    ((Number) baseline.summary().metadata().get("mixedPrecisionLossScale")).doubleValue(),
                    1e-9);
        }

        Linear resumedModel = new Linear(1, 1);
        copyParameters(seed, resumedModel);
        try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                .model(resumedModel)
                .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .gradScaler(GradScaler.builder().initScale(4.0).growthInterval(2).build())
                .build()) {
            resumed.fit(train, null);
            TrainingSummary summary = resumed.summary();
            assertEquals(Boolean.TRUE, summary.metadata().get("gradScalerCheckpointLoaded"));
            assertEquals(Boolean.FALSE, summary.metadata().get("gradScalerCheckpointLoadFailed"));
            assertEquals(Boolean.FALSE, summary.metadata().get("gradScalerCheckpointFallbackUsed"));
            assertEquals(8.0, ((Number) summary.metadata().get("mixedPrecisionLossScale")).doubleValue(), 1e-9);
        }

        assertParametersClose(baselineModel, resumedModel, 1e-5f);
    }

    @Test
    void canonicalTrainerRejectsIncompatibleGradScalerCheckpointByDefault() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-grad-scaler-mismatch");
        Linear seed = new Linear(1, 1);
        Linear interruptedModel = new Linear(1, 1);
        copyParameters(seed, interruptedModel);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        TrainingListener stopAfterFirstEpoch = new TrainingListener() {
            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                if (epoch == 0) {
                    session.stop();
                }
            }
        };

        try (CanonicalTrainer interrupted = CanonicalTrainer.builder()
                .model(interruptedModel)
                .optimizer(SGD.builder(interruptedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .gradScaler(GradScaler.builder().initScale(4.0).growthInterval(2).build())
                .listener(stopAfterFirstEpoch)
                .build()) {
            interrupted.fit(train, null);
            assertTrue(Files.isRegularFile(checkpointDir.resolve("canonical-grad-scaler.state")));
        }

        Linear resumedModel = new Linear(1, 1);
        copyParameters(seed, resumedModel);
        IllegalStateException error = assertThrows(IllegalStateException.class, () -> {
            try (CanonicalTrainer resumed = CanonicalTrainer.builder()
                    .model(resumedModel)
                    .optimizer(SGD.builder(resumedModel.parameters(), 0.01f).build())
                    .loss(mseLoss::compute)
                    .epochs(2)
                    .checkpointDir(checkpointDir)
                    .resumeFromCheckpoint()
                    .gradScaler(GradScaler.builder().initScale(4.0).growthInterval(3).build())
                    .build()) {
                resumed.fit(train, null);
            }
        });
        assertTrue(error.getMessage().contains("GradScaler checkpoint"));
        assertNotNull(error.getCause());
        assertTrue(error.getCause().getMessage().contains("growthInterval"));
    }

    @Test
    void canonicalTrainerCanContinueLenientlyAfterIncompatibleGradScalerCheckpoint() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-grad-scaler-lenient");
        Linear seed = new Linear(1, 1);
        Linear interruptedModel = new Linear(1, 1);
        copyParameters(seed, interruptedModel);
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2));

        TrainingListener stopAfterFirstEpoch = new TrainingListener() {
            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                if (epoch == 0) {
                    session.stop();
                }
            }
        };

        try (CanonicalTrainer interrupted = CanonicalTrainer.builder()
                .model(interruptedModel)
                .optimizer(SGD.builder(interruptedModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .gradScaler(GradScaler.builder().initScale(4.0).growthInterval(2).build())
                .listener(stopAfterFirstEpoch)
                .build()) {
            interrupted.fit(train, null);
        }

        Linear lenientModel = new Linear(1, 1);
        copyParameters(seed, lenientModel);
        try (CanonicalTrainer lenient = CanonicalTrainer.builder()
                .model(lenientModel)
                .optimizer(SGD.builder(lenientModel.parameters(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(2)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .gradScaler(GradScaler.builder().initScale(4.0).growthInterval(3).build())
                .build()) {
            lenient.fit(train, null);
            TrainingSummary summary = lenient.summary();
            assertEquals(Boolean.FALSE, summary.metadata().get("gradScalerCheckpointLoaded"));
            assertEquals(Boolean.TRUE, summary.metadata().get("gradScalerCheckpointLoadFailed"));
            assertEquals(Boolean.TRUE, summary.metadata().get("gradScalerCheckpointFallbackUsed"));
            assertTrue(String.valueOf(summary.metadata().get("gradScalerCheckpointLoadError"))
                    .contains("growthInterval"));
            assertEquals(0, ((Number) summary.metadata().get("mixedPrecisionOverflowSkipCount")).intValue());
        }
    }

    @Test
    void canonicalTrainerReportsRegressionMetricsForTrainAndValidation() {
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(
                batch(new float[] {1f, 3f}, new float[] {2f, 1f}, 2),
                batch(new float[] {2f}, new float[] {5f}, 1));
        List<Batch> validation = List.of(batch(new float[] {0f, 10f}, new float[] {1f, 7f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .metric(TrainingMetrics.meanAbsoluteError())
                .metric(TrainingMetrics.meanSquaredError())
                .build()) {
            trainer.fit(train, validation);

            TrainingSummary summary = trainer.summary();
            assertEquals(Boolean.TRUE, summary.metadata().get("metricsEnabled"));
            assertEquals(2.0, metric(summary, "trainMetric.mae"), 1e-6);
            assertEquals(14.0 / 3.0, metric(summary, "trainMetric.mse"), 1e-6);
            assertEquals(2.0, metric(summary, "validationMetric.mae"), 1e-6);
            assertEquals(5.0, metric(summary, "validationMetric.mse"), 1e-6);
            assertEquals(2.0, metricMap(summary, "latestTrainMetrics").get("mae"), 1e-6);
            assertEquals(5.0, metricMap(summary, "latestValidationMetrics").get("mse"), 1e-6);
        }
    }

    @Test
    void canonicalTrainerAcceptsTopLevelMetricContracts() {
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f, 3f}, new float[] {2f, 1f}, 2));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .metric(TrainingMetrics.meanAbsoluteError())
                .metric(() -> new ConstantMetric("constant_metric", 1.0))
                .metric(TrainingMetrics.custom("mean_bias", BiasMetricState::new)
                        .reset(BiasMetricState::reset)
                        .update(BiasMetricState::update)
                        .value(BiasMetricState::meanBias)
                        .details(state -> Map.of(
                                "count", state.count,
                                "sumBias", state.sumBias))
                        .build())
                .build()) {
            trainer.fit(train, null);

            TrainingSummary summary = trainer.summary();
            assertEquals(1.5, metric(summary, "trainMetric.mae"), 1e-6);
            assertEquals(1.0, metric(summary, "trainMetric.constant_metric"), 1e-6);
            assertEquals(0.5, metric(summary, "trainMetric.mean_bias"), 1e-6);
            Map<String, Object> details = metricDetails(summary, "latestTrainMetricDetails", "mean_bias");
            assertEquals(2L, details.get("count"));
            assertEquals(1.0, ((Number) details.get("sumBias")).doubleValue(), 1e-6);
            assertEquals(details, summary.metadata().get("trainMetricDetails.mean_bias"));
        }
    }

    @Test
    @SuppressWarnings("deprecation")
    void canonicalTrainerKeepsLegacyNestedMetricAliasCompatible() {
        MSELoss mseLoss = new MSELoss();
        List<Batch> train = List.of(batch(new float[] {1f}, new float[] {2f}, 1));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(mseLoss::compute)
                .epochs(1)
                .metric(LegacyNestedMetric::new)
                .build()) {
            trainer.fit(train, null);

            assertEquals(2.0, metric(trainer.summary(), "trainMetric.legacy_metric"), 1e-6);
        }
    }

    @Test
    @SuppressWarnings("deprecation")
    void canonicalTrainerKeepsLegacyDetailedMetricAliasCompatible() {
        CanonicalTrainer.Metric metric = CanonicalTrainer.Metrics.binaryConfusionMatrix().get();
        assertTrue(metric instanceof CanonicalTrainer.DetailedMetric);

        metric.reset();
        metric.update(
                GradTensor.of(new float[] {2f, -1f}, 2),
                GradTensor.of(new float[] {1f, 0f}, 2));

        CanonicalTrainer.DetailedMetric detailedMetric = (CanonicalTrainer.DetailedMetric) metric;
        assertEquals(1.0, metric.value(), 1e-6);
        assertEquals("binary_confusion_matrix", detailedMetric.details().get("type"));
    }

    @Test
    void canonicalTrainerReportsClassificationAccuracyMetric() {
        CrossEntropyLoss crossEntropy = new CrossEntropyLoss();
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {
                        3f, 1f, 0f,
                        0f, 1f, 4f,
                        0f, 5f, 1f
                }, 3, 3),
                GradTensor.of(new float[] {0f, 1f, 1f}, 3)));
        List<Batch> validation = List.of(new Batch(
                GradTensor.of(new float[] {
                        0f, 5f, 1f,
                        4f, 1f, 0f
                }, 2, 3),
                GradTensor.of(new float[] {1f, 1f}, 2)));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(crossEntropy::compute)
                .epochs(1)
                .metric(Gollek.DL.accuracyMetric())
                .metric(Gollek.DL.topKAccuracyMetric(2))
                .metric(Gollek.DL.precisionMetric())
                .metric(Gollek.DL.recallMetric())
                .metric(Gollek.DL.f1Metric())
                .build()) {
            trainer.fit(train, validation);

            TrainingSummary summary = trainer.summary();
            assertEquals(2.0 / 3.0, metric(summary, "trainMetric.accuracy"), 1e-6);
            assertEquals(1.0, metric(summary, "trainMetric.top2_accuracy"), 1e-6);
            assertEquals(2.0 / 3.0, metric(summary, "trainMetric.precision"), 1e-6);
            assertEquals(0.5, metric(summary, "trainMetric.recall"), 1e-6);
            assertEquals(5.0 / 9.0, metric(summary, "trainMetric.f1"), 1e-6);
            assertEquals(0.5, metric(summary, "validationMetric.accuracy"), 1e-6);
            assertEquals(1.0, metric(summary, "validationMetric.top2_accuracy"), 1e-6);
            assertEquals(1.0 / 3.0, metric(summary, "validationMetric.precision"), 1e-6);
            assertEquals(1.0 / 6.0, metric(summary, "validationMetric.recall"), 1e-6);
            assertEquals(2.0 / 9.0, metric(summary, "validationMetric.f1"), 1e-6);
            assertEquals(2.0 / 3.0, metricMap(summary, "latestTrainMetrics").get("accuracy"), 1e-6);
            assertEquals(5.0 / 9.0, metricMap(summary, "latestTrainMetrics").get("f1"), 1e-6);
        }
    }

    @Test
    void canonicalTrainerRecordsConfusionMatrixMetricDetails() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-typed-trainer-confusion-matrix");
        CrossEntropyLoss crossEntropy = new CrossEntropyLoss();
        List<Batch> loader = List.of(new Batch(
                GradTensor.of(new float[] {
                        5f, 1f, 0f,
                        0f, 4f, 1f,
                        0f, 1f, 4f,
                        3f, 2f, 1f
                }, 4, 3),
                Gollek.DL.classLabels(0, 2, 2, 1)));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(crossEntropy::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .metric(Gollek.DL.confusionMatrixMetric())
                .build()) {
            trainer.fit(loader, loader);

            TrainingSummary summary = trainer.summary();
            assertEquals(0.5, metric(summary, "trainMetric.confusion_matrix_accuracy"), 1e-6);
            assertEquals(0.5, metric(summary, "validationMetric.confusion_matrix_accuracy"), 1e-6);

            Map<String, Object> trainDetails = metricDetails(
                    summary,
                    "latestTrainMetricDetails",
                    "confusion_matrix_accuracy");
            assertEquals("classification_confusion_matrix", trainDetails.get("type"));
            assertEquals(3, ((Number) trainDetails.get("classes")).intValue());
            assertEquals(4L, ((Number) trainDetails.get("total")).longValue());
            assertEquals(2L, ((Number) trainDetails.get("correct")).longValue());
            assertEquals(0.5, ((Number) trainDetails.get("accuracy")).doubleValue(), 1e-6);
            assertEquals("actual_class", trainDetails.get("rowMeaning"));
            assertEquals("predicted_class", trainDetails.get("columnMeaning"));
            assertEquals(List.of(0, 1, 2), trainDetails.get("labels"));
            assertEquals(List.of(
                    List.of(1L, 0L, 0L),
                    List.of(1L, 0L, 0L),
                    List.of(0L, 1L, 1L)), trainDetails.get("matrix"));
            assertEquals(List.of(0.5, 0.0, 1.0), trainDetails.get("perClassPrecision"));
            assertEquals(List.of(1.0, 0.0, 0.5), trainDetails.get("perClassRecall"));

            Map<String, Object> validationDetails = metricDetails(
                    summary,
                    "latestValidationMetricDetails",
                    "confusion_matrix_accuracy");
            assertEquals(trainDetails.get("matrix"), validationDetails.get("matrix"));
            assertEquals(trainDetails, summary.metadata().get("trainMetricDetails.confusion_matrix_accuracy"));
            assertEquals(validationDetails, summary.metadata().get("validationMetricDetails.confusion_matrix_accuracy"));

            Map<String, Object> firstEpoch = epochHistory(summary).get(0);
            Map<String, Object> epochTrainDetails = metricDetails(
                    firstEpoch,
                    "trainMetricDetails",
                    "confusion_matrix_accuracy");
            assertEquals(trainDetails.get("matrix"), epochTrainDetails.get("matrix"));
        }

        String reportJson = Files.readString(checkpointDir.resolve("canonical-report.json"));
        assertTrue(reportJson.contains("\"confusion_matrix_accuracy\""));
        assertTrue(reportJson.contains("\"matrix\":[[1,0,0],[1,0,0],[0,1,1]]"));

        var evaluation = Gollek.DL.evaluate(
                new IdentityModel(),
                loader,
                crossEntropy::compute,
                "cpu",
                Gollek.DL.confusionMatrixMetric());
        assertEquals(0.5, evaluation.metric("confusion_matrix_accuracy"), 1e-6);
        assertEquals(List.of(
                List.of(1L, 0L, 0L),
                List.of(1L, 0L, 0L),
                List.of(0L, 1L, 1L)),
                metricDetails(evaluation.metricDetails(), "confusion_matrix_accuracy").get("matrix"));
    }

    @Test
    void canonicalTrainerReportsBinaryClassificationMetrics() {
        var bce = Gollek.DL.bceWithLogitsLoss();
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {2f, -1f, 0.5f, -2f}, 4, 1),
                Gollek.DL.binaryLabels(1, 0, 0, 1)));
        List<Batch> validation = List.of(new Batch(
                GradTensor.of(new float[] {-2f, 2f}, 2, 1),
                Gollek.DL.binaryLabels(0, 1)));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(bce::compute)
                .epochs(1)
                .metric(Gollek.DL.binaryAccuracyMetric())
                .metric(Gollek.DL.binaryPrecisionMetric())
                .metric(Gollek.DL.binaryRecallMetric())
                .metric(Gollek.DL.binaryF1Metric())
                .metric(Gollek.DL.binaryRocAucMetric())
                .metric(Gollek.DL.binaryAveragePrecisionMetric())
                .build()) {
            trainer.fit(train, validation);

            TrainingSummary summary = trainer.summary();
            assertEquals(0.5, metric(summary, "trainMetric.binary_accuracy"), 1e-6);
            assertEquals(0.5, metric(summary, "trainMetric.binary_precision"), 1e-6);
            assertEquals(0.5, metric(summary, "trainMetric.binary_recall"), 1e-6);
            assertEquals(0.5, metric(summary, "trainMetric.binary_f1"), 1e-6);
            assertEquals(0.5, metric(summary, "trainMetric.binary_roc_auc"), 1e-6);
            assertEquals(0.75, metric(summary, "trainMetric.binary_average_precision"), 1e-6);
            assertEquals(1.0, metric(summary, "validationMetric.binary_accuracy"), 1e-6);
            assertEquals(1.0, metric(summary, "validationMetric.binary_precision"), 1e-6);
            assertEquals(1.0, metric(summary, "validationMetric.binary_recall"), 1e-6);
            assertEquals(1.0, metric(summary, "validationMetric.binary_f1"), 1e-6);
            assertEquals(1.0, metric(summary, "validationMetric.binary_roc_auc"), 1e-6);
            assertEquals(1.0, metric(summary, "validationMetric.binary_average_precision"), 1e-6);
        }
    }

    @Test
    void canonicalTrainerRecordsBinaryConfusionMatrixMetricDetails() throws Exception {
        var bce = Gollek.DL.bceWithLogitsLoss();
        Path checkpointDir = Files.createTempDirectory("gollek-binary-confusion");
        List<Batch> train = List.of(new Batch(
                GradTensor.of(new float[] {2f, -1f, 0.5f, -2f}, 4, 1),
                Gollek.DL.binaryLabels(1, 0, 0, 1)));
        List<Batch> validation = List.of(new Batch(
                GradTensor.of(new float[] {-2f, 2f}, 2, 1),
                Gollek.DL.binaryLabels(0, 1)));

        try (CanonicalTrainer trainer = CanonicalTrainer.builder()
                .model(new IdentityModel())
                .optimizer(SGD.builder(List.of(), 0.01f).build())
                .loss(bce::compute)
                .epochs(1)
                .checkpointDir(checkpointDir)
                .metric(Gollek.DL.binaryConfusionMatrixMetric())
                .build()) {
            trainer.fit(train, validation);

            TrainingSummary summary = trainer.summary();
            assertEquals(0.5, metric(summary, "trainMetric.binary_confusion_matrix_accuracy"), 1e-6);
            assertEquals(1.0, metric(summary, "validationMetric.binary_confusion_matrix_accuracy"), 1e-6);

            Map<String, Object> trainDetails = metricDetails(
                    summary,
                    "latestTrainMetricDetails",
                    "binary_confusion_matrix_accuracy");
            assertEquals("binary_confusion_matrix", trainDetails.get("type"));
            assertEquals(0.0, ((Number) trainDetails.get("threshold")).doubleValue(), 1e-6);
            assertEquals(4L, ((Number) trainDetails.get("total")).longValue());
            assertEquals(1L, ((Number) trainDetails.get("trueNegative")).longValue());
            assertEquals(1L, ((Number) trainDetails.get("falsePositive")).longValue());
            assertEquals(1L, ((Number) trainDetails.get("falseNegative")).longValue());
            assertEquals(1L, ((Number) trainDetails.get("truePositive")).longValue());
            assertEquals(0.5, ((Number) trainDetails.get("accuracy")).doubleValue(), 1e-6);
            assertEquals(0.5, ((Number) trainDetails.get("precision")).doubleValue(), 1e-6);
            assertEquals(0.5, ((Number) trainDetails.get("recall")).doubleValue(), 1e-6);
            assertEquals(0.5, ((Number) trainDetails.get("f1")).doubleValue(), 1e-6);
            assertEquals(0.5, ((Number) trainDetails.get("specificity")).doubleValue(), 1e-6);
            assertEquals(0.5, ((Number) trainDetails.get("balancedAccuracy")).doubleValue(), 1e-6);
            assertEquals("actual_label", trainDetails.get("rowMeaning"));
            assertEquals("predicted_label", trainDetails.get("columnMeaning"));
            assertEquals(List.of(0, 1), trainDetails.get("labels"));
            assertEquals(List.of(
                    List.of(1L, 1L),
                    List.of(1L, 1L)), trainDetails.get("matrix"));

            Map<String, Object> validationDetails = metricDetails(
                    summary,
                    "latestValidationMetricDetails",
                    "binary_confusion_matrix_accuracy");
            assertEquals(List.of(
                    List.of(1L, 0L),
                    List.of(0L, 1L)), validationDetails.get("matrix"));
            assertEquals(trainDetails, summary.metadata().get("trainMetricDetails.binary_confusion_matrix_accuracy"));
            assertEquals(
                    validationDetails,
                    summary.metadata().get("validationMetricDetails.binary_confusion_matrix_accuracy"));

            Map<String, Object> firstEpoch = epochHistory(summary).get(0);
            Map<String, Object> epochTrainDetails = metricDetails(
                    firstEpoch,
                    "trainMetricDetails",
                    "binary_confusion_matrix_accuracy");
            assertEquals(trainDetails.get("matrix"), epochTrainDetails.get("matrix"));
        }

        String reportJson = Files.readString(checkpointDir.resolve("canonical-report.json"));
        assertTrue(reportJson.contains("\"binary_confusion_matrix_accuracy\""));
        assertTrue(reportJson.contains("\"matrix\":[[1,1],[1,1]]"));

        var evaluation = Gollek.DL.evaluate(
                new IdentityModel(),
                train,
                bce::compute,
                "cpu",
                Gollek.DL.binaryConfusionMatrixMetric());
        assertEquals(0.5, evaluation.metric("binary_confusion_matrix_accuracy"), 1e-6);
        assertEquals(List.of(
                List.of(1L, 1L),
                List.of(1L, 1L)),
                metricDetails(evaluation.metricDetails(), "binary_confusion_matrix_accuracy").get("matrix"));
    }

    private static void assertLabelCounts(DataLoader.TensorDataset dataset, int expectedZeros, int expectedOnes) {
        List<Float> labels = new java.util.ArrayList<>();
        for (int i = 0; i < dataset.size(); i++) {
            for (float value : dataset.get(i)[1].data()) {
                labels.add(value);
            }
        }
        assertEquals(expectedZeros, Collections.frequency(labels, 0f));
        assertEquals(expectedOnes, Collections.frequency(labels, 1f));
    }

    private static void assertPositiveCounts(DataLoader.TensorDataset dataset, int[] expected) {
        int[] counts = new int[expected.length];
        for (int i = 0; i < dataset.size(); i++) {
            float[] labels = dataset.get(i)[1].data();
            assertEquals(expected.length, labels.length);
            for (int column = 0; column < expected.length; column++) {
                if (labels[column] >= 0.5f) {
                    counts[column]++;
                }
            }
        }
        assertArrayEquals(expected, counts);
    }

    private static void copyParameters(Linear source, Linear target) {
        List<Parameter> sourceParams = source.parameters();
        List<Parameter> targetParams = target.parameters();
        assertEquals(sourceParams.size(), targetParams.size());
        for (int i = 0; i < sourceParams.size(); i++) {
            float[] sourceData = sourceParams.get(i).data().data();
            float[] targetData = targetParams.get(i).data().data();
            assertEquals(sourceData.length, targetData.length);
            System.arraycopy(sourceData, 0, targetData, 0, sourceData.length);
        }
    }

    private static void assertParametersClose(Linear expected, Linear actual, float tolerance) {
        List<Parameter> expectedParams = expected.parameters();
        List<Parameter> actualParams = actual.parameters();
        assertEquals(expectedParams.size(), actualParams.size());
        for (int i = 0; i < expectedParams.size(); i++) {
            assertArrayEquals(
                    expectedParams.get(i).data().data(),
                    actualParams.get(i).data().data(),
                    tolerance);
        }
    }

    private static Batch batch(float[] inputs, float[] targets, int rows) {
        return new Batch(
                GradTensor.of(inputs, rows, 1),
                GradTensor.of(targets, rows, 1));
    }

    private static List<Float> flattenInputs(Iterable<Batch> batches) {
        List<Float> values = new ArrayList<>();
        for (Batch batch : batches) {
            for (float value : batch.inputs().data()) {
                values.add(value);
            }
        }
        return values;
    }

    private static float logit(double probability) {
        return (float) Math.log(probability / (1.0 - probability));
    }

    private static float logProb(double probability) {
        return (float) Math.log(probability);
    }

    private static void writeCorruptCheckpoint(Path checkpointFile) {
        try {
            Files.write(checkpointFile, new byte[] {9, 8, 7, 6});
        } catch (Exception error) {
            throw new RuntimeException(error);
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Double> metricMap(TrainingSummary summary, String metadataKey) {
        Object value = summary.metadata().get(metadataKey);
        assertTrue(value instanceof Map<?, ?>);
        return (Map<String, Double>) value;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Double> metricMap(Map<String, Object> metadata, String metadataKey) {
        Object value = metadata.get(metadataKey);
        assertTrue(value instanceof Map<?, ?>);
        return (Map<String, Double>) value;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> metricDetails(
            TrainingSummary summary,
            String metadataKey,
            String metricName) {
        Object value = summary.metadata().get(metadataKey);
        assertTrue(value instanceof Map<?, ?>);
        return metricDetails((Map<String, Object>) value, metricName);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> metricDetails(
            Map<String, Object> metadata,
            String metadataKey,
            String metricName) {
        Object value = metadata.get(metadataKey);
        assertTrue(value instanceof Map<?, ?>);
        return metricDetails((Map<String, Object>) value, metricName);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> metricDetails(Map<String, Object> detailMap, String metricName) {
        Object value = detailMap.get(metricName);
        assertTrue(value instanceof Map<?, ?>);
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> epochHistory(TrainingSummary summary) {
        Object value = summary.metadata().get("epochHistory");
        assertTrue(value instanceof List<?>);
        return (List<Map<String, Object>>) value;
    }

    private static double metric(TrainingSummary summary, String metadataKey) {
        Object value = summary.metadata().get(metadataKey);
        assertTrue(value instanceof Number, "expected numeric metric for " + metadataKey);
        return ((Number) value).doubleValue();
    }

    private static double trainMsle() {
        return (square(Math.log1p(1.0) - Math.log1p(2.0))
                + square(Math.log1p(3.0) - Math.log1p(1.0))
                + square(Math.log1p(2.0) - Math.log1p(5.0))) / 3.0;
    }

    private static double validationMsle() {
        return (square(Math.log1p(0.0) - Math.log1p(1.0))
                + square(Math.log1p(10.0) - Math.log1p(7.0))) / 2.0;
    }

    private static double square(double value) {
        return value * value;
    }

    private static final class IdentityModel extends NNModule {
        @Override
        public GradTensor forward(GradTensor input) {
            return input;
        }
    }

    private static final class NonFinitePredictionModel extends NNModule {
        private final float value;

        NonFinitePredictionModel(float value) {
            this.value = value;
        }

        @Override
        public GradTensor forward(GradTensor input) {
            return GradTensor.full(value, input.shape());
        }
    }

    private static final class SentinelNonFinitePredictionModel extends NNModule {
        private final float sentinel;
        private final float value;

        SentinelNonFinitePredictionModel(float sentinel, float value) {
            this.sentinel = sentinel;
            this.value = value;
        }

        @Override
        public GradTensor forward(GradTensor input) {
            return input.item() == sentinel ? GradTensor.full(value, input.shape()) : input;
        }
    }

    private static final class ScoredIdentityModel extends NNModule {
        ScoredIdentityModel() {
            registerParameter("dummy", GradTensor.of(new float[] {1f}, 1));
        }

        @Override
        public GradTensor forward(GradTensor input) {
            return input;
        }
    }

    private static final class MeanPredictionMetric implements TrainingMetric {
        private double sum;
        private long count;

        @Override
        public String name() {
            return "mean_prediction";
        }

        @Override
        public void reset() {
            sum = 0.0;
            count = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            for (float value : predictions.data()) {
                sum += value;
                count++;
            }
        }

        @Override
        public double value() {
            return count == 0 ? Double.NaN : sum / count;
        }
    }

    private static final class ConstantMetric implements TrainingMetric {
        private final String name;
        private final double value;

        ConstantMetric(String name, double value) {
            this.name = name;
            this.value = value;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }

        @Override
        public double value() {
            return value;
        }
    }

    private static final class BiasMetricState {
        private double sumBias;
        private long count;

        private void reset() {
            sumBias = 0.0;
            count = 0L;
        }

        private void update(GradTensor predictions, GradTensor targets) {
            float[] predictionValues = predictions.data();
            float[] targetValues = targets.data();
            for (int i = 0; i < predictionValues.length; i++) {
                sumBias += predictionValues[i] - targetValues[i];
                count++;
            }
        }

        private double meanBias() {
            return count == 0L ? 0.0 : sumBias / count;
        }
    }

    @SuppressWarnings("deprecation")
    private static final class LegacyNestedMetric implements CanonicalTrainer.Metric {
        @Override
        public String name() {
            return "legacy_metric";
        }

        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }

        @Override
        public double value() {
            return 2.0;
        }
    }

    private static final class MatrixDetailMetric implements DetailedTrainingMetric {
        @Override
        public String name() {
            return "matrix_detail";
        }

        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }

        @Override
        public double value() {
            return 1.0;
        }

        @Override
        public Map<String, Object> details() {
            return Map.of(
                    "label", "matrix-detail",
                    "matrix", List.of(List.of(1, 2), List.of(3, 4)));
        }
    }

    private static final class DynamicNameMetric implements TrainingMetric {
        private final String initialName;
        private final String updatedName;
        private boolean updated;

        DynamicNameMetric(String initialName, String updatedName) {
            this.initialName = initialName;
            this.updatedName = updatedName;
        }

        @Override
        public String name() {
            return updated ? updatedName : initialName;
        }

        @Override
        public void reset() {
            updated = false;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            updated = true;
        }

        @Override
        public double value() {
            return 1.0;
        }
    }

    private static final class ThrowingMetric implements TrainingMetric {
        @Override
        public String name() {
            return "throwing_metric";
        }

        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }

        @Override
        public double value() {
            throw new IllegalStateException("boom");
        }
    }

    private static final class NonFiniteDetailMetric implements DetailedTrainingMetric {
        @Override
        public String name() {
            return "detail_metric";
        }

        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }

        @Override
        public double value() {
            return 1.0;
        }

        @Override
        public Map<String, Object> details() {
            return Map.of("bad", Double.NaN);
        }
    }

    private static final class ThrowingDetailMetric implements DetailedTrainingMetric {
        @Override
        public String name() {
            return "throwing_detail_metric";
        }

        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }

        @Override
        public double value() {
            return 1.0;
        }

        @Override
        public Map<String, Object> details() {
            throw new IllegalStateException("boom");
        }
    }

    private static final class SentinelNonFiniteMetric implements TrainingMetric {
        private final float sentinel;
        private boolean seenSentinel;
        private long count;

        SentinelNonFiniteMetric(float sentinel) {
            this.sentinel = sentinel;
        }

        @Override
        public String name() {
            return "sentinel_metric";
        }

        @Override
        public void reset() {
            seenSentinel = false;
            count = 0L;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            for (float value : predictions.data()) {
                if (value == sentinel) {
                    seenSentinel = true;
                }
                count++;
            }
        }

        @Override
        public double value() {
            if (seenSentinel) {
                return Double.NaN;
            }
            return count == 0L ? 0.0 : 1.0;
        }
    }

    private static final class EpochBatches implements Iterable<Batch> {
        private final List<List<Batch>> epochs;
        private int cursor;

        EpochBatches(List<List<Batch>> epochs) {
            this.epochs = List.copyOf(epochs);
        }

        @Override
        public java.util.Iterator<Batch> iterator() {
            List<Batch> batches = epochs.get(Math.min(cursor, epochs.size() - 1));
            cursor++;
            return batches.iterator();
        }
    }
}
