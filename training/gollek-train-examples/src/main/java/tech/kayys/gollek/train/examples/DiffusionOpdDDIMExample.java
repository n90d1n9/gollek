package tech.kayys.gollek.train.examples;

import java.nio.file.Path;
import java.util.List;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.diffusion.model.UNetModel;
import tech.kayys.gollek.diffusion.scheduler.DDIMScheduler;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdListener;
import tech.kayys.gollek.train.diffusion.api.DiffusionPromptSample;
import tech.kayys.gollek.train.diffusion.api.DiffusionSamplerType;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;
import tech.kayys.gollek.train.diffusion.api.DiffusionTeacherBindings;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSession;
import tech.kayys.gollek.trainer.api.TrainingSummary;
import tech.kayys.gollek.train.diffusion.opd.adapter.RunnerDiffusionAdapters;

/**
 * Minimal Java-first DiffusionOPD example using the Gollek diffusion runner
 * adapters and a DDIM-style scheduler.
 *
 * <p>Reference:
 * Quanhao Li et al., "DiffusionOPD: A Unified Perspective of On-Policy
 * Distillation in Diffusion Models", arXiv:2605.15055, 2026.
 */
public final class DiffusionOpdDDIMExample {

    private DiffusionOpdDDIMExample() {
    }

    public static void main(String[] args) {
        UNetModel studentUnet = new StubUNetModel(0.10f);
        UNetModel ocrEarlyTeacher = new StubUNetModel(0.07f);
        UNetModel ocrLateTeacher = new StubUNetModel(0.09f);
        UNetModel aestheticsEarlyTeacher = new StubUNetModel(0.11f);
        UNetModel aestheticsLateTeacher = new StubUNetModel(0.13f);

        float[] alphasCumprod = new float[] {
                0.9990f, 0.9950f, 0.9880f, 0.9760f, 0.9600f, 0.9400f, 0.9150f, 0.8850f
        };
        DDIMScheduler ddimScheduler = new DDIMScheduler(alphasCumprod, 4, null);
        var diffusionScheduler = RunnerDiffusionAdapters.scheduler(ddimScheduler);

        List<DiffusionPromptSample> ocrPrompts = List.of(
                new DiffusionPromptSample("street sign that clearly reads GOLLEK", "", 7L, null),
                new DiffusionPromptSample("receipt with bold total amount and crisp text", "", 11L, null));
        List<DiffusionPromptSample> aestheticsPrompts = List.of(
                new DiffusionPromptSample("cinematic portrait with dramatic rim lighting", "", 13L, null),
                new DiffusionPromptSample("editorial product shot on textured marble", "", 17L, null));

        var trainer = Gollek.DL.diffusionOpdTrainer()
                .samplerType(DiffusionSamplerType.ODE)
                .student(RunnerDiffusionAdapters.denoiser(studentUnet))
                .teacher("ocr-early", RunnerDiffusionAdapters.denoiser(ocrEarlyTeacher))
                .teacher("ocr-late", RunnerDiffusionAdapters.denoiser(ocrLateTeacher))
                .teacher("aesthetics-early", RunnerDiffusionAdapters.denoiser(aestheticsEarlyTeacher))
                .teacher("aesthetics-late", RunnerDiffusionAdapters.denoiser(aestheticsLateTeacher))
                .scheduler(diffusionScheduler)
                .conditioningResolver(sample -> Tensor.zeros(1, 4, 8, 8))
                .optimizationStep(loss -> {
                    // The example focuses on Java-side wiring and rollout shape.
                    // Real training code should connect this to an optimizer step.
                })
                .checkpointDir(Path.of("build", "diffusion-opd-example"))
                .adaptiveStageWeighting(true)
                .adaptiveStageWeightMomentum(0.60d)
                .adaptiveStageWeightRange(0.80d, 1.35d)
                .listener(new ConsoleDiffusionListener())
                .task(new DiffusionTask(
                        "ocr",
                        "OCR",
                        "ocr",
                        "teacher-ocr",
                        DiffusionTeacherBindings.splitEarlyLate(
                                diffusionScheduler,
                                "ocr-early",
                                "ocr-late",
                                1.25d,
                                0.85d),
                        ocrPrompts))
                .task(new DiffusionTask(
                        "aesthetics",
                        "Aesthetics",
                        "aesthetic",
                        "teacher-aesthetics",
                        DiffusionTeacherBindings.splitEarlyLate(
                                diffusionScheduler,
                                "aesthetics-early",
                                "aesthetics-late",
                                0.90d,
                                1.15d),
                        aestheticsPrompts))
                .latentShape(1, 4, 8, 8)
                .batchSize(1)
                .maxRounds(2)
                .build();

        var summary = trainer.fit();
        System.out.println("DiffusionOPD example completed.");
        System.out.println("Rounds: " + summary.epochCount());
        System.out.println("Mean loss: " + summary.latestTrainLoss());
        System.out.println("Partitioning: " + DiffusionTeacherBindings.partitionSummary(diffusionScheduler));
        System.out.println("Metadata: " + summary.metadata());
    }

    private static final class ConsoleDiffusionListener implements DiffusionOpdListener {
        @Override
        public void onRoundEnd(DiffusionOpdSession session, int round, double meanLoss, int optimizationSteps) {
            System.out.printf(
                    "round=%d meanLoss=%.6f steps=%d%n",
                    round,
                    meanLoss,
                    optimizationSteps);
        }

        @Override
        public void onTrainingEnd(DiffusionOpdSession session, TrainingSummary summary) {
            System.out.println("summary file: " + summary.metadata().get("summaryFile"));
            System.out.println("history file: " + summary.metadata().get("historyFile"));
            System.out.println("teacher usage: " + summary.metadata().get("teacherUsage"));
            System.out.println("stage usage: " + summary.metadata().get("stageUsage"));
            System.out.println("stage weighted loss: " + summary.metadata().get("stageWeightedLoss"));
            System.out.println("adaptive stage factors: " + summary.metadata().get("adaptiveStageFactors"));
        }
    }

    private static final class StubUNetModel implements UNetModel {
        private final float scale;

        private StubUNetModel(float scale) {
            this.scale = scale;
        }

        @Override
        public Tensor predict(Tensor latents, Tensor embedding, int timestep) {
            float timestepBias = 1.0f / Math.max(1, timestep + 1);
            return latents.mul(scale).add(timestepBias);
        }
    }
}
