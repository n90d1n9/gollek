///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-ml-diffusion-opd:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-diffusion:0.1.0-SNAPSHOT

import java.util.List;
import java.util.Locale;
import java.nio.file.Path;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.diffusion.model.UNetModel;
import tech.kayys.gollek.diffusion.scheduler.DDIMScheduler;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdListener;
import tech.kayys.gollek.train.diffusion.api.DiffusionPromptSample;
import tech.kayys.gollek.train.diffusion.api.DiffusionSamplerType;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSession;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;
import tech.kayys.gollek.train.diffusion.api.DiffusionTeacherBindings;
import tech.kayys.gollek.trainer.api.TrainingSummary;
import tech.kayys.gollek.train.diffusion.opd.adapter.RunnerDiffusionAdapters;

/**
 * JBang demo for the Java-first DiffusionOPD scaffold in Gollek.
 *
 * <p>Reference:
 * Quanhao Li et al., "DiffusionOPD: A Unified Perspective of On-Policy
 * Distillation in Diffusion Models", arXiv:2605.15055, 2026.
 */
public class trainer_diffusion_opd_ddim {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        int rounds = args.length > 0 ? parsePositiveInt(args[0], 2) : 2;

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

        System.out.println("====================================================");
        System.out.println(" Gollek JBang DiffusionOPD Demo");
        System.out.println("====================================================");
        System.out.println("sampler=ODE (DDIM-style mean matching)");
        System.out.println("reference=arXiv:2605.15055");
        System.out.println("rounds=" + rounds);

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
                    // Demo mode: structural walkthrough only.
                    // Replace with a real optimizer-backed update step for
                    // actual training.
                })
                .checkpointDir(Path.of("jbang-diffusion-opd"))
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
                .maxRounds(rounds)
                .build();

        var summary = trainer.fit();
        System.out.println("----------------------------------------------------");
        System.out.println("DiffusionOPD demo completed.");
        System.out.println("epochCount=" + summary.epochCount());
        System.out.println("latestTrainLoss=" + summary.latestTrainLoss());
        System.out.println("partitioning=" + DiffusionTeacherBindings.partitionSummary(diffusionScheduler));
        System.out.println("metadata=" + summary.metadata());
    }

    private static final class ConsoleDiffusionListener implements DiffusionOpdListener {
        @Override
        public void onRoundEnd(DiffusionOpdSession session, int round, double meanLoss, int optimizationSteps) {
            System.out.printf("round=%d meanLoss=%.6f steps=%d%n", round, meanLoss, optimizationSteps);
        }

        @Override
        public void onTrainingEnd(DiffusionOpdSession session, TrainingSummary summary) {
            System.out.println("summaryFile=" + summary.metadata().get("summaryFile"));
            System.out.println("historyFile=" + summary.metadata().get("historyFile"));
            System.out.println("teacherUsage=" + summary.metadata().get("teacherUsage"));
            System.out.println("stageUsage=" + summary.metadata().get("stageUsage"));
            System.out.println("stageWeightedLoss=" + summary.metadata().get("stageWeightedLoss"));
            System.out.println("adaptiveStageFactors=" + summary.metadata().get("adaptiveStageFactors"));
        }
    }

    private static int parsePositiveInt(String raw, int fallback) {
        try {
            return Math.max(1, Integer.parseInt(raw.trim()));
        } catch (NumberFormatException ignored) {
            return fallback;
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
