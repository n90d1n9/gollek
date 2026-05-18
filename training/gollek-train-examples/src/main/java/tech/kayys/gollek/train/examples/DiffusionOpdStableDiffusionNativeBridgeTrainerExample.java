package tech.kayys.gollek.train.examples;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.runner.sd.PNDMScheduler;
import tech.kayys.gollek.safetensor.runner.sd.UNetModel;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdListener;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSession;
import tech.kayys.gollek.train.diffusion.api.DiffusionPromptSample;
import tech.kayys.gollek.train.diffusion.api.DiffusionSamplerType;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;
import tech.kayys.gollek.train.diffusion.api.DiffusionTeacherBindings;
import tech.kayys.gollek.train.diffusion.opd.DiffusionOpdReports;
import tech.kayys.gollek.train.diffusion.opd.DiffusionOpdDiagnosticsPacks;
import tech.kayys.gollek.train.diffusion.opd.adapter.StableDiffusionRunnerAdapters;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * End-to-end Java-first DiffusionOPD demo that uses both stable-diffusion-
 * native bridge surfaces together: the executable PNDM scheduler adapter and
 * the UNet denoiser adapter.
 *
 * <p>Reference:
 * Quanhao Li et al., "DiffusionOPD: A Unified Perspective of On-Policy
 * Distillation in Diffusion Models", arXiv:2605.15055, 2026.
 */
public final class DiffusionOpdStableDiffusionNativeBridgeTrainerExample {

    private DiffusionOpdStableDiffusionNativeBridgeTrainerExample() {
    }

    public static void main(String[] args) {
        MetalBinding.initialize();
        MetalBinding metalBinding = MetalBinding.getInstance();
        metalBinding.init();
        StableDiffusionNativeRoleFixtures roleFixtures = StableDiffusionNativeRoleFixtures.detect(
                args.length > 0 ? Path.of(args[0]) : null);
        StableDiffusionNativeModelFixture fixture = roleFixtures.sharedFixture();
        StableDiffusionNativeTextFixtures textFixtures = StableDiffusionNativeTextFixtures.detect(
                args.length > 0 ? Path.of(args[0]) : null);

        var bridge = StableDiffusionRunnerAdapters.metalDefaultTensorFloat32Bridge(metalBinding);
        PNDMScheduler scheduler = new PNDMScheduler(4);
        var diffusionScheduler = StableDiffusionRunnerAdapters.scheduler(scheduler, bridge);

        List<DiffusionPromptSample> ocrPrompts = List.of(
                promptSample("clean poster typography with readable GOLLEK headline", 21L, StableDiffusionNativeTextFixtures.OCR),
                promptSample("document scan with crisp invoice text and layout", 22L, StableDiffusionNativeTextFixtures.OCR));
        List<DiffusionPromptSample> aestheticsPrompts = List.of(
                promptSample("premium watch campaign with dramatic reflections", 31L, StableDiffusionNativeTextFixtures.AESTHETICS),
                promptSample("magazine beauty portrait with soft cinematic bloom", 32L, StableDiffusionNativeTextFixtures.AESTHETICS));
        DiffusionTask ocrTask = new DiffusionTask(
                "ocr",
                "OCR",
                "ocr",
                "teacher-ocr",
                DiffusionTeacherBindings.splitEarlyLate(
                        diffusionScheduler,
                        "ocr-early",
                        "ocr-late",
                        1.20d,
                        0.90d),
                ocrPrompts);
        DiffusionTask aestheticsTask = new DiffusionTask(
                "aesthetics",
                "Aesthetics",
                "aesthetic",
                "teacher-aesthetics",
                DiffusionTeacherBindings.splitEarlyLate(
                        diffusionScheduler,
                        "aesthetics-early",
                        "aesthetics-late",
                        0.90d,
                        1.10d),
                aestheticsPrompts);
        List<DiffusionTask> tasks = List.of(ocrTask, aestheticsTask);
        boolean useAnyRealUnet = roleFixtures.fixturesByRole().values().stream().anyMatch(StableDiffusionNativeModelFixture::isUsable);
        boolean useAnyRealTextConditioning = textFixtures.fixturesByTask().values().stream().anyMatch(StableDiffusionNativeModelFixture::isUsable);
        try (RoleLoadedUnets loadedUnets = loadRoleUnets(roleFixtures);
                TaskLoadedTextConditioners textConditioners = loadTaskTextConditioners(textFixtures)) {
            UNetModel student = loadedUnets.modelOrFallback(
                    StableDiffusionNativeRoleFixtures.STUDENT,
                    new StubStableDiffusionUnet(0.08f, 0.010f));
            UNetModel ocrEarlyTeacher = loadedUnets.modelOrFallback(
                    StableDiffusionNativeRoleFixtures.OCR_EARLY,
                    new StubStableDiffusionUnet(0.06f, 0.008f));
            UNetModel ocrLateTeacher = loadedUnets.modelOrFallback(
                    StableDiffusionNativeRoleFixtures.OCR_LATE,
                    new StubStableDiffusionUnet(0.07f, 0.012f));
            UNetModel aestheticsEarlyTeacher = loadedUnets.modelOrFallback(
                    StableDiffusionNativeRoleFixtures.AESTHETICS_EARLY,
                    new StubStableDiffusionUnet(0.09f, 0.014f));
            UNetModel aestheticsLateTeacher = loadedUnets.modelOrFallback(
                    StableDiffusionNativeRoleFixtures.AESTHETICS_LATE,
                    new StubStableDiffusionUnet(0.10f, 0.016f));

            var trainer = Gollek.DL.diffusionOpdTrainer()
                    .samplerType(DiffusionSamplerType.ODE)
                    .student(StableDiffusionRunnerAdapters.denoiser(student, bridge))
                    .teacher("ocr-early", StableDiffusionRunnerAdapters.denoiser(ocrEarlyTeacher, bridge))
                    .teacher("ocr-late", StableDiffusionRunnerAdapters.denoiser(ocrLateTeacher, bridge))
                    .teacher("aesthetics-early", StableDiffusionRunnerAdapters.denoiser(aestheticsEarlyTeacher, bridge))
                    .teacher("aesthetics-late", StableDiffusionRunnerAdapters.denoiser(aestheticsLateTeacher, bridge))
                    .scheduler(diffusionScheduler)
                    .conditioningResolver(sample -> conditioningTensor(sample, textConditioners))
                    .optimizationStep(loss -> {
                        // Structural example only.
                        // Replace with a real optimizer-backed update step for
                        // actual trainable stable-diffusion-native OPD.
                    })
                    .checkpointDir(Path.of("build", "diffusion-opd-stable-diffusion-native-bridge"))
                    .adaptiveStageWeighting(true)
                    .adaptiveStageWeightMomentum(0.60d)
                    .adaptiveStageWeightRange(0.80d, 1.35d)
                    .runtimeObservers(DiffusionOpdDiagnosticsPacks.standardTaskDiagnostics(
                            tasks,
                            conditioningModes(textFixtures),
                            conditioningFixtureBaseDirs(textFixtures),
                            textFixtures.summary()))
                    .listener(new ConsoleDiffusionListener())
                    .task(ocrTask)
                    .task(aestheticsTask)
                    .latentShape(1, 4, 8, 8)
                    .batchSize(1)
                    .maxRounds(2)
                    .build();

            TrainingSummary summary = trainer.fit();
            System.out.println("DiffusionOPD stable-diffusion-native bridge trainer completed.");
            System.out.println("reference=arXiv:2605.15055");
            System.out.println("fixture=" + fixture.summary());
            System.out.println("fixtureRoles=" + roleFixtures.summary());
            System.out.println("textFixtures=" + textFixtures.summary());
            System.out.println("fixtureMode=" + (useAnyRealUnet ? "mixed-or-real-unet" : "stub-unet"));
            System.out.println("conditioningMode=" + (useAnyRealTextConditioning ? "task-real-clip-or-shared" : "synthetic"));
            System.out.println("fixtureOverride=" + fixture.overrideHint());
            System.out.println("textFixtureOverrideExamples="
                    + StableDiffusionNativeTextFixtures.envNameForTask(StableDiffusionNativeTextFixtures.OCR) + ", "
                    + StableDiffusionNativeTextFixtures.envNameForTask(StableDiffusionNativeTextFixtures.AESTHETICS));
            System.out.println("epochCount=" + summary.epochCount());
            System.out.println("latestTrainLoss=" + summary.latestTrainLoss());
            System.out.println("acceleratorMetadata=" + bridge.acceleratorMetadata());
            System.out.println("schedulerMetadata=" + StableDiffusionRunnerAdapters.schedulerMetadata(scheduler));
            System.out.println("partitioning=" + DiffusionTeacherBindings.partitionSummary(diffusionScheduler));
            Object reportFile = summary.metadata().get("reportFile");
            if (reportFile instanceof String reportPath && !reportPath.isBlank()) {
                var report = DiffusionOpdReports.load(Path.of(reportPath));
                System.out.println("reportRun=" + report.run().asMap());
            }
            System.out.println("metadata=" + summary.metadata());
        }
    }

    private static RoleLoadedUnets loadRoleUnets(StableDiffusionNativeRoleFixtures roleFixtures) {
        RoleLoadedUnets loaded = new RoleLoadedUnets();
        roleFixtures.fixturesByRole().forEach((role, roleFixture) -> {
            if (roleFixture.isUsable()) {
                loaded.put(role, StableDiffusionNativeFixtureLoader.loadUnet(roleFixture.baseDir()));
            }
        });
        return loaded;
    }

    private static TaskLoadedTextConditioners loadTaskTextConditioners(
            StableDiffusionNativeTextFixtures textFixtures) {
        TaskLoadedTextConditioners loaded = new TaskLoadedTextConditioners();
        textFixtures.fixturesByTask().forEach((taskId, taskFixture) -> {
            if (taskFixture.isUsable()) {
                loaded.put(taskId, StableDiffusionNativeFixtureLoader.loadTextConditioner(taskFixture.baseDir()));
            }
        });
        return loaded;
    }

    private static DiffusionPromptSample promptSample(String prompt, long seed, String taskId) {
        return new DiffusionPromptSample(
                prompt,
                "",
                seed,
                Map.of("taskId", taskId));
    }

    private static Tensor conditioningTensor(
            DiffusionPromptSample sample,
            TaskLoadedTextConditioners textConditioners) {
        String taskId = String.valueOf(sample.metadata().getOrDefault("taskId", ""));
        StableDiffusionNativeFixtureLoader.LoadedTextConditioner textConditioner =
                textConditioners.get(taskId);
        if (textConditioner == null) {
            return Tensor.full(0.25f, 1, 77, 768);
        }
        return Tensor.of(textConditioner.encodePromptToFloatArray(sample.prompt()), 1, 77, 768);
    }

    private static Map<String, String> conditioningModes(
            StableDiffusionNativeTextFixtures textFixtures) {
        Map<String, String> modes = new LinkedHashMap<>();
        textFixtures.fixturesByTask().forEach((taskId, fixture) ->
                modes.put(taskId, fixture.isUsable() ? "real-clip" : "synthetic"));
        return Map.copyOf(modes);
    }

    private static Map<String, String> conditioningFixtureBaseDirs(
            StableDiffusionNativeTextFixtures textFixtures) {
        Map<String, String> baseDirs = new LinkedHashMap<>();
        textFixtures.fixturesByTask().forEach((taskId, fixture) ->
                baseDirs.put(taskId, fixture.baseDir().toString()));
        return Map.copyOf(baseDirs);
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
            System.out.println("reportFile=" + summary.metadata().get("reportFile"));
            System.out.println("teacherUsage=" + summary.metadata().get("teacherUsage"));
            System.out.println("stageUsage=" + summary.metadata().get("stageUsage"));
            System.out.println("stageWeightedLoss=" + summary.metadata().get("stageWeightedLoss"));
            System.out.println("adaptiveStageFactors=" + summary.metadata().get("adaptiveStageFactors"));
        }
    }

    private static final class StubStableDiffusionUnet extends UNetModel {
        private final float latentScale;
        private final float conditioningScale;

        private StubStableDiffusionUnet(float latentScale, float conditioningScale) {
            super(Map.of());
            this.latentScale = latentScale;
            this.conditioningScale = conditioningScale;
        }

        @Override
        public AccelTensor predict(AccelTensor sample, long timestep, AccelTensor encoderHiddenStates) {
            float[] latents = sample.toFloatArray();
            float[] conditioning = encoderHiddenStates.toFloatArray();
            float conditioningMean = mean(conditioning);
            float timestepBias = 1.0f / Math.max(1L, timestep + 1L);
            float[] values = new float[latents.length];
            for (int i = 0; i < latents.length; i++) {
                values[i] = latents[i] * latentScale
                        + conditioningMean * conditioningScale
                        + timestepBias;
            }
            return AccelTensor.fromFloatArray(values, sample.shape());
        }

        private static float mean(float[] values) {
            float sum = 0.0f;
            for (float value : values) {
                sum += value;
            }
            return values.length == 0 ? 0.0f : sum / values.length;
        }
    }

    private static final class RoleLoadedUnets implements AutoCloseable {
        private final Map<String, StableDiffusionNativeFixtureLoader.LoadedUnet> loaded = new java.util.LinkedHashMap<>();

        private void put(String role, StableDiffusionNativeFixtureLoader.LoadedUnet unet) {
            loaded.put(role, unet);
        }

        private UNetModel modelOrFallback(String role, UNetModel fallback) {
            StableDiffusionNativeFixtureLoader.LoadedUnet loadedUnet = loaded.get(role);
            return loadedUnet == null ? fallback : loadedUnet.model();
        }

        @Override
        public void close() {
            loaded.values().forEach(StableDiffusionNativeFixtureLoader.LoadedUnet::close);
        }
    }

    private static final class TaskLoadedTextConditioners implements AutoCloseable {
        private final Map<String, StableDiffusionNativeFixtureLoader.LoadedTextConditioner> loaded =
                new LinkedHashMap<>();

        private void put(String taskId, StableDiffusionNativeFixtureLoader.LoadedTextConditioner textConditioner) {
            loaded.put(taskId, textConditioner);
        }

        private StableDiffusionNativeFixtureLoader.LoadedTextConditioner get(String taskId) {
            return loaded.get(taskId);
        }

        @Override
        public void close() {
            loaded.values().forEach(StableDiffusionNativeFixtureLoader.LoadedTextConditioner::close);
        }
    }
}
