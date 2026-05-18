///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-ml-diffusion-opd:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-backend-metal:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-runner-stable-diffusion:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-safetensor-core:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-safetensor-loader:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-safetensor-quantization:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-tokenizer-core:0.1.0-SNAPSHOT

import java.nio.file.Path;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.runner.sd.CLIPModel;
import tech.kayys.gollek.safetensor.runner.sd.PNDMScheduler;
import tech.kayys.gollek.safetensor.runner.sd.UNetModel;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdListener;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSession;
import tech.kayys.gollek.train.diffusion.api.DiffusionPromptSample;
import tech.kayys.gollek.train.diffusion.api.DiffusionSamplerType;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;
import tech.kayys.gollek.train.diffusion.api.DiffusionTeacherBindings;
import tech.kayys.gollek.train.diffusion.opd.DiffusionOpdDiagnosticsPacks;
import tech.kayys.gollek.train.diffusion.opd.adapter.StableDiffusionRunnerAdapters;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * JBang end-to-end DiffusionOPD demo over the stable-diffusion-native bridge
 * surfaces.
 *
 * <p>Reference:
 * Quanhao Li et al., "DiffusionOPD: A Unified Perspective of On-Policy
 * Distillation in Diffusion Models", arXiv:2605.15055, 2026.
 */
public class trainer_diffusion_opd_stable_diffusion_native_bridge {
    private static final String ENV_MODEL_DIR = "GOLLEK_SD_NATIVE_MODEL_DIR";

    public static void main(String[] args) {
        MetalBinding.initialize();
        MetalBinding metalBinding = MetalBinding.getInstance();
        metalBinding.init();
        FixtureSet fixtures = args.length > 0
                ? detectRoleFixtures(Path.of(args[0]))
                : detectRoleFixtures(null);
        FixtureStatus fixture = fixtures.shared;
        TextFixtureSet textFixtures = args.length > 0
                ? detectTextFixtures(Path.of(args[0]))
                : detectTextFixtures(null);

        var bridge = StableDiffusionRunnerAdapters.metalDefaultTensorFloat32Bridge(metalBinding);
        PNDMScheduler scheduler = new PNDMScheduler(4);
        var diffusionScheduler = StableDiffusionRunnerAdapters.scheduler(scheduler, bridge);

        List<DiffusionPromptSample> ocrPrompts = List.of(
                promptSample("clean poster typography with readable GOLLEK headline", 21L, "ocr"),
                promptSample("document scan with crisp invoice text and layout", 22L, "ocr"));
        List<DiffusionPromptSample> aestheticsPrompts = List.of(
                promptSample("premium watch campaign with dramatic reflections", 31L, "aesthetics"),
                promptSample("magazine beauty portrait with soft cinematic bloom", 32L, "aesthetics"));
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
        boolean useAnyRealUnet = fixtures.byRole.values().stream().anyMatch(FixtureStatus::isUsable);
        boolean useAnyRealTextConditioning = textFixtures.byTask.values().stream().anyMatch(FixtureStatus::isUsable);
        try (LoadedUnets loadedUnets = loadRoleUnets(fixtures);
                LoadedTextConditioners textConditioners = loadTaskTextConditioners(textFixtures)) {
            UNetModel student = loadedUnets.modelOrFallback("student", new StubStableDiffusionUnet(0.08f, 0.010f));
            UNetModel ocrEarlyTeacher = loadedUnets.modelOrFallback("ocr-early", new StubStableDiffusionUnet(0.06f, 0.008f));
            UNetModel ocrLateTeacher = loadedUnets.modelOrFallback("ocr-late", new StubStableDiffusionUnet(0.07f, 0.012f));
            UNetModel aestheticsEarlyTeacher = loadedUnets.modelOrFallback("aesthetics-early", new StubStableDiffusionUnet(0.09f, 0.014f));
            UNetModel aestheticsLateTeacher = loadedUnets.modelOrFallback("aesthetics-late", new StubStableDiffusionUnet(0.10f, 0.016f));

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
                        // Structural demo only.
                    })
                    .checkpointDir(Path.of("jbang-diffusion-opd-stable-diffusion-native-bridge"))
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
            System.out.println("====================================================");
            System.out.println(" Gollek JBang Stable Diffusion Native OPD Bridge");
            System.out.println("====================================================");
            System.out.println("reference=arXiv:2605.15055");
            System.out.println("fixture=" + fixture.summary());
            System.out.println("fixtureRoles=" + fixtures.summary());
            System.out.println("textFixtures=" + textFixtures.summary());
            System.out.println("fixtureMode=" + (useAnyRealUnet ? "mixed-or-real-unet" : "stub-unet"));
            System.out.println("conditioningMode=" + (useAnyRealTextConditioning ? "task-real-clip-or-shared" : "synthetic"));
            System.out.println("fixtureOverride=Set " + ENV_MODEL_DIR
                    + " to a Stable Diffusion repo containing text_encoder/, unet/, vae/, and tokenizer/.");
            System.out.println("fixtureRoleOverrideExamples="
                    + envNameForRole("student") + ", "
                    + envNameForRole("ocr-early") + ", "
                    + envNameForRole("ocr-late"));
            System.out.println("textFixtureOverrideExamples="
                    + envNameForTextTask("ocr") + ", "
                    + envNameForTextTask("aesthetics"));
            System.out.println("epochCount=" + summary.epochCount());
            System.out.println("latestTrainLoss=" + summary.latestTrainLoss());
            System.out.println("acceleratorMetadata=" + bridge.acceleratorMetadata());
            System.out.println("schedulerMetadata=" + StableDiffusionRunnerAdapters.schedulerMetadata(scheduler));
            System.out.println("partitioning=" + DiffusionTeacherBindings.partitionSummary(diffusionScheduler));
            System.out.println("metadata=" + summary.metadata());
        }
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

    private static Path detectFixtureBaseDir() {
        String envPath = System.getenv(ENV_MODEL_DIR);
        if (envPath != null && !envPath.isBlank()) {
            return Path.of(envPath);
        }
        List<Path> candidates = List.of(
                Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors", "CompVis", "stable-diffusion-v1-4"),
                Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors", "stable-diffusion-v1-4"),
                Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs", "CompVis", "stable-diffusion-v1-4"));
        for (Path candidate : candidates) {
            if (Files.isDirectory(candidate)) {
                return candidate;
            }
        }
        return candidates.get(0);
    }

    private static FixtureStatus fixtureStatus(Path baseDir) {
        Path normalized = baseDir.toAbsolutePath().normalize();
        Map<String, Path> required = new LinkedHashMap<>();
        required.put("textEncoderDir", normalized.resolve("text_encoder"));
        required.put("unetDir", normalized.resolve("unet"));
        required.put("vaeDir", normalized.resolve("vae"));
        required.put("tokenizerDir", normalized.resolve("tokenizer"));
        List<String> missing = new ArrayList<>();
        required.forEach((key, path) -> {
            if (!Files.isDirectory(path)) {
                missing.add(key + "=" + path);
            }
        });
        return new FixtureStatus(normalized, missing);
    }

    private static FixtureSet detectRoleFixtures(Path sharedBaseOverride) {
        FixtureStatus shared = sharedBaseOverride == null
                ? fixtureStatus(detectFixtureBaseDir())
                : fixtureStatus(sharedBaseOverride);
        Map<String, FixtureStatus> byRole = new LinkedHashMap<>();
        for (String role : List.of("student", "ocr-early", "ocr-late", "aesthetics-early", "aesthetics-late")) {
            byRole.put(role, resolveRoleFixture(role, shared));
        }
        return new FixtureSet(shared, byRole);
    }

    private static TextFixtureSet detectTextFixtures(Path sharedBaseOverride) {
        FixtureStatus shared = sharedBaseOverride == null
                ? fixtureStatus(detectFixtureBaseDir())
                : fixtureStatus(sharedBaseOverride);
        Map<String, FixtureStatus> byTask = new LinkedHashMap<>();
        for (String taskId : List.of("ocr", "aesthetics")) {
            byTask.put(taskId, resolveTextTaskFixture(taskId, shared));
        }
        return new TextFixtureSet(shared, byTask);
    }

    private static FixtureStatus resolveRoleFixture(String role, FixtureStatus shared) {
        String env = System.getenv(envNameForRole(role));
        if (env != null && !env.isBlank()) {
            return fixtureStatus(Path.of(env));
        }
        Path roleDir = shared.baseDir().resolve(role);
        if (Files.isDirectory(roleDir)) {
            FixtureStatus candidate = fixtureStatus(roleDir);
            if (candidate.isUsable()) {
                return candidate;
            }
        }
        return shared;
    }

    private static String envNameForRole(String role) {
        return ENV_MODEL_DIR + "_" + role.toUpperCase().replace('-', '_');
    }

    private static FixtureStatus resolveTextTaskFixture(String taskId, FixtureStatus shared) {
        String env = System.getenv(envNameForTextTask(taskId));
        if (env != null && !env.isBlank()) {
            return fixtureStatus(Path.of(env));
        }
        Path taskDir = shared.baseDir().resolve("text-" + taskId);
        if (Files.isDirectory(taskDir)) {
            FixtureStatus candidate = fixtureStatus(taskDir);
            if (candidate.isUsable()) {
                return candidate;
            }
        }
        return shared;
    }

    private static String envNameForTextTask(String taskId) {
        return ENV_MODEL_DIR + "_TEXT_" + taskId.toUpperCase().replace('-', '_');
    }

    private static DiffusionPromptSample promptSample(String prompt, long seed, String taskId) {
        return new DiffusionPromptSample(prompt, "", seed, Map.of("taskId", taskId));
    }

    private static Tensor conditioningTensor(
            DiffusionPromptSample sample,
            LoadedTextConditioners textConditioners) {
        String taskId = String.valueOf(sample.metadata().getOrDefault("taskId", ""));
        LoadedTextConditioner textConditioner = textConditioners.get(taskId);
        if (textConditioner == null) {
            return Tensor.full(0.25f, 1, 77, 768);
        }
        return Tensor.of(textConditioner.encodePromptToFloatArray(sample.prompt()), 1, 77, 768);
    }

    private static Map<String, String> conditioningModes(TextFixtureSet textFixtures) {
        Map<String, String> modes = new LinkedHashMap<>();
        textFixtures.byTask.forEach((taskId, fixture) ->
                modes.put(taskId, fixture.isUsable() ? "real-clip" : "synthetic"));
        return Map.copyOf(modes);
    }

    private static Map<String, String> conditioningFixtureBaseDirs(TextFixtureSet textFixtures) {
        Map<String, String> baseDirs = new LinkedHashMap<>();
        textFixtures.byTask.forEach((taskId, fixture) ->
                baseDirs.put(taskId, fixture.baseDir().toString()));
        return Map.copyOf(baseDirs);
    }

    private record FixtureStatus(Path baseDir, List<String> missing) {
        private boolean isUsable() {
            return missing.isEmpty();
        }

        private String summary() {
            if (missing.isEmpty()) {
                return "real-fixture-ready baseDir=" + baseDir;
            }
            return "stub-fallback missing=" + missing + " baseDir=" + baseDir;
        }
    }

    private record FixtureSet(FixtureStatus shared, Map<String, FixtureStatus> byRole) {
        private String summary() {
            Map<String, String> summary = new LinkedHashMap<>();
            byRole.forEach((role, status) -> summary.put(role, status.summary()));
            return summary.toString();
        }
    }

    private record TextFixtureSet(FixtureStatus shared, Map<String, FixtureStatus> byTask) {
        private String summary() {
            Map<String, String> summary = new LinkedHashMap<>();
            byTask.forEach((taskId, status) -> summary.put(taskId, status.summary()));
            return summary.toString();
        }
    }

    private static LoadedUnets loadRoleUnets(FixtureSet fixtures) {
        LoadedUnets loaded = new LoadedUnets();
        fixtures.byRole.forEach((role, status) -> {
            if (status.isUsable()) {
                loaded.put(role, loadUnetFixture(status.baseDir()));
            }
        });
        return loaded;
    }

    private static LoadedTextConditioners loadTaskTextConditioners(TextFixtureSet fixtures) {
        LoadedTextConditioners loaded = new LoadedTextConditioners();
        fixtures.byTask.forEach((taskId, status) -> {
            if (status.isUsable()) {
                loaded.put(taskId, loadTextConditioner(status.baseDir()));
            }
        });
        return loaded;
    }

    private static LoadedUnet loadUnetFixture(Path modelBaseDir) {
        try {
            StandaloneSafetensorStack stack = createStandaloneStack();
            tech.kayys.gollek.safetensor.quantization.bridge.AccelWeightBridge weightBridge =
                    new tech.kayys.gollek.safetensor.quantization.bridge.AccelWeightBridge();
            try (tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession session =
                         stack.shardLoader.open(modelBaseDir.resolve("unet"))) {
                java.util.Map<String, AccelTensor> weights = weightBridge.bridgeAll(session);
                return new LoadedUnet(new UNetModel(weights), weights);
            }
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to load stable-diffusion-native UNet fixture from " + modelBaseDir, e);
        }
    }

    private static LoadedTextConditioner loadTextConditioner(Path modelBaseDir) {
        try {
            StandaloneSafetensorStack stack = createStandaloneStack();
            tech.kayys.gollek.safetensor.quantization.bridge.AccelWeightBridge weightBridge =
                    new tech.kayys.gollek.safetensor.quantization.bridge.AccelWeightBridge();
            try (tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession session =
                         stack.shardLoader.open(modelBaseDir.resolve("text_encoder"))) {
                java.util.Map<String, AccelTensor> weights = weightBridge.bridgeAll(session);
                Tokenizer tokenizer = TokenizerFactory.load(modelBaseDir.resolve("tokenizer"), null);
                return new LoadedTextConditioner(new CLIPModel(weights), weights, tokenizer);
            }
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to load stable-diffusion-native CLIP fixture from " + modelBaseDir, e);
        }
    }

    private static StandaloneSafetensorStack createStandaloneStack() throws Exception {
        tech.kayys.gollek.safetensor.loader.SafetensorFFMLoader ffmLoader =
                new tech.kayys.gollek.safetensor.loader.SafetensorFFMLoader();
        setField(ffmLoader, "config", standaloneConfig());
        setField(ffmLoader, "objectMapper", newObjectMapper());

        tech.kayys.gollek.safetensor.loader.SafetensorMetrics metrics =
                new tech.kayys.gollek.safetensor.loader.SafetensorMetrics();
        setField(metrics, "registry", newSimpleMeterRegistry());
        invoke(metrics, "registerGauge");
        setField(ffmLoader, "metrics", metrics);

        tech.kayys.gollek.safetensor.loader.SafetensorShardLoader shardLoader =
                new tech.kayys.gollek.safetensor.loader.SafetensorShardLoader();
        setField(shardLoader, "ffmLoader", ffmLoader);
        setField(shardLoader, "objectMapper", newObjectMapper());
        return new StandaloneSafetensorStack(ffmLoader, shardLoader);
    }

    private static Object standaloneConfig() {
        Class<?> loaderConfigType = loadClass("tech.kayys.gollek.safetensor.loader.SafetensorLoaderConfig");
        Class<?> validationType = loadClass("tech.kayys.gollek.safetensor.loader.SafetensorLoaderConfig$Validation");
        Class<?> cacheType = loadClass("tech.kayys.gollek.safetensor.loader.SafetensorLoaderConfig$Cache");
        java.lang.reflect.InvocationHandler handler = (proxy, method, methodArgs) -> switch (method.getName()) {
            case "preferMmap", "strict", "warnOnEmptyTensors", "enabled", "logLoadSummary" -> true;
            case "readChunkBytes" -> 8 * 1024 * 1024;
            case "maxHeaderBytes" -> 104_857_600L;
            case "maxSize" -> 8;
            case "ttlSeconds" -> 300L;
            case "validation" -> java.lang.reflect.Proxy.newProxyInstance(
                    trainer_diffusion_opd_stable_diffusion_native_bridge.class.getClassLoader(),
                    new Class<?>[] { validationType },
                    (java.lang.reflect.InvocationHandler) (p, nestedMethod, nestedArgs) ->
                            handlerResult(proxy, nestedMethod.getName(), nestedArgs));
            case "cache" -> java.lang.reflect.Proxy.newProxyInstance(
                    trainer_diffusion_opd_stable_diffusion_native_bridge.class.getClassLoader(),
                    new Class<?>[] { cacheType },
                    (java.lang.reflect.InvocationHandler) (p, nestedMethod, nestedArgs) ->
                            handlerResult(proxy, nestedMethod.getName(), nestedArgs));
            case "toString" -> "StandaloneSafetensorLoaderConfig";
            case "hashCode" -> System.identityHashCode(proxy);
            case "equals" -> proxy == methodArgs[0];
            default -> throw new UnsupportedOperationException("Unsupported config method: " + method);
        };
        return java.lang.reflect.Proxy.newProxyInstance(
                trainer_diffusion_opd_stable_diffusion_native_bridge.class.getClassLoader(),
                new Class<?>[] { loaderConfigType },
                handler);
    }

    private static Object handlerResult(Object proxy, String methodName, Object[] args) {
        return switch (methodName) {
            case "preferMmap", "strict", "warnOnEmptyTensors", "enabled", "logLoadSummary" -> true;
            case "readChunkBytes" -> 8 * 1024 * 1024;
            case "maxHeaderBytes" -> 104_857_600L;
            case "maxSize" -> 8;
            case "ttlSeconds" -> 300L;
            case "toString" -> "StandaloneSafetensorLoaderConfig";
            case "hashCode" -> System.identityHashCode(proxy);
            case "equals" -> proxy == args[0];
            default -> throw new UnsupportedOperationException("Unsupported config method: " + methodName);
        };
    }

    private static Object newObjectMapper() throws Exception {
        return Class.forName("com.fasterxml.jackson.databind.ObjectMapper")
                .getDeclaredConstructor()
                .newInstance();
    }

    private static Object newSimpleMeterRegistry() throws Exception {
        return Class.forName("io.micrometer.core.instrument.simple.SimpleMeterRegistry")
                .getDeclaredConstructor()
                .newInstance();
    }

    private static Class<?> loadClass(String className) {
        try {
            return Class.forName(className);
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException("Missing class on classpath: " + className, e);
        }
    }

    private static void setField(Object target, String fieldName, Object value) throws Exception {
        java.lang.reflect.Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static void invoke(Object target, String methodName) throws Exception {
        java.lang.reflect.Method method = target.getClass().getDeclaredMethod(methodName);
        method.setAccessible(true);
        method.invoke(target);
    }

    private static final class LoadedUnet implements AutoCloseable {
        private final UNetModel model;
        private final java.util.Map<String, AccelTensor> weights;

        private LoadedUnet(UNetModel model, java.util.Map<String, AccelTensor> weights) {
            this.model = model;
            this.weights = java.util.Map.copyOf(weights);
        }

        @Override
        public void close() {
            weights.values().forEach(AccelTensor::close);
        }
    }

    private static final class LoadedTextConditioner implements AutoCloseable {
        private static final int MAX_TOKENS = 77;
        private static final long PAD_TOKEN_ID = 49407L;

        private final CLIPModel clipModel;
        private final java.util.Map<String, AccelTensor> weights;
        private final Tokenizer tokenizer;

        private LoadedTextConditioner(CLIPModel clipModel, java.util.Map<String, AccelTensor> weights, Tokenizer tokenizer) {
            this.clipModel = clipModel;
            this.weights = java.util.Map.copyOf(weights);
            this.tokenizer = tokenizer;
        }

        private float[] encodePromptToFloatArray(String prompt) {
            long[] ids = tokenizer.encode(prompt == null ? "" : prompt, EncodeOptions.defaultOptions());
            long[] padded = new long[MAX_TOKENS];
            Arrays.fill(padded, PAD_TOKEN_ID);
            for (int i = 0; i < Math.min(ids.length, MAX_TOKENS); i++) {
                padded[i] = ids[i];
            }
            try (AccelTensor encoded = clipModel.encode(padded)) {
                return encoded.toFloatArray();
            }
        }

        @Override
        public void close() {
            weights.values().forEach(AccelTensor::close);
        }
    }

    private static final class LoadedTextConditioners implements AutoCloseable {
        private final Map<String, LoadedTextConditioner> loaded = new LinkedHashMap<>();

        private void put(String taskId, LoadedTextConditioner textConditioner) {
            loaded.put(taskId, textConditioner);
        }

        private LoadedTextConditioner get(String taskId) {
            return loaded.get(taskId);
        }

        @Override
        public void close() {
            loaded.values().forEach(LoadedTextConditioner::close);
        }
    }

    private static final class LoadedUnets implements AutoCloseable {
        private final Map<String, LoadedUnet> loaded = new LinkedHashMap<>();

        private void put(String role, LoadedUnet unet) {
            loaded.put(role, unet);
        }

        private UNetModel modelOrFallback(String role, UNetModel fallback) {
            LoadedUnet loadedUnet = loaded.get(role);
            return loadedUnet == null ? fallback : loadedUnet.model;
        }

        @Override
        public void close() {
            loaded.values().forEach(LoadedUnet::close);
        }
    }

    private record StandaloneSafetensorStack(
            tech.kayys.gollek.safetensor.loader.SafetensorFFMLoader ffmLoader,
            tech.kayys.gollek.safetensor.loader.SafetensorShardLoader shardLoader) {
    }
}
