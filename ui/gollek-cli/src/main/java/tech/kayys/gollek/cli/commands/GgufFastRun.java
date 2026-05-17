package tech.kayys.gollek.cli.commands;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.ConnectException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

import tech.kayys.gollek.gguf.runtime.GgufRuntimeProfile;
import tech.kayys.gollek.gguf.runtime.GgufRuntimeProbe;

/**
 * Small GGUF/llama.cpp fast path used by the local macOS shim for simple
 * {@code gollek run} calls.
 */
public final class GgufFastRun {
    private static final int FALLBACK_TO_FULL_CLI = 42;
    private static final int COMMAND_ERROR = 2;
    private static final String DAEMON_MAGIC = "GOLLEK_GGUF_FAST_DAEMON_V1";
    private static final String DAEMON_STOP_MAGIC = "GOLLEK_GGUF_FAST_DAEMON_STOP_V1";
    private static final ThreadLocal<Boolean> REQUEST_TIMING = new ThreadLocal<>();

    private GgufFastRun() {
    }

    public static void main(String[] args) {
        if (args.length > 0 && "__daemon".equals(args[0])) {
            Runtime.getRuntime().halt(runDaemon());
            return;
        }
        if (args.length > 0 && ("__daemon-stop".equals(args[0]) || "__gguf-daemon-stop".equals(args[0]))) {
            System.exit(stopDaemon());
            return;
        }
        int status = run(args);
        if (status != 0) {
            System.exit(status);
        }
    }

    private static int run(String[] args) {
        try {
            FastArgs parsed = FastArgs.parse(args);
            if (!parsed.supported()) {
                return FALLBACK_TO_FULL_CLI;
            }
            Optional<Path> modelPath = resolveGgufModel(parsed);
            if (modelPath.isEmpty()) {
                return FALLBACK_TO_FULL_CLI;
            }
            EngineMode engine = parsed.engineMode();
            if (daemonEnabled() && engine != EngineMode.JAVA && engine != EngineMode.BENCHMARK) {
                OptionalInt daemonStatus = requestDaemon(args);
                if (daemonStatus.isPresent()) {
                    return daemonStatus.getAsInt();
                }
            }
            return generate(modelPath.get(), parsed);
        } catch (Throwable throwable) {
            if (Boolean.getBoolean("gollek.gguf.fast_run.debug")) {
                throwable.printStackTrace(System.err);
            }
            return FALLBACK_TO_FULL_CLI;
        }
    }

    private static Optional<Path> resolveGgufModel(FastArgs args) {
        if (args.modelFile != null) {
            Optional<Path> explicit = resolvePath(Path.of(args.modelFile));
            if (explicit.isPresent()) {
                return findGguf(explicit.get());
            }
        }
        if (args.model != null) {
            Optional<Path> direct = resolvePath(Path.of(args.model));
            if (direct.isPresent()) {
                Optional<Path> found = findGguf(direct.get());
                if (found.isPresent()) {
                    return found;
                }
            }
            try {
                Optional<LocalModelIndex.Entry> indexed = LocalModelIndex.find(args.model);
                if (indexed.isPresent() && indexed.get().path != null && !indexed.get().path.isBlank()) {
                    return findGguf(Path.of(indexed.get().path));
                }
            } catch (Throwable ignored) {
                // Full CLI has the complete resolver if the lightweight index read fails.
            }
        }
        return Optional.empty();
    }

    private static Optional<Path> resolvePath(Path raw) {
        Path expanded = expandHome(raw);
        if (Files.exists(expanded)) {
            return Optional.of(expanded.toAbsolutePath().normalize());
        }
        return Optional.empty();
    }

    private static Path expandHome(Path path) {
        String text = path.toString();
        if (text.equals("~")) {
            return Path.of(System.getProperty("user.home"));
        }
        if (text.startsWith("~/")) {
            return Path.of(System.getProperty("user.home"), text.substring(2));
        }
        return path;
    }

    private static Optional<Path> findGguf(Path path) {
        try {
            if (Files.isRegularFile(path) && isGguf(path)) {
                return Optional.of(path.toAbsolutePath().normalize());
            }
            Path searchDir = Files.isDirectory(path) ? path : path.getParent();
            if (searchDir == null || !Files.isDirectory(searchDir)) {
                return Optional.empty();
            }
            try (var stream = Files.list(searchDir)) {
                return stream
                        .filter(Files::isRegularFile)
                        .filter(GgufFastRun::isGguf)
                        .findFirst()
                        .map(candidate -> candidate.toAbsolutePath().normalize());
            }
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static boolean isGguf(Path path) {
        return path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    private static int generate(Path modelPath, FastArgs args) throws Throwable {
        return generate(modelPath, args, System.out, System.err, null, "fast path");
    }

    private static int generate(
            Path modelPath,
            FastArgs args,
            PrintStream out,
            PrintStream err,
            GgufSessionCache sessionCache,
            String runnerName) throws Throwable {
        EngineMode engine = args.engineMode();
        if (engine == EngineMode.JAVA) {
            printJavaNativeProbe(modelPath, "Java-native GGUF loader");
            err.println("Java-native GGUF generation is not enabled yet; refusing to silently use llama.cpp. "
                    + "Use --engine benchmark to compare the Java loader with the llama.cpp fallback.");
            return COMMAND_ERROR;
        }
        if (engine == EngineMode.BENCHMARK) {
            out.println("GGUF engine benchmark: Java-native loader/probe vs llama.cpp generation fallback.");
            printJavaNativeProbe(modelPath, "Java-native GGUF loader");
        }
        generateWithLlamaCpp(modelPath, args, engine == EngineMode.BENCHMARK, out, err, sessionCache, runnerName);
        return 0;
    }

    private static void printJavaNativeProbe(Path modelPath, String label) {
        try {
            int probeRows = Math.max(1, Integer.getInteger("gollek.gguf.java_probe_rows", 16));
            int probeMatVecRows = Math.max(1, Integer.getInteger("gollek.gguf.java_probe_matvec_rows", 4096));
            GgufRuntimeProbe probe = GgufRuntimeProbe.load(modelPath, probeRows, probeMatVecRows);
            printProfile(label, probe.profile());
            printRowDotProbe(label, probe);
        } catch (Throwable throwable) {
            System.out.printf("%s: unavailable, reason=%s.%n", label, briefMessage(throwable));
            if (Boolean.getBoolean("gollek.gguf.fast_run.debug")) {
                throwable.printStackTrace(System.err);
            }
        }
    }

    private static void generateWithLlamaCpp(
            Path modelPath,
            FastArgs args,
            boolean benchmarkMode,
            PrintStream out,
            PrintStream err,
            GgufSessionCache sessionCache,
            String runnerName) throws Throwable {
        long startNanos = System.nanoTime();
        String requestedBackend = normalizedBackend(args.backend);
        int nGpuLayers = "cpu".equals(requestedBackend)
                ? 0
                : Integer.getInteger("gollek.gguf.fast_run.gpu_layers", -1);
        int threads = Integer.getInteger("gollek.gguf.fast_run.threads", defaultThreads());
        int batchThreads = Integer.getInteger("gollek.gguf.fast_run.batch_threads", threads);
        String prompt = formatPromptForModel(args.prompt, modelPath);
        int context = fastRunContext(prompt, args.maxTokens);
        int batch = boundedBatch("gollek.gguf.fast_run.batch", Math.min(context, 1024), context);
        int microBatch = boundedBatch("gollek.gguf.fast_run.ubatch", Math.min(batch, 512), batch);
        boolean swaFull = fastRunSwaFull();

        try {
            generateWithLlamaCppSession(
                    modelPath,
                    args,
                    benchmarkMode,
                    startNanos,
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    nGpuLayers,
                    swaFull,
                    prompt,
                    false,
                    out,
                    err,
                    sessionCache,
                    runnerName);
        } catch (Throwable throwable) {
            if (nGpuLayers == 0 || !fastRunCpuFallback()) {
                throw throwable;
            }
            err.printf("WARN: GGUF Metal fast path failed (%s); retrying with CPU fallback.%n",
                    briefMessage(throwable));
            generateWithLlamaCppSession(
                    modelPath,
                    args,
                    benchmarkMode,
                    startNanos,
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    0,
                    swaFull,
                    prompt,
                    true,
                    out,
                    err,
                    sessionCache,
                    runnerName);
        }
    }

    private static void generateWithLlamaCppSession(
            Path modelPath,
            FastArgs args,
            boolean benchmarkMode,
            long startNanos,
            int context,
            int batch,
            int microBatch,
            int threads,
            int batchThreads,
            int nGpuLayers,
            boolean swaFull,
            String prompt,
            boolean cpuFallback,
            PrintStream out,
            PrintStream err,
            GgufSessionCache sessionCache,
            String runnerName) throws Throwable {
        long openStartNanos = System.nanoTime();
        boolean useMmap = Boolean.parseBoolean(System.getProperty("gollek.gguf.fast_run.mmap", "true"));
        boolean useMlock = Boolean.parseBoolean(System.getProperty("gollek.gguf.fast_run.mlock", "false"));
        SessionLease opened = sessionCache == null
                ? SessionLease.open(modelPath, context, batch, microBatch, threads, batchThreads, nGpuLayers,
                        useMmap, useMlock, swaFull)
                : sessionCache.acquire(modelPath, context, batch, microBatch, threads, batchThreads, nGpuLayers,
                        useMmap, useMlock, swaFull);
        long openNanos = System.nanoTime() - openStartNanos;
        try (SessionLease lease = opened) {
            NativeGgufSession session = lease.session();
            String warmSuffix = sessionCache == null ? "" : ", warmSession=" + lease.reused();
            out.printf("Using llama.cpp GGUF %s for %s "
                            + "(backend=%s, nGpuLayers=%d, threads=%d, context=%d, batch=%d, ubatch=%d, "
                            + "swaFull=%s, cpuFallback=%s%s).%n",
                    runnerName,
                    modelPath.getFileName(),
                    session.backendName(),
                    nGpuLayers,
                    threads,
                    context,
                    batch,
                    microBatch,
                    swaFull,
                    cpuFallback,
                    warmSuffix);
            long generateStartNanos = System.nanoTime();
            String output = session.generate(prompt, args.maxTokens, args.temperature, args.topK, args.topP);
            long generateNanos = System.nanoTime() - generateStartNanos;
            out.print(output);
            long durationMs = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
            String label = benchmarkMode ? "llama.cpp fallback" : "Fast GGUF";
            out.printf("%n[%s, Duration: %.2fs]%n", label, durationMs / 1000.0d);
            if (fastRunTiming()) {
                String nativeMetrics = session.metrics().map(metrics -> ", native={" + metrics + "}").orElse("");
                err.printf("GGUF timing: open=%.3fms, generateCall=%.3fms%s%n",
                        openNanos / 1_000_000.0d,
                        generateNanos / 1_000_000.0d,
                        nativeMetrics);
            }
        }
    }

    private enum EngineMode {
        AUTO,
        JAVA,
        LLAMA_CPP,
        BENCHMARK;

        static EngineMode parse(String value) {
            String normalized = value == null ? "auto" : value.trim().toLowerCase(Locale.ROOT);
            return switch (normalized) {
                case "java", "java-native", "jvm" -> JAVA;
                case "llamacpp", "llama.cpp", "llama-cpp", "binding", "native" -> LLAMA_CPP;
                case "bench", "benchmark", "compare" -> BENCHMARK;
                default -> AUTO;
            };
        }

        static EngineMode effective(String provider, String engine) {
            EngineMode requested = parse(engine);
            if (requested != AUTO) {
                return requested;
            }
            String normalizedProvider = provider == null ? "" : provider.trim().toLowerCase(Locale.ROOT);
            return switch (normalizedProvider) {
                case "java", "java-native", "jvm" -> JAVA;
                case "llamacpp", "llama.cpp", "llama-cpp", "binding", "native" -> LLAMA_CPP;
                default -> AUTO;
            };
        }
    }

    private static void printProfile(String label, GgufRuntimeProfile profile) {
        System.out.printf(
                "%s: architecture=%s, ggufVersion=%d, tensors=%d, metadata=%d, size=%.2fGiB, "
                        + "load=%.2fs, knownTypes=%.1f%%, decoderTensors=%d/%d, types=%s, status=%s.%n",
                label,
                profile.architecture(),
                profile.ggufVersion(),
                profile.tensorCount(),
                profile.metadataCount(),
                profile.modelBytes() / 1024.0d / 1024.0d / 1024.0d,
                profile.loadMillis() / 1000.0d,
                profile.knownTensorTypeRatio() * 100.0d,
                profile.presentDecoderTensorCount(),
                profile.requiredDecoderTensorCount(),
                profile.compactTypeSummary(4),
                profile.javaStatus());
    }

    private static void printRowDotProbe(String label, GgufRuntimeProbe probe) {
        if (!probe.hasTensorProbe()) {
            System.out.printf("%s row-dot probe: unavailable, no supported matrix tensor.%n", label);
            return;
        }
        System.out.printf(
                "%s tensor probe: tensor=%s, type=%s, rows=%d, cols=%d, sampledRows=%d, "
                        + "dot=%.3fms, dotChecksum=%.6g, matVecRows=%d, cache=%.3fms, "
                        + "parallelMatVec=%.3fms, matVecChecksum=%.6g, cachedGenericMatVec=%.3fms, "
                        + "cachedChecksum=%.6g.%n",
                label,
                probe.tensorName(),
                probe.tensorType(),
                probe.rows(),
                probe.columns(),
                probe.sampledRows(),
                probe.rowDotMillis(),
                probe.rowDotChecksum(),
                probe.matVecRows(),
                probe.matrixCacheMillis(),
                probe.matVecMillis(),
                probe.matVecChecksum(),
                probe.cachedMatVecMillis(),
                probe.cachedMatVecChecksum());
    }

    static String formatPromptForModel(String prompt, Path modelPath) {
        String mode = System.getProperty("gollek.gguf.fast_run.prompt_format", "auto")
                .trim()
                .toLowerCase(Locale.ROOT);
        if (mode.equals("raw") || mode.equals("none") || prompt == null || prompt.isBlank()) {
            return prompt;
        }
        if (prompt.contains("<|turn>") || prompt.contains("<start_of_turn>")) {
            return prompt;
        }
        boolean gemma4 = mode.equals("gemma4") || (mode.equals("auto") && looksLikeGemma4(modelPath));
        if (!gemma4) {
            return prompt;
        }
        String userPrompt = prompt;
        if (useConciseQuestionInstruction(prompt)) {
            userPrompt = "Answer directly and concisely.\nQuestion: " + prompt;
        }
        return "<|turn>user\n" + userPrompt + "<turn|>\n<|turn>model\n";
    }

    static String effectiveEngineModeName(String provider, String engine) {
        return EngineMode.effective(provider, engine).name();
    }

    private static boolean looksLikeGemma4(Path modelPath) {
        if (modelPath == null || modelPath.getFileName() == null) {
            return false;
        }
        String fileName = modelPath.getFileName().toString().toLowerCase(Locale.ROOT);
        return fileName.contains("gemma-4") || fileName.contains("gemma4");
    }

    private static boolean useConciseQuestionInstruction(String prompt) {
        if (!Boolean.parseBoolean(System.getProperty("gollek.gguf.fast_run.concise_qa_prompt", "true"))) {
            return false;
        }
        if (prompt == null) {
            return false;
        }
        String trimmed = prompt.trim();
        if (trimmed.isEmpty() || trimmed.getBytes(StandardCharsets.UTF_8).length > 256) {
            return false;
        }
        String normalized = trimmed.toLowerCase(Locale.ROOT);
        return normalized.endsWith("?")
                || normalized.startsWith("where ")
                || normalized.startsWith("what ")
                || normalized.startsWith("who ")
                || normalized.startsWith("when ")
                || normalized.startsWith("why ")
                || normalized.startsWith("how ")
                || normalized.startsWith("which ");
    }

    private static String normalizedBackend(String backend) {
        if (backend == null || backend.isBlank()) {
            return "metal";
        }
        String normalized = backend.trim().toLowerCase(Locale.ROOT);
        if (normalized.equals("gpu")) {
            return "metal";
        }
        return normalized;
    }

    private static int defaultThreads() {
        int cpus = Runtime.getRuntime().availableProcessors();
        return Math.max(1, Math.min(8, cpus));
    }

    static int boundedBatch(String property, int defaultValue, int maxValue) {
        int requested = Integer.getInteger(property, defaultValue);
        int upperBound = Math.max(1, maxValue);
        return Math.max(1, Math.min(requested, upperBound));
    }

    static int fastRunContext(String prompt, int maxTokens) {
        String configured = System.getProperty("gollek.gguf.fast_run.context");
        if (configured != null && !configured.isBlank() && !configured.trim().equalsIgnoreCase("auto")) {
            return parsePositiveInt(configured, 2048);
        }

        int maxAutoContext = parsePositiveInt(System.getProperty("gollek.gguf.fast_run.max_auto_context"), 2048);
        int promptBytes = prompt == null ? 0 : prompt.getBytes(StandardCharsets.UTF_8).length;
        int requested = (int) Math.min(Integer.MAX_VALUE, (long) promptBytes + Math.max(1, maxTokens) + 128L);
        return nextPowerOfTwoBounded(requested, 512, maxAutoContext);
    }

    private static int parsePositiveInt(String value, int fallback) {
        if (value == null || value.isBlank()) {
            return Math.max(1, fallback);
        }
        try {
            return Math.max(1, Integer.parseInt(value.trim()));
        } catch (NumberFormatException ignored) {
            return Math.max(1, fallback);
        }
    }

    private static int nextPowerOfTwoBounded(int value, int minimum, int maximum) {
        int lowerBound = Math.max(1, minimum);
        int upperBound = Math.max(lowerBound, maximum);
        int context = lowerBound;
        int target = Math.max(lowerBound, value);
        while (context < target && context < upperBound) {
            context = Math.min(context << 1, upperBound);
        }
        return context;
    }

    static boolean fastRunSwaFull() {
        return Boolean.parseBoolean(System.getProperty("gollek.gguf.fast_run.swa_full", "false"));
    }

    static boolean fastRunCpuFallback() {
        return Boolean.parseBoolean(System.getProperty("gollek.gguf.fast_run.cpu_fallback", "true"));
    }

    static boolean fastRunTiming() {
        Boolean requestTiming = REQUEST_TIMING.get();
        if (requestTiming != null) {
            return requestTiming;
        }
        return Boolean.parseBoolean(firstNonBlank(
                System.getProperty("gollek.gguf.fast_run.timing"),
                System.getenv("GOLLEK_GGUF_FAST_RUN_TIMING"),
                "false"));
    }

    private static boolean daemonEnabled() {
        return Boolean.parseBoolean(System.getProperty("gollek.gguf.fast_run.daemon", "false"));
    }

    private record SessionKey(
            Path modelPath,
            int context,
            int batch,
            int microBatch,
            int threads,
            int batchThreads,
            int gpuLayers,
            boolean useMmap,
            boolean useMlock,
            boolean swaFull) {
    }

    private record SessionLease(NativeGgufSession session, boolean reused, boolean closeOnClose) implements AutoCloseable {
        static SessionLease open(
                Path modelPath,
                int context,
                int batch,
                int microBatch,
                int threads,
                int batchThreads,
                int gpuLayers,
                boolean useMmap,
                boolean useMlock,
                boolean swaFull) throws Throwable {
            return new SessionLease(NativeGgufSession.open(
                    modelPath,
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    gpuLayers,
                    useMmap,
                    useMlock,
                    swaFull), false, true);
        }

        @Override
        public void close() {
            if (closeOnClose) {
                session.close();
            }
        }
    }

    private static final class GgufSessionCache implements AutoCloseable {
        private final Map<SessionKey, NativeGgufSession> sessions = new HashMap<>();

        private SessionLease acquire(
                Path modelPath,
                int context,
                int batch,
                int microBatch,
                int threads,
                int batchThreads,
                int gpuLayers,
                boolean useMmap,
                boolean useMlock,
                boolean swaFull) throws Throwable {
            SessionKey key = new SessionKey(
                    modelPath.toAbsolutePath().normalize(),
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    gpuLayers,
                    useMmap,
                    useMlock,
                    swaFull);
            NativeGgufSession existing = sessions.get(key);
            if (existing != null) {
                daemonLog("session cache hit: " + key.modelPath().getFileName()
                        + ", context=" + context
                        + ", gpuLayers=" + gpuLayers
                        + ", sessions=" + sessions.size());
                return new SessionLease(existing, true, false);
            }
            daemonLog("session cache miss: " + key.modelPath().getFileName()
                    + ", context=" + context
                    + ", gpuLayers=" + gpuLayers
                    + ", sessions=" + sessions.size());
            NativeGgufSession opened = NativeGgufSession.open(
                    key.modelPath(),
                    context,
                    batch,
                    microBatch,
                    threads,
                    batchThreads,
                    gpuLayers,
                    useMmap,
                    useMlock,
                    swaFull);
            sessions.put(key, opened);
            return new SessionLease(opened, false, false);
        }

        @Override
        public void close() {
            for (NativeGgufSession session : sessions.values()) {
                session.close();
            }
            sessions.clear();
        }
    }

    private static final class NativeGgufSession implements AutoCloseable {
        private final MemorySegment handle;
        private final String backendName;

        private NativeGgufSession(MemorySegment handle, String backendName) {
            this.handle = handle;
            this.backendName = backendName;
        }

        private static NativeGgufSession open(
                Path modelPath,
                int context,
                int batch,
                int microBatch,
                int threads,
                int batchThreads,
                int gpuLayers,
                boolean useMmap,
                boolean useMlock,
                boolean swaFull) throws Throwable {
            NativeGgufBinding binding = NativeGgufBinding.INSTANCE;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment model = arena.allocateFrom(modelPath.toString(), StandardCharsets.UTF_8);
                MemorySegment backendDir = arena.allocateFrom(llamaLibDir().toString(), StandardCharsets.UTF_8);
                MemorySegment error = arena.allocate(8192);
                MemorySegment handle = (MemorySegment) binding.open.invokeExact(
                        model,
                        backendDir,
                        context,
                        batch,
                        microBatch,
                        threads,
                        batchThreads,
                        gpuLayers,
                        useMmap ? 1 : 0,
                        useMlock ? 1 : 0,
                        swaFull ? 1 : 0,
                        error,
                        8192L);
                if (isNull(handle)) {
                    throw new IllegalStateException(error.getString(0));
                }
                String backendName = "unknown";
                MemorySegment backendPtr = (MemorySegment) binding.backendName.invokeExact(handle);
                if (!isNull(backendPtr)) {
                    backendName = backendPtr.reinterpret(128).getString(0);
                }
                return new NativeGgufSession(handle, backendName);
            }
        }

        private String generate(String prompt, int maxTokens, double temperature, int topK, double topP) throws Throwable {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment promptSegment = arena.allocateFrom(prompt, StandardCharsets.UTF_8);
                int outputBytes = Math.max(64 * 1024, Math.max(1, maxTokens) * 512);
                MemorySegment output = arena.allocate(outputBytes);
                MemorySegment error = arena.allocate(8192);
                int generated = (int) NativeGgufBinding.INSTANCE.generate.invokeExact(
                        handle,
                        promptSegment,
                        maxTokens,
                        (float) temperature,
                        topK,
                        (float) topP,
                        0,
                        output,
                        (long) outputBytes,
                        error,
                        8192L);
                if (generated < 0) {
                    throw new IllegalStateException(error.getString(0));
                }
                return output.getString(0, StandardCharsets.UTF_8);
            }
        }

        private String backendName() {
            return backendName;
        }

        private Optional<String> metrics() throws Throwable {
            NativeGgufBinding binding = NativeGgufBinding.INSTANCE;
            if (binding.metrics == null) {
                return Optional.empty();
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment output = arena.allocate(1024);
                int written = (int) binding.metrics.invokeExact(handle, output, 1024L);
                if (written <= 0) {
                    return Optional.empty();
                }
                return Optional.of(output.getString(0, StandardCharsets.UTF_8));
            }
        }

        @Override
        public void close() {
            try {
                NativeGgufBinding.INSTANCE.close.invokeExact(handle);
            } catch (Throwable ignored) {
            }
        }
    }

    private static Path llamaLibDir() {
        String configured = System.getProperty("gollek.gguf.fast_run.llama_lib_dir");
        if (configured == null || configured.isBlank()) {
            configured = System.getenv("GOLLEK_LLAMA_LIB_DIR");
        }
        if (configured != null && !configured.isBlank()) {
            return Path.of(configured).toAbsolutePath();
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "libs", "llama").toAbsolutePath();
    }

    private static Path bridgePath() {
        String configured = System.getProperty("gollek.gguf.fast_run.bridge");
        if (configured == null || configured.isBlank()) {
            configured = System.getenv("GOLLEK_GGUF_FAST_BRIDGE");
        }
        if (configured != null && !configured.isBlank()) {
            return Path.of(configured).toAbsolutePath();
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "libs", "libgollek_gguf_fast.dylib").toAbsolutePath();
    }

    private static boolean isNull(MemorySegment segment) {
        return segment == null || segment.equals(MemorySegment.NULL) || segment.address() == 0L;
    }

    private static String briefMessage(Throwable throwable) {
        Throwable cursor = throwable;
        while (cursor.getCause() != null) {
            cursor = cursor.getCause();
        }
        String message = cursor.getMessage();
        if (message == null || message.isBlank()) {
            return cursor.getClass().getSimpleName();
        }
        return message.replace('\n', ' ').replace('\r', ' ');
    }

    private static final class NativeGgufBinding {
        private static final NativeGgufBinding INSTANCE = new NativeGgufBinding();

        private final MethodHandle open;
        private final MethodHandle generate;
        private final MethodHandle backendName;
        private final MethodHandle metrics;
        private final MethodHandle hardExit;
        private final MethodHandle close;

        private NativeGgufBinding() {
            try {
                Path bridge = bridgePath();
                if (!Files.isRegularFile(bridge)) {
                    throw new IOException("GGUF fast bridge is not installed at " + bridge);
                }
                System.load(bridge.toString());
                Linker linker = Linker.nativeLinker();
                SymbolLookup lookup = SymbolLookup.libraryLookup(bridge, Arena.global());
                this.open = linker.downcallHandle(
                        lookup.find("gollek_gguf_open").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_LONG));
                this.generate = linker.downcallHandle(
                        lookup.find("gollek_gguf_generate").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_FLOAT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_FLOAT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_LONG,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_LONG));
                this.backendName = linker.downcallHandle(
                        lookup.find("gollek_gguf_backend_name").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
                Optional<MemorySegment> metricsSymbol = lookup.find("gollek_gguf_last_metrics");
                this.metrics = metricsSymbol.isPresent()
                        ? linker.downcallHandle(
                                metricsSymbol.get(),
                                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                                        ValueLayout.ADDRESS,
                                        ValueLayout.ADDRESS,
                                        ValueLayout.JAVA_LONG))
                        : null;
                Optional<MemorySegment> hardExitSymbol = lookup.find("gollek_gguf_hard_exit");
                this.hardExit = hardExitSymbol.isPresent()
                        ? linker.downcallHandle(
                                hardExitSymbol.get(),
                                FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT))
                        : null;
                this.close = linker.downcallHandle(
                        lookup.find("gollek_gguf_close").orElseThrow(),
                        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            } catch (Throwable throwable) {
                throw new ExceptionInInitializerError(throwable);
            }
        }
    }

    private static OptionalInt requestDaemon(String[] rawArgs) {
        OptionalInt status = sendDaemonRequest(rawArgs);
        if (status.isPresent()) {
            return status;
        }
        if (!startDaemon()) {
            return OptionalInt.empty();
        }
        return sendDaemonRequest(rawArgs);
    }

    private static OptionalInt sendDaemonRequest(String[] rawArgs) {
        OptionalInt port = readDaemonPort();
        if (port.isEmpty()) {
            return OptionalInt.empty();
        }
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getLoopbackAddress(), port.getAsInt()),
                    daemonConnectTimeoutMillis());
            socket.setSoTimeout((int) Math.min(Integer.MAX_VALUE, timeoutMillis()));
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            DataInputStream in = new DataInputStream(socket.getInputStream());
            writeString(out, DAEMON_MAGIC);
            out.writeInt(rawArgs.length);
            for (String arg : rawArgs) {
                writeString(out, arg);
            }
            out.writeBoolean(fastRunTiming());
            out.flush();
            while (true) {
                int channel = in.read();
                if (channel < 0) {
                    return OptionalInt.empty();
                }
                int length = in.readInt();
                if (channel == 0) {
                    return OptionalInt.of(length);
                }
                byte[] payload = in.readNBytes(length);
                PrintStream target = channel == 2 ? System.err : System.out;
                target.write(payload);
                target.flush();
            }
        } catch (ConnectException ignored) {
            deleteDaemonPortFile();
            return OptionalInt.empty();
        } catch (Exception e) {
            if (Boolean.getBoolean("gollek.gguf.fast_run.debug")) {
                System.err.println("GGUF daemon request failed: " + briefMessage(e));
            }
            return OptionalInt.empty();
        }
    }

    private static boolean startDaemon() {
        deleteDaemonPortFile();
        try {
            Files.createDirectories(daemonRunDir());
            Files.createDirectories(daemonLogFile().getParent());
            List<String> command = new ArrayList<>();
            command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
            command.add("-Xmx" + System.getProperty("gollek.gguf.fast_run.daemon_heap", "1g"));
            command.add("-XX:MaxDirectMemorySize=" + System.getProperty("gollek.gguf.fast_run.daemon_direct_memory", "8g"));
            command.add("--enable-preview");
            command.add("--add-modules");
            command.add("jdk.incubator.vector");
            command.add("--enable-native-access=ALL-UNNAMED");
            copyGollekSystemProperties(command);
            command.add("-Djava.library.path=" + System.getProperty("java.library.path", ""));
            command.add("-cp");
            command.add(System.getProperty("java.class.path"));
            command.add(GgufFastRun.class.getName());
            command.add("__daemon");

            if (!startDetachedDaemon(command)) {
                return false;
            }

            long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(daemonStartTimeoutMillis());
            while (System.nanoTime() < deadlineNanos) {
                if (readDaemonPort().isPresent()) {
                    return true;
                }
                Thread.sleep(100L);
            }
            return readDaemonPort().isPresent();
        } catch (Exception e) {
            if (Boolean.getBoolean("gollek.gguf.fast_run.debug")) {
                System.err.println("GGUF daemon failed to start: " + briefMessage(e));
            }
            return false;
        }
    }

    private static boolean startDetachedDaemon(List<String> command) throws IOException, InterruptedException {
        String launcher = System.getProperty("gollek.gguf.fast_run.daemon_launcher", "auto")
                .trim()
                .toLowerCase(Locale.ROOT);
        if (launcher.equals("launchctl")) {
            return isMacOs() && startDetachedDaemonWithLaunchctl(command);
        }
        if (launcher.equals("auto") && isMacOs() && startDetachedDaemonWithLaunchctl(command)) {
            return true;
        }
        return startDetachedDaemonWithNohup(command);
    }

    private static boolean startDetachedDaemonWithLaunchctl(List<String> command) throws IOException, InterruptedException {
        String launchctl = "/bin/launchctl";
        if (!Files.isExecutable(Path.of(launchctl))) {
            return false;
        }
        removeLaunchctlDaemon();
        Process launcher = new ProcessBuilder(
                launchctl,
                "submit",
                "-l",
                launchctlLabel(),
                "--",
                "/bin/sh",
                "-c",
                daemonShellCommand(command, false))
                .redirectOutput(ProcessBuilder.Redirect.appendTo(daemonLogFile().toFile()))
                .redirectError(ProcessBuilder.Redirect.appendTo(daemonLogFile().toFile()))
                .start();
        if (!launcher.waitFor(3, TimeUnit.SECONDS)) {
            return true;
        }
        boolean launched = launcher.exitValue() == 0;
        if (!launched && Boolean.getBoolean("gollek.gguf.fast_run.debug")) {
            System.err.println("GGUF launchctl daemon submit failed; falling back to nohup");
        }
        return launched;
    }

    private static boolean startDetachedDaemonWithNohup(List<String> command) throws IOException, InterruptedException {
        Process launcher = new ProcessBuilder("/bin/sh", "-c", daemonShellCommand(command, true)).start();
        launcher.waitFor(2, TimeUnit.SECONDS);
        return true;
    }

    private static String daemonShellCommand(List<String> command, boolean useNohup) {
        String home = System.getProperty("user.home");
        String nativePaths = home + "/.gollek/libs" + File.pathSeparator + home + "/.gollek/libs/llama";
        StringBuilder shellCommand = new StringBuilder();
        shellCommand.append("export DYLD_LIBRARY_PATH=")
                .append(shellQuote(nativePaths))
                .append("${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}; ");
        shellCommand.append("export DYLD_FALLBACK_LIBRARY_PATH=")
                .append(shellQuote(nativePaths))
                .append("${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}; ");
        if (useNohup) {
            shellCommand.append("nohup ");
        } else {
            shellCommand.append("exec ");
        }
        for (int i = 0; i < command.size(); i++) {
            if (i > 0 || useNohup) {
                shellCommand.append(' ');
            }
            shellCommand.append(shellQuote(command.get(i)));
        }
        shellCommand.append(" >> ").append(shellQuote(daemonLogFile().toString()))
                .append(" 2>&1 < /dev/null");
        if (useNohup) {
            shellCommand.append(" &");
        }
        return shellCommand.toString();
    }

    private static int runDaemon() {
        int port = -1;
        try {
            Files.createDirectories(daemonRunDir());
            try (ServerSocket server = new ServerSocket(0, 16, InetAddress.getLoopbackAddress());
                    GgufSessionCache sessionCache = new GgufSessionCache()) {
                port = server.getLocalPort();
                writeDaemonPort(port);
                long idleNanos = TimeUnit.SECONDS.toNanos(
                        Long.getLong("gollek.gguf.fast_run.daemon_idle_seconds", 900L));
                long lastRequestNanos = System.nanoTime();
                server.setSoTimeout(1000);
                daemonLog("daemon started: pid=" + ProcessHandle.current().pid()
                        + ", port=" + port
                        + ", idleSeconds=" + TimeUnit.NANOSECONDS.toSeconds(idleNanos));
                boolean stop = false;
                while (!stop) {
                    try (Socket socket = server.accept()) {
                        lastRequestNanos = System.nanoTime();
                        stop = handleDaemonClient(socket, sessionCache);
                    } catch (SocketTimeoutException ignored) {
                        if (System.nanoTime() - lastRequestNanos > idleNanos) {
                            daemonLog("daemon idle timeout");
                            break;
                        }
                    }
                }
                daemonLog("daemon stopping");
                if (daemonHardExit()) {
                    hardExitDaemon(0, port);
                }
            }
            return 0;
        } catch (Throwable throwable) {
            System.err.println("GGUF daemon failed: " + briefMessage(throwable));
            if (Boolean.getBoolean("gollek.gguf.fast_run.debug")) {
                throwable.printStackTrace(System.err);
            }
            return COMMAND_ERROR;
        } finally {
            deleteDaemonPortFileIfPort(port);
        }
    }

    private static boolean handleDaemonClient(Socket socket, GgufSessionCache sessionCache) {
        try {
            socket.setSoTimeout((int) Math.min(Integer.MAX_VALUE, timeoutMillis()));
            DataInputStream in = new DataInputStream(socket.getInputStream());
            DataOutputStream framed = new DataOutputStream(socket.getOutputStream());
            String magic = readString(in);
            if (DAEMON_STOP_MAGIC.equals(magic)) {
                writeStatusFrame(framed, 0);
                daemonLog("daemon stop requested");
                return true;
            }
            if (!DAEMON_MAGIC.equals(magic)) {
                writeStatusFrame(framed, FALLBACK_TO_FULL_CLI);
                return false;
            }
            int argCount = in.readInt();
            if (argCount < 0 || argCount > 512) {
                throw new IOException("Invalid GGUF daemon arg count: " + argCount);
            }
            String[] rawArgs = new String[argCount];
            for (int i = 0; i < argCount; i++) {
                rawArgs[i] = readString(in);
            }
            boolean requestTiming = in.readBoolean();

            PrintStream out = new PrintStream(new FramedOutputStream(framed, 1), true, StandardCharsets.UTF_8);
            PrintStream err = new PrintStream(new FramedOutputStream(framed, 2), true, StandardCharsets.UTF_8);
            int status;
            REQUEST_TIMING.set(requestTiming);
            try {
                FastArgs parsed = FastArgs.parse(rawArgs);
                if (!parsed.supported()) {
                    status = FALLBACK_TO_FULL_CLI;
                } else {
                    Optional<Path> modelPath = resolveGgufModel(parsed);
                    if (modelPath.isEmpty()) {
                        status = FALLBACK_TO_FULL_CLI;
                    } else {
                        status = generate(modelPath.get(), parsed, out, err, sessionCache, "daemon");
                    }
                }
            } catch (Throwable throwable) {
                err.println("GGUF daemon request failed: " + briefMessage(throwable));
                if (Boolean.getBoolean("gollek.gguf.fast_run.debug")) {
                    throwable.printStackTrace(err);
                }
                status = COMMAND_ERROR;
            } finally {
                REQUEST_TIMING.remove();
            }
            out.flush();
            err.flush();
            writeStatusFrame(framed, status);
            return false;
        } catch (Exception e) {
            if (Boolean.getBoolean("gollek.gguf.fast_run.debug")) {
                System.err.println("GGUF daemon client failed: " + briefMessage(e));
            }
            return false;
        }
    }

    private static int stopDaemon() {
        OptionalInt port = readDaemonPort();
        if (port.isEmpty()) {
            removeLaunchctlDaemon();
            return 0;
        }
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getLoopbackAddress(), port.getAsInt()),
                    daemonConnectTimeoutMillis());
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            DataInputStream in = new DataInputStream(socket.getInputStream());
            writeString(out, DAEMON_STOP_MAGIC);
            out.flush();
            int channel = in.read();
            int status = in.readInt();
            return channel == 0 ? status : COMMAND_ERROR;
        } catch (Exception ignored) {
            return 0;
        } finally {
            deleteDaemonPortFile();
            removeLaunchctlDaemon();
        }
    }

    private static void copyGollekSystemProperties(List<String> command) {
        Properties properties = System.getProperties();
        for (String name : properties.stringPropertyNames()) {
            if (name.startsWith("gollek.gguf.fast_run.")) {
                command.add("-D" + name + "=" + properties.getProperty(name));
            }
        }
    }

    private static Path daemonRunDir() {
        return Path.of(System.getProperty("user.home"), ".gollek", "run");
    }

    private static Path daemonPortFile() {
        return daemonRunDir().resolve("gguf-fast-daemon.port");
    }

    private static Path daemonLogFile() {
        return Path.of(System.getProperty("user.home"), ".gollek", "logs", "gguf-fast-daemon.log");
    }

    private static OptionalInt readDaemonPort() {
        Optional<DaemonInfo> info = readDaemonInfo();
        if (info.isEmpty()) {
            return OptionalInt.empty();
        }
        if (!daemonKey().equals(info.get().key()) || !isProcessAlive(info.get().pid())) {
            discardStaleDaemon(info.get());
            return OptionalInt.empty();
        }
        return OptionalInt.of(info.get().port());
    }

    private static Optional<DaemonInfo> readDaemonInfo() {
        try {
            Path portFile = daemonPortFile();
            if (!Files.isRegularFile(portFile)) {
                return Optional.empty();
            }
            List<String> lines = Files.readAllLines(portFile, StandardCharsets.UTF_8);
            if (lines.size() < 3) {
                return Optional.empty();
            }
            int port = Integer.parseInt(lines.get(0).trim());
            long pid = Long.parseLong(lines.get(1).trim());
            String key = lines.get(2);
            return port > 0 && port <= 65535
                    ? Optional.of(new DaemonInfo(port, pid, key))
                    : Optional.empty();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static void writeDaemonPort(int port) throws IOException {
        Files.writeString(daemonPortFile(),
                port + System.lineSeparator()
                        + ProcessHandle.current().pid() + System.lineSeparator()
                        + daemonKey() + System.lineSeparator(),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static String daemonKey() {
        return classPathKey(System.getProperty("java.class.path", ""))
                + "|" + pathKey(bridgePath())
                + "|" + pathKey(llamaLibDir());
    }

    private static String classPathKey(String classPath) {
        if (classPath == null || classPath.isBlank()) {
            return "";
        }
        String[] entries = classPath.split(java.util.regex.Pattern.quote(File.pathSeparator));
        StringBuilder key = new StringBuilder();
        for (int i = 0; i < entries.length; i++) {
            if (i > 0) {
                key.append(File.pathSeparator);
            }
            String entry = entries[i];
            if (entry == null || entry.isBlank()) {
                continue;
            }
            Path path = Path.of(entry);
            key.append(Files.exists(path) ? pathKey(path) : entry);
        }
        return key.toString();
    }

    private static String pathKey(Path path) {
        try {
            long modified = Files.exists(path) ? Files.getLastModifiedTime(path).toMillis() : -1L;
            long size = Files.isRegularFile(path) ? Files.size(path) : -1L;
            return path.toAbsolutePath().normalize() + ":size=" + size + ":mtime=" + modified;
        } catch (Exception ignored) {
            return path.toAbsolutePath().normalize().toString();
        }
    }

    private static boolean isProcessAlive(long pid) {
        if (pid <= 0 || pid == ProcessHandle.current().pid()) {
            return false;
        }
        try {
            Optional<ProcessHandle> handle = ProcessHandle.of(pid);
            return handle.isPresent() && handle.get().isAlive();
        } catch (Exception ignored) {
            return false;
        }
    }

    private static void discardStaleDaemon(DaemonInfo info) {
        deleteDaemonPortFile();
        removeLaunchctlDaemon();
        if (!isProcessAlive(info.pid())) {
            return;
        }
        try {
            ProcessHandle process = ProcessHandle.of(info.pid()).orElse(null);
            if (process == null || !process.isAlive()) {
                return;
            }
            process.destroy();
            try {
                process.onExit().get(2, TimeUnit.SECONDS);
            } catch (Exception ignored) {
                process.destroyForcibly();
            }
        } catch (Exception ignored) {
        }
    }

    private static void daemonLog(String message) {
        System.err.printf("%tF %<tT [gguf-daemon] %s%n", System.currentTimeMillis(), message);
    }

    private static boolean daemonHardExit() {
        return Boolean.parseBoolean(System.getProperty("gollek.gguf.fast_run.daemon_hard_exit", "true"));
    }

    private static void hardExitDaemon(int status, int port) {
        deleteDaemonPortFileIfPort(port);
        System.out.flush();
        System.err.flush();
        try {
            MethodHandle hardExit = NativeGgufBinding.INSTANCE.hardExit;
            if (hardExit != null) {
                hardExit.invokeExact(status);
            }
        } catch (Throwable ignored) {
        }
        Runtime.getRuntime().halt(status);
    }

    private static boolean isMacOs() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac");
    }

    private static String launchctlLabel() {
        return System.getProperty("gollek.gguf.fast_run.daemon_launchctl_label",
                "tech.kayys.gollek.gguf-fast-daemon");
    }

    private static void removeLaunchctlDaemon() {
        if (!isMacOs()) {
            return;
        }
        try {
            new ProcessBuilder("/bin/launchctl", "remove", launchctlLabel())
                    .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                    .redirectError(ProcessBuilder.Redirect.DISCARD)
                    .start()
                    .waitFor(2, TimeUnit.SECONDS);
        } catch (Exception ignored) {
        }
    }

    private static String shellQuote(String value) {
        return "'" + value.replace("'", "'\\''") + "'";
    }

    private static void deleteDaemonPortFile() {
        try {
            Files.deleteIfExists(daemonPortFile());
        } catch (Exception ignored) {
        }
    }

    private static void deleteDaemonPortFileIfPort(int port) {
        if (port <= 0) {
            return;
        }
        OptionalInt current = readDaemonPort();
        if (current.isPresent() && current.getAsInt() == port) {
            deleteDaemonPortFile();
        }
    }

    private static int daemonConnectTimeoutMillis() {
        return Integer.getInteger("gollek.gguf.fast_run.daemon_connect_timeout_ms", 250);
    }

    private static int daemonStartTimeoutMillis() {
        return Integer.getInteger("gollek.gguf.fast_run.daemon_start_timeout_ms", 20_000);
    }

    private static long timeoutMillis() {
        return TimeUnit.SECONDS.toMillis(Long.getLong("gollek.gguf.fast_run.timeout_seconds", 180L));
    }

    private static void writeStatusFrame(DataOutputStream out, int status) throws IOException {
        synchronized (out) {
            out.write(0);
            out.writeInt(status);
            out.flush();
        }
    }

    private static void writeFrame(DataOutputStream out, int channel, byte[] payload) throws IOException {
        if (payload.length == 0) {
            return;
        }
        synchronized (out) {
            out.write(channel);
            out.writeInt(payload.length);
            out.write(payload);
            out.flush();
        }
    }

    private static void writeString(DataOutputStream out, String value) throws IOException {
        byte[] bytes = (value == null ? "" : value).getBytes(StandardCharsets.UTF_8);
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    private static String readString(DataInputStream in) throws IOException {
        int length = in.readInt();
        if (length < 0 || length > 1_048_576) {
            throw new IOException("Invalid GGUF daemon string length: " + length);
        }
        byte[] bytes = in.readNBytes(length);
        if (bytes.length != length) {
            throw new IOException("Unexpected end of GGUF daemon string");
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private record DaemonInfo(int port, long pid, String key) {
    }

    private static final class FramedOutputStream extends OutputStream {
        private final DataOutputStream target;
        private final int channel;
        private final ByteArrayOutputStream buffer = new ByteArrayOutputStream();

        private FramedOutputStream(DataOutputStream target, int channel) {
            this.target = target;
            this.channel = channel;
        }

        @Override
        public void write(int b) {
            buffer.write(b);
        }

        @Override
        public void write(byte[] b, int off, int len) {
            buffer.write(b, off, len);
        }

        @Override
        public void flush() throws IOException {
            byte[] payload = buffer.toByteArray();
            buffer.reset();
            writeFrame(target, channel, payload);
        }
    }

    private static final class FastArgs {
        private String model;
        private String modelFile;
        private String prompt;
        private int maxTokens = 256;
        private double temperature = 0.2d;
        private double topP = 0.9d;
        private int topK = 40;
        private String provider;
        private String backend = System.getProperty("gollek.gguf.fast_run.backend", "metal");
        private String engine = firstNonBlank(
                System.getProperty("gollek.gguf.fast_run.engine"),
                System.getenv("GOLLEK_GGUF_ENGINE"),
                "auto");
        private boolean supported = true;

        private boolean supported() {
            if (!supported || prompt == null || prompt.isBlank()) {
                return false;
            }
            if ((model == null || model.isBlank()) && (modelFile == null || modelFile.isBlank())) {
                return false;
            }
            if (provider == null || provider.isBlank()) {
                return true;
            }
            String normalized = provider.toLowerCase(Locale.ROOT);
            return normalized.equals("gguf")
                    || normalized.equals("native")
                    || normalized.equals("llamacpp")
                    || normalized.equals("llama.cpp")
                    || normalized.equals("llama-cpp")
                    || normalized.equals("java")
                    || normalized.equals("java-native")
                    || normalized.equals("jvm");
        }

        private EngineMode engineMode() {
            return EngineMode.effective(provider, engine);
        }

        private static FastArgs parse(String[] args) {
            FastArgs parsed = new FastArgs();
            if (args.length == 0 || !"run".equals(args[0])) {
                parsed.supported = false;
                return parsed;
            }
            for (int i = 1; i < args.length; i++) {
                String arg = args[i];
                String value = null;
                int eq = arg.indexOf('=');
                if (arg.startsWith("--") && eq > 0) {
                    value = arg.substring(eq + 1);
                    arg = arg.substring(0, eq);
                }
                switch (arg) {
                    case "--model", "-m" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.model = next.value();
                        i = next.index();
                    }
                    case "--modelFile", "--model-file", "--model-path" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.modelFile = next.value();
                        i = next.index();
                    }
                    case "--prompt", "-p" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.prompt = next.value();
                        i = next.index();
                    }
                    case "--provider" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.provider = next.value();
                        i = next.index();
                    }
                    case "--backend", "--platform" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.backend = next.value();
                        i = next.index();
                    }
                    case "--gguf-engine", "--engine" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.engine = next.value();
                        i = next.index();
                    }
                    case "--java-native" -> parsed.engine = "java";
                    case "--llamacpp", "--llama-cpp" -> parsed.engine = "llamacpp";
                    case "--benchmark", "--bench" -> parsed.engine = "benchmark";
                    case "--use-cpu" -> parsed.backend = "cpu";
                    case "--max-tokens" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.maxTokens = parseInt(next.value(), parsed.maxTokens);
                        i = next.index();
                    }
                    case "--temperature" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.temperature = parseDouble(next.value(), parsed.temperature);
                        i = next.index();
                    }
                    case "--top-p" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.topP = parseDouble(next.value(), parsed.topP);
                        i = next.index();
                    }
                    case "--top-k" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.topK = parseInt(next.value(), parsed.topK);
                        i = next.index();
                    }
                    default -> {
                        if (arg.startsWith("--")) {
                            parsed.supported = false;
                        }
                    }
                }
            }
            parsed.maxTokens = Math.max(1, parsed.maxTokens);
            parsed.topK = Math.max(1, parsed.topK);
            parsed.topP = Math.max(0.0d, Math.min(1.0d, parsed.topP));
            parsed.temperature = Math.max(0.0d, parsed.temperature);
            return parsed;
        }

        private static Value valueOrNext(String value, String[] args, int index) {
            if (value != null) {
                return new Value(value, index);
            }
            if (index + 1 >= args.length) {
                return new Value("", index);
            }
            return new Value(args[index + 1], index + 1);
        }

        private static int parseInt(String value, int fallback) {
            try {
                return Integer.parseInt(value);
            } catch (Exception ignored) {
                return fallback;
            }
        }

        private static double parseDouble(String value, double fallback) {
            try {
                return Double.parseDouble(value);
            } catch (Exception ignored) {
                return fallback;
            }
        }

        private record Value(String value, int index) {
        }
    }

    private static String firstNonBlank(String first, String second, String fallback) {
        if (first != null && !first.isBlank()) {
            return first;
        }
        if (second != null && !second.isBlank()) {
            return second;
        }
        return fallback;
    }
}
