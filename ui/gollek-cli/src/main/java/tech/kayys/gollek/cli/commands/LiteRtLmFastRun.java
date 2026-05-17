package tech.kayys.gollek.cli.commands;

import com.google.ai.edge.litertlm.Backend;
import com.google.ai.edge.litertlm.Conversation;
import com.google.ai.edge.litertlm.ConversationConfig;
import com.google.ai.edge.litertlm.Contents;
import com.google.ai.edge.litertlm.Engine;
import com.google.ai.edge.litertlm.EngineConfig;
import com.google.ai.edge.litertlm.ExperimentalFlags;
import com.google.ai.edge.litertlm.LogSeverity;
import com.google.ai.edge.litertlm.MessageCallback;
import com.google.ai.edge.litertlm.SamplerConfig;

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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tiny LiteRT-LM text runner used by the local macOS shim for simple
 * {@code gollek run} calls.
 *
 * <p>The full Quarkus CLI is still the source of truth for general commands.
 * This entrypoint intentionally supports only local text LiteRT-LM inference so
 * short prompts do not pay the whole CLI/SDK startup tax.</p>
 */
public final class LiteRtLmFastRun {
    private static final int FALLBACK_TO_FULL_CLI = 42;
    private static final String DAEMON_MAGIC = "GOLLEK_LITERT_FAST_DAEMON_V1";
    private static final String DAEMON_STOP_MAGIC = "GOLLEK_LITERT_FAST_DAEMON_STOP_V1";
    private static final int DEFAULT_MAX_NUM_TOKENS = 2048;
    private static final int DEFAULT_MAX_NUM_TOKENS_CEILING = 2048;
    private static final int DEFAULT_MAX_NUM_TOKENS_FLOOR = 512;
    private static final Pattern JSON_OBJECT_PATTERN = Pattern.compile("\\{.*?}", Pattern.DOTALL);

    private LiteRtLmFastRun() {
    }

    public static void main(String[] args) {
        if (args.length > 0 && "__daemon".equals(args[0])) {
            System.exit(runDaemon());
            return;
        }
        if (args.length > 0 && "__daemon-stop".equals(args[0])) {
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
            Optional<Path> modelPath = resolveLiteRtLmModel(parsed);
            if (modelPath.isEmpty()) {
                return FALLBACK_TO_FULL_CLI;
            }
            if (daemonEnabled()) {
                OptionalInt daemonStatus = requestDaemon(args);
                if (daemonStatus.isPresent()) {
                    return daemonStatus.getAsInt();
                }
            }
            generate(modelPath.get(), parsed);
            return 0;
        } catch (Throwable throwable) {
            if (Boolean.getBoolean("gollek.litert.fast_run.debug")) {
                throwable.printStackTrace(System.err);
            }
            return FALLBACK_TO_FULL_CLI;
        }
    }

    private static Optional<Path> resolveLiteRtLmModel(FastArgs args) {
        if (args.modelFile != null) {
            Optional<Path> explicit = resolvePath(Path.of(args.modelFile));
            if (explicit.isPresent()) {
                return findLiteRtLmFromExplicitPath(explicit.get());
            }
        }
        if (args.model != null) {
            Optional<Path> direct = resolvePath(Path.of(args.model));
            if (direct.isPresent()) {
                Optional<Path> found = findLiteRtLmFromExplicitPath(direct.get());
                if (found.isPresent()) {
                    return found;
                }
            }
            Optional<Path> fastIndexed = findIndexedLiteRtModelPath(args.model);
            if (fastIndexed.isPresent()) {
                Optional<Path> found = findLiteRtLmFromExplicitPath(fastIndexed.get());
                if (found.isPresent()) {
                    return found;
                }
            }
            try {
                Optional<LocalModelIndex.Entry> indexed = LocalModelIndex.find(args.model);
                if (indexed.isPresent()
                        && isLiteRtIndexEntry(indexed.get())
                        && indexed.get().path != null
                        && !indexed.get().path.isBlank()) {
                    return findLiteRtLmFromExplicitPath(Path.of(indexed.get().path));
                }
            } catch (Throwable ignored) {
                // Full CLI has the complete resolver if the lightweight index read fails.
            }
        }
        return Optional.empty();
    }

    private static Optional<Path> resolvePath(Path path) {
        try {
            Path normalized = path.isAbsolute()
                    ? path.toAbsolutePath().normalize()
                    : Path.of("").toAbsolutePath().resolve(path).normalize();
            if (Files.exists(normalized)) {
                return Optional.of(normalized);
            }
        } catch (Exception ignored) {
        }
        return Optional.empty();
    }

    private static Optional<Path> findIndexedLiteRtModelPath(String ref) {
        if (ref == null || ref.isBlank()) {
            return Optional.empty();
        }
        Path indexPath = Path.of(System.getProperty("user.home"), ".gollek", "models", "index.json");
        if (!Files.isRegularFile(indexPath)) {
            return Optional.empty();
        }
        try {
            String needle = ref.trim();
            String json = Files.readString(indexPath);
            Matcher objects = JSON_OBJECT_PATTERN.matcher(json);
            while (objects.find()) {
                String object = objects.group();
                String path = jsonStringField(object, "path").orElse(null);
                if (path == null || path.isBlank()) {
                    continue;
                }
                if (matchesIndexEntry(object, needle, path) && isLiteRtIndexObject(object, path)) {
                    return Optional.of(Path.of(path));
                }
            }
        } catch (Exception ignored) {
            // Full CLI resolver remains available if the lightweight parse fails.
        }
        return Optional.empty();
    }

    private static boolean matchesIndexEntry(String object, String ref, String path) {
        if (ref.equals(jsonStringField(object, "id").orElse(null))
                || ref.equals(jsonStringField(object, "shortId").orElse(null))
                || ref.equals(jsonStringField(object, "name").orElse(null))
                || ref.equals(path)) {
            return true;
        }
        String normalized = ref.replace("\\", "/").toLowerCase(Locale.ROOT);
        String normalizedPath = path.replace("\\", "/").toLowerCase(Locale.ROOT);
        return normalizedPath.endsWith(normalized);
    }

    private static boolean isLiteRtIndexEntry(LocalModelIndex.Entry entry) {
        if (entry == null) {
            return false;
        }
        if (isLiteRtFormat(entry.format)) {
            return true;
        }
        return entry.path != null && isLiteRtPathString(entry.path);
    }

    private static boolean isLiteRtIndexObject(String object, String path) {
        return isLiteRtFormat(jsonStringField(object, "format").orElse(null))
                || isLiteRtPathString(path);
    }

    private static boolean isLiteRtFormat(String format) {
        if (format == null || format.isBlank()) {
            return false;
        }
        String normalized = format.trim().toLowerCase(Locale.ROOT);
        return normalized.equals("litert")
                || normalized.equals("litertlm")
                || normalized.equals("tflite")
                || normalized.equals("task");
    }

    private static boolean isLiteRtPathString(String path) {
        if (path == null || path.isBlank()) {
            return false;
        }
        return path.replace("\\", "/").toLowerCase(Locale.ROOT).endsWith(".litertlm");
    }

    private static Optional<String> jsonStringField(String object, String field) {
        String quotedField = Pattern.quote(field);
        Pattern fieldPattern = Pattern.compile("\"" + quotedField + "\"\\s*:\\s*\"((?:\\\\.|[^\"])*)\"");
        Matcher matcher = fieldPattern.matcher(object);
        if (!matcher.find()) {
            return Optional.empty();
        }
        return Optional.of(unescapeJsonString(matcher.group(1)));
    }

    private static String unescapeJsonString(String value) {
        StringBuilder out = new StringBuilder(value.length());
        boolean escaping = false;
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            if (!escaping) {
                if (ch == '\\') {
                    escaping = true;
                } else {
                    out.append(ch);
                }
                continue;
            }
            switch (ch) {
                case '"', '\\', '/' -> out.append(ch);
                case 'b' -> out.append('\b');
                case 'f' -> out.append('\f');
                case 'n' -> out.append('\n');
                case 'r' -> out.append('\r');
                case 't' -> out.append('\t');
                default -> out.append(ch);
            }
            escaping = false;
        }
        if (escaping) {
            out.append('\\');
        }
        return out.toString();
    }

    private static Optional<Path> findLiteRtLmFromExplicitPath(Path path) {
        try {
            if (Files.isRegularFile(path) && isLiteRtLm(path)) {
                return Optional.of(path.toAbsolutePath().normalize());
            }
            if (Files.isRegularFile(path)) {
                return Optional.empty();
            }
            Path searchDir = Files.isDirectory(path) ? path : path.getParent();
            if (searchDir == null || !Files.isDirectory(searchDir)) {
                return Optional.empty();
            }
            try (var stream = Files.list(searchDir)) {
                return stream
                        .filter(Files::isRegularFile)
                        .filter(LiteRtLmFastRun::isLiteRtLm)
                        .sorted((left, right) -> Integer.compare(
                                liteRtLmPreference(left),
                                liteRtLmPreference(right)))
                        .findFirst()
                        .map(candidate -> candidate.toAbsolutePath().normalize());
            }
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static boolean isLiteRtLm(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".litertlm");
    }

    private static int liteRtLmPreference(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        int score = 0;
        if (name.contains("qualcomm")) {
            score += 100;
        }
        if (!name.contains("gemma-4-e2b-it.litertlm")) {
            score += 10;
        }
        return score;
    }

    private static void generate(Path modelPath, FastArgs args) throws Exception {
        generate(modelPath, args, System.out, System.err, null, "fast path");
    }

    private static void generate(
            Path modelPath,
            FastArgs args,
            PrintStream out,
            PrintStream err,
            EngineCache engineCache,
            String runnerName) throws Exception {
        long startNanos = System.nanoTime();
        Files.createDirectories(cacheDir());
        Engine.Companion.setNativeMinLogSeverity(nativeLogSeverity());

        int engineMaxNumTokens = maxNumTokens(args);
        long beforeEngineInitNanos = System.nanoTime();
        String requestedBackend = normalizedBackend(args.backend);
        try (EngineLease lease = engineCache == null
                ? openEngine(modelPath, requestedBackend, engineMaxNumTokens, true, false)
                : engineCache.acquire(modelPath, requestedBackend, engineMaxNumTokens)) {
            long afterEngineInitNanos = System.nanoTime();
            String warmSuffix = engineCache == null ? "" : ", warmEngine=" + lease.reused;
            out.printf("Using official LiteRT-LM JVM %s for %s (backend=%s, speculativeDecoding=%s, engineMaxNumTokens=%d%s).%n",
                    runnerName,
                    modelPath.getFileName(),
                    lease.backendName,
                    lease.speculativeDecoding,
                    engineMaxNumTokens,
                    warmSuffix);
            if (lease.fallbackReason != null) {
                err.printf("LiteRT-LM GPU backend failed; using official CPU fallback instead: %s%n",
                        lease.fallbackReason);
            }

            SamplerConfig sampler = new SamplerConfig(
                    Math.max(1, args.topK),
                    Math.max(0.0d, Math.min(1.0d, args.topP)),
                    Math.max(0.0d, args.temperature),
                    0);
            ConversationConfig conversationConfig = new ConversationConfig(
                    null,
                    List.of(),
                    List.of(),
                    sampler,
                    false,
                    null,
                    Map.of());
            try (Conversation conversation = lease.engine.createConversation(conversationConfig)) {
                long afterConversationCreateNanos = System.nanoTime();
                CountDownLatch done = new CountDownLatch(1);
                AtomicBoolean cancelled = new AtomicBoolean(false);
                AtomicLong firstChunkNanos = new AtomicLong(0L);
                AtomicReference<Throwable> error = new AtomicReference<>();
                StringBuilder accumulated = new StringBuilder();
                StringBuilder emitted = new StringBuilder();
                conversation.sendMessageAsync(Contents.Companion.of(args.prompt), new MessageCallback() {
                    @Override
                    public void onMessage(com.google.ai.edge.litertlm.Message message) {
                        if (cancelled.get()) {
                            return;
                        }
                        firstChunkNanos.compareAndSet(0L, System.nanoTime());
                        accumulated.append(message.toString());
                        String limited = limitByApproximateTokens(accumulated.toString(), args.maxTokens);
                        if (limited.length() > emitted.length()) {
                            String delta = limited.substring(emitted.length());
                            out.print(delta);
                            out.flush();
                            emitted.setLength(0);
                            emitted.append(limited);
                        }
                        if (approximateTokenCount(accumulated) >= args.maxTokens
                                && cancelled.compareAndSet(false, true)) {
                            conversation.cancelProcess();
                            done.countDown();
                        }
                    }

                    @Override
                    public void onDone() {
                        done.countDown();
                    }

                    @Override
                    public void onError(Throwable throwable) {
                        if (!cancelled.get()) {
                            error.compareAndSet(null, throwable);
                        }
                        done.countDown();
                    }
                }, Map.of());
                if (!done.await(timeout().toMillis(), TimeUnit.MILLISECONDS)) {
                    conversation.cancelProcess();
                    throw new IllegalStateException("LiteRT-LM fast path timed out");
                }
                if (error.get() != null && !cancelled.get()) {
                    throw new IllegalStateException(error.get());
                }
                long durationMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);
                out.printf("%n[Fast LiteRT-LM, Duration: %.2fs]%n", durationMs / 1000.0d);
                if (Boolean.getBoolean("gollek.litert.fast_run.profile")) {
                    long first = firstChunkNanos.get();
                    err.printf(Locale.ROOT,
                            "[Fast LiteRT-LM profile] cache+config=%.3fs engineInit=%.3fs conversation=%.3fs firstChunk=%.3fs total=%.3fs%n",
                            seconds(beforeEngineInitNanos - startNanos),
                            seconds(afterEngineInitNanos - beforeEngineInitNanos),
                            seconds(afterConversationCreateNanos - afterEngineInitNanos),
                            first == 0L ? -1.0d : seconds(first - afterConversationCreateNanos),
                            seconds(System.nanoTime() - startNanos));
                }
            }
        }
    }

    private static EngineLease openEngine(
            Path modelPath,
            String requestedBackend,
            int engineMaxNumTokens,
            boolean closeOnClose,
            boolean reused) {
        boolean speculativeDecoding = speculativeDecodingEnabled(requestedBackend);
        ExperimentalFlags.INSTANCE.setEnableSpeculativeDecoding(speculativeDecoding);
        EngineConfig engineConfig = new EngineConfig(
                modelPath.toString(),
                backend(requestedBackend),
                null,
                null,
                engineMaxNumTokens,
                null,
                cacheDir().toString());

        try {
            return new EngineLease(
                    initializeEngine(engineConfig),
                    backendName(requestedBackend),
                    speculativeDecoding,
                    null,
                    closeOnClose,
                    reused);
        } catch (Throwable gpuFailure) {
            if (!"gpu".equals(requestedBackend)
                    || !Boolean.parseBoolean(System.getProperty("gollek.litert.fast_run.cpu_fallback", "true"))) {
                throw gpuFailure;
            }
            String fallbackReason = summarize(gpuFailure);
            speculativeDecoding = speculativeDecodingEnabled("cpu");
            ExperimentalFlags.INSTANCE.setEnableSpeculativeDecoding(speculativeDecoding);
            EngineConfig cpuEngineConfig = new EngineConfig(
                    modelPath.toString(),
                    new Backend.CPU(),
                    null,
                    null,
                    engineMaxNumTokens,
                    null,
                    cacheDir().toString());
            return new EngineLease(
                    initializeEngine(cpuEngineConfig),
                    "CPU",
                    speculativeDecoding,
                    fallbackReason,
                    closeOnClose,
                    reused);
        }
    }

    private static Path cacheDir() {
        String configured = System.getProperty("gollek.litert.lm_jvm_bridge.cache_dir");
        if (configured != null && !configured.isBlank()) {
            return Path.of(configured).toAbsolutePath();
        }
        return Path.of(System.getProperty("user.home"), ".gollek", "cache", "litert-lm-jvm").toAbsolutePath();
    }

    private static Engine initializeEngine(EngineConfig config) {
        Engine engine = new Engine(config);
        try {
            engine.initialize();
            return engine;
        } catch (Throwable failure) {
            try {
                engine.close();
            } catch (Throwable ignored) {
            }
            throw failure;
        }
    }

    private static LogSeverity nativeLogSeverity() {
        String configured = System.getProperty("gollek.litert.fast_run.native_log_severity", "ERROR");
        try {
            return LogSeverity.valueOf(configured.trim().toUpperCase(Locale.ROOT));
        } catch (Exception ignored) {
            return LogSeverity.ERROR;
        }
    }

    private static String metalAwareGpuBackendName() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        return os.contains("mac") ? "GPU/Metal" : "GPU";
    }

    private static Backend backend(String normalizedBackend) {
        return "cpu".equals(normalizedBackend) ? new Backend.CPU() : new Backend.GPU();
    }

    private static String backendName(String normalizedBackend) {
        return "cpu".equals(normalizedBackend) ? "CPU" : metalAwareGpuBackendName();
    }

    private static String normalizedBackend(String configured) {
        String normalized = configured == null ? "gpu" : configured.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "cpu" -> "cpu";
            case "gpu", "metal", "auto", "" -> "gpu";
            default -> "gpu";
        };
    }

    private static boolean speculativeDecodingEnabled(String normalizedBackend) {
        String configured = System.getProperty("gollek.litert.fast_run.speculative_decoding");
        if (configured != null && !configured.isBlank()) {
            return Boolean.parseBoolean(configured);
        }
        return "gpu".equals(normalizedBackend);
    }

    private static String summarize(Throwable throwable) {
        if (throwable == null) {
            return "unknown failure";
        }
        String message = throwable.getMessage();
        if (message == null || message.isBlank()) {
            message = throwable.getClass().getSimpleName();
        }
        return message.replace('\n', ' ').strip();
    }

    private static int maxNumTokens(FastArgs args) {
        Integer configured = Integer.getInteger("gollek.litert.lm_jvm_bridge.max_num_tokens");
        if (configured != null && configured > 0) {
            return configured;
        }
        if (!Boolean.getBoolean("gollek.litert.fast_run.dynamic_engine_tokens")) {
            return DEFAULT_MAX_NUM_TOKENS;
        }
        int requested = Math.max(1, args.maxTokens);
        int promptEstimate = approximateTokenCount(args.prompt);
        int dynamic = requested + promptEstimate + 64;
        int floor = Integer.getInteger("gollek.litert.fast_run.min_engine_tokens", DEFAULT_MAX_NUM_TOKENS_FLOOR);
        int ceiling = Integer.getInteger("gollek.litert.fast_run.max_engine_tokens", DEFAULT_MAX_NUM_TOKENS_CEILING);
        return Math.max(1, Math.min(Math.max(floor, dynamic), Math.max(floor, ceiling)));
    }

    private static Duration timeout() {
        return Duration.ofSeconds(Long.getLong("gollek.litert.lm_jvm_bridge.timeout_seconds", 180L));
    }

    private static double seconds(long nanos) {
        return nanos / 1_000_000_000.0d;
    }

    private static int approximateTokenCount(CharSequence text) {
        int tokens = 0;
        boolean inToken = false;
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (Character.isWhitespace(ch)) {
                inToken = false;
            } else if (!inToken) {
                tokens++;
                inToken = true;
            }
        }
        return tokens;
    }

    private static String limitByApproximateTokens(String text, int maxTokens) {
        int tokens = 0;
        boolean inToken = false;
        int end = text.length();
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (Character.isWhitespace(ch)) {
                inToken = false;
            } else if (!inToken) {
                tokens++;
                inToken = true;
                if (tokens > maxTokens) {
                    end = i;
                    break;
                }
            }
        }
        return text.substring(0, end).stripTrailing();
    }

    private static boolean daemonEnabled() {
        return Boolean.parseBoolean(System.getProperty("gollek.litert.fast_run.daemon", "false"));
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
            socket.setSoTimeout((int) timeout().toMillis());
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            DataInputStream in = new DataInputStream(socket.getInputStream());
            writeString(out, DAEMON_MAGIC);
            out.writeInt(rawArgs.length);
            for (String arg : rawArgs) {
                writeString(out, arg);
            }
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
            if (Boolean.getBoolean("gollek.litert.fast_run.debug")) {
                System.err.println("LiteRT-LM daemon request failed: " + summarize(e));
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
            command.add("-Xmx" + System.getProperty("gollek.litert.fast_run.daemon_heap",
                    System.getProperty("gollek.litert.fast_run.heap", "8g")));
            command.add("-XX:MaxDirectMemorySize=" + System.getProperty("gollek.litert.fast_run.daemon_direct_memory", "24g"));
            command.add("--enable-preview");
            command.add("--add-modules");
            command.add("jdk.incubator.vector");
            command.add("--enable-native-access=ALL-UNNAMED");
            copyGollekSystemProperties(command);
            command.add("-Djava.library.path=" + System.getProperty("java.library.path", ""));
            command.add("-cp");
            command.add(System.getProperty("java.class.path"));
            command.add(LiteRtLmFastRun.class.getName());
            command.add("__daemon");

            if (!startDetachedDaemon(command)) {
                return false;
            }
            long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(daemonStartTimeoutMillis());
            while (System.nanoTime() < deadlineNanos) {
                OptionalInt port = readDaemonPort();
                if (port.isPresent()) {
                    return true;
                }
                Thread.sleep(100L);
            }
            return readDaemonPort().isPresent();
        } catch (Exception e) {
            if (Boolean.getBoolean("gollek.litert.fast_run.debug")) {
                System.err.println("LiteRT-LM daemon failed to start: " + summarize(e));
            }
            return false;
        }
    }

    private static boolean startDetachedDaemon(List<String> command) throws IOException, InterruptedException {
        if (isMacOs() && startDetachedDaemonWithLaunchctl(command)) {
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
        String shellCommand = daemonShellCommand(command, false);
        Process launcher = new ProcessBuilder(
                launchctl,
                "submit",
                "-l",
                launchctlLabel(),
                "--",
                "/bin/sh",
                "-c",
                shellCommand)
                .redirectOutput(ProcessBuilder.Redirect.appendTo(daemonLogFile().toFile()))
                .redirectError(ProcessBuilder.Redirect.appendTo(daemonLogFile().toFile()))
                .start();
        if (!launcher.waitFor(3, TimeUnit.SECONDS)) {
            return true;
        }
        boolean launched = launcher.exitValue() == 0;
        if (!launched && Boolean.getBoolean("gollek.litert.fast_run.debug")) {
            System.err.println("LiteRT-LM launchctl daemon submit failed; falling back to nohup");
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
        StringBuilder shellCommand = new StringBuilder();
        shellCommand.append("export DYLD_LIBRARY_PATH=")
                .append(shellQuote(home + "/.gollek/libs"))
                .append("${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}; ");
        shellCommand.append("export DYLD_FALLBACK_LIBRARY_PATH=")
                .append(shellQuote(home + "/.gollek/libs"))
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

    private static boolean isMacOs() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac");
    }

    private static String launchctlLabel() {
        return System.getProperty("gollek.litert.fast_run.daemon_launchctl_label",
                "tech.kayys.gollek.litert-fast-daemon");
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

    private static void copyGollekSystemProperties(List<String> command) {
        Properties properties = System.getProperties();
        for (String name : properties.stringPropertyNames()) {
            if (!name.startsWith("gollek.litert.")) {
                continue;
            }
            String value = properties.getProperty(name);
            if (value != null) {
                command.add("-D" + name + "=" + value);
            }
        }
    }

    private static int runDaemon() {
        try {
            Files.createDirectories(daemonRunDir());
            Files.createDirectories(cacheDir());
        } catch (IOException e) {
            System.err.println("Failed to create LiteRT-LM daemon directories: " + summarize(e));
            return 1;
        }
        AtomicBoolean stop = new AtomicBoolean(false);
        int port = -1;
        try (ServerSocket server = new ServerSocket(0, 16, InetAddress.getLoopbackAddress());
             EngineCache engineCache = new EngineCache()) {
            port = server.getLocalPort();
            writeDaemonPort(port);
            server.setSoTimeout(1000);
            long lastRequestNanos = System.nanoTime();
            long idleNanos = TimeUnit.SECONDS.toNanos(
                    Long.getLong("gollek.litert.fast_run.daemon_idle_seconds", 900L));
            while (!stop.get()) {
                try (Socket socket = server.accept()) {
                    lastRequestNanos = System.nanoTime();
                    stop.set(handleDaemonClient(socket, engineCache));
                } catch (SocketTimeoutException ignored) {
                    if (System.nanoTime() - lastRequestNanos > idleNanos) {
                        break;
                    }
                }
            }
            return 0;
        } catch (Throwable throwable) {
            System.err.println("LiteRT-LM daemon failed: " + summarize(throwable));
            if (Boolean.getBoolean("gollek.litert.fast_run.debug")) {
                throwable.printStackTrace(System.err);
            }
            return 1;
        } finally {
            deleteDaemonPortFileIfPort(port);
        }
    }

    private static boolean handleDaemonClient(Socket socket, EngineCache engineCache) {
        try {
            DataInputStream in = new DataInputStream(socket.getInputStream());
            DataOutputStream framed = new DataOutputStream(socket.getOutputStream());
            String magic = readString(in);
            if (DAEMON_STOP_MAGIC.equals(magic)) {
                writeStatus(framed, 0);
                return true;
            }
            if (!DAEMON_MAGIC.equals(magic)) {
                writeStatus(framed, FALLBACK_TO_FULL_CLI);
                return false;
            }
            int argCount = in.readInt();
            if (argCount < 0 || argCount > 256) {
                writeStatus(framed, FALLBACK_TO_FULL_CLI);
                return false;
            }
            String[] rawArgs = new String[argCount];
            for (int i = 0; i < argCount; i++) {
                rawArgs[i] = readString(in);
            }
            FastArgs parsed = FastArgs.parse(rawArgs);
            if (!parsed.supported()) {
                writeStatus(framed, FALLBACK_TO_FULL_CLI);
                return false;
            }
            Optional<Path> modelPath = resolveLiteRtLmModel(parsed);
            if (modelPath.isEmpty()) {
                writeStatus(framed, FALLBACK_TO_FULL_CLI);
                return false;
            }
            try (PrintStream out = new PrintStream(new FramedOutputStream(framed, 1), true, StandardCharsets.UTF_8);
                 PrintStream err = new PrintStream(new FramedOutputStream(framed, 2), true, StandardCharsets.UTF_8)) {
                generate(modelPath.get(), parsed, out, err, engineCache, "daemon");
                out.flush();
                err.flush();
            }
            writeStatus(framed, 0);
        } catch (Throwable throwable) {
            try {
                DataOutputStream framed = new DataOutputStream(socket.getOutputStream());
                writeFrame(framed, 2, ("LiteRT-LM daemon request failed: " + summarize(throwable) + System.lineSeparator())
                        .getBytes(StandardCharsets.UTF_8));
                writeStatus(framed, FALLBACK_TO_FULL_CLI);
            } catch (IOException ignored) {
            }
        }
        return false;
    }

    private static int stopDaemon() {
        OptionalInt port = readDaemonPort();
        if (port.isEmpty()) {
            return 0;
        }
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getLoopbackAddress(), port.getAsInt()),
                    daemonConnectTimeoutMillis());
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            DataInputStream in = new DataInputStream(socket.getInputStream());
            writeString(out, DAEMON_STOP_MAGIC);
            out.flush();
            while (true) {
                int channel = in.read();
                if (channel < 0) {
                    break;
                }
                int length = in.readInt();
                if (channel == 0) {
                    break;
                }
                in.skipNBytes(length);
            }
        } catch (Exception ignored) {
        } finally {
            deleteDaemonPortFile();
            removeLaunchctlDaemon();
        }
        return 0;
    }

    private static Path daemonRunDir() {
        return Path.of(System.getProperty("user.home"), ".gollek", "run");
    }

    private static Path daemonPortFile() {
        return daemonRunDir().resolve("litert-fast-daemon.port");
    }

    private static Path daemonLogFile() {
        return Path.of(System.getProperty("user.home"), ".gollek", "logs", "litert-fast-daemon.log");
    }

    private static OptionalInt readDaemonPort() {
        Optional<DaemonInfo> info = readDaemonInfo();
        if (info.isEmpty()) {
            return OptionalInt.empty();
        }
        if (!daemonKey().equals(info.get().key())) {
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
            if (lines.isEmpty()) {
                return Optional.empty();
            }
            int port = Integer.parseInt(lines.get(0).trim());
            long pid = lines.size() > 1 ? Long.parseLong(lines.get(1).trim()) : -1L;
            String key = lines.size() > 2 ? lines.get(2).trim() : "";
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
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static String daemonKey() {
        String classPath = System.getProperty("java.class.path", "");
        if (!classPath.isBlank() && classPath.indexOf(File.pathSeparatorChar) < 0) {
            try {
                Path jar = Path.of(classPath);
                if (Files.isRegularFile(jar)) {
                    Path normalized = jar.toAbsolutePath().normalize();
                    return normalized
                            + "|size=" + Files.size(normalized)
                            + "|mtime=" + Files.getLastModifiedTime(normalized).toMillis();
                }
            } catch (Exception ignored) {
                // Fall through to the raw classpath identity below.
            }
        }
        return classPath;
    }

    private static void discardStaleDaemon(DaemonInfo info) {
        deleteDaemonPortFile();
        removeLaunchctlDaemon();
        if (info.pid() <= 0 || info.pid() == ProcessHandle.current().pid()) {
            return;
        }
        try {
            Optional<ProcessHandle> handle = ProcessHandle.of(info.pid());
            if (handle.isEmpty()) {
                return;
            }
            ProcessHandle process = handle.get();
            if (!process.isAlive()) {
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

    private static void deleteDaemonPortFile() {
        try {
            Files.deleteIfExists(daemonPortFile());
        } catch (IOException ignored) {
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
        return Integer.getInteger("gollek.litert.fast_run.daemon_connect_timeout_ms", 250);
    }

    private static int daemonStartTimeoutMillis() {
        return Integer.getInteger("gollek.litert.fast_run.daemon_start_timeout_ms", 20_000);
    }

    private record DaemonInfo(int port, long pid, String key) {
    }

    private static void writeString(DataOutputStream out, String value) throws IOException {
        byte[] bytes = (value == null ? "" : value).getBytes(StandardCharsets.UTF_8);
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    private static String readString(DataInputStream in) throws IOException {
        int length = in.readInt();
        if (length < 0 || length > 16 * 1024 * 1024) {
            throw new IOException("Invalid LiteRT-LM daemon string length: " + length);
        }
        byte[] bytes = in.readNBytes(length);
        if (bytes.length != length) {
            throw new IOException("Unexpected end of LiteRT-LM daemon string");
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static void writeStatus(DataOutputStream out, int status) throws IOException {
        synchronized (out) {
            out.writeByte(0);
            out.writeInt(status);
            out.flush();
        }
    }

    private static void writeFrame(DataOutputStream out, int channel, byte[] payload) throws IOException {
        synchronized (out) {
            out.writeByte(channel);
            out.writeInt(payload.length);
            out.write(payload);
            out.flush();
        }
    }

    private static final class FramedOutputStream extends OutputStream {
        private final DataOutputStream out;
        private final int channel;
        private final ByteArrayOutputStream buffer = new ByteArrayOutputStream(4096);

        private FramedOutputStream(DataOutputStream out, int channel) {
            this.out = out;
            this.channel = channel;
        }

        @Override
        public void write(int value) throws IOException {
            buffer.write(value);
            if (value == '\n' || buffer.size() >= 4096) {
                flush();
            }
        }

        @Override
        public void write(byte[] bytes, int offset, int length) throws IOException {
            if (length <= 0) {
                return;
            }
            if (buffer.size() == 0 && length >= 4096) {
                byte[] copy = new byte[length];
                System.arraycopy(bytes, offset, copy, 0, length);
                writeFrame(out, channel, copy);
                return;
            }
            buffer.write(bytes, offset, length);
            if (buffer.size() >= 4096 || bytes[offset + length - 1] == '\n') {
                flush();
            }
        }

        @Override
        public void flush() throws IOException {
            if (buffer.size() == 0) {
                return;
            }
            byte[] payload = buffer.toByteArray();
            buffer.reset();
            writeFrame(out, channel, payload);
        }
    }

    private static final class EngineLease implements AutoCloseable {
        private final Engine engine;
        private final String backendName;
        private final boolean speculativeDecoding;
        private final String fallbackReason;
        private final boolean closeOnClose;
        private final boolean reused;

        private EngineLease(
                Engine engine,
                String backendName,
                boolean speculativeDecoding,
                String fallbackReason,
                boolean closeOnClose,
                boolean reused) {
            this.engine = engine;
            this.backendName = backendName;
            this.speculativeDecoding = speculativeDecoding;
            this.fallbackReason = fallbackReason;
            this.closeOnClose = closeOnClose;
            this.reused = reused;
        }

        private EngineLease asBorrowed(boolean reused) {
            return new EngineLease(engine, backendName, speculativeDecoding, fallbackReason, false, reused);
        }

        @Override
        public void close() {
            if (!closeOnClose) {
                return;
            }
            try {
                engine.close();
            } catch (Throwable ignored) {
            }
        }

        private void forceClose() {
            try {
                engine.close();
            } catch (Throwable ignored) {
            }
        }
    }

    private static final class EngineCache implements AutoCloseable {
        private final Map<String, EngineLease> engines = new HashMap<>();

        private synchronized EngineLease acquire(Path modelPath, String requestedBackend, int engineMaxNumTokens) {
            String key = modelPath.toAbsolutePath().normalize()
                    + "|" + requestedBackend
                    + "|" + engineMaxNumTokens
                    + "|" + cacheDir()
                    + "|" + System.getProperty("gollek.litert.fast_run.speculative_decoding", "");
            EngineLease existing = engines.get(key);
            if (existing != null) {
                return existing.asBorrowed(true);
            }
            EngineLease created = openEngine(modelPath, requestedBackend, engineMaxNumTokens, false, false);
            engines.put(key, created);
            return created.asBorrowed(false);
        }

        @Override
        public synchronized void close() {
            for (EngineLease engine : engines.values()) {
                engine.forceClose();
            }
            engines.clear();
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
        private String backend = System.getProperty("gollek.litert.fast_run.backend", "gpu");
        private boolean supported = true;

        private boolean supported() {
            return supported && prompt != null && !prompt.isBlank()
                    && ((model != null && !model.isBlank()) || (modelFile != null && !modelFile.isBlank()));
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
                    case "--backend" -> {
                        Value next = valueOrNext(value, args, i);
                        parsed.backend = next.value();
                        i = next.index();
                    }
                    case "--use-cpu" -> parsed.backend = "cpu";
                    case "--enable-cpu" -> {
                        // CPU fallback is already enabled by default for GPU init failures.
                    }
                    case "--platform" -> {
                        Value next = valueOrNext(value, args, i);
                        String platform = next.value();
                        i = next.index();
                        if ("cpu".equalsIgnoreCase(platform)) {
                            parsed.backend = "cpu";
                        } else if ("metal".equalsIgnoreCase(platform) || "gpu".equalsIgnoreCase(platform)) {
                            parsed.backend = "gpu";
                        } else {
                            parsed.supported = false;
                        }
                    }
                    case "--provider" -> {
                        Value next = valueOrNext(value, args, i);
                        String provider = next.value();
                        i = next.index();
                        if (provider != null && !provider.isBlank() && !"litert".equalsIgnoreCase(provider)) {
                            parsed.supported = false;
                        }
                    }
                    case "--stream" -> {
                        String stream = value != null ? value : "true";
                        if ("false".equalsIgnoreCase(stream)) {
                            parsed.supported = false;
                        }
                    }
                    case "--no-cache", "--offline", "--local" -> {
                        // Safe no-op for the local fast path.
                    }
                    default -> {
                        if (arg.startsWith("-")) {
                            parsed.supported = false;
                        }
                    }
                }
            }
            return parsed;
        }

        private record Value(String value, int index) {
        }

        private static Value valueOrNext(String inline, String[] args, int currentIndex) {
            if (inline != null) {
                return new Value(inline, currentIndex);
            }
            int nextIndex = currentIndex + 1;
            if (nextIndex >= args.length) {
                return new Value("", currentIndex);
            }
            return new Value(args[nextIndex], nextIndex);
        }

        private static int parseInt(String value, int fallback) {
            try {
                return Math.max(1, Integer.parseInt(value));
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
    }
}
