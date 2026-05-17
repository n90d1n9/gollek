package tech.kayys.gollek.ml.autograd;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

final class AcceleratedOps {

    private static final String AUTO = "auto";
    private static final String CPU = "cpu";
    private static final String METAL = "metal";
    private static final String CUDA = "cuda";
    private static final Backend CPU_BACKEND = new CpuBackend();
    private static final ThreadLocal<String> PREFERRED_DEVICE = new ThreadLocal<>();
    private static final AtomicLong ACCELERATED_MATMUL_CALLS = new AtomicLong();
    private static final Object BACKEND_LOCK = new Object();
    private static volatile Backend metalBackend;
    private static volatile boolean metalResolved;
    private static volatile Backend cudaBackend;
    private static volatile boolean cudaResolved;

    private AcceleratedOps() {
    }

    static Acceleration.Scope prefer(String deviceId) {
        String previous = PREFERRED_DEVICE.get();
        String normalized = normalizeDevice(deviceId);
        if (AUTO.equals(normalized)) {
            PREFERRED_DEVICE.remove();
        } else {
            PREFERRED_DEVICE.set(normalized);
        }
        return () -> {
            if (previous == null) {
                PREFERRED_DEVICE.remove();
            } else {
                PREFERRED_DEVICE.set(previous);
            }
        };
    }

    static Acceleration.BackendStatus status(String requestedDevice) {
        Selection selection = select(requestedDevice);
        Backend backend = selection.backend();
        return new Acceleration.BackendStatus(
                backend.id(),
                backend.deviceName(),
                backend.accelerated(),
                selection.requestedAvailable(),
                ACCELERATED_MATMUL_CALLS.get());
    }

    static long acceleratedMatmulCalls() {
        return ACCELERATED_MATMUL_CALLS.get();
    }

    static float[] matmul(float[] left, long[] leftShape, float[] right, long[] rightShape) {
        MatmulSpec spec = MatmulSpec.from(left, leftShape, right, rightShape);
        Backend backend = select(null).backend();
        if (backend.accelerated()) {
            try {
                float[] out = backend.matmul(left, right, spec);
                ACCELERATED_MATMUL_CALLS.incrementAndGet();
                return out;
            } catch (LinkageError | RuntimeException ignored) {
                disableBackend(backend);
                // Backend modules are optional; dispatch failures must not break Java training.
            }
        }
        return cpuMatmul(left, right, spec);
    }

    private static void disableBackend(Backend backend) {
        synchronized (BACKEND_LOCK) {
            if (METAL.equals(backend.id())) {
                metalBackend = null;
                metalResolved = true;
            } else if (CUDA.equals(backend.id())) {
                cudaBackend = null;
                cudaResolved = true;
            }
        }
    }

    private static Selection select(String requestedDevice) {
        String requested = requestedDevice(requestedDevice);
        return switch (requested) {
            case CPU -> new Selection(CPU_BACKEND, true);
            case METAL -> metalBackend()
                    .map(backend -> new Selection(backend, true))
                    .orElseGet(() -> new Selection(CPU_BACKEND, false));
            case CUDA -> cudaBackend()
                    .map(backend -> new Selection(backend, true))
                    .orElseGet(() -> new Selection(CPU_BACKEND, false));
            case AUTO -> selectAuto();
            default -> new Selection(CPU_BACKEND, false);
        };
    }

    private static Selection selectAuto() {
        List<String> preference = isMacHost() ? List.of(METAL, CUDA) : List.of(CUDA, METAL);
        for (String device : preference) {
            Optional<Backend> backend = METAL.equals(device) ? metalBackend() : cudaBackend();
            if (backend.isPresent()) {
                return new Selection(backend.get(), true);
            }
        }
        return new Selection(CPU_BACKEND, true);
    }

    private static Optional<Backend> metalBackend() {
        if (!metalResolved) {
            synchronized (BACKEND_LOCK) {
                if (!metalResolved) {
                    metalBackend = MetalBackend.tryCreate().orElse(null);
                    metalResolved = true;
                }
            }
        }
        return Optional.ofNullable(metalBackend);
    }

    private static Optional<Backend> cudaBackend() {
        if (!cudaResolved) {
            synchronized (BACKEND_LOCK) {
                if (!cudaResolved) {
                    cudaBackend = CudaBackend.tryCreate().orElse(null);
                    cudaResolved = true;
                }
            }
        }
        return Optional.ofNullable(cudaBackend);
    }

    private static String requestedDevice(String explicitDevice) {
        String explicit = normalizeDevice(explicitDevice);
        if (!AUTO.equals(explicit)) {
            return explicit;
        }
        String preferred = PREFERRED_DEVICE.get();
        if (preferred != null && !AUTO.equals(preferred)) {
            return preferred;
        }
        for (String property : List.of(
                "gollek.ml.device",
                "gollek.ml.acceleration",
                "gollek.device",
                "gollek.kernel.platform")) {
            String value = normalizeDevice(System.getProperty(property));
            if (!AUTO.equals(value)) {
                return value;
            }
        }
        for (String env : List.of(
                "GOLLEK_ML_DEVICE",
                "GOLLEK_ML_ACCELERATION",
                "GOLLEK_DEVICE",
                "GOLLEK_ACCELERATOR")) {
            String value = normalizeDevice(System.getenv(env));
            if (!AUTO.equals(value)) {
                return value;
            }
        }
        return AUTO;
    }

    private static String normalizeDevice(String deviceId) {
        if (deviceId == null || deviceId.isBlank()) {
            return AUTO;
        }
        String value = deviceId.trim().toLowerCase(Locale.ROOT).replace('_', '-');
        return switch (value) {
            case "auto", "gpu", "accelerated", "default" -> AUTO;
            case "cpu", "java", "none", "off", "disable", "disabled" -> CPU;
            case "metal", "mps", "apple", "apple-metal" -> METAL;
            case "cuda", "nvidia", "gpu-nvidia", "nvidia-cuda" -> CUDA;
            default -> value;
        };
    }

    private static boolean isMacHost() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac");
    }

    private static boolean looksLikeCpu(String deviceName) {
        return deviceName == null || deviceName.toLowerCase(Locale.ROOT).startsWith("cpu");
    }

    private static float[] cpuMatmul(float[] left, float[] right, MatmulSpec spec) {
        float[] out = new float[Math.multiplyExact(spec.batches(), spec.outBlock())];
        for (int batch = 0; batch < spec.batches(); batch++) {
            int leftOffset = batch * spec.leftBlock();
            int rightOffset = spec.sharedRight() ? 0 : batch * spec.rightBlock();
            int outOffset = batch * spec.outBlock();
            for (int row = 0; row < spec.m(); row++) {
                for (int col = 0; col < spec.n(); col++) {
                    float acc = 0f;
                    for (int inner = 0; inner < spec.k(); inner++) {
                        acc += left[leftOffset + row * spec.k() + inner]
                                * right[rightOffset + inner * spec.n() + col];
                    }
                    out[outOffset + row * spec.n() + col] = acc;
                }
            }
        }
        return out;
    }

    private static MemorySegment copyToSegment(Arena arena, float[] values) {
        long bytes = Math.multiplyExact((long) values.length, Float.BYTES);
        MemorySegment segment = arena.allocate(bytes, 64);
        MemorySegment.copy(MemorySegment.ofArray(values), 0, segment, 0, bytes);
        return segment;
    }

    private static int invokeInt(Object receiver, Method method, Object... args) {
        try {
            Object value = method.invoke(receiver, args);
            if (value instanceof Number number) {
                return number.intValue();
            }
            throw new IllegalStateException("Native method did not return an integer status");
        } catch (IllegalAccessException error) {
            throw new IllegalStateException("Cannot call native backend method", error);
        } catch (InvocationTargetException error) {
            throw unwrap(error);
        }
    }

    private static Object invokeObject(Object receiver, Method method, Object... args) {
        try {
            return method.invoke(receiver, args);
        } catch (IllegalAccessException error) {
            throw new IllegalStateException("Cannot call native backend method", error);
        } catch (InvocationTargetException error) {
            throw unwrap(error);
        }
    }

    private static RuntimeException unwrap(InvocationTargetException error) {
        Throwable cause = error.getCause();
        if (cause instanceof RuntimeException runtime) {
            return runtime;
        }
        if (cause instanceof LinkageError linkage) {
            throw linkage;
        }
        return new IllegalStateException("Native backend call failed", cause);
    }

    private static boolean invokeBoolean(Object receiver, Method method) throws ReflectiveOperationException {
        Object value = method.invoke(receiver);
        return value instanceof Boolean bool && bool;
    }

    private static String invokeString(Object receiver, Method method) throws ReflectiveOperationException {
        Object value = method.invoke(receiver);
        return value == null ? "" : value.toString();
    }

    private static Optional<Path> findCudaLibrary() {
        for (Path candidate : cudaLibraryCandidates()) {
            if (Files.isRegularFile(candidate)) {
                return Optional.of(candidate);
            }
        }
        return Optional.empty();
    }

    private static List<Path> cudaLibraryCandidates() {
        List<Path> candidates = new ArrayList<>();
        for (String key : List.of(
                "gollek.cuda.library",
                "gollek.cuda.library.path",
                "gollek.kernel.cuda.library")) {
            addCudaCandidate(candidates, System.getProperty(key));
        }
        for (String key : List.of(
                "GOLLEK_CUDA_LIBRARY",
                "CUDA_LIBRARY_PATH",
                "CUDA_PATH",
                "CUDA_HOME")) {
            addCudaCandidate(candidates, System.getenv(key));
        }
        candidates.add(Path.of("/usr/local/cuda/lib64/libgollek_cuda.so"));
        candidates.add(Path.of("/usr/local/lib/libgollek_cuda.so"));
        candidates.add(Path.of("/opt/cuda/lib64/libgollek_cuda.so"));
        return candidates;
    }

    private static void addCudaCandidate(List<Path> candidates, String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return;
        }
        Path path = Path.of(rawPath.trim());
        candidates.add(path);
        if (Files.isDirectory(path)) {
            candidates.add(path.resolve("libgollek_cuda.so"));
            candidates.add(path.resolve("lib64/libgollek_cuda.so"));
            candidates.add(path.resolve("bin/gollek_cuda.dll"));
        }
    }

    private interface Backend {
        String id();

        String deviceName();

        boolean accelerated();

        float[] matmul(float[] left, float[] right, MatmulSpec spec);
    }

    private record Selection(Backend backend, boolean requestedAvailable) {
    }

    private record MatmulSpec(
            int batches,
            int m,
            int k,
            int n,
            int leftBlock,
            int rightBlock,
            int outBlock,
            boolean sharedRight) {

        static MatmulSpec from(float[] left, long[] leftShape, float[] right, long[] rightShape) {
            if (leftShape.length < 2 || rightShape.length < 2) {
                throw new IllegalArgumentException("matmul requires rank >= 2");
            }
            if (rightShape.length == 2) {
                int m = toInt(leftShape[leftShape.length - 2], "m");
                int k = toInt(leftShape[leftShape.length - 1], "k");
                int otherK = toInt(rightShape[0], "right k");
                int n = toInt(rightShape[1], "n");
                if (k != otherK) {
                    throw new IllegalArgumentException("matmul inner dimension mismatch");
                }
                long[] prefix = Arrays.copyOf(leftShape, leftShape.length - 2);
                int batches = numel(prefix);
                requireLength(left, Math.multiplyExact(batches, Math.multiplyExact(m, k)), "left");
                requireLength(right, Math.multiplyExact(k, n), "right");
                return new MatmulSpec(
                        batches,
                        m,
                        k,
                        n,
                        Math.multiplyExact(m, k),
                        Math.multiplyExact(k, n),
                        Math.multiplyExact(m, n),
                        true);
            }
            if (leftShape.length == 3 && rightShape.length == 3) {
                int batch = toInt(leftShape[0], "batch");
                int m = toInt(leftShape[1], "m");
                int k = toInt(leftShape[2], "k");
                int rightBatch = toInt(rightShape[0], "right batch");
                int otherK = toInt(rightShape[1], "right k");
                int n = toInt(rightShape[2], "n");
                if (batch != rightBatch || k != otherK) {
                    throw new IllegalArgumentException("batched matmul shape mismatch");
                }
                requireLength(left, Math.multiplyExact(batch, Math.multiplyExact(m, k)), "left");
                requireLength(right, Math.multiplyExact(batch, Math.multiplyExact(k, n)), "right");
                return new MatmulSpec(
                        batch,
                        m,
                        k,
                        n,
                        Math.multiplyExact(m, k),
                        Math.multiplyExact(k, n),
                        Math.multiplyExact(m, n),
                        false);
            }
            throw new UnsupportedOperationException(
                    "matmul compatibility currently supports [..,m,k]x[k,n] and [b,m,k]x[b,k,n]");
        }

        private static int numel(long[] shape) {
            long total = 1L;
            for (long dimension : shape) {
                total = Math.multiplyExact(total, dimension);
            }
            return Math.toIntExact(total);
        }

        private static int toInt(long value, String name) {
            if (value < 0) {
                throw new IllegalArgumentException(name + " dimension must be non-negative");
            }
            return Math.toIntExact(value);
        }

        private static void requireLength(float[] values, int expected, String name) {
            if (values.length != expected) {
                throw new IllegalArgumentException(
                        name + " data length mismatch: expected " + expected + " but got " + values.length);
            }
        }
    }

    private static final class CpuBackend implements Backend {
        @Override
        public String id() {
            return CPU;
        }

        @Override
        public String deviceName() {
            return "Java CPU fallback";
        }

        @Override
        public boolean accelerated() {
            return false;
        }

        @Override
        public float[] matmul(float[] left, float[] right, MatmulSpec spec) {
            return cpuMatmul(left, right, spec);
        }
    }

    private abstract static class NativeBindingBackend implements Backend {
        private final Object binding;
        private final Method matmulMethod;
        private final String deviceName;

        NativeBindingBackend(Object binding, Method matmulMethod, String deviceName) {
            this.binding = binding;
            this.matmulMethod = matmulMethod;
            this.deviceName = deviceName;
        }

        @Override
        public String deviceName() {
            return deviceName;
        }

        @Override
        public boolean accelerated() {
            return true;
        }

        @Override
        public float[] matmul(float[] left, float[] right, MatmulSpec spec) {
            long outBytes = Math.multiplyExact((long) spec.batches() * spec.outBlock(), Float.BYTES);
            try (Arena arena = Arena.ofShared()) {
                MemorySegment leftSegment = copyToSegment(arena, left);
                MemorySegment rightSegment = copyToSegment(arena, right);
                MemorySegment outSegment = arena.allocate(outBytes, 64);
                for (int batch = 0; batch < spec.batches(); batch++) {
                    MemorySegment leftSlice = leftSegment.asSlice(
                            (long) batch * spec.leftBlock() * Float.BYTES,
                            (long) spec.leftBlock() * Float.BYTES);
                    MemorySegment rightSlice = rightSegment.asSlice(
                            spec.sharedRight() ? 0L : (long) batch * spec.rightBlock() * Float.BYTES,
                            (long) spec.rightBlock() * Float.BYTES);
                    MemorySegment outSlice = outSegment.asSlice(
                            (long) batch * spec.outBlock() * Float.BYTES,
                            (long) spec.outBlock() * Float.BYTES);
                    int rc = invokeInt(
                            binding,
                            matmulMethod,
                            outSlice,
                            leftSlice,
                            rightSlice,
                            spec.m(),
                            spec.k(),
                            spec.n(),
                            1.0f,
                            0.0f);
                    if (rc != 0) {
                        throw new IllegalStateException(id() + " matmul failed with status " + rc);
                    }
                }
                return outSegment.asSlice(0, outBytes).toArray(ValueLayout.JAVA_FLOAT);
            }
        }
    }

    private static final class MetalBackend extends NativeBindingBackend {
        private MetalBackend(Object binding, Method matmulMethod, String deviceName) {
            super(binding, matmulMethod, deviceName);
        }

        static Optional<Backend> tryCreate() {
            try {
                Class<?> type = Class.forName("tech.kayys.gollek.metal.binding.MetalBinding");
                Method initialize = type.getMethod("initialize");
                Object loaded = initialize.invoke(null);
                if (!(loaded instanceof Boolean bool) || !bool) {
                    return Optional.empty();
                }
                Object binding = type.getMethod("getInstance").invoke(null);
                int initCode = invokeInt(binding, type.getMethod("init"));
                if (initCode != 0 || !invokeBoolean(binding, type.getMethod("isNativeAvailable"))) {
                    return Optional.empty();
                }
                String deviceName = invokeString(binding, type.getMethod("deviceName"));
                if (looksLikeCpu(deviceName)) {
                    return Optional.empty();
                }
                Method matmul = type.getMethod(
                        "matmul",
                        MemorySegment.class,
                        MemorySegment.class,
                        MemorySegment.class,
                        int.class,
                        int.class,
                        int.class,
                        float.class,
                        float.class);
                return Optional.of(new MetalBackend(binding, matmul, deviceName));
            } catch (ReflectiveOperationException | RuntimeException | LinkageError ignored) {
                return Optional.empty();
            }
        }

        @Override
        public String id() {
            return METAL;
        }
    }

    private static final class CudaBackend extends NativeBindingBackend {
        private final Object binding;
        private final Method mallocMethod;
        private final Method memcpyH2DMethod;
        private final Method memcpyD2HMethod;
        private final Method freeMethod;
        private final Method matmulMethod;

        private CudaBackend(
                Object binding,
                Method matmulMethod,
                Method mallocMethod,
                Method memcpyH2DMethod,
                Method memcpyD2HMethod,
                Method freeMethod,
                String deviceName) {
            super(binding, matmulMethod, deviceName);
            this.binding = binding;
            this.matmulMethod = matmulMethod;
            this.mallocMethod = mallocMethod;
            this.memcpyH2DMethod = memcpyH2DMethod;
            this.memcpyD2HMethod = memcpyD2HMethod;
            this.freeMethod = freeMethod;
        }

        static Optional<Backend> tryCreate() {
            try {
                Optional<Path> library = findCudaLibrary();
                if (library.isEmpty()) {
                    return Optional.empty();
                }
                Class<?> type = Class.forName("tech.kayys.gollek.cuda.binding.CudaBinding");
                Method initialize = type.getMethod("initialize", Path.class);
                Object loaded = initialize.invoke(null, library.get());
                if (!(loaded instanceof Boolean bool) || !bool) {
                    return Optional.empty();
                }
                Object binding = type.getMethod("getInstance").invoke(null);
                if (!invokeBoolean(binding, type.getMethod("isNativeAvailable"))) {
                    return Optional.empty();
                }
                int deviceCount = invokeInt(binding, type.getMethod("deviceCount"));
                if (deviceCount <= 0) {
                    return Optional.empty();
                }
                int initCode = invokeInt(binding, type.getMethod("init", int.class), 0);
                if (initCode != 0) {
                    return Optional.empty();
                }
                String deviceName = invokeString(binding, type.getMethod("deviceName", int.class), 0);
                if (looksLikeCpu(deviceName)) {
                    return Optional.empty();
                }
                Method matmul = type.getMethod(
                        "matmul",
                        MemorySegment.class,
                        MemorySegment.class,
                        MemorySegment.class,
                        int.class,
                        int.class,
                        int.class,
                        float.class,
                        float.class);
                Method malloc = type.getMethod("malloc", long.class);
                Method memcpyH2D = type.getMethod("memcpyH2D", MemorySegment.class, MemorySegment.class, long.class);
                Method memcpyD2H = type.getMethod("memcpyD2H", MemorySegment.class, MemorySegment.class, long.class);
                Method free = type.getMethod("free", MemorySegment.class);
                return Optional.of(new CudaBackend(binding, matmul, malloc, memcpyH2D, memcpyD2H, free, deviceName));
            } catch (ReflectiveOperationException | RuntimeException | LinkageError ignored) {
                return Optional.empty();
            }
        }

        @Override
        public float[] matmul(float[] left, float[] right, MatmulSpec spec) {
            long leftBytes = Math.multiplyExact((long) spec.leftBlock(), Float.BYTES);
            long rightBytes = Math.multiplyExact((long) spec.rightBlock(), Float.BYTES);
            long outBytes = Math.multiplyExact((long) spec.outBlock(), Float.BYTES);
            float[] output = new float[Math.multiplyExact(spec.batches(), spec.outBlock())];
            MemorySegment deviceLeft = null;
            MemorySegment deviceRight = null;
            MemorySegment deviceOut = null;
            try (Arena arena = Arena.ofShared()) {
                MemorySegment heapLeft = MemorySegment.ofArray(left);
                MemorySegment heapRight = MemorySegment.ofArray(right);
                MemorySegment heapOutput = MemorySegment.ofArray(output);
                MemorySegment hostLeft = arena.allocate(leftBytes, 64);
                MemorySegment hostRight = arena.allocate(rightBytes, 64);
                MemorySegment hostOut = arena.allocate(outBytes, 64);
                deviceLeft = requireSegment(invokeObject(binding, mallocMethod, leftBytes), "cuda left allocation");
                deviceRight = requireSegment(invokeObject(binding, mallocMethod, rightBytes), "cuda right allocation");
                deviceOut = requireSegment(invokeObject(binding, mallocMethod, outBytes), "cuda output allocation");
                for (int batch = 0; batch < spec.batches(); batch++) {
                    MemorySegment.copy(
                            heapLeft,
                            (long) batch * spec.leftBlock() * Float.BYTES,
                            hostLeft,
                            0L,
                            leftBytes);
                    MemorySegment.copy(
                            heapRight,
                            spec.sharedRight() ? 0L : (long) batch * spec.rightBlock() * Float.BYTES,
                            hostRight,
                            0L,
                            rightBytes);
                    invokeObject(binding, memcpyH2DMethod, deviceLeft, hostLeft, leftBytes);
                    invokeObject(binding, memcpyH2DMethod, deviceRight, hostRight, rightBytes);
                    int rc = invokeInt(
                            binding,
                            matmulMethod,
                            deviceOut,
                            deviceLeft,
                            deviceRight,
                            spec.m(),
                            spec.k(),
                            spec.n(),
                            1.0f,
                            0.0f);
                    if (rc != 0) {
                        throw new IllegalStateException(id() + " matmul failed with status " + rc);
                    }
                    invokeObject(binding, memcpyD2HMethod, hostOut, deviceOut, outBytes);
                    MemorySegment.copy(
                            hostOut,
                            0L,
                            heapOutput,
                            (long) batch * spec.outBlock() * Float.BYTES,
                            outBytes);
                }
                return output;
            } finally {
                freeQuietly(deviceLeft);
                freeQuietly(deviceRight);
                freeQuietly(deviceOut);
            }
        }

        private static MemorySegment requireSegment(Object value, String operation) {
            if (value instanceof MemorySegment segment && !MemorySegment.NULL.equals(segment)) {
                return segment;
            }
            throw new IllegalStateException(operation + " failed");
        }

        private void freeQuietly(MemorySegment segment) {
            if (segment != null && !MemorySegment.NULL.equals(segment)) {
                try {
                    freeMethod.invoke(binding, segment);
                } catch (ReflectiveOperationException ignored) {
                    // Best effort cleanup for a backend that is about to be disabled.
                }
            }
        }

        private static String invokeString(Object receiver, Method method, Object... args)
                throws ReflectiveOperationException {
            Object value = method.invoke(receiver, args);
            return value == null ? "" : value.toString();
        }

        @Override
        public String id() {
            return CUDA;
        }
    }
}
