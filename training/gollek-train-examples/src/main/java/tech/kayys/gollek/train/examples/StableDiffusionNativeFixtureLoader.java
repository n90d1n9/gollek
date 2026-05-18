package tech.kayys.gollek.train.examples;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorFFMLoader;
import tech.kayys.gollek.safetensor.loader.SafetensorMetrics;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.gollek.safetensor.quantization.bridge.AccelWeightBridge;
import tech.kayys.gollek.safetensor.runner.sd.CLIPModel;
import tech.kayys.gollek.safetensor.runner.sd.UNetModel;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

/**
 * Plain-Java utility that wires the safetensor-native UNet loader path without
 * a CDI container, so trainer examples can upgrade from stub to real local
 * weights when a valid fixture is present.
 */
public final class StableDiffusionNativeFixtureLoader {

    private StableDiffusionNativeFixtureLoader() {
    }

    public static LoadedUnet loadUnet(Path modelBaseDir) {
        try {
            StandaloneSafetensorStack stack = createStandaloneStack();
            try (SafetensorShardSession session = stack.shardLoader.open(modelBaseDir.resolve("unet"))) {
                AccelWeightBridge bridge = new AccelWeightBridge();
                Map<String, AccelTensor> weights = bridge.bridgeAll(session);
                return new LoadedUnet(new UNetModel(weights), weights);
            }
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to load stable-diffusion-native UNet fixture from " + modelBaseDir, e);
        }
    }

    public static LoadedTextConditioner loadTextConditioner(Path modelBaseDir) {
        try {
            StandaloneSafetensorStack stack = createStandaloneStack();
            try (SafetensorShardSession session = stack.shardLoader.open(modelBaseDir.resolve("text_encoder"))) {
                AccelWeightBridge bridge = new AccelWeightBridge();
                Map<String, AccelTensor> weights = bridge.bridgeAll(session);
                Tokenizer tokenizer = TokenizerFactory.load(modelBaseDir.resolve("tokenizer"), null);
                return new LoadedTextConditioner(new CLIPModel(weights), weights, tokenizer);
            }
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to load stable-diffusion-native CLIP fixture from " + modelBaseDir, e);
        }
    }

    public static final class LoadedUnet implements AutoCloseable {
        private final UNetModel model;
        private final Map<String, AccelTensor> weights;

        private LoadedUnet(UNetModel model, Map<String, AccelTensor> weights) {
            this.model = model;
            this.weights = Map.copyOf(weights);
        }

        public UNetModel model() {
            return model;
        }

        @Override
        public void close() {
            weights.values().forEach(AccelTensor::close);
        }
    }

    public static final class LoadedTextConditioner implements AutoCloseable {
        private static final int MAX_TOKENS = 77;
        private static final long PAD_TOKEN_ID = 49407L;

        private final CLIPModel clipModel;
        private final Map<String, AccelTensor> weights;
        private final Tokenizer tokenizer;

        private LoadedTextConditioner(CLIPModel clipModel, Map<String, AccelTensor> weights, Tokenizer tokenizer) {
            this.clipModel = clipModel;
            this.weights = Map.copyOf(weights);
            this.tokenizer = tokenizer;
        }

        public float[] encodePromptToFloatArray(String prompt) {
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

    private static StandaloneSafetensorStack createStandaloneStack() throws Exception {
        SafetensorFFMLoader ffmLoader = new SafetensorFFMLoader();
        setField(ffmLoader, "config", standaloneConfig());
        setField(ffmLoader, "objectMapper", newObjectMapper());

        SafetensorMetrics metrics = new SafetensorMetrics();
        setField(metrics, "registry", newSimpleMeterRegistry());
        invoke(metrics, "registerGauge");
        setField(ffmLoader, "metrics", metrics);

        SafetensorShardLoader shardLoader = new SafetensorShardLoader();
        setField(shardLoader, "ffmLoader", ffmLoader);
        setField(shardLoader, "objectMapper", newObjectMapper());
        return new StandaloneSafetensorStack(ffmLoader, shardLoader);
    }

    private record StandaloneSafetensorStack(
            SafetensorFFMLoader ffmLoader,
            SafetensorShardLoader shardLoader) {
    }

    private static Object standaloneConfig() {
        Class<?> loaderConfigType = loadClass("tech.kayys.gollek.safetensor.loader.SafetensorLoaderConfig");
        Class<?> validationType = loadClass("tech.kayys.gollek.safetensor.loader.SafetensorLoaderConfig$Validation");
        Class<?> cacheType = loadClass("tech.kayys.gollek.safetensor.loader.SafetensorLoaderConfig$Cache");
        InvocationHandler handler = new InvocationHandler() {
            @Override
            public Object invoke(Object proxy, Method method, Object[] args) {
                return switch (method.getName()) {
                    case "preferMmap", "strict", "warnOnEmptyTensors", "enabled", "logLoadSummary" -> true;
                    case "readChunkBytes" -> 8 * 1024 * 1024;
                    case "maxHeaderBytes" -> 104_857_600L;
                    case "maxSize" -> 8;
                    case "ttlSeconds" -> 300L;
                    case "validation" -> nestedProxy(validationType, this);
                    case "cache" -> nestedProxy(cacheType, this);
                    case "toString" -> "StandaloneSafetensorLoaderConfig";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    default -> throw new UnsupportedOperationException("Unsupported config method: " + method);
                };
            }
        };
        return Proxy.newProxyInstance(
                StableDiffusionNativeFixtureLoader.class.getClassLoader(),
                new Class<?>[] { loaderConfigType },
                handler);
    }

    private static Object nestedProxy(Class<?> type, InvocationHandler handler) {
        return Proxy.newProxyInstance(
                StableDiffusionNativeFixtureLoader.class.getClassLoader(),
                new Class<?>[] { type },
                handler);
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
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static void invoke(Object target, String methodName) throws Exception {
        Method method = target.getClass().getDeclaredMethod(methodName);
        method.setAccessible(true);
        method.invoke(target);
    }
}
