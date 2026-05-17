package tech.kayys.gollek.trainer;

import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import tech.kayys.gollek.trainer.api.TrainerRuntimeMode;
import tech.kayys.gollek.trainer.api.TrainerSession;
import tech.kayys.gollek.trainer.api.TrainingListener;

/**
 * Canonical package-level entrypoint for the Gollek trainer runtime.
 *
 * <p>This is the forward-looking surface for package migration. It uses a
 * reflective bridge to the richer {@code tech.kayys.gollek.ml.train.Trainer}
 * implementation while the physical sources are still being moved out of
 * {@code training/}.
 */
public final class Trainers {

    private static final String LEGACY_TRAINER_CLASS = "tech.kayys.gollek.ml.train.Trainer";

    private Trainers() {
    }

    /**
     * Returns true when the legacy trainer runtime implementation is present on
     * the classpath and can be bridged from the canonical package.
     */
    public static boolean runtimeAvailable() {
        return legacyRuntimeAvailable() || canonicalFallbackAvailable();
    }

    /**
     * Returns the active runtime mode selected for canonical entrypoints.
     */
    public static TrainerRuntimeMode runtimeModeType() {
        return legacyRuntimeAvailable() ? TrainerRuntimeMode.LEGACY_BRIDGE : TrainerRuntimeMode.CANONICAL_FALLBACK;
    }

    /**
     * Returns the active runtime mode selected for canonical entrypoints.
     */
    public static String runtimeMode() {
        return runtimeModeType().value();
    }

    private static boolean legacyRuntimeAvailable() {
        try {
            Class.forName(LEGACY_TRAINER_CLASS);
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    private static boolean canonicalFallbackAvailable() {
        try {
            Class.forName(CanonicalTrainerRuntime.class.getName());
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    /**
     * Loads the legacy trainer builder reflectively while the trainer runtime is
     * still being migrated into the canonical Gradle namespace.
     *
     * @return the legacy trainer builder instance
     */
    public static Object builder() {
        if (!legacyRuntimeAvailable()) {
            return CanonicalTrainerRuntime.builder();
        }
        try {
            Class<?> trainerClass = Class.forName(LEGACY_TRAINER_CLASS);
            return trainerClass.getMethod("builder").invoke(null);
        } catch (ReflectiveOperationException e) {
            return CanonicalTrainerRuntime.builder();
        }
    }

    /**
     * Returns a typed builder for the canonical trainer runtime.
     *
     * <p>This entrypoint is stable during migration because it does not depend
     * on whether the legacy runtime is present on the classpath.
     */
    public static CanonicalTrainerRuntime.Builder canonicalBuilder() {
        return CanonicalTrainerRuntime.builder();
    }

    /**
     * Create a canonical trainer session with default settings.
     */
    public static TrainerSession newSession() {
        return canonicalBuilder().build();
    }

    /**
     * Create a canonical trainer session with the requested epoch count.
     */
    public static TrainerSession newSession(int epochs) {
        return canonicalBuilder().epochs(epochs).build();
    }

    /**
     * Create a typed trainer session builder that can use legacy runtime when
     * all required runtime objects are provided, and otherwise falls back to the
     * canonical in-module trainer runtime.
     */
    public static RuntimeBuilder sessionBuilder() {
        return new RuntimeBuilder();
    }

    private static TrainerSession tryBuildLegacySession(RuntimeBuilder builderState) {
        if (!legacyRuntimeAvailable()) {
            return null;
        }
        if (builderState.model == null || builderState.optimizer == null || builderState.loss == null) {
            return null;
        }
        try {
            Class<?> trainerClass = Class.forName(LEGACY_TRAINER_CLASS);
            Object legacyBuilder = trainerClass.getMethod("builder").invoke(null);

            invokeBuilderMethod(legacyBuilder, "model", builderState.model);
            invokeBuilderMethod(legacyBuilder, "optimizer", builderState.optimizer);
            invokeBuilderMethod(legacyBuilder, "loss", builderState.loss);
            invokeBuilderMethod(legacyBuilder, "epochs", builderState.epochs);
            invokeBuilderMethod(legacyBuilder, "gradientClip", builderState.gradientClip);
            invokeBuilderMethod(legacyBuilder, "mixedPrecision", builderState.mixedPrecision);
            if (builderState.checkpointDir != null) {
                invokeBuilderMethod(legacyBuilder, "checkpointDir", builderState.checkpointDir);
            }
            if (!builderState.listeners.isEmpty()) {
                invokeBuilderMethod(legacyBuilder, "listeners", builderState.listeners);
            }

            Object session = legacyBuilder.getClass().getMethod("build").invoke(legacyBuilder);
            if (session instanceof TrainerSession trainerSession) {
                return trainerSession;
            }
        } catch (ReflectiveOperationException ignored) {
            // Fallback is handled by caller.
        }
        return null;
    }

    private static void invokeBuilderMethod(Object builder, String methodName, Object argument)
            throws ReflectiveOperationException {
        Method method = findCompatibleMethod(builder.getClass(), methodName, argument);
        if (method == null) {
            throw new NoSuchMethodException(
                    "No compatible method '" + methodName + "' found on " + builder.getClass().getName());
        }
        method.invoke(builder, argument);
    }

    private static Method findCompatibleMethod(Class<?> type, String name, Object argument) {
        for (Method method : type.getMethods()) {
            if (!method.getName().equals(name) || method.getParameterCount() != 1) {
                continue;
            }
            Class<?> parameterType = wrapPrimitive(method.getParameterTypes()[0]);
            if (argument == null) {
                return method;
            }
            if (parameterType.isAssignableFrom(argument.getClass())) {
                return method;
            }
        }
        return null;
    }

    private static Class<?> wrapPrimitive(Class<?> type) {
        if (!type.isPrimitive()) {
            return type;
        }
        if (type == int.class) {
            return Integer.class;
        }
        if (type == long.class) {
            return Long.class;
        }
        if (type == double.class) {
            return Double.class;
        }
        if (type == float.class) {
            return Float.class;
        }
        if (type == boolean.class) {
            return Boolean.class;
        }
        if (type == byte.class) {
            return Byte.class;
        }
        if (type == short.class) {
            return Short.class;
        }
        if (type == char.class) {
            return Character.class;
        }
        return type;
    }

    public static final class RuntimeBuilder {
        private Object model;
        private Object optimizer;
        private Object loss;
        private int epochs = 1;
        private double gradientClip = 0.0;
        private boolean mixedPrecision = false;
        private Path checkpointDir;
        private boolean resumeFromCheckpoint = false;
        private boolean failOnCheckpointLoadError = true;
        private final List<TrainingListener> listeners = new ArrayList<>();

        private RuntimeBuilder() {
        }

        public RuntimeBuilder model(Object model) {
            this.model = model;
            return this;
        }

        public RuntimeBuilder optimizer(Object optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public RuntimeBuilder loss(Object loss) {
            this.loss = loss;
            return this;
        }

        public RuntimeBuilder epochs(int epochs) {
            this.epochs = Math.max(1, epochs);
            return this;
        }

        public RuntimeBuilder gradientClip(double gradientClip) {
            this.gradientClip = Math.max(0.0, gradientClip);
            return this;
        }

        public RuntimeBuilder mixedPrecision(boolean mixedPrecision) {
            this.mixedPrecision = mixedPrecision;
            return this;
        }

        public RuntimeBuilder checkpointDir(Path checkpointDir) {
            this.checkpointDir = checkpointDir;
            return this;
        }

        public RuntimeBuilder resumeFromCheckpoint() {
            return resumeFromCheckpoint(true);
        }

        public RuntimeBuilder resumeFromCheckpoint(boolean resumeFromCheckpoint) {
            this.resumeFromCheckpoint = resumeFromCheckpoint;
            return this;
        }

        public RuntimeBuilder failOnCheckpointLoadError(boolean failOnCheckpointLoadError) {
            this.failOnCheckpointLoadError = failOnCheckpointLoadError;
            return this;
        }

        public RuntimeBuilder listener(TrainingListener listener) {
            if (listener == null) {
                return this;
            }
            this.listeners.add(listener);
            return this;
        }

        public RuntimeBuilder listeners(List<? extends TrainingListener> listeners) {
            if (listeners == null) {
                return this;
            }
            for (TrainingListener listener : listeners) {
                if (listener != null) {
                    this.listeners.add(listener);
                }
            }
            return this;
        }

        public TrainerSession build() {
            if (!resumeFromCheckpoint) {
                TrainerSession legacy = tryBuildLegacySession(this);
                if (legacy != null) {
                    return legacy;
                }
            }

            CanonicalTrainerRuntime.Builder fallback = canonicalBuilder()
                    .model(model)
                    .optimizer(optimizer)
                    .loss(loss)
                    .epochs(epochs)
                    .gradientClip(gradientClip)
                    .mixedPrecision(mixedPrecision)
                    .resumeFromCheckpoint(resumeFromCheckpoint)
                    .failOnCheckpointLoadError(failOnCheckpointLoadError);
            if (checkpointDir != null) {
                fallback.checkpointDir(checkpointDir);
            }
            if (!listeners.isEmpty()) {
                fallback.listeners(listeners);
            }
            return fallback.build();
        }
    }
}
