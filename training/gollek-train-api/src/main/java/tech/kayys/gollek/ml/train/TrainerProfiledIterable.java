package tech.kayys.gollek.ml.train;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Iterator;
import java.util.Objects;

/**
 * Profiles data-loader iteration without changing the lower-level trainer runtime contract.
 */
public final class TrainerProfiledIterable implements Iterable<Object> {
    private final Iterable<?> delegate;
    private final TrainerRuntimeProfiler profiler;
    private final TrainerRuntimeProfiler.Phase iteratorPhase;
    private final TrainerRuntimeProfiler.Phase hasNextPhase;
    private final TrainerRuntimeProfiler.Phase nextPhase;

    private TrainerProfiledIterable(
            Iterable<?> delegate,
            TrainerRuntimeProfiler profiler,
            TrainerRuntimeProfiler.Phase iteratorPhase,
            TrainerRuntimeProfiler.Phase hasNextPhase,
            TrainerRuntimeProfiler.Phase nextPhase) {
        this.delegate = Objects.requireNonNull(delegate, "delegate must not be null");
        this.profiler = Objects.requireNonNull(profiler, "profiler must not be null");
        this.iteratorPhase = Objects.requireNonNull(iteratorPhase, "iteratorPhase must not be null");
        this.hasNextPhase = Objects.requireNonNull(hasNextPhase, "hasNextPhase must not be null");
        this.nextPhase = Objects.requireNonNull(nextPhase, "nextPhase must not be null");
    }

    static Iterable<?> train(Iterable<?> delegate, TrainerRuntimeProfiler profiler) {
        return delegate == null ? null : new TrainerProfiledIterable(
                delegate,
                profiler,
                TrainerRuntimeProfiler.Phase.INPUT_TRAIN_ITERATOR,
                TrainerRuntimeProfiler.Phase.INPUT_TRAIN_HAS_NEXT,
                TrainerRuntimeProfiler.Phase.INPUT_TRAIN_NEXT);
    }

    static Iterable<?> validation(Iterable<?> delegate, TrainerRuntimeProfiler profiler) {
        return delegate == null ? null : new TrainerProfiledIterable(
                delegate,
                profiler,
                TrainerRuntimeProfiler.Phase.INPUT_VALIDATION_ITERATOR,
                TrainerRuntimeProfiler.Phase.INPUT_VALIDATION_HAS_NEXT,
                TrainerRuntimeProfiler.Phase.INPUT_VALIDATION_NEXT);
    }

    @Override
    public Iterator<Object> iterator() {
        Iterator<?> iterator = profiler.time(iteratorPhase, delegate::iterator);
        return new ProfiledIterator(iterator, profiler, hasNextPhase, nextPhase);
    }

    public Iterable<?> epoch(long epoch) {
        Method epochMethod = findEpochMethod(delegate.getClass());
        if (epochMethod == null) {
            return this;
        }
        Object epochView = invokeEpochMethod(epochMethod, epoch);
        if (!(epochView instanceof Iterable<?> iterable)) {
            throw new IllegalStateException(epochMethodDescription(epochMethod)
                    + " returned " + typeName(epochView) + " instead of Iterable");
        }
        return new TrainerProfiledIterable(iterable, profiler, iteratorPhase, hasNextPhase, nextPhase);
    }

    private Method findEpochMethod(Class<?> type) {
        try {
            Method method = type.getMethod("epoch", long.class);
            if (Modifier.isStatic(method.getModifiers())
                    || !Iterable.class.isAssignableFrom(method.getReturnType())) {
                return null;
            }
            return method;
        } catch (NoSuchMethodException ignored) {
            return null;
        }
    }

    private Object invokeEpochMethod(Method method, long epoch) {
        try {
            return method.invoke(delegate, epoch);
        } catch (IllegalAccessException | IllegalArgumentException error) {
            throw new IllegalStateException(epochMethodDescription(method) + " could not be invoked", error);
        } catch (InvocationTargetException error) {
            Throwable cause = error.getCause();
            if (cause instanceof RuntimeException runtimeException) {
                throw runtimeException;
            }
            if (cause instanceof Error fatal) {
                throw fatal;
            }
            throw new IllegalStateException(epochMethodDescription(method) + " failed", cause);
        }
    }

    private String epochMethodDescription(Method method) {
        return delegate.getClass().getName() + "." + method.getName() + "(long)";
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getName();
    }

    private record ProfiledIterator(
            Iterator<?> delegate,
            TrainerRuntimeProfiler profiler,
            TrainerRuntimeProfiler.Phase hasNextPhase,
            TrainerRuntimeProfiler.Phase nextPhase) implements Iterator<Object> {
        private ProfiledIterator {
            Objects.requireNonNull(delegate, "delegate must not be null");
            Objects.requireNonNull(profiler, "profiler must not be null");
            Objects.requireNonNull(hasNextPhase, "hasNextPhase must not be null");
            Objects.requireNonNull(nextPhase, "nextPhase must not be null");
        }

        @Override
        public boolean hasNext() {
            return profiler.time(hasNextPhase, delegate::hasNext);
        }

        @Override
        public Object next() {
            return profiler.time(nextPhase, delegate::next);
        }
    }
}
