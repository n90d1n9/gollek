package tech.kayys.gollek.trainer;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Objects;

final class TrainerEpochLoaderViews {

    private TrainerEpochLoaderViews() {
    }

    static View resolve(Iterable<?> loader, int epoch) {
        Objects.requireNonNull(loader, "loader must not be null");
        if (epoch < 0) {
            throw new IllegalArgumentException("epoch must be non-negative, got: " + epoch);
        }

        Method epochMethod = findEpochMethod(loader.getClass());
        if (epochMethod == null) {
            return new View(loader, false);
        }

        Object epochView = invokeEpochMethod(loader, epochMethod, epoch);
        if (!(epochView instanceof Iterable<?> iterable)) {
            throw new IllegalStateException(epochMethodDescription(loader, epochMethod)
                    + " returned " + typeName(epochView) + " instead of Iterable");
        }
        return new View(iterable, true);
    }

    private static Method findEpochMethod(Class<?> loaderType) {
        try {
            Method method = loaderType.getMethod("epoch", long.class);
            if (Modifier.isStatic(method.getModifiers())
                    || !Iterable.class.isAssignableFrom(method.getReturnType())) {
                return null;
            }
            return method;
        } catch (NoSuchMethodException ignored) {
            return null;
        }
    }

    private static Object invokeEpochMethod(Iterable<?> loader, Method method, int epoch) {
        try {
            return method.invoke(loader, (long) epoch);
        } catch (IllegalAccessException | IllegalArgumentException error) {
            throw new IllegalStateException(epochMethodDescription(loader, method) + " could not be invoked", error);
        } catch (InvocationTargetException error) {
            Throwable cause = error.getCause();
            if (cause instanceof RuntimeException runtimeException) {
                throw runtimeException;
            }
            if (cause instanceof Error fatal) {
                throw fatal;
            }
            throw new IllegalStateException(epochMethodDescription(loader, method) + " failed", cause);
        }
    }

    private static String epochMethodDescription(Iterable<?> loader, Method method) {
        return loader.getClass().getName() + "." + method.getName() + "(long)";
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getName();
    }

    record View(Iterable<?> iterable, boolean explicitEpoch) {
    }
}
