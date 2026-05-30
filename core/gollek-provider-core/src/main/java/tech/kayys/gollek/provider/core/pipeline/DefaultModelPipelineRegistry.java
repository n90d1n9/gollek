package tech.kayys.gollek.provider.core.pipeline;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.pipeline.ModelPipeline;
import tech.kayys.gollek.spi.pipeline.ModelPipelineRegistry;
import tech.kayys.gollek.spi.pipeline.ModelPipelineRequest;

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Default pipeline registry for backend-agnostic Gollek extensions.
 *
 * <p>Extension projects can contribute pipelines either as CDI beans or via Java
 * ServiceLoader. The registry deduplicates by id and orders by priority.</p>
 */
@ApplicationScoped
public class DefaultModelPipelineRegistry implements ModelPipelineRegistry {

    private static final Logger LOG = Logger.getLogger(DefaultModelPipelineRegistry.class);
    private static final String EXTENSION_PATH_PROPERTY = "gollek.extensions.path";
    private static final String EXTENSION_PATH_ENV = "GOLLEK_EXTENSION_PATH";
    private static final String LEGACY_FEATURE_PATH_PROPERTY = "gollek.features.path";
    private static final String LEGACY_FEATURE_PATH_ENV = "GOLLEK_FEATURE_PATH";

    @Inject
    Instance<ModelPipeline> cdiPipelines;

    private volatile List<ModelPipeline> runtimeFeaturePipelines;
    private volatile URLClassLoader runtimeFeatureClassLoader;

    @Override
    public List<ModelPipeline> all() {
        Map<String, ModelPipeline> discovered = new LinkedHashMap<>();
        if (cdiPipelines != null) {
            for (ModelPipeline pipeline : cdiPipelines) {
                addPipeline(discovered, pipeline);
            }
        }
        ServiceLoader.load(ModelPipeline.class).forEach(pipeline -> addPipeline(discovered, pipeline));
        runtimeFeaturePipelines().forEach(pipeline -> addPipeline(discovered, pipeline));
        return discovered.values().stream()
                .sorted(Comparator
                        .comparingInt(ModelPipeline::priority)
                        .reversed()
                        .thenComparing(ModelPipeline::id))
                .toList();
    }

    @Override
    public Optional<ModelPipeline> select(ModelPipelineRequest request) {
        for (ModelPipeline pipeline : all()) {
            try {
                if (pipeline.supports(request)) {
                    return Optional.of(pipeline);
                }
            } catch (Throwable failure) {
                // Keep discovery resilient: one experimental feature must not
                // prevent core providers from handling the request.
            }
        }
        return Optional.empty();
    }

    private List<ModelPipeline> runtimeFeaturePipelines() {
        List<ModelPipeline> cached = runtimeFeaturePipelines;
        if (cached != null) {
            return cached;
        }
        synchronized (this) {
            if (runtimeFeaturePipelines != null) {
                return runtimeFeaturePipelines;
            }
            List<Path> jars = runtimeFeatureJars();
            if (jars.isEmpty()) {
                runtimeFeaturePipelines = List.of();
                return runtimeFeaturePipelines;
            }
            if (isNativeImageRuntime()) {
                LOG.warnf("Dynamic Gollek extension jar loading is disabled in native image runtime. "
                        + "The following jars are manifest-inspectable only unless included ahead of time: %s", jars);
                runtimeFeaturePipelines = List.of();
                return runtimeFeaturePipelines;
            }
            try {
                URL[] urls = new URL[jars.size()];
                for (int i = 0; i < jars.size(); i++) {
                    urls[i] = jars.get(i).toUri().toURL();
                }
                ClassLoader parent = Thread.currentThread().getContextClassLoader();
                if (parent == null) {
                    parent = ModelPipeline.class.getClassLoader();
                }
                runtimeFeatureClassLoader = new URLClassLoader(urls, parent);
                List<ModelPipeline> loaded = new ArrayList<>();
                ServiceLoader.load(ModelPipeline.class, runtimeFeatureClassLoader)
                        .forEach(loaded::add);
                runtimeFeaturePipelines = List.copyOf(loaded);
                if (!loaded.isEmpty()) {
                    LOG.infof("Loaded %d Gollek extension pipeline(s) from %d jar(s)",
                            loaded.size(), jars.size());
                }
                return runtimeFeaturePipelines;
            } catch (Throwable failure) {
                LOG.warnf(failure, "Failed to load Gollek extension pipelines from %s", jars);
                runtimeFeaturePipelines = List.of();
                return runtimeFeaturePipelines;
            }
        }
    }

    private static List<Path> runtimeFeatureJars() {
        Set<Path> jars = new LinkedHashSet<>();
        for (Path root : runtimeFeatureRoots()) {
            if (Files.isRegularFile(root) && isJar(root)) {
                jars.add(normalize(root));
            } else if (Files.isDirectory(root)) {
                try (var stream = Files.list(root)) {
                    stream.filter(path -> Files.isRegularFile(path) && isJar(path))
                            .map(DefaultModelPipelineRegistry::normalize)
                            .sorted()
                            .forEach(jars::add);
                } catch (Exception e) {
                    LOG.warnf(e, "Failed to inspect Gollek extension directory: %s", root);
                }
            }
        }
        return List.copyOf(jars);
    }

    private static List<Path> runtimeFeatureRoots() {
        warnIfLegacyFeaturePathConfigured();
        List<Path> roots = new ArrayList<>();
        addConfiguredRoots(roots, System.getProperty(EXTENSION_PATH_PROPERTY));
        addConfiguredRoots(roots, System.getenv(EXTENSION_PATH_ENV));
        String home = System.getProperty("user.home");
        if (home != null && !home.isBlank()) {
            roots.add(Path.of(home, ".gollek", "extensions"));
        }
        return roots.stream()
                .map(DefaultModelPipelineRegistry::normalize)
                .distinct()
                .toList();
    }

    private static void warnIfLegacyFeaturePathConfigured() {
        String property = System.getProperty(LEGACY_FEATURE_PATH_PROPERTY);
        if (property != null && !property.isBlank()) {
            LOG.warnf("Ignoring legacy Gollek feature path property -D%s=%s. "
                            + "Use -D%s or %s instead.",
                    LEGACY_FEATURE_PATH_PROPERTY, property.trim(), EXTENSION_PATH_PROPERTY, EXTENSION_PATH_ENV);
        }
        String env = System.getenv(LEGACY_FEATURE_PATH_ENV);
        if (env != null && !env.isBlank()) {
            LOG.warnf("Ignoring legacy Gollek feature path environment variable %s=%s. "
                            + "Use %s or -D%s instead.",
                    LEGACY_FEATURE_PATH_ENV, env.trim(), EXTENSION_PATH_ENV, EXTENSION_PATH_PROPERTY);
        }
    }

    private static void addConfiguredRoots(List<Path> roots, String value) {
        if (value == null || value.isBlank()) {
            return;
        }
        String normalizedSeparators = value.contains(File.pathSeparator)
                ? value
                : value.replace(',', File.pathSeparatorChar);
        String[] parts = normalizedSeparators.split(Pattern.quote(File.pathSeparator));
        for (String part : parts) {
            String text = part == null ? "" : part.trim();
            if (text.isEmpty()) {
                continue;
            }
            try {
                roots.add(Path.of(text));
            } catch (InvalidPathException e) {
                LOG.warnf(e, "Ignoring invalid Gollek extension path: %s", text);
            }
        }
    }

    private static boolean isJar(Path path) {
        String name = path.getFileName() == null ? "" : path.getFileName().toString();
        return name.endsWith(".jar");
    }

    private static Path normalize(Path path) {
        return path.toAbsolutePath().normalize();
    }

    private static boolean isNativeImageRuntime() {
        String imageCode = System.getProperty("org.graalvm.nativeimage.imagecode", "");
        String imageKind = System.getProperty("org.graalvm.nativeimage.kind", "");
        return "runtime".equalsIgnoreCase(imageCode) || !imageKind.isBlank();
    }

    private static void addPipeline(Map<String, ModelPipeline> target, ModelPipeline pipeline) {
        if (pipeline == null || pipeline.id() == null || pipeline.id().isBlank()) {
            return;
        }
        ModelPipeline existing = target.get(pipeline.id());
        if (existing == null || pipeline.priority() > existing.priority()) {
            target.put(pipeline.id(), pipeline);
        }
    }
}
