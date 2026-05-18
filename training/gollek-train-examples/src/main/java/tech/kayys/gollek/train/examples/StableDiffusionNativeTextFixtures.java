package tech.kayys.gollek.train.examples;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Resolves per-task Stable Diffusion native text-conditioning fixtures.
 *
 * <p>Resolution order for each task:
 * 1. Task-specific env var such as {@code GOLLEK_SD_NATIVE_MODEL_DIR_TEXT_OCR}
 * 2. A {@code text-<taskId>} subdirectory under the shared base dir
 * 3. Shared base fixture fallback
 */
public final class StableDiffusionNativeTextFixtures {
    public static final String OCR = "ocr";
    public static final String AESTHETICS = "aesthetics";

    private static final List<String> ORDER = List.of(OCR, AESTHETICS);

    private final StableDiffusionNativeModelFixture sharedFixture;
    private final Map<String, StableDiffusionNativeModelFixture> fixturesByTask;

    private StableDiffusionNativeTextFixtures(
            StableDiffusionNativeModelFixture sharedFixture,
            Map<String, StableDiffusionNativeModelFixture> fixturesByTask) {
        this.sharedFixture = sharedFixture;
        this.fixturesByTask = Map.copyOf(fixturesByTask);
    }

    public static StableDiffusionNativeTextFixtures detect(
            Path sharedBaseDirOverride) {
        StableDiffusionNativeModelFixture sharedFixture = sharedBaseDirOverride == null
                ? StableDiffusionNativeModelFixture.detect()
                : StableDiffusionNativeModelFixture.fromBaseDir(sharedBaseDirOverride);
        Map<String, StableDiffusionNativeModelFixture> fixtures = new LinkedHashMap<>();
        for (String taskId : ORDER) {
            fixtures.put(taskId, resolveTaskFixture(taskId, sharedFixture));
        }
        return new StableDiffusionNativeTextFixtures(sharedFixture, fixtures);
    }

    public StableDiffusionNativeModelFixture sharedFixture() {
        return sharedFixture;
    }

    public StableDiffusionNativeModelFixture fixture(String taskId) {
        return fixturesByTask.get(taskId);
    }

    public Map<String, StableDiffusionNativeModelFixture> fixturesByTask() {
        return fixturesByTask;
    }

    public String summary() {
        Map<String, String> summary = new LinkedHashMap<>();
        fixturesByTask.forEach((taskId, fixture) -> summary.put(taskId, fixture.summary()));
        return summary.toString();
    }

    public static String envNameForTask(String taskId) {
        return "GOLLEK_SD_NATIVE_MODEL_DIR_TEXT_" + taskId.toUpperCase().replace('-', '_');
    }

    private static StableDiffusionNativeModelFixture resolveTaskFixture(
            String taskId,
            StableDiffusionNativeModelFixture sharedFixture) {
        String env = System.getenv(envNameForTask(taskId));
        if (env != null && !env.isBlank()) {
            return StableDiffusionNativeModelFixture.fromBaseDir(Path.of(env));
        }

        Path sharedBaseDir = sharedFixture.baseDir();
        Path taskDir = sharedBaseDir.resolve("text-" + taskId);
        if (Files.isDirectory(taskDir)) {
            StableDiffusionNativeModelFixture candidate =
                    StableDiffusionNativeModelFixture.fromBaseDir(taskDir);
            if (candidate.isUsable()) {
                return candidate;
            }
        }
        return sharedFixture;
    }
}
