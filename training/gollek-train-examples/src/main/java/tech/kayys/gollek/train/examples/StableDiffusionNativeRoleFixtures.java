package tech.kayys.gollek.train.examples;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Resolves per-role Stable Diffusion native fixtures for student and teachers.
 *
 * <p>Resolution order for each role:
 * 1. Role-specific env var such as {@code GOLLEK_SD_NATIVE_MODEL_DIR_OCR_EARLY}
 * 2. A role-named subdirectory under the shared base dir
 * 3. Shared base fixture fallback
 */
public final class StableDiffusionNativeRoleFixtures {
    public static final String STUDENT = "student";
    public static final String OCR_EARLY = "ocr-early";
    public static final String OCR_LATE = "ocr-late";
    public static final String AESTHETICS_EARLY = "aesthetics-early";
    public static final String AESTHETICS_LATE = "aesthetics-late";

    private static final List<String> ORDER = List.of(
            STUDENT,
            OCR_EARLY,
            OCR_LATE,
            AESTHETICS_EARLY,
            AESTHETICS_LATE);

    private final StableDiffusionNativeModelFixture sharedFixture;
    private final Map<String, StableDiffusionNativeModelFixture> fixturesByRole;

    private StableDiffusionNativeRoleFixtures(
            StableDiffusionNativeModelFixture sharedFixture,
            Map<String, StableDiffusionNativeModelFixture> fixturesByRole) {
        this.sharedFixture = sharedFixture;
        this.fixturesByRole = Map.copyOf(fixturesByRole);
    }

    public static StableDiffusionNativeRoleFixtures detect(Path sharedBaseDirOverride) {
        StableDiffusionNativeModelFixture sharedFixture = sharedBaseDirOverride == null
                ? StableDiffusionNativeModelFixture.detect()
                : StableDiffusionNativeModelFixture.fromBaseDir(sharedBaseDirOverride);
        Map<String, StableDiffusionNativeModelFixture> fixtures = new LinkedHashMap<>();
        for (String role : ORDER) {
            fixtures.put(role, resolveRoleFixture(role, sharedFixture));
        }
        return new StableDiffusionNativeRoleFixtures(sharedFixture, fixtures);
    }

    public StableDiffusionNativeModelFixture sharedFixture() {
        return sharedFixture;
    }

    public StableDiffusionNativeModelFixture fixture(String role) {
        return fixturesByRole.get(role);
    }

    public Map<String, StableDiffusionNativeModelFixture> fixturesByRole() {
        return fixturesByRole;
    }

    public String summary() {
        Map<String, String> summary = new LinkedHashMap<>();
        fixturesByRole.forEach((role, fixture) -> summary.put(role, fixture.summary()));
        return summary.toString();
    }

    private static StableDiffusionNativeModelFixture resolveRoleFixture(
            String role,
            StableDiffusionNativeModelFixture sharedFixture) {
        String env = System.getenv(envNameForRole(role));
        if (env != null && !env.isBlank()) {
            return StableDiffusionNativeModelFixture.fromBaseDir(Path.of(env));
        }

        Path sharedBaseDir = sharedFixture.baseDir();
        Path roleDir = sharedBaseDir.resolve(role);
        if (Files.isDirectory(roleDir)) {
            StableDiffusionNativeModelFixture candidate = StableDiffusionNativeModelFixture.fromBaseDir(roleDir);
            if (candidate.isUsable()) {
                return candidate;
            }
        }
        return sharedFixture;
    }

    public static String envNameForRole(String role) {
        return "GOLLEK_SD_NATIVE_MODEL_DIR_" + role.toUpperCase().replace('-', '_');
    }
}
