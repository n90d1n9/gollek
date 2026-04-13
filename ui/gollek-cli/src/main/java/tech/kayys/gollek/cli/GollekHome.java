package tech.kayys.gollek.cli;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Resolves Gollek runtime home with Wayang-first defaults and legacy fallback.
 */
public final class GollekHome {
    private static final String GOLLEK_HOME_PROP = "gollek.home";
    private static final String WAYANG_HOME_PROP = "wayang.home";

    private GollekHome() {
    }

    public static Path resolve() {
        String explicit = firstNonBlank(
                System.getProperty(GOLLEK_HOME_PROP),
                System.getenv("GOLLEK_HOME"),
                System.getenv("WAYANG_GOLLEK_HOME"));
        if (hasText(explicit)) {
            return Paths.get(explicit).toAbsolutePath().normalize();
        }

        String userHome = System.getProperty("user.home");
        String wayangHome = firstNonBlank(System.getProperty(WAYANG_HOME_PROP), System.getenv("WAYANG_HOME"));
        Path preferred = hasText(wayangHome)
                ? Paths.get(wayangHome, "gollek")
                : Paths.get(userHome, ".wayang", "gollek");

        Path legacy = Paths.get(userHome, ".gollek");
        if (Files.isDirectory(preferred) || !Files.isDirectory(legacy)) {
            return preferred.toAbsolutePath().normalize();
        }
        return legacy.toAbsolutePath().normalize();
    }

    public static Path path(String... parts) {
        Path root = resolve();
        for (String part : parts) {
            root = root.resolve(part);
        }
        return root;
    }

    public static void applySystemProperties() {
        Path resolved = resolve();
        if (!hasText(System.getProperty(GOLLEK_HOME_PROP))) {
            System.setProperty(GOLLEK_HOME_PROP, resolved.toString());
        }
        if (!hasText(System.getProperty(WAYANG_HOME_PROP))) {
            Path wayangHome = resolved;
            if ("gollek".equals(resolved.getFileName() != null ? resolved.getFileName().toString() : "")) {
                wayangHome = resolved.getParent();
            }
            if (wayangHome != null) {
                System.setProperty(WAYANG_HOME_PROP, wayangHome.toString());
            }
        }
    }

    private static boolean hasText(String value) {
        return value != null && !value.trim().isEmpty();
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (hasText(value)) {
                return value.trim();
            }
        }
        return null;
    }
}
