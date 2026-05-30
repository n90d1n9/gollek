package tech.kayys.gollek.jupyter;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;

final class NotebookDependencies {

    private NotebookDependencies() {
    }

    record MavenLookup(Path jar, Path m2Candidate, Path gradleCandidateDir) {}

    record MavenMagicArgs(
            boolean allowRemote,
            boolean fetch,
            boolean explain,
            String coordinates,
            String groupId,
            String artifactId,
            String version,
            String error
    ) {
        boolean valid() {
            return error == null;
        }
    }

    static List<String> classpathEntries(Collection<String> dynamicClasspathEntries) {
        String[] entries = System.getProperty("java.class.path", "").split(File.pathSeparator);
        List<String> nonBlankEntries = Arrays.stream(entries)
                .filter(entry -> entry != null && !entry.isBlank())
                .collect(Collectors.toCollection(java.util.ArrayList::new));
        nonBlankEntries.addAll(dynamicClasspathEntries);
        return nonBlankEntries;
    }

    static MavenMagicArgs parseMavenMagicArgs(String raw) {
        boolean allowRemote = false;
        boolean fetch = false;
        boolean explain = false;
        String remaining = raw == null ? "" : raw.trim();

        while (remaining.startsWith("--")) {
            int split = remaining.indexOf(' ');
            String option = split < 0 ? remaining : remaining.substring(0, split);
            switch (option) {
                case "--allow-remote" -> allowRemote = true;
                case "--fetch" -> fetch = true;
                case "--explain" -> explain = true;
                default -> {
                    return invalidMavenArgs("Unknown %maven option: " + option
                            + "\nUsage: %maven [--allow-remote] [--fetch] [--explain] <groupId:artifactId:version>");
                }
            }
            remaining = split < 0 ? "" : remaining.substring(split + 1).trim();
        }

        String[] parts = remaining.split(":");
        if (parts.length != 3 || Arrays.stream(parts).anyMatch(String::isBlank)) {
            return invalidMavenArgs("Invalid coordinates. Expected: groupId:artifactId:version");
        }
        return new MavenMagicArgs(
                allowRemote,
                fetch,
                explain,
                remaining,
                parts[0],
                parts[1],
                parts[2],
                null
        );
    }

    static MavenLookup resolveMavenArtifact(String groupId, String artifactId, String version) {
        String relative = groupId.replace('.', File.separatorChar)
                + File.separator + artifactId
                + File.separator + version
                + File.separator + artifactId + "-" + version + ".jar";
        Path m2Root = Path.of(System.getProperty("user.home"), ".m2", "repository");
        Path gradleCacheRoot = Path.of(System.getProperty("user.home"), ".gradle", "caches", "modules-2", "files-2.1");
        List<Path> roots = List.of(
                m2Root,
                gradleCacheRoot
        );
        Path m2Jar = roots.getFirst().resolve(relative);
        if (Files.isRegularFile(m2Jar)) {
            return new MavenLookup(m2Jar.toAbsolutePath().normalize(), m2Jar.toAbsolutePath().normalize(), gradleCacheRoot.resolve(groupId).resolve(artifactId).resolve(version));
        }
        Path gradleRoot = roots.get(1).resolve(groupId).resolve(artifactId).resolve(version);
        if (Files.isDirectory(gradleRoot)) {
            try (Stream<Path> stream = Files.walk(gradleRoot, 3)) {
                Path found = stream
                        .filter(Files::isRegularFile)
                        .filter(path -> path.getFileName().toString().equals(artifactId + "-" + version + ".jar"))
                        .findFirst()
                        .map(path -> path.toAbsolutePath().normalize())
                        .orElse(null);
                return new MavenLookup(found, m2Jar.toAbsolutePath().normalize(), gradleRoot.toAbsolutePath().normalize());
            } catch (Exception ignored) {
                return new MavenLookup(null, m2Jar.toAbsolutePath().normalize(), gradleRoot.toAbsolutePath().normalize());
            }
        }
        return new MavenLookup(null, m2Jar.toAbsolutePath().normalize(), gradleRoot.toAbsolutePath().normalize());
    }

    static String validateJarPath(Path jar) {
        if (!Files.exists(jar)) {
            return "Jar not found: " + jar;
        }
        if (!Files.isRegularFile(jar)) {
            return "Not a file: " + jar;
        }
        if (!jar.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".jar")) {
            return "Not a jar file: " + jar;
        }
        return null;
    }

    static String fetchMavenArtifactError(String coords) {
        String enabled = System.getenv("GOLLEK_JUPYTER_ENABLE_REMOTE_MAVEN");
        if (!"true".equalsIgnoreCase(enabled)) {
            return "Remote Maven fetch is disabled. Set GOLLEK_JUPYTER_ENABLE_REMOTE_MAVEN=true and retry:\n"
                    + "%maven --allow-remote --fetch " + coords;
        }
        ProcessBuilder builder = new ProcessBuilder("mvn", "dependency:get", "-Dartifact=" + coords);
        builder.redirectErrorStream(true);
        try {
            Process process = builder.start();
            boolean finished = process.waitFor(120, TimeUnit.SECONDS);
            String output = new String(process.getInputStream().readAllBytes());
            if (!finished) {
                process.destroyForcibly();
                return "Remote Maven fetch timed out for " + coords + "\n" + output;
            }
            if (process.exitValue() != 0) {
                return "Remote Maven fetch failed for " + coords + "\n" + output;
            }
            return null;
        } catch (Exception e) {
            return "Remote Maven fetch failed for " + coords + "\n" + e.getMessage();
        }
    }

    private static MavenMagicArgs invalidMavenArgs(String error) {
        return new MavenMagicArgs(false, false, false, "", "", "", "", error);
    }
}
