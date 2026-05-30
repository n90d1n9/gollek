package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Properties;

/**
 * Writes and verifies a provenance manifest for promotion gate artifacts.
 */
public final class TrainingReportPromotionGateArtifactManifest {
    public static final int FORMAT_VERSION = 1;
    public static final String DEFAULT_FILE_NAME = "promotion-gate.manifest.properties";

    private static final String JSON_ARTIFACT = "json";
    private static final String MARKDOWN_ARTIFACT = "markdown";
    private static final String JUNIT_XML_ARTIFACT = "junitXml";

    private TrainingReportPromotionGateArtifactManifest() {
    }

    public record Options(String fileName) {
        public Options {
            fileName = normalizeFileName(fileName, DEFAULT_FILE_NAME, "fileName");
        }

        public static Options defaults() {
            return new Options(DEFAULT_FILE_NAME);
        }
    }

    public record ArtifactEntry(String name, Path file, long bytes, String sha256) {
        public ArtifactEntry {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("name must not be blank");
            }
            file = Objects.requireNonNull(file, "file must not be null").toAbsolutePath().normalize();
            if (bytes < 0) {
                throw new IllegalArgumentException("bytes must be non-negative");
            }
            if (sha256 == null || sha256.isBlank()) {
                throw new IllegalArgumentException("sha256 must not be blank");
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("name", name);
            map.put("file", file.toString());
            map.put("bytes", bytes);
            map.put("sha256", sha256);
            return Map.copyOf(map);
        }
    }

    public record ManifestBundle(
            Path directory,
            Path manifestFile,
            String manifestSha256,
            Instant generatedAt,
            TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts) {
        public ManifestBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            manifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            if (manifestSha256 == null || manifestSha256.isBlank()) {
                throw new IllegalArgumentException("manifestSha256 must not be blank");
            }
            generatedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
            artifacts = Objects.requireNonNull(artifacts, "artifacts must not be null");
        }

        public boolean passed() {
            return artifacts.passed();
        }

        public boolean promotable() {
            return artifacts.promotable();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("manifestFile", manifestFile.toString());
            map.put("manifestSha256", manifestSha256);
            map.put("generatedAt", generatedAt.toString());
            map.put("passed", passed());
            map.put("promotable", promotable());
            map.put("artifacts", artifacts.toMap());
            return Map.copyOf(map);
        }
    }

    public record ManifestInspection(
            Path directory,
            Path manifestFile,
            int formatVersion,
            Instant generatedAt,
            String manifestSha256,
            Map<String, ArtifactEntry> artifacts,
            Map<String, String> metadata) {
        public ManifestInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            manifestFile = Objects.requireNonNull(manifestFile, "manifestFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            generatedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
            if (manifestSha256 == null || manifestSha256.isBlank()) {
                throw new IllegalArgumentException("manifestSha256 must not be blank");
            }
            artifacts = Map.copyOf(Objects.requireNonNull(artifacts, "artifacts must not be null"));
            metadata = Map.copyOf(Objects.requireNonNull(metadata, "metadata must not be null"));
        }

        public boolean passed() {
            return Boolean.parseBoolean(metadata.getOrDefault("gate.passed", "false"));
        }

        public boolean promotable() {
            return Boolean.parseBoolean(metadata.getOrDefault("gate.promotable", "false"));
        }

        public String decisionStatus() {
            return metadata.getOrDefault("gate.decision.status", "UNKNOWN");
        }

        public Optional<String> decisionCandidate() {
            String candidate = metadata.getOrDefault("gate.decision.candidate", "").trim();
            return candidate.isEmpty() ? Optional.empty() : Optional.of(candidate);
        }

        public Optional<ArtifactEntry> artifact(String name) {
            return Optional.ofNullable(artifacts.get(name));
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("manifestFile", manifestFile.toString());
            map.put("formatVersion", formatVersion);
            map.put("generatedAt", generatedAt.toString());
            map.put("manifestSha256", manifestSha256);
            map.put("passed", passed());
            map.put("promotable", promotable());
            map.put("decisionStatus", decisionStatus());
            decisionCandidate().ifPresent(candidate -> map.put("decisionCandidate", candidate));
            map.put("artifacts", artifactsToMap(artifacts));
            map.put("metadata", metadata);
            return Map.copyOf(map);
        }
    }

    public record ManifestVerification(
            ManifestInspection inspection,
            String expectedManifestSha256,
            boolean manifestSha256Matches,
            boolean artifactBytesMatch,
            boolean artifactSha256Match,
            TrainingReportPromotionGateArtifacts.ArtifactVerification artifactVerification,
            List<String> failures) {
        public ManifestVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedManifestSha256 = normalizeChecksum(expectedManifestSha256);
            failures = failures == null ? List.of() : List.copyOf(failures);
        }

        public boolean passed() {
            return failures.isEmpty();
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public String message() {
            if (passed()) {
                return "Promotion gate artifact manifest verified for " + inspection.directory() + ".";
            }
            return "Promotion gate artifact manifest verification failed: " + String.join("; ", failures) + ".";
        }

        public Optional<TrainingReportPromotionGateArtifacts.ArtifactVerification> artifactVerificationOptional() {
            return Optional.ofNullable(artifactVerification);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("manifestSha256Matches", manifestSha256Matches);
            map.put("artifactBytesMatch", artifactBytesMatch);
            map.put("artifactSha256Match", artifactSha256Match);
            map.put("actualManifestSha256", inspection.manifestSha256());
            if (expectedManifestSha256 != null) {
                map.put("expectedManifestSha256", expectedManifestSha256);
            }
            if (artifactVerification != null) {
                map.put("artifactVerification", artifactVerification.toMap());
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }
    }

    public static ManifestBundle write(
            TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts) throws IOException {
        return write(artifacts, Options.defaults(), Instant.now());
    }

    public static ManifestBundle write(
            TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts,
            Instant generatedAt) throws IOException {
        return write(artifacts, Options.defaults(), generatedAt);
    }

    public static ManifestBundle write(
            TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts,
            Options options) throws IOException {
        return write(artifacts, options, Instant.now());
    }

    public static ManifestBundle write(
            TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts,
            Options options,
            Instant generatedAt) throws IOException {
        return write(artifacts, Map.of(), options, generatedAt);
    }

    public static ManifestBundle write(
            TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts,
            Map<String, Path> extraArtifacts,
            Options options,
            Instant generatedAt) throws IOException {
        TrainingReportPromotionGateArtifacts.ArtifactBundle resolvedArtifacts =
                Objects.requireNonNull(artifacts, "artifacts must not be null");
        Map<String, Path> resolvedExtraArtifacts =
                extraArtifacts == null ? Map.of() : Map.copyOf(extraArtifacts);
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Path manifestFile = resolvedArtifacts.directory().resolve(resolvedOptions.fileName());
        Properties manifest = TrainerArtifactManifest.build(
                artifactPaths(resolvedArtifacts, resolvedExtraArtifacts),
                FORMAT_VERSION,
                resolvedGeneratedAt);
        addGateMetadata(manifest, resolvedArtifacts);
        TrainerCheckpointIO.writePropertiesAtomically(
                manifestFile,
                manifest,
                "Gollek training promotion gate artifact manifest");
        return new ManifestBundle(
                resolvedArtifacts.directory(),
                manifestFile,
                TrainerCheckpointIO.sha256Hex(manifestFile),
                resolvedGeneratedAt,
                resolvedArtifacts);
    }

    public static ManifestInspection read(Path directory) throws IOException {
        return read(directory, Options.defaults());
    }

    public static ManifestInspection read(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return readFile(resolvedDirectory.resolve(resolvedOptions.fileName()));
    }

    public static ManifestInspection readFile(Path manifestFile) throws IOException {
        TrainerArtifactManifest.Inspection manifest = TrainerArtifactManifest.read(manifestFile);
        Path directory = manifest.manifestFile().getParent();
        if (directory == null) {
            directory = Path.of(".").toAbsolutePath().normalize();
        }
        return new ManifestInspection(
                directory,
                manifest.manifestFile(),
                manifest.formatVersion(),
                manifest.generatedAt(),
                manifest.sha256(),
                publicEntries(directory, manifest.artifacts()),
                gateMetadata(manifest.properties()));
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactInspection readArtifacts(
            ManifestInspection inspection) throws IOException {
        Objects.requireNonNull(inspection, "inspection must not be null");
        return TrainingReportPromotionGateArtifacts.readFiles(
                requiredArtifact(inspection, JSON_ARTIFACT).file(),
                requiredArtifact(inspection, MARKDOWN_ARTIFACT).file(),
                requiredArtifact(inspection, JUNIT_XML_ARTIFACT).file());
    }

    public static ManifestInspection refresh(
            ManifestInspection inspection,
            Instant generatedAt) throws IOException {
        ManifestInspection resolvedInspection = Objects.requireNonNull(
                inspection,
                "inspection must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Properties manifest = TrainerArtifactManifest.build(
                artifactPaths(resolvedInspection),
                FORMAT_VERSION,
                resolvedGeneratedAt);
        copyGateMetadata(manifest, resolvedInspection.metadata());
        TrainerCheckpointIO.writePropertiesAtomically(
                resolvedInspection.manifestFile(),
                manifest,
                "Gollek training promotion gate artifact manifest");
        return readFile(resolvedInspection.manifestFile());
    }

    public static ManifestVerification verify(ManifestBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verifyFile(bundle.manifestFile(), bundle.manifestSha256());
    }

    public static ManifestVerification verify(Path directory) throws IOException {
        return verify(directory, null, Options.defaults());
    }

    public static ManifestVerification verify(
            Path directory,
            String expectedManifestSha256) throws IOException {
        return verify(directory, expectedManifestSha256, Options.defaults());
    }

    public static ManifestVerification verify(
            Path directory,
            String expectedManifestSha256,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return verifyFile(resolvedDirectory.resolve(resolvedOptions.fileName()), expectedManifestSha256);
    }

    public static ManifestVerification verifyFile(
            Path manifestFile,
            String expectedManifestSha256) throws IOException {
        ManifestInspection inspection = readFile(manifestFile);
        String expectedManifest = normalizeChecksum(expectedManifestSha256);
        boolean manifestMatches = expectedManifest == null
                || expectedManifest.equalsIgnoreCase(inspection.manifestSha256());
        List<String> failures = new ArrayList<>();
        if (!manifestMatches) {
            failures.add("Manifest checksum mismatch for " + inspection.manifestFile());
        }
        if (inspection.formatVersion() != FORMAT_VERSION) {
            failures.add("Unsupported promotion gate artifact manifest format version "
                    + inspection.formatVersion() + " (supported: " + FORMAT_VERSION + ")");
        }

        ArtifactIntegrity artifactIntegrity = verifyArtifactIntegrity(inspection, failures);
        TrainingReportPromotionGateArtifacts.ArtifactVerification artifactVerification = null;
        if (hasRequiredArtifacts(inspection, failures)) {
            TrainingReportPromotionGateArtifacts.ArtifactInspection artifacts = TrainingReportPromotionGateArtifacts.readFiles(
                    requiredArtifact(inspection, JSON_ARTIFACT).file(),
                    requiredArtifact(inspection, MARKDOWN_ARTIFACT).file(),
                    requiredArtifact(inspection, JUNIT_XML_ARTIFACT).file());
            artifactVerification = TrainingReportPromotionGateArtifacts.verify(
                    artifacts,
                    requiredArtifact(inspection, JSON_ARTIFACT).sha256(),
                    requiredArtifact(inspection, MARKDOWN_ARTIFACT).sha256(),
                    requiredArtifact(inspection, JUNIT_XML_ARTIFACT).sha256());
            failures.addAll(artifactVerification.failures());
        }
        return new ManifestVerification(
                inspection,
                expectedManifest,
                manifestMatches,
                artifactIntegrity.bytesMatch(),
                artifactIntegrity.sha256Match(),
                artifactVerification,
                failures);
    }

    private static Map<String, Path> artifactPaths(
            TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts,
            Map<String, Path> extraArtifacts) {
        Map<String, Path> paths = new LinkedHashMap<>();
        paths.put(JSON_ARTIFACT, artifacts.jsonFile());
        paths.put(MARKDOWN_ARTIFACT, artifacts.markdownFile());
        paths.put(JUNIT_XML_ARTIFACT, artifacts.junitXmlFile());
        for (Map.Entry<String, Path> entry : extraArtifacts.entrySet()) {
            String name = entry.getKey();
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("extra artifact name must not be blank");
            }
            String normalizedName = name.trim();
            if (paths.containsKey(normalizedName)) {
                throw new IllegalArgumentException("duplicate artifact manifest entry: " + normalizedName);
            }
            paths.put(normalizedName, Objects.requireNonNull(entry.getValue(), "extra artifact path must not be null"));
        }
        return paths;
    }

    private static Map<String, Path> artifactPaths(ManifestInspection inspection) {
        Map<String, Path> paths = new LinkedHashMap<>();
        for (Map.Entry<String, ArtifactEntry> entry : inspection.artifacts().entrySet()) {
            paths.put(entry.getKey(), entry.getValue().file());
        }
        return Map.copyOf(paths);
    }

    private static void addGateMetadata(
            Properties manifest,
            TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts) {
        manifest.setProperty("gate.passed", Boolean.toString(artifacts.passed()));
        manifest.setProperty("gate.promotable", Boolean.toString(artifacts.promotable()));
        manifest.setProperty("gate.decision.status", artifacts.result().decision().status().name());
        artifacts.result().decision().candidate()
                .ifPresent(candidate -> manifest.setProperty("gate.decision.candidate", candidate.name()));
    }

    private static void copyGateMetadata(Properties manifest, Map<String, String> metadata) {
        for (Map.Entry<String, String> entry : metadata.entrySet()) {
            if (entry.getKey().startsWith("gate.")) {
                manifest.setProperty(entry.getKey(), entry.getValue());
            }
        }
    }

    private static Map<String, ArtifactEntry> publicEntries(
            Path directory,
            Map<String, TrainerArtifactManifest.ArtifactEntry> entries) {
        Map<String, ArtifactEntry> values = new LinkedHashMap<>();
        for (Map.Entry<String, TrainerArtifactManifest.ArtifactEntry> entry : entries.entrySet()) {
            TrainerArtifactManifest.ArtifactEntry artifact = entry.getValue();
            values.put(entry.getKey(), new ArtifactEntry(
                    artifact.name(),
                    artifact.resolve(directory),
                    artifact.bytes(),
                    artifact.sha256()));
        }
        return Map.copyOf(values);
    }

    private static Map<String, String> gateMetadata(Map<String, String> properties) {
        Map<String, String> metadata = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : properties.entrySet()) {
            if (entry.getKey().startsWith("gate.")) {
                metadata.put(entry.getKey(), entry.getValue());
            }
        }
        return Map.copyOf(metadata);
    }

    private static ArtifactIntegrity verifyArtifactIntegrity(ManifestInspection inspection, List<String> failures) {
        boolean bytesMatch = true;
        boolean sha256Match = true;
        for (ArtifactEntry artifact : inspection.artifacts().values()) {
            if (!Files.isRegularFile(artifact.file())) {
                failures.add(artifact.name() + " artifact is missing: " + artifact.file());
                bytesMatch = false;
                sha256Match = false;
                continue;
            }
            try {
                long actualBytes = Files.size(artifact.file());
                if (actualBytes != artifact.bytes()) {
                    failures.add(artifact.name() + " artifact byte size mismatch for " + artifact.file()
                            + " (expected " + artifact.bytes() + " bytes, got " + actualBytes + " bytes)");
                    bytesMatch = false;
                }
            } catch (IOException error) {
                failures.add(artifact.name() + " artifact byte size could not be read: " + error.getMessage());
                bytesMatch = false;
            }
            try {
                String actualSha256 = TrainerCheckpointIO.sha256Hex(artifact.file());
                if (!artifact.sha256().equalsIgnoreCase(actualSha256)) {
                    failures.add(artifact.name() + " artifact SHA-256 mismatch for " + artifact.file()
                            + " (expected " + artifact.sha256() + ", got " + actualSha256 + ")");
                    sha256Match = false;
                }
            } catch (IOException error) {
                failures.add(artifact.name() + " artifact SHA-256 could not be read: " + error.getMessage());
                sha256Match = false;
            }
        }
        return new ArtifactIntegrity(bytesMatch, sha256Match);
    }

    private static boolean hasRequiredArtifacts(ManifestInspection inspection, List<String> failures) {
        boolean present = true;
        for (String name : List.of(JSON_ARTIFACT, MARKDOWN_ARTIFACT, JUNIT_XML_ARTIFACT)) {
            ArtifactEntry artifact = inspection.artifacts().get(name);
            if (artifact == null) {
                failures.add("Manifest is missing required " + name + " artifact entry");
                present = false;
            } else if (!Files.isRegularFile(artifact.file())) {
                present = false;
            }
        }
        return present;
    }

    private static ArtifactEntry requiredArtifact(ManifestInspection inspection, String name) {
        return inspection.artifact(name)
                .orElseThrow(() -> new IllegalArgumentException("Manifest is missing required " + name + " artifact"));
    }

    private static Map<String, Object> artifactsToMap(Map<String, ArtifactEntry> artifacts) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<String, ArtifactEntry> entry : artifacts.entrySet()) {
            values.put(entry.getKey(), entry.getValue().toMap());
        }
        return Map.copyOf(values);
    }

    private static String normalizeFileName(String value, String fallback, String fieldName) {
        String normalized = value == null || value.isBlank() ? fallback : value.trim();
        Path path = Path.of(normalized);
        if (path.isAbsolute() || path.getParent() != null || ".".equals(normalized) || "..".equals(normalized)) {
            throw new IllegalArgumentException(fieldName + " must be a file name, not a path: " + value);
        }
        return normalized;
    }

    private static String normalizeChecksum(String checksum) {
        if (checksum == null || checksum.isBlank()) {
            return null;
        }
        return checksum.trim().toLowerCase(java.util.Locale.ROOT);
    }

    private record ArtifactIntegrity(boolean bytesMatch, boolean sha256Match) {
    }
}
