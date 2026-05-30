package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Reader/verifier for the promotion-gate evidence manifest.
 */
final class TrainingReportPromotionGateEvidenceManifestVerifier {
    private TrainingReportPromotionGateEvidenceManifestVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection read(Path evidenceFile)
            throws IOException {
        Path resolvedEvidenceFile = Objects.requireNonNull(evidenceFile, "evidenceFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedEvidenceFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification evidence JSON at "
                    + resolvedEvidenceFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification evidence at "
                    + resolvedEvidenceFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection(
                resolvedEvidenceFile,
                TrainerCheckpointIO.sha256Hex(resolvedEvidenceFile),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verify(
            Path evidenceFile,
            String expectedEvidenceSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection inspection =
                read(evidenceFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedEvidenceSha256 == null || expectedEvidenceSha256.isBlank()
                ? null
                : expectedEvidenceSha256.trim();
        boolean evidenceSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.evidenceSha256());
        if (!evidenceSha256Matches) {
            failures.add("Verification evidence checksum mismatch for " + inspection.evidenceFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.evidenceSha256());
        }
        boolean schemaValid = verifySchema(inspection, failures);
        boolean evidenceFilesValid = verifyEvidenceFilesReferences(inspection, failures);
        boolean packageArtifactsValid = verifyPackageArtifactReferences(inspection, failures);
        return new TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification(
                inspection,
                normalizedExpectedSha256,
                evidenceSha256Matches,
                schemaValid,
                schemaValid && evidenceFilesValid,
                schemaValid && packageArtifactsValid,
                failures);
    }

    private static boolean verifySchema(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> evidence = inspection.evidence();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_FORMAT.equals(inspection.format())) {
            failures.add("Verification evidence format mismatch for " + inspection.evidenceFile()
                    + ": expected " + TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_FORMAT
                    + " but found " + inspection.format());
        }
        requireInstant(evidence, "generatedAt", "verification evidence", failures);
        requireDirectoryPath(evidence, "packageDirectory", "verification evidence", failures);
        requireDirectoryPath(evidence, "reportDirectory", "verification evidence", failures);
        requireBoolean(evidence, "passed", "verification evidence", failures);
        requireBoolean(evidence, "promotable", "verification evidence", failures);
        requireString(evidence, "decisionStatus", "verification evidence", failures);
        requireOptionalString(evidence, "decisionCandidate", "verification evidence", failures);
        Map<String, Object> evidenceFiles = requireObject(evidence, "evidenceFiles", "verification evidence", failures);
        if (evidenceFiles != null) {
            for (String required : List.of(
                    "manifest",
                    "verificationJson",
                    "verificationMarkdown",
                    "verificationJunitXml")) {
                requireFileReference(evidenceFiles, required, "verification evidence evidenceFiles", failures);
            }
            for (String optional : List.of(
                    "verificationReportBundleReceipt",
                    "verificationIndex",
                    "verificationIndexReceipt",
                    "verificationIndexPackageAudit")) {
                objectValue(evidenceFiles, optional).ifPresent(reference ->
                        requireFileReferenceFields(
                                reference,
                                "verification evidence evidenceFiles." + optional,
                                failures));
            }
            validateEvidenceFileReferenceRoots(evidence, evidenceFiles, failures);
        }
        Map<String, Object> packageArtifacts =
                requireObject(evidence, "packageArtifacts", "verification evidence", failures);
        if (packageArtifacts != null) {
            for (Map.Entry<String, Object> artifact : packageArtifacts.entrySet()) {
                if (!(artifact.getValue() instanceof Map<?, ?> artifactMap)) {
                    failures.add("verification evidence packageArtifacts." + artifact.getKey()
                            + " must be an object");
                    continue;
                }
                Map<String, Object> artifactReference = immutableMap(artifactMap);
                requireString(
                        artifactReference,
                        "file",
                        "verification evidence packageArtifacts." + artifact.getKey(),
                        failures);
                requireSha256(
                        artifactReference,
                        "sha256",
                        "verification evidence packageArtifacts." + artifact.getKey(),
                        failures);
                requireNonNegativeNumber(
                        artifactReference,
                        "bytes",
                        "verification evidence packageArtifacts." + artifact.getKey(),
                        failures);
            }
            validatePackageArtifactReferenceRoots(evidence, packageArtifacts, failures);
        }
        Map<String, Object> sourceReportSnapshots =
                requireObject(evidence, "sourceReportSnapshots", "verification evidence", failures);
        if (sourceReportSnapshots != null) {
            requireBoolean(sourceReportSnapshots, "passed", "verification evidence sourceReportSnapshots", failures);
            requireIterable(sourceReportSnapshots, "snapshots", "verification evidence sourceReportSnapshots", failures);
            requireIterable(
                    sourceReportSnapshots,
                    "expectedSourceReportArtifacts",
                    "verification evidence sourceReportSnapshots",
                    failures);
            requireIterable(
                    sourceReportSnapshots,
                    "presentSourceReportArtifacts",
                    "verification evidence sourceReportSnapshots",
                    failures);
            requireIterable(
                    sourceReportSnapshots,
                    "missingSourceReportArtifacts",
                    "verification evidence sourceReportSnapshots",
                    failures);
            requireIterable(
                    sourceReportSnapshots,
                    "unexpectedSourceReportArtifacts",
                    "verification evidence sourceReportSnapshots",
                    failures);
            requireIterable(sourceReportSnapshots, "failures", "verification evidence sourceReportSnapshots", failures);
            requireObject(sourceReportSnapshots, "inspection", "verification evidence sourceReportSnapshots", failures);
            validateSourceSnapshotEntries(sourceReportSnapshots, packageArtifacts, failures);
            validateSourceSnapshotInventory(sourceReportSnapshots, failures);
        }
        return failures.size() == before;
    }

    private static void validateEvidenceFileReferenceRoots(
            Map<String, ?> evidence,
            Map<String, ?> evidenceFiles,
            List<String> failures) {
        Optional<Path> packageDirectory = pathValue(evidence, "packageDirectory");
        Optional<Path> reportDirectory = pathValue(evidence, "reportDirectory");
        for (Map.Entry<String, ?> entry : evidenceFiles.entrySet()) {
            if (!(entry.getValue() instanceof Map<?, ?> referenceMap)) {
                continue;
            }
            String key = entry.getKey();
            Optional<Path> root = "manifest".equals(key) ? packageDirectory : reportDirectory;
            root.ifPresent(path -> validateReferencePathWithin(
                    path,
                    immutableMap(referenceMap),
                    "file",
                    "verification evidence evidenceFiles." + key,
                    "manifest".equals(key) ? "packageDirectory" : "reportDirectory",
                    failures));
        }
    }

    private static void validatePackageArtifactReferenceRoots(
            Map<String, ?> evidence,
            Map<String, ?> packageArtifacts,
            List<String> failures) {
        Optional<Path> packageDirectory = pathValue(evidence, "packageDirectory");
        if (packageDirectory.isEmpty()) {
            return;
        }
        for (Map.Entry<String, ?> entry : packageArtifacts.entrySet()) {
            if (entry.getValue() instanceof Map<?, ?> referenceMap) {
                validateReferencePathWithin(
                        packageDirectory.orElseThrow(),
                        immutableMap(referenceMap),
                        "file",
                        "verification evidence packageArtifacts." + entry.getKey(),
                        "packageDirectory",
                        failures);
            }
        }
    }

    private static void validateReferencePathWithin(
            Path root,
            Map<String, ?> reference,
            String key,
            String owner,
            String rootLabel,
            List<String> failures) {
        Optional<Path> path = pathValue(reference, key);
        if (path.isPresent() && !isWithin(root, path.orElseThrow())) {
            failures.add(owner + "." + key + " is outside " + rootLabel + ": " + path.orElseThrow());
        }
    }

    private static boolean isWithin(Path root, Path candidate) {
        return candidate.toAbsolutePath().normalize().startsWith(root.toAbsolutePath().normalize());
    }

    private static void validateSourceSnapshotEntries(
            Map<String, ?> sourceReportSnapshots,
            Map<String, ?> packageArtifacts,
            List<String> failures) {
        List<?> snapshots = iterableValue(sourceReportSnapshots, "snapshots").orElse(List.of());
        int index = 0;
        for (Object item : snapshots) {
            if (!(item instanceof Map<?, ?> snapshotMap)) {
                failures.add("verification evidence sourceReportSnapshots.snapshots["
                        + index + "] must be an object");
                index++;
                continue;
            }
            Map<String, Object> snapshot = immutableMap(snapshotMap);
            String owner = "verification evidence sourceReportSnapshots.snapshots[" + index + "]";
            requireString(snapshot, "role", owner, failures);
            requireString(snapshot, "name", owner, failures);
            requireString(snapshot, "snapshotArtifact", owner, failures);
            requireString(snapshot, "snapshotFile", owner, failures);
            requireNonNegativeNumber(snapshot, "snapshotBytes", owner, failures);
            requireSha256(snapshot, "snapshotSha256", owner, failures);
            requireBoolean(snapshot, "manifestBytesMatchSource", owner, failures);
            requireBoolean(snapshot, "manifestSha256MatchesSource", owner, failures);
            Map<String, Object> sourceReport = requireObject(snapshot, "sourceReport", owner, failures);
            if (sourceReport != null) {
                validateEmbeddedSourceReport(sourceReport, owner + ".sourceReport", failures);
            }
            validateSourceSnapshotPackageArtifactReference(snapshot, owner, packageArtifacts, failures);
            index++;
        }
    }

    private static void validateSourceSnapshotPackageArtifactReference(
            Map<String, ?> snapshot,
            String owner,
            Map<String, ?> packageArtifacts,
            List<String> failures) {
        Optional<String> artifactName = stringValue(snapshot, "snapshotArtifact");
        if (artifactName.isEmpty()) {
            return;
        }
        Object artifact = packageArtifacts == null ? null : packageArtifacts.get(artifactName.orElseThrow());
        if (artifact == null) {
            failures.add(owner + " references missing package artifact " + artifactName.orElseThrow());
            return;
        }
        if (!(artifact instanceof Map<?, ?> artifactMap)) {
            return;
        }
        Map<String, Object> artifactReference = immutableMap(artifactMap);
        comparePathField(
                snapshot,
                "snapshotFile",
                artifactReference,
                "file",
                owner,
                "packageArtifacts." + artifactName.orElseThrow() + ".file",
                failures);
        compareLongField(
                snapshot,
                "snapshotBytes",
                artifactReference,
                "bytes",
                owner,
                "packageArtifacts." + artifactName.orElseThrow() + ".bytes",
                failures);
        compareSha256Field(
                snapshot,
                "snapshotSha256",
                artifactReference,
                "sha256",
                owner,
                "packageArtifacts." + artifactName.orElseThrow() + ".sha256",
                failures);
    }

    private static void validateSourceSnapshotInventory(
            Map<String, ?> sourceReportSnapshots,
            List<String> failures) {
        String owner = "verification evidence sourceReportSnapshots";
        List<String> snapshotArtifacts = snapshotArtifactNames(sourceReportSnapshots, failures);
        Optional<List<String>> expected =
                stringListValue(sourceReportSnapshots, "expectedSourceReportArtifacts", owner, failures);
        Optional<List<String>> present =
                stringListValue(sourceReportSnapshots, "presentSourceReportArtifacts", owner, failures);
        Optional<List<String>> missing =
                stringListValue(sourceReportSnapshots, "missingSourceReportArtifacts", owner, failures);
        Optional<List<String>> unexpected =
                stringListValue(sourceReportSnapshots, "unexpectedSourceReportArtifacts", owner, failures);

        expected.ifPresent(values -> rejectDuplicateStrings(values, owner + ".expectedSourceReportArtifacts", failures));
        present.ifPresent(values -> rejectDuplicateStrings(values, owner + ".presentSourceReportArtifacts", failures));
        missing.ifPresent(values -> rejectDuplicateStrings(values, owner + ".missingSourceReportArtifacts", failures));
        unexpected.ifPresent(values -> rejectDuplicateStrings(values, owner + ".unexpectedSourceReportArtifacts", failures));
        rejectDuplicateStrings(snapshotArtifacts, owner + ".snapshots[].snapshotArtifact", failures);

        if (present.isPresent() && !sameStringMembers(present.orElseThrow(), snapshotArtifacts)) {
            failures.add(owner + ".presentSourceReportArtifacts does not match snapshots[].snapshotArtifact");
        }

        if (expected.isPresent() && present.isPresent() && missing.isPresent()) {
            List<String> expectedMissing = difference(expected.orElseThrow(), present.orElseThrow());
            if (!missing.orElseThrow().equals(expectedMissing)) {
                failures.add(owner + ".missingSourceReportArtifacts does not match expected-minus-present artifacts");
            }
        }
        if (expected.isPresent() && present.isPresent() && unexpected.isPresent()) {
            List<String> expectedUnexpected = difference(present.orElseThrow(), expected.orElseThrow());
            if (!unexpected.orElseThrow().equals(expectedUnexpected)) {
                failures.add(owner + ".unexpectedSourceReportArtifacts does not match present-minus-expected artifacts");
            }
        }
    }

    private static List<String> snapshotArtifactNames(
            Map<String, ?> sourceReportSnapshots,
            List<String> failures) {
        List<?> snapshots = iterableValue(sourceReportSnapshots, "snapshots").orElse(List.of());
        List<String> artifactNames = new ArrayList<>();
        int index = 0;
        for (Object item : snapshots) {
            if (item instanceof Map<?, ?> snapshotMap) {
                Optional<String> artifactName = stringValue(immutableMap(snapshotMap), "snapshotArtifact");
                artifactName.ifPresent(artifactNames::add);
            } else {
                failures.add("verification evidence sourceReportSnapshots.snapshots["
                        + index + "] cannot be included in snapshot artifact inventory");
            }
            index++;
        }
        return List.copyOf(artifactNames);
    }

    private static boolean sameStringMembers(List<String> left, List<String> right) {
        return new LinkedHashSet<>(left).equals(new LinkedHashSet<>(right));
    }

    private static void validateEmbeddedSourceReport(
            Map<String, ?> sourceReport,
            String owner,
            List<String> failures) {
        requireString(sourceReport, "role", owner, failures);
        requireString(sourceReport, "name", owner, failures);
        requireOptionalString(sourceReport, "source", owner, failures);
        requireOptionalNumber(sourceReport, "bytes", owner, failures);
        requireOptionalSha256(sourceReport, "sha256", owner, failures);
    }

    private static void comparePathField(
            Map<String, ?> left,
            String leftKey,
            Map<String, ?> right,
            String rightKey,
            String owner,
            String rightLabel,
            List<String> failures) {
        Optional<Path> leftPath = pathValue(left, leftKey);
        Optional<Path> rightPath = pathValue(right, rightKey);
        if (leftPath.isPresent() && rightPath.isPresent() && !leftPath.orElseThrow().equals(rightPath.orElseThrow())) {
            failures.add(owner + " " + leftKey + " does not match " + rightLabel);
        }
    }

    private static void compareLongField(
            Map<String, ?> left,
            String leftKey,
            Map<String, ?> right,
            String rightKey,
            String owner,
            String rightLabel,
            List<String> failures) {
        Optional<Long> leftValue = longValue(left, leftKey);
        Optional<Long> rightValue = longValue(right, rightKey);
        if (leftValue.isPresent()
                && rightValue.isPresent()
                && leftValue.orElseThrow().longValue() != rightValue.orElseThrow().longValue()) {
            failures.add(owner + " " + leftKey + " does not match " + rightLabel);
        }
    }

    private static void compareSha256Field(
            Map<String, ?> left,
            String leftKey,
            Map<String, ?> right,
            String rightKey,
            String owner,
            String rightLabel,
            List<String> failures) {
        Optional<String> leftValue = stringValue(left, leftKey);
        Optional<String> rightValue = stringValue(right, rightKey);
        if (leftValue.isPresent()
                && rightValue.isPresent()
                && !leftValue.orElseThrow().equalsIgnoreCase(rightValue.orElseThrow())) {
            failures.add(owner + " " + leftKey + " does not match " + rightLabel);
        }
    }

    private static boolean verifyEvidenceFilesReferences(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection inspection,
            List<String> failures) throws IOException {
        int before = failures.size();
        Map<String, Object> evidenceFiles = objectValue(inspection.evidence(), "evidenceFiles").orElse(Map.of());
        for (Map.Entry<String, Object> entry : evidenceFiles.entrySet()) {
            if (!(entry.getValue() instanceof Map<?, ?> referenceMap)) {
                failures.add("Verification evidence evidenceFiles." + entry.getKey() + " must be an object");
                continue;
            }
            verifyFileReference(
                    "evidenceFiles." + entry.getKey(),
                    immutableMap(referenceMap),
                    null,
                    failures);
        }
        return failures.size() == before;
    }

    private static boolean verifyPackageArtifactReferences(
            TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection inspection,
            List<String> failures) throws IOException {
        int before = failures.size();
        Map<String, Object> packageArtifacts =
                objectValue(inspection.evidence(), "packageArtifacts").orElse(Map.of());
        for (Map.Entry<String, Object> entry : packageArtifacts.entrySet()) {
            if (!(entry.getValue() instanceof Map<?, ?> referenceMap)) {
                failures.add("Verification evidence packageArtifacts." + entry.getKey() + " must be an object");
                continue;
            }
            Map<String, Object> reference = immutableMap(referenceMap);
            verifyFileReference(
                    "packageArtifacts." + entry.getKey(),
                    reference,
                    longValue(reference, "bytes").orElse(null),
                    failures);
        }
        return failures.size() == before;
    }

    private static boolean verifyFileReference(
            String label,
            Map<String, ?> reference,
            Long expectedBytes,
            List<String> failures) throws IOException {
        int before = failures.size();
        String file = stringValue(reference, "file").orElse(null);
        String expectedSha256 = stringValue(reference, "sha256").orElse(null);
        if (file == null || file.isBlank()) {
            failures.add("Verification evidence " + label + " is missing file");
            return false;
        }
        if (expectedSha256 == null || expectedSha256.isBlank()) {
            failures.add("Verification evidence " + label + " is missing sha256");
            return false;
        }
        Path path;
        try {
            path = Path.of(file).toAbsolutePath().normalize();
        } catch (InvalidPathException error) {
            failures.add("Verification evidence " + label + " has invalid file path: " + file);
            return false;
        }
        if (!Files.isRegularFile(path)) {
            failures.add("Verification evidence " + label + " file is missing: " + path);
            return false;
        }
        if (expectedBytes != null) {
            long actualBytes = Files.size(path);
            if (actualBytes != expectedBytes.longValue()) {
                failures.add("Verification evidence " + label + " byte count mismatch for " + path
                        + ": expected " + expectedBytes + " but found " + actualBytes);
            }
        }
        String actualSha256 = TrainerCheckpointIO.sha256Hex(path);
        if (!expectedSha256.equalsIgnoreCase(actualSha256)) {
            failures.add("Verification evidence " + label + " checksum mismatch for " + path
                    + ": expected " + expectedSha256 + " but found " + actualSha256);
        }
        return failures.size() == before;
    }

    private static void requireString(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (stringValue(map, key).isEmpty()) {
            failures.add(owner + " is missing string field " + key);
        }
    }

    private static void requireDirectoryPath(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<Path> path = pathValue(map, key);
        if (path.isEmpty()) {
            failures.add(owner + " is missing directory path field " + key);
            return;
        }
        if (!Files.isDirectory(path.orElseThrow())) {
            failures.add(owner + " directory path field " + key + " is not a directory: "
                    + path.orElseThrow());
        }
    }

    private static Map<String, Object> requireObject(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Map<String, Object> value = objectValue(map, key).orElse(null);
        if (value == null) {
            failures.add(owner + " is missing object field " + key);
        }
        return value;
    }

    private static Map<String, Object> requireFileReference(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Map<String, Object> value = requireObject(map, key, owner, failures);
        if (value != null) {
            requireFileReferenceFields(value, owner + "." + key, failures);
        }
        return value;
    }

    private static void requireFileReferenceFields(
            Map<String, ?> reference,
            String owner,
            List<String> failures) {
        requireString(reference, "file", owner, failures);
        requireSha256(reference, "sha256", owner, failures);
    }

    private static void requireNumber(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (longValue(map, key).isEmpty()) {
            failures.add(owner + " is missing numeric field " + key);
        }
    }

    private static void requireNonNegativeNumber(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<Long> value = longValue(map, key);
        if (value.isEmpty()) {
            failures.add(owner + " is missing numeric field " + key);
            return;
        }
        if (value.orElseThrow().longValue() < 0L) {
            failures.add(owner + " has negative numeric field " + key);
        }
    }

    private static void requireSha256(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<String> value = stringValue(map, key);
        if (value.isEmpty()) {
            failures.add(owner + " is missing SHA-256 field " + key);
            return;
        }
        if (!isSha256Hex(value.orElseThrow())) {
            failures.add(owner + " has invalid SHA-256 field " + key);
        }
    }

    private static void requireInstant(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<String> value = stringValue(map, key);
        if (value.isEmpty()) {
            failures.add(owner + " is missing instant field " + key);
            return;
        }
        try {
            Instant.parse(value.orElseThrow());
        } catch (DateTimeParseException error) {
            failures.add(owner + " has invalid instant field " + key);
        }
    }

    private static void requireOptionalString(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (map.containsKey(key) && stringValue(map, key).isEmpty()) {
            failures.add(owner + " has invalid string field " + key);
        }
    }

    private static void requireOptionalNumber(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (map.containsKey(key) && longValue(map, key).isEmpty()) {
            failures.add(owner + " has invalid numeric field " + key);
        }
    }

    private static void requireOptionalSha256(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (!map.containsKey(key)) {
            return;
        }
        Optional<String> value = stringValue(map, key);
        if (value.isEmpty() || !isSha256Hex(value.orElseThrow())) {
            failures.add(owner + " has invalid SHA-256 field " + key);
        }
    }

    private static void requireBoolean(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (booleanValue(map, key).isEmpty()) {
            failures.add(owner + " is missing boolean field " + key);
        }
    }

    private static void requireIterable(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        if (iterableValue(map, key).isEmpty()) {
            failures.add(owner + " is missing array field " + key);
        }
    }

    private static Optional<List<String>> stringListValue(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<List<?>> values = iterableValue(map, key);
        if (values.isEmpty()) {
            return Optional.empty();
        }
        List<String> strings = new ArrayList<>();
        int index = 0;
        for (Object value : values.orElseThrow()) {
            if (value instanceof String text && !text.isBlank()) {
                strings.add(text);
            } else {
                failures.add(owner + "." + key + "[" + index + "] must be a non-blank string");
            }
            index++;
        }
        return Optional.of(List.copyOf(strings));
    }

    private static void rejectDuplicateStrings(
            List<String> values,
            String owner,
            List<String> failures) {
        Set<String> seen = new LinkedHashSet<>();
        for (String value : values) {
            if (!seen.add(value)) {
                failures.add(owner + " contains duplicate entry " + value);
            }
        }
    }

    private static List<String> difference(List<String> left, List<String> right) {
        Set<String> rightSet = new LinkedHashSet<>(right);
        List<String> values = new ArrayList<>();
        for (String value : left) {
            if (!rightSet.contains(value)) {
                values.add(value);
            }
        }
        return List.copyOf(values);
    }

    private static Optional<String> stringValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof String text && !text.isBlank()) {
            return Optional.of(text);
        }
        return Optional.empty();
    }

    private static Optional<Map<String, Object>> objectValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Map<?, ?> object) {
            return Optional.of(immutableMap(object));
        }
        return Optional.empty();
    }

    private static Optional<Long> longValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Number number) {
            return Optional.of(number.longValue());
        }
        if (value instanceof String text && !text.isBlank()) {
            try {
                return Optional.of(Long.parseLong(text));
            } catch (NumberFormatException ignored) {
                return Optional.empty();
            }
        }
        return Optional.empty();
    }

    private static Optional<Path> pathValue(Map<String, ?> map, String key) {
        Optional<String> value = stringValue(map, key);
        if (value.isEmpty()) {
            return Optional.empty();
        }
        try {
            return Optional.of(Path.of(value.orElseThrow()).toAbsolutePath().normalize());
        } catch (InvalidPathException error) {
            return Optional.empty();
        }
    }

    private static Optional<Boolean> booleanValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Boolean flag) {
            return Optional.of(flag);
        }
        if (value instanceof String text && !text.isBlank()) {
            String normalized = text.trim().toLowerCase();
            if ("true".equals(normalized)) {
                return Optional.of(Boolean.TRUE);
            }
            if ("false".equals(normalized)) {
                return Optional.of(Boolean.FALSE);
            }
        }
        return Optional.empty();
    }

    private static Optional<List<?>> iterableValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof List<?> list) {
            return Optional.of(List.copyOf(list));
        }
        if (value instanceof Iterable<?> iterable) {
            List<Object> values = new ArrayList<>();
            for (Object item : iterable) {
                values.add(item);
            }
            return Optional.of(List.copyOf(values));
        }
        return Optional.empty();
    }

    private static boolean isSha256Hex(String value) {
        if (value.length() != 64) {
            return false;
        }
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            boolean hex = (ch >= '0' && ch <= '9')
                    || (ch >= 'a' && ch <= 'f')
                    || (ch >= 'A' && ch <= 'F');
            if (!hex) {
                return false;
            }
        }
        return true;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableMap(Map<?, ?> map) {
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }
}
