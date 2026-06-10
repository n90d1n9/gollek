package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Reusable catalog, lock, and lock-validation helper for trainer schema bundles.
 */
public record DiscreteTokenDatasetSchemaBundle(
        Class<?> resourceOwner,
        String catalogKind,
        String catalogSchemaVersion,
        String catalogSchemaId,
        String catalogSchemaResource,
        String lockKind,
        String lockSchemaVersion,
        String lockSchemaId,
        String lockSchemaResource,
        String lockValidationKind,
        String lockValidationSchemaVersion,
        String lockValidationSchemaId,
        String lockValidationSchemaResource,
        String catalogLabel,
        String lockLabel,
        String lockValidationLabel,
        String validCode,
        String mismatchCode,
        List<DiscreteTokenDatasetSchemaContract> contracts) {
    public DiscreteTokenDatasetSchemaBundle {
        resourceOwner = Objects.requireNonNull(resourceOwner, "resourceOwner must not be null");
        catalogKind = DiscreteTokenDatasetMetadataSupport.requireText(catalogKind, "catalogKind");
        catalogSchemaVersion =
                DiscreteTokenDatasetMetadataSupport.requireText(catalogSchemaVersion, "catalogSchemaVersion");
        catalogSchemaId = DiscreteTokenDatasetMetadataSupport.requireText(catalogSchemaId, "catalogSchemaId");
        catalogSchemaResource =
                DiscreteTokenDatasetMetadataSupport.requireText(catalogSchemaResource, "catalogSchemaResource");
        lockKind = DiscreteTokenDatasetMetadataSupport.requireText(lockKind, "lockKind");
        lockSchemaVersion =
                DiscreteTokenDatasetMetadataSupport.requireText(lockSchemaVersion, "lockSchemaVersion");
        lockSchemaId = DiscreteTokenDatasetMetadataSupport.requireText(lockSchemaId, "lockSchemaId");
        lockSchemaResource =
                DiscreteTokenDatasetMetadataSupport.requireText(lockSchemaResource, "lockSchemaResource");
        lockValidationKind =
                DiscreteTokenDatasetMetadataSupport.requireText(lockValidationKind, "lockValidationKind");
        lockValidationSchemaVersion = DiscreteTokenDatasetMetadataSupport.requireText(
                lockValidationSchemaVersion,
                "lockValidationSchemaVersion");
        lockValidationSchemaId =
                DiscreteTokenDatasetMetadataSupport.requireText(lockValidationSchemaId, "lockValidationSchemaId");
        lockValidationSchemaResource = DiscreteTokenDatasetMetadataSupport.requireText(
                lockValidationSchemaResource,
                "lockValidationSchemaResource");
        catalogLabel = DiscreteTokenDatasetMetadataSupport.requireText(catalogLabel, "catalogLabel");
        lockLabel = DiscreteTokenDatasetMetadataSupport.requireText(lockLabel, "lockLabel");
        lockValidationLabel =
                DiscreteTokenDatasetMetadataSupport.requireText(lockValidationLabel, "lockValidationLabel");
        validCode = DiscreteTokenDatasetMetadataSupport.requireText(validCode, "validCode");
        mismatchCode = DiscreteTokenDatasetMetadataSupport.requireText(mismatchCode, "mismatchCode");
        contracts = validateContracts(contracts);
    }

    public static Builder builder(Class<?> resourceOwner) {
        return new Builder(resourceOwner);
    }

    public Map<String, Object> catalogMetadata() {
        return catalogMetadata(false);
    }

    public Map<String, Object> catalogMetadata(boolean includeSchemas) {
        List<Map<String, Object>> contractMetadata = contractsMetadata(includeSchemas);
        String catalogSchemaText = jsonSchemaText();
        String catalogSchemaSha256 = DiscreteTokenDatasetSchemaContract.sha256Hex(catalogSchemaText);
        int catalogSchemaByteCount = DiscreteTokenDatasetSchemaContract.utf8ByteCount(catalogSchemaText);
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("kind", catalogKind);
        metadata.put("schemaVersion", catalogSchemaVersion);
        metadata.put("catalogSchemaId", catalogSchemaId);
        metadata.put("catalogSchemaResource", catalogSchemaResource);
        metadata.put("catalogSchemaSha256", catalogSchemaSha256);
        metadata.put("catalogSchemaByteCount", catalogSchemaByteCount);
        metadata.put("schemaSetSha256", schemaSetSha256(catalogSchemaSha256, contractMetadata));
        metadata.put("schemaSetByteCount", schemaSetByteCount(catalogSchemaByteCount, contractMetadata));
        metadata.put("schemaCount", contractMetadata.size());
        metadata.put("contracts", contractMetadata);
        metadata.put("summary", catalogLabel + " with " + contractMetadata.size() + " contract(s)");
        return Collections.unmodifiableMap(metadata);
    }

    public DiscreteTokenDatasetSchemaCatalogSnapshot catalogSnapshot() {
        return catalogSnapshot(false);
    }

    public DiscreteTokenDatasetSchemaCatalogSnapshot catalogSnapshot(boolean includeSchemas) {
        return DiscreteTokenDatasetSchemaCatalogSnapshot.fromMetadata(catalogMetadata(includeSchemas));
    }

    public Map<String, Object> lockMetadata() {
        Map<String, Object> catalog = catalogMetadata(false);
        List<?> contractMetadata = (List<?>) catalog.get("contracts");
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("kind", lockKind);
        metadata.put("schemaVersion", lockSchemaVersion);
        metadata.put("schemaSetSha256", catalog.get("schemaSetSha256"));
        metadata.put("schemaSetByteCount", catalog.get("schemaSetByteCount"));
        metadata.put("schemaCount", catalog.get("schemaCount"));
        metadata.put("catalogSchemaId", catalog.get("catalogSchemaId"));
        metadata.put("catalogSchemaSha256", catalog.get("catalogSchemaSha256"));
        metadata.put("lockSchemaId", lockSchemaId);
        metadata.put("contracts", contractMetadata.stream()
                .map(DiscreteTokenDatasetSchemaBundle::lockContract)
                .toList());
        metadata.put("summary", lockLabel + " for " + contractMetadata.size() + " contract(s)");
        return Collections.unmodifiableMap(metadata);
    }

    public DiscreteTokenDatasetSchemaLockSnapshot lockSnapshot() {
        return DiscreteTokenDatasetSchemaLockSnapshot.fromMetadata(lockMetadata());
    }

    public Map<String, Object> validateCurrentLock() {
        return validateLockMetadata(lockMetadata());
    }

    public DiscreteTokenDatasetSchemaLockValidationReport validateCurrentLockReport() {
        return DiscreteTokenDatasetSchemaLockValidationReport.fromMetadata(validateCurrentLock());
    }

    public Map<String, Object> requireCurrentLockValid() {
        return requireValidLockMetadata(lockMetadata());
    }

    public DiscreteTokenDatasetSchemaLockValidationReport requireCurrentLockValidReport() {
        return DiscreteTokenDatasetSchemaLockValidationReport.fromMetadata(requireCurrentLockValid());
    }

    public Map<String, Object> requireValidLockMetadata(Map<?, ?> expectedLock) {
        Map<String, Object> validation = validateLockMetadata(expectedLock);
        requireValidLockValidation(validation);
        return validation;
    }

    public DiscreteTokenDatasetSchemaLockValidationReport validateLockMetadataReport(Map<?, ?> expectedLock) {
        return DiscreteTokenDatasetSchemaLockValidationReport.fromMetadata(validateLockMetadata(expectedLock));
    }

    public DiscreteTokenDatasetSchemaLockValidationReport requireValidLockMetadataReport(Map<?, ?> expectedLock) {
        DiscreteTokenDatasetSchemaLockValidationReport report = validateLockMetadataReport(expectedLock);
        report.requireValid();
        return report;
    }

    public Map<String, Object> validateLockSnapshot(DiscreteTokenDatasetSchemaLockSnapshot expectedLock) {
        Objects.requireNonNull(expectedLock, "expectedLock must not be null");
        return validateLockMetadata(expectedLock.toMetadata());
    }

    public DiscreteTokenDatasetSchemaLockValidationReport validateLockSnapshotReport(
            DiscreteTokenDatasetSchemaLockSnapshot expectedLock) {
        return DiscreteTokenDatasetSchemaLockValidationReport.fromMetadata(validateLockSnapshot(expectedLock));
    }

    public Map<String, Object> requireValidLockSnapshot(DiscreteTokenDatasetSchemaLockSnapshot expectedLock) {
        Map<String, Object> validation = validateLockSnapshot(expectedLock);
        requireValidLockValidation(validation);
        return validation;
    }

    public DiscreteTokenDatasetSchemaLockValidationReport requireValidLockSnapshotReport(
            DiscreteTokenDatasetSchemaLockSnapshot expectedLock) {
        DiscreteTokenDatasetSchemaLockValidationReport report = validateLockSnapshotReport(expectedLock);
        report.requireValid();
        return report;
    }

    public Map<String, Object> validateLockJson(String expectedLockJson) {
        return validateLockMetadata(DiscreteTokenDatasetCheckpointMetadataJson.fromJson(expectedLockJson));
    }

    public DiscreteTokenDatasetSchemaLockValidationReport validateLockJsonReport(String expectedLockJson) {
        return DiscreteTokenDatasetSchemaLockValidationReport.fromMetadata(validateLockJson(expectedLockJson));
    }

    public Map<String, Object> requireValidLockJson(String expectedLockJson) {
        Map<String, Object> validation = validateLockJson(expectedLockJson);
        requireValidLockValidation(validation);
        return validation;
    }

    public DiscreteTokenDatasetSchemaLockValidationReport requireValidLockJsonReport(String expectedLockJson) {
        DiscreteTokenDatasetSchemaLockValidationReport report = validateLockJsonReport(expectedLockJson);
        report.requireValid();
        return report;
    }

    public Map<String, Object> validateLockPath(Path expectedLockPath) throws IOException {
        return validateLockMetadata(DiscreteTokenDatasetCheckpointMetadataJson.read(expectedLockPath));
    }

    public DiscreteTokenDatasetSchemaLockValidationReport validateLockPathReport(Path expectedLockPath)
            throws IOException {
        return DiscreteTokenDatasetSchemaLockValidationReport.fromMetadata(validateLockPath(expectedLockPath));
    }

    public Map<String, Object> requireValidLockPath(Path expectedLockPath) throws IOException {
        Map<String, Object> validation = validateLockPath(expectedLockPath);
        requireValidLockValidation(validation);
        return validation;
    }

    public DiscreteTokenDatasetSchemaLockValidationReport requireValidLockPathReport(Path expectedLockPath)
            throws IOException {
        DiscreteTokenDatasetSchemaLockValidationReport report = validateLockPathReport(expectedLockPath);
        report.requireValid();
        return report;
    }

    public void requireValidLockValidation(Map<?, ?> validation) {
        Objects.requireNonNull(validation, "validation must not be null");
        if (Boolean.TRUE.equals(validation.get("valid"))) {
            return;
        }
        throw new IllegalStateException(validationFailureMessage(validation));
    }

    public Map<String, Object> validateLockMetadata(Map<?, ?> expectedLock) {
        Objects.requireNonNull(expectedLock, "expectedLock must not be null");
        Map<String, Object> currentLock = lockMetadata();
        List<Map<String, Object>> mismatches = new ArrayList<>();
        compareField(mismatches, "kind", expectedLock.get("kind"), lockKind);
        compareField(mismatches, "schemaVersion", expectedLock.get("schemaVersion"), lockSchemaVersion);
        compareField(
                mismatches,
                "schemaSetSha256",
                expectedLock.get("schemaSetSha256"),
                currentLock.get("schemaSetSha256"));
        compareField(
                mismatches,
                "schemaSetByteCount",
                expectedLock.get("schemaSetByteCount"),
                currentLock.get("schemaSetByteCount"));
        compareField(
                mismatches,
                "schemaCount",
                expectedLock.get("schemaCount"),
                currentLock.get("schemaCount"));
        compareField(
                mismatches,
                "catalogSchemaId",
                expectedLock.get("catalogSchemaId"),
                currentLock.get("catalogSchemaId"));
        compareField(
                mismatches,
                "catalogSchemaSha256",
                expectedLock.get("catalogSchemaSha256"),
                currentLock.get("catalogSchemaSha256"));
        compareField(
                mismatches,
                "lockSchemaId",
                expectedLock.get("lockSchemaId"),
                currentLock.get("lockSchemaId"));
        compareContracts(mismatches, expectedLock.get("contracts"), currentLock.get("contracts"));

        boolean valid = mismatches.isEmpty();
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("kind", lockValidationKind);
        metadata.put("schemaVersion", lockValidationSchemaVersion);
        metadata.put("validationSchemaId", lockValidationSchemaId);
        metadata.put("validationSchemaResource", lockValidationSchemaResource);
        metadata.put("status", valid ? "valid" : "invalid");
        metadata.put("valid", valid);
        metadata.put("code", valid ? validCode : mismatchCode);
        metadata.put("message", valid
                ? lockLabel + " matches current schema set"
                : lockLabel + " does not match current schema set");
        metadata.put("expectedSchemaSetSha256", expectedLock.get("schemaSetSha256"));
        metadata.put("actualSchemaSetSha256", currentLock.get("schemaSetSha256"));
        metadata.put(
                "schemaSetMatch",
                Objects.equals(expectedLock.get("schemaSetSha256"), currentLock.get("schemaSetSha256")));
        metadata.put("expectedSchemaCount", expectedLock.get("schemaCount"));
        metadata.put("actualSchemaCount", currentLock.get("schemaCount"));
        metadata.put("expectedLockSchemaId", expectedLock.get("lockSchemaId"));
        metadata.put("actualLockSchemaId", currentLock.get("lockSchemaId"));
        metadata.put("mismatchCount", mismatches.size());
        metadata.put("mismatches", List.copyOf(mismatches));
        metadata.put("summary", valid
                ? lockLabel + " valid"
                : lockLabel + " mismatch: " + mismatches.size() + " issue(s)");
        return Collections.unmodifiableMap(metadata);
    }

    public List<Map<String, Object>> contractsMetadata() {
        return contractsMetadata(false);
    }

    public List<Map<String, Object>> contractsMetadata(boolean includeSchemas) {
        return contracts.stream()
                .map(contract -> contract.toMetadata(includeSchemas))
                .toList();
    }

    public String jsonSchemaText() {
        return DiscreteTokenDatasetSchemaContract.resourceText(
                resourceOwner,
                catalogSchemaResource,
                catalogLabel + " JSON schema");
    }

    public Map<String, Object> jsonSchemaMetadata() {
        return DiscreteTokenDatasetSchemaContract.jsonMetadata(
                resourceOwner,
                catalogSchemaResource,
                catalogLabel + " JSON schema");
    }

    public String lockJsonSchemaText() {
        return DiscreteTokenDatasetSchemaContract.resourceText(
                resourceOwner,
                lockSchemaResource,
                lockLabel + " JSON schema");
    }

    public Map<String, Object> lockJsonSchemaMetadata() {
        return DiscreteTokenDatasetSchemaContract.jsonMetadata(
                resourceOwner,
                lockSchemaResource,
                lockLabel + " JSON schema");
    }

    public String lockValidationJsonSchemaText() {
        return DiscreteTokenDatasetSchemaContract.resourceText(
                resourceOwner,
                lockValidationSchemaResource,
                lockValidationLabel + " JSON schema");
    }

    public Map<String, Object> lockValidationJsonSchemaMetadata() {
        return DiscreteTokenDatasetSchemaContract.jsonMetadata(
                resourceOwner,
                lockValidationSchemaResource,
                lockValidationLabel + " JSON schema");
    }

    public static final class Builder {
        private final Class<?> resourceOwner;
        private final List<DiscreteTokenDatasetSchemaContract> contracts = new ArrayList<>();
        private String catalogKind;
        private String catalogSchemaVersion;
        private String catalogSchemaId;
        private String catalogSchemaResource;
        private String lockKind;
        private String lockSchemaVersion;
        private String lockSchemaId;
        private String lockSchemaResource;
        private String lockValidationKind;
        private String lockValidationSchemaVersion;
        private String lockValidationSchemaId;
        private String lockValidationSchemaResource;
        private String catalogLabel;
        private String lockLabel;
        private String lockValidationLabel;
        private String validCode;
        private String mismatchCode;

        private Builder(Class<?> resourceOwner) {
            this.resourceOwner = Objects.requireNonNull(resourceOwner, "resourceOwner must not be null");
        }

        public Builder catalog(String kind, String schemaVersion, String schemaId, String schemaResource) {
            this.catalogKind = kind;
            this.catalogSchemaVersion = schemaVersion;
            this.catalogSchemaId = schemaId;
            this.catalogSchemaResource = schemaResource;
            return this;
        }

        public Builder lock(String kind, String schemaVersion, String schemaId, String schemaResource) {
            this.lockKind = kind;
            this.lockSchemaVersion = schemaVersion;
            this.lockSchemaId = schemaId;
            this.lockSchemaResource = schemaResource;
            return this;
        }

        public Builder lockValidation(String kind, String schemaVersion, String schemaId, String schemaResource) {
            this.lockValidationKind = kind;
            this.lockValidationSchemaVersion = schemaVersion;
            this.lockValidationSchemaId = schemaId;
            this.lockValidationSchemaResource = schemaResource;
            return this;
        }

        public Builder labels(String catalogLabel, String lockLabel, String lockValidationLabel) {
            this.catalogLabel = catalogLabel;
            this.lockLabel = lockLabel;
            this.lockValidationLabel = lockValidationLabel;
            return this;
        }

        public Builder codes(String validCode, String mismatchCode) {
            this.validCode = validCode;
            this.mismatchCode = mismatchCode;
            return this;
        }

        public Builder addContract(DiscreteTokenDatasetSchemaContract contract) {
            contracts.add(Objects.requireNonNull(contract, "contract must not be null"));
            return this;
        }

        public Builder contracts(List<DiscreteTokenDatasetSchemaContract> contracts) {
            this.contracts.clear();
            this.contracts.addAll(validateContracts(contracts));
            return this;
        }

        public DiscreteTokenDatasetSchemaBundle build() {
            return new DiscreteTokenDatasetSchemaBundle(
                    resourceOwner,
                    catalogKind,
                    catalogSchemaVersion,
                    catalogSchemaId,
                    catalogSchemaResource,
                    lockKind,
                    lockSchemaVersion,
                    lockSchemaId,
                    lockSchemaResource,
                    lockValidationKind,
                    lockValidationSchemaVersion,
                    lockValidationSchemaId,
                    lockValidationSchemaResource,
                    catalogLabel,
                    lockLabel,
                    lockValidationLabel,
                    validCode,
                    mismatchCode,
                    contracts);
        }
    }

    private static Map<String, Object> lockContract(Object value) {
        Map<?, ?> contract = (Map<?, ?>) value;
        Map<String, Object> lock = new LinkedHashMap<>();
        lock.put("name", contract.get("name"));
        lock.put("payloadKind", contract.get("payloadKind"));
        lock.put("payloadSchemaVersion", contract.get("payloadSchemaVersion"));
        lock.put("jsonSchemaId", contract.get("jsonSchemaId"));
        lock.put("jsonSchemaSha256", contract.get("jsonSchemaSha256"));
        lock.put("jsonSchemaByteCount", contract.get("jsonSchemaByteCount"));
        return Collections.unmodifiableMap(lock);
    }

    private static void compareField(
            List<Map<String, Object>> mismatches,
            String field,
            Object expected,
            Object actual) {
        if (!valuesEqual(expected, actual)) {
            mismatches.add(mismatch(field, field, expected, actual));
        }
    }

    private static void compareContracts(List<Map<String, Object>> mismatches, Object expected, Object actual) {
        if (!(expected instanceof List<?> expectedContracts)) {
            mismatches.add(mismatch("contracts", "contracts", expected, actual));
            return;
        }
        if (!(actual instanceof List<?> actualContracts)) {
            mismatches.add(mismatch("contracts", "contracts", expected, actual));
            return;
        }
        Map<String, Map<?, ?>> expectedByName = contractsByName(expectedContracts);
        Map<String, Map<?, ?>> actualByName = contractsByName(actualContracts);
        for (Map.Entry<String, Map<?, ?>> entry : actualByName.entrySet()) {
            String name = entry.getKey();
            Map<?, ?> expectedContract = expectedByName.get(name);
            Map<?, ?> actualContract = entry.getValue();
            if (expectedContract == null) {
                mismatches.add(mismatch(
                        "missing-contract",
                        "contracts." + name,
                        null,
                        contractSummary(actualContract)));
                continue;
            }
            compareContractField(mismatches, name, expectedContract, actualContract, "payloadKind");
            compareContractField(mismatches, name, expectedContract, actualContract, "payloadSchemaVersion");
            compareContractField(mismatches, name, expectedContract, actualContract, "jsonSchemaId");
            compareContractField(mismatches, name, expectedContract, actualContract, "jsonSchemaSha256");
            compareContractField(mismatches, name, expectedContract, actualContract, "jsonSchemaByteCount");
        }
        for (String name : expectedByName.keySet()) {
            if (!actualByName.containsKey(name)) {
                mismatches.add(mismatch(
                        "extra-contract",
                        "contracts." + name,
                        contractSummary(expectedByName.get(name)),
                        null));
            }
        }
    }

    private static void compareContractField(
            List<Map<String, Object>> mismatches,
            String contractName,
            Map<?, ?> expected,
            Map<?, ?> actual,
            String field) {
        Object expectedValue = expected.get(field);
        Object actualValue = actual.get(field);
        if (!valuesEqual(expectedValue, actualValue)) {
            mismatches.add(mismatch(
                    "contract-" + field,
                    "contracts." + contractName + "." + field,
                    expectedValue,
                    actualValue));
        }
    }

    private static Map<String, Map<?, ?>> contractsByName(List<?> contractMetadata) {
        Map<String, Map<?, ?>> byName = new LinkedHashMap<>();
        for (Object value : contractMetadata) {
            if (value instanceof Map<?, ?> contract && contract.get("name") != null) {
                byName.put(String.valueOf(contract.get("name")), contract);
            }
        }
        return byName;
    }

    private static Map<String, Object> contractSummary(Map<?, ?> contract) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("name", contract.get("name"));
        summary.put("payloadSchemaVersion", contract.get("payloadSchemaVersion"));
        summary.put("jsonSchemaId", contract.get("jsonSchemaId"));
        summary.put("jsonSchemaSha256", contract.get("jsonSchemaSha256"));
        return Collections.unmodifiableMap(summary);
    }

    private static boolean valuesEqual(Object expected, Object actual) {
        if (expected instanceof Number expectedNumber && actual instanceof Number actualNumber) {
            return expectedNumber.longValue() == actualNumber.longValue();
        }
        return Objects.equals(expected, actual);
    }

    private static Map<String, Object> mismatch(String type, String path, Object expected, Object actual) {
        Map<String, Object> mismatch = new LinkedHashMap<>();
        mismatch.put("type", type);
        mismatch.put("path", path);
        mismatch.put("expected", expected);
        mismatch.put("actual", actual);
        return Collections.unmodifiableMap(mismatch);
    }

    private String validationFailureMessage(Map<?, ?> validation) {
        StringBuilder builder = new StringBuilder(lockLabel + " invalid");
        Object summary = validation.get("summary");
        if (summary != null) {
            builder.append(": ").append(summary);
        }
        Object mismatches = validation.get("mismatches");
        if (mismatches instanceof List<?> list && !list.isEmpty()) {
            builder.append("; mismatches=");
            int count = 0;
            for (Object mismatch : list) {
                if (count > 0) {
                    builder.append(", ");
                }
                if (count >= 5) {
                    builder.append("...");
                    break;
                }
                builder.append(mismatchFailureSummary(mismatch));
                count++;
            }
        }
        return builder.toString();
    }

    private static String mismatchFailureSummary(Object mismatch) {
        if (mismatch instanceof Map<?, ?> map) {
            Object path = map.get("path");
            Object type = map.get("type");
            if (path != null && type != null) {
                return path + " (" + type + ")";
            }
            if (path != null) {
                return String.valueOf(path);
            }
        }
        return String.valueOf(mismatch);
    }

    private String schemaSetSha256(String catalogSchemaSha256, List<Map<String, Object>> contractMetadata) {
        StringBuilder builder = new StringBuilder();
        builder.append("catalog|")
                .append(catalogSchemaId)
                .append('|')
                .append(catalogSchemaSha256)
                .append('\n');
        for (Map<String, Object> contract : contractMetadata) {
            builder.append(contract.get("name"))
                    .append('|')
                    .append(contract.get("payloadSchemaVersion"))
                    .append('|')
                    .append(contract.get("jsonSchemaId"))
                    .append('|')
                    .append(contract.get("jsonSchemaSha256"))
                    .append('\n');
        }
        return DiscreteTokenDatasetSchemaContract.sha256Hex(builder.toString());
    }

    private static int schemaSetByteCount(int catalogSchemaByteCount, List<Map<String, Object>> contractMetadata) {
        int total = catalogSchemaByteCount;
        for (Map<String, Object> contract : contractMetadata) {
            total += ((Number) contract.get("jsonSchemaByteCount")).intValue();
        }
        return total;
    }

    private static List<DiscreteTokenDatasetSchemaContract> validateContracts(
            List<DiscreteTokenDatasetSchemaContract> source) {
        Objects.requireNonNull(source, "contracts must not be null");
        if (source.isEmpty()) {
            throw new IllegalArgumentException("contracts must not be empty");
        }
        List<DiscreteTokenDatasetSchemaContract> copy = new ArrayList<>(source.size());
        Set<String> names = new LinkedHashSet<>();
        for (DiscreteTokenDatasetSchemaContract contract : source) {
            DiscreteTokenDatasetSchemaContract checked =
                    Objects.requireNonNull(contract, "contracts must not contain null");
            if (!names.add(checked.name())) {
                throw new IllegalArgumentException("contracts contains duplicate contract name: " + checked.name());
            }
            copy.add(checked);
        }
        return List.copyOf(copy);
    }

}
