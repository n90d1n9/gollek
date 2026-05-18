package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Properties;

/**
 * Compatibility checks for model checkpoint metadata saved by the trainer.
 */
final class TrainerModelCheckpointMetadata {
    private TrainerModelCheckpointMetadata() {
    }

    static CompatibilityCheck check(
            Path metadataFile,
            Path checkpointFile,
            ExpectedModel expectedModel,
            int supportedMetadataVersion) {
        if (metadataFile == null) {
            return CompatibilityCheck.ok(false, false, null);
        }
        if (!Files.isRegularFile(metadataFile)) {
            return CompatibilityCheck.ok(false, true, null);
        }

        Properties metadata = new Properties();
        try (Reader reader = Files.newBufferedReader(metadataFile, StandardCharsets.UTF_8)) {
            metadata.load(reader);
        } catch (IOException error) {
            String message = "model metadata could not be read: " + error.getMessage();
            return CompatibilityCheck.incompatible(message, false, false, error.getMessage());
        }

        String mismatch = compatibilityMismatch(
                metadata,
                checkpointFile,
                expectedModel,
                supportedMetadataVersion);
        if (mismatch != null) {
            return CompatibilityCheck.incompatible(mismatch, true, false, null);
        }
        return CompatibilityCheck.ok(true, false, null);
    }

    static void write(
            Path metadataFile,
            Path checkpointFile,
            ExpectedModel expectedModel,
            int metadataVersion) throws IOException {
        if (metadataFile == null) {
            return;
        }
        Properties metadata = new Properties();
        metadata.setProperty("formatVersion", Integer.toString(metadataVersion));
        metadata.setProperty("modelClass", expectedModel.modelClass());
        metadata.setProperty("modelParameterCount", Long.toString(expectedModel.parameterCount()));
        metadata.setProperty("modelParameterSignature", expectedModel.parameterSignature());
        if (checkpointFile != null && Files.isRegularFile(checkpointFile)) {
            metadata.setProperty("modelCheckpointBytes", Long.toString(Files.size(checkpointFile)));
            metadata.setProperty("modelCheckpointSha256", TrainerCheckpointIO.sha256Hex(checkpointFile));
        }
        TrainerCheckpointIO.writePropertiesAtomically(
                metadataFile,
                metadata,
                "Gollek canonical trainer model checkpoint metadata");
    }

    static String compatibilityMismatch(
            Properties metadata,
            Path checkpointFile,
            ExpectedModel expectedModel,
            int supportedMetadataVersion) {
        String versionMismatch = TrainerCheckpointFileIntegrity.formatVersionMismatch(
                metadata.getProperty("formatVersion"),
                "model metadata",
                supportedMetadataVersion);
        if (versionMismatch != null) {
            return versionMismatch;
        }

        String integrityMismatch = TrainerCheckpointFileIntegrity.modelCheckpointMismatch(
                metadata,
                checkpointFile);
        if (integrityMismatch != null) {
            return integrityMismatch;
        }

        String savedModelClass = metadata.getProperty("modelClass");
        if (savedModelClass != null && !savedModelClass.equals(expectedModel.modelClass())) {
            return "model class mismatch (expected " + expectedModel.modelClass()
                    + ", got " + savedModelClass + ")";
        }

        String savedSignature = metadata.getProperty("modelParameterSignature");
        if (savedSignature != null && !savedSignature.equals(expectedModel.parameterSignature())) {
            return "model parameter signature mismatch";
        }

        String savedParameterCount = metadata.getProperty("modelParameterCount");
        String expectedParameterCount = Long.toString(expectedModel.parameterCount());
        if (savedParameterCount != null && !savedParameterCount.equals(expectedParameterCount)) {
            return "model parameter count mismatch (expected "
                    + expectedParameterCount + ", got " + savedParameterCount + ")";
        }

        return null;
    }

    record ExpectedModel(String modelClass, String parameterSignature, long parameterCount) {
        ExpectedModel {
            Objects.requireNonNull(modelClass, "modelClass must not be null");
            Objects.requireNonNull(parameterSignature, "parameterSignature must not be null");
        }
    }

    record CompatibilityCheck(
            TrainerCheckpointCompatibilityReport report,
            boolean loaded,
            boolean missing,
            String loadError) {
        static CompatibilityCheck ok(boolean loaded, boolean missing, String loadError) {
            return new CompatibilityCheck(
                    TrainerCheckpointCompatibilityReport.ok(),
                    loaded,
                    missing,
                    loadError);
        }

        static CompatibilityCheck incompatible(
                String error,
                boolean loaded,
                boolean missing,
                String loadError) {
            return new CompatibilityCheck(
                    TrainerCheckpointCompatibilityReport.incompatible(error),
                    loaded,
                    missing,
                    loadError);
        }
    }
}
