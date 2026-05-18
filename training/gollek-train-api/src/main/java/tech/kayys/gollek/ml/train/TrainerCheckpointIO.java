package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

final class TrainerCheckpointIO {
    private TrainerCheckpointIO() {
    }

    static void writeStringAtomically(Path target, String content) throws IOException {
        writeAtomically(target, tempFile -> Files.writeString(tempFile, content, StandardCharsets.UTF_8));
    }

    static void writePropertiesAtomically(Path target, Properties properties, String comment) throws IOException {
        writeAtomically(target, tempFile -> {
            try (Writer writer = Files.newBufferedWriter(
                    tempFile,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE)) {
                properties.store(writer, comment);
            }
        });
    }

    static void writeMapAtomically(Path target, Map<String, Object> state) throws IOException {
        writeAtomically(target, tempFile -> {
            try (ObjectOutputStream output = new ObjectOutputStream(Files.newOutputStream(tempFile))) {
                output.writeObject(state);
            }
        });
    }

    static Map<String, Object> readMap(Path checkpointFile, String artifactName)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream input = new ObjectInputStream(Files.newInputStream(checkpointFile))) {
            Object payload = input.readObject();
            if (!(payload instanceof Map<?, ?> rawMap)) {
                throw new IOException("Invalid " + artifactName + " checkpoint payload: expected Map");
            }
            Map<String, Object> state = new HashMap<>();
            for (Map.Entry<?, ?> entry : rawMap.entrySet()) {
                if (!(entry.getKey() instanceof String key)) {
                    throw new IOException("Invalid " + artifactName + " checkpoint payload: non-string key");
                }
                state.put(key, entry.getValue());
            }
            return state;
        }
    }

    static void writeAtomically(Path target, CheckedPathWriter writer) throws IOException {
        Path parent = target.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path tempFile = target.resolveSibling(target.getFileName() + ".tmp");
        try {
            writer.write(tempFile);
            moveIntoPlace(tempFile, target);
        } catch (IOException | RuntimeException error) {
            deleteTempFile(tempFile);
            throw error;
        } catch (Exception error) {
            deleteTempFile(tempFile);
            throw new IOException("Failed to write checkpoint atomically to " + target, error);
        }
    }

    static void moveIntoPlace(Path tempFile, Path target) throws IOException {
        try {
            Files.move(tempFile, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException atomicMoveUnsupported) {
            Files.move(tempFile, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    static String sha256Hex(Path path) throws IOException {
        MessageDigest digest = newSha256Digest();
        byte[] buffer = new byte[64 * 1024];
        try (InputStream input = Files.newInputStream(path)) {
            int read;
            while ((read = input.read(buffer)) >= 0) {
                if (read > 0) {
                    digest.update(buffer, 0, read);
                }
            }
        }
        return hex(digest.digest());
    }

    private static void deleteTempFile(Path tempFile) {
        try {
            Files.deleteIfExists(tempFile);
        } catch (IOException ignored) {
            // Preserve the original checkpoint write failure.
        }
    }

    private static MessageDigest newSha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 digest is not available", error);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            result.append(Character.forDigit((value >>> 4) & 0x0f, 16));
            result.append(Character.forDigit(value & 0x0f, 16));
        }
        return result.toString();
    }

    @FunctionalInterface
    interface CheckedPathWriter {
        void write(Path tempFile) throws Exception;
    }
}
