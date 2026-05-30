package tech.kayys.gollek.train.data.multimodal;

import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * JSONL-backed dataset for real multimodal corpora.
 *
 * <p>Each non-blank line is one sample. The loader supports a concise shorthand:
 * {@code {"text":"caption","image":"images/cat.png","audio":"clips/cat.wav"}}
 * and an explicit {@code parts} form for mixed or future modalities.</p>
 *
 * <p>Relative asset paths are resolved against the manifest directory by default and
 * cannot escape that root. Binary assets are referenced as {@code file://} URIs unless
 * {@link Options#inlineBinaryAssets()} is enabled.</p>
 */
public final class MultimodalManifestDataset implements Dataset<List<MultimodalContent>> {
    public record Options(
            Path assetRoot,
            boolean inlineBinaryAssets,
            boolean requireExistingFiles,
            boolean allowAbsolutePaths) {
        public Options {
            assetRoot = Objects.requireNonNull(assetRoot, "assetRoot must not be null")
                    .toAbsolutePath()
                    .normalize();
        }

        public static Options forManifest(Path manifestPath) {
            Path manifest = Objects.requireNonNull(manifestPath, "manifestPath must not be null")
                    .toAbsolutePath()
                    .normalize();
            Path parent = manifest.getParent();
            return new Options(parent == null ? Path.of(".") : parent, false, true, false);
        }

        public Options withAssetRoot(Path assetRoot) {
            return new Options(assetRoot, inlineBinaryAssets, requireExistingFiles, allowAbsolutePaths);
        }

        public Options withInlineBinaryAssets(boolean inlineBinaryAssets) {
            return new Options(assetRoot, inlineBinaryAssets, requireExistingFiles, allowAbsolutePaths);
        }

        public Options withRequireExistingFiles(boolean requireExistingFiles) {
            return new Options(assetRoot, inlineBinaryAssets, requireExistingFiles, allowAbsolutePaths);
        }

        public Options withAllowAbsolutePaths(boolean allowAbsolutePaths) {
            return new Options(assetRoot, inlineBinaryAssets, requireExistingFiles, allowAbsolutePaths);
        }
    }

    private final Path manifestPath;
    private final Options options;
    private final List<List<MultimodalContent>> samples;

    public MultimodalManifestDataset(Path manifestPath) throws IOException {
        this(manifestPath, Options.forManifest(manifestPath));
    }

    public MultimodalManifestDataset(Path manifestPath, Options options) throws IOException {
        this.manifestPath = Objects.requireNonNull(manifestPath, "manifestPath must not be null")
                .toAbsolutePath()
                .normalize();
        if (!Files.isRegularFile(this.manifestPath)) {
            throw new IllegalArgumentException("manifestPath must be an existing file: " + this.manifestPath);
        }
        this.options = Objects.requireNonNull(options, "options must not be null");
        this.samples = Collections.unmodifiableList(load());
    }

    @Override
    public List<MultimodalContent> get(int index) {
        return samples.get(index);
    }

    @Override
    public int size() {
        return samples.size();
    }

    public Path manifestPath() {
        return manifestPath;
    }

    public Options options() {
        return options;
    }

    public List<List<MultimodalContent>> samples() {
        return samples;
    }

    private List<List<MultimodalContent>> load() throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        MultimodalManifestParser parser = new MultimodalManifestParser(
                mapper,
                new MultimodalManifestAssetResolver(options));
        List<List<MultimodalContent>> loaded = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(manifestPath, StandardCharsets.UTF_8)) {
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                List<MultimodalContent> sample = parser.parse(mapper.readTree(trimmed), lineNumber);
                loaded.add(MultimodalDatasetSupport.immutableSample(sample, "manifest line " + lineNumber));
            }
        }
        return loaded;
    }
}
