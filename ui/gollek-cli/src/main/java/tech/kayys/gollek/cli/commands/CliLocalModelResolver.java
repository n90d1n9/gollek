package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.gollek.safetensor.engine.route.LocalModelResolver;
import java.nio.file.Path;
import java.util.Optional;
import java.util.stream.Stream;

public class CliLocalModelResolver implements LocalModelResolver {

    @Override
    public Stream<LocalModelCandidate> localModels() {
        return LocalModelIndex.entries().stream().map(entry -> new LocalModelCandidate() {
            @Override
            public String format() { return entry.format; }
            @Override
            public String quantization() { return entry.quantStrategy; }
            @Override
            public String architecture() { return entry.architecture; }
            @Override
            public String provider() { return entry.source; }
            @Override
            public String id() { return entry.id; }
            @Override
            public String name() { return entry.name; }
            @Override
            public String shortId() { return entry.shortId; }
            @Override
            public Path path() { return entry.path == null ? null : Path.of(entry.path); }
        });
    }

    @Override
    public Optional<Path> resolveLiteRtModel(String modelId) {
        return LiteRtLmFastRun.findIndexedLiteRtModelPath(modelId);
    }
}
