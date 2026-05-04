package tech.kayys.gollek.adapter;

import tech.kayys.gollek.model.ModelAdapter;
import tech.kayys.gollek.model.ModelLoader;
import java.nio.file.Path;

public final class GGUFModelLoader implements ModelLoader {
    @Override
    public boolean supports(Path path) {
        return path.toString().endsWith(".gguf");
    }

    @Override
    public ModelAdapter load(Path path) {
        return new GGUFAdapter();
    }
}
