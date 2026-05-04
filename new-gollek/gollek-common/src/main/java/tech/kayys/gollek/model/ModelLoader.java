package tech.kayys.gollek.model;

import java.nio.file.Path;

public interface ModelLoader {
    boolean supports(Path path);
    ModelAdapter load(Path path);
}