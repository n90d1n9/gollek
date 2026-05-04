package tech.kayys.gollek.ir;
import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.model.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.model.*;

import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.model.*;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;


public final class ModelRegistry {
    private final List<ModelLoader> loaders;

    public ModelAdapter load(Path path) {
        for (ModelLoader l : loaders) {
            if (l.supports(path)) {
                return l.load(path);
            }
        }
        throw new RuntimeException("Unsupported format");
    }
}