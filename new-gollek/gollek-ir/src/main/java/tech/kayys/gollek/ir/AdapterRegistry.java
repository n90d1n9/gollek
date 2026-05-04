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


import java.nio.file.Path;
import java.util.List;

/**
 * CONVERSION BETWEEN MODELS
 * ❗ Reality
 * ✅ Possible
 * LLM → LLM (LLaMA ↔ Mistral ↔ GPT-style)
 * FP32 ↔ FP16 ↔ INT8
 * ONNX ↔ IR ↔ runtime
 * ⚠ Partial
 * Diffusion ↔ Diffusion (with scheduler differences)
 * Vision ↔ Vision
 * ❌ Not universal
 * Diffusion ↔ LLM
 * CNN ↔ Transformer (without redesign)
 * ✅ So define:
 * “Convertible within domain”
 * 
 */
public final class AdapterRegistry {
    private final List<ModelLoader> loaders;

    public GollekGraph load(Path path) {
        for (ModelLoader l : loaders) {
            if (l.supports(path)) {
                return l.load(path);
            }
        }
        throw new RuntimeException("Unsupported format");
    }
}