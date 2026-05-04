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


import java.util.*;

public final class GGraphBuilder {
    private final List<GOp> ops = new ArrayList<>();
    private final Set<String> defined = new HashSet<>();
    private final List<GValueId> inputs = new ArrayList<>();
    private final List<GValueId> outputs = new ArrayList<>();

    public GValueId input(String name, GType type, Shape shape) {
        GValueId id = new GValueId(name);
        defined.add(name);
        inputs.add(id);
        return id;
    }

    public GValueId op(
            String opType,
            List<GValueId> inputs,
            String outputName,
            Map<String, GAttrValue> attrs) {
        for (GValueId in : inputs) {
            if (!defined.contains(in.id()))
                throw new RuntimeException("Undefined input: " + in);
        }
        GValueId out = new GValueId(outputName);
        ops.add(new GOp(
                opType,
                outputName,
                inputs.stream().map(i -> new GValueRef(i.id())).toList(),
                List.of(outputName), attrs));
        defined.add(outputName);
        return out;
    }

    public void output(GValueId id) {
        outputs.add(id);
    }

    public GGraph build() {
        return new GGraph(ops, inputs, outputs);
    }

}
