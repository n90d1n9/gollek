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

public final class OpSchemaRegistry {
    private final Map<String, OpSchema> schemas = new HashMap<>();

    public void register(OpSchema schema) {
        schemas.put(schema.opType, schema);
    }

    public OpSchema get(String opType) {
        return schemas.get(opType);
    }
}