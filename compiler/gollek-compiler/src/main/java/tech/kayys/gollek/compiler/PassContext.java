package tech.kayys.gollek.compiler;

import tech.kayys.gollek.ir.OpSchemaRegistry;

public final class PassContext {
    public final OpSchemaRegistry schemaRegistry;

    public PassContext(OpSchemaRegistry schemaRegistry) {
        this.schemaRegistry = schemaRegistry;
    }
}