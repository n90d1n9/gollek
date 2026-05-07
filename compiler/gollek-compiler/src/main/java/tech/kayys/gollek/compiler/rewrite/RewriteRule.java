package tech.kayys.gollek.compiler.rewrite;

import tech.kayys.gollek.ir.*;

public interface RewriteRule {
    boolean matches(GGraph graph, int index);

    RewriteResult apply(GGraph graph, int index);
}