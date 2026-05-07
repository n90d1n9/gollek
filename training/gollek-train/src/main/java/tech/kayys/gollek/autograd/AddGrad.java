package tech.kayys.gollek.autograd;

import tech.kayys.gollek.ir.*;
import java.util.*;

public final class AddGrad implements GradFn {
    @Override
    public Map<GValueId, GValueId> backward(
            GOp op,
            GValueId gradOut,
            GradContext ctx) {
        GValueRef a = op.inputs().get(0);
        GValueRef b = op.inputs().get(1);

        // For addition, gradient flows equally to both inputs
        return Map.of(
                a.id(), gradOut,
                b.id(), gradOut);
    }
}