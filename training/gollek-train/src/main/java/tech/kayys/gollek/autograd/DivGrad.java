package tech.kayys.gollek.autograd;

import tech.kayys.gollek.ir.*;
import java.util.*;

public final class DivGrad implements GradFn {
    @Override
    public Map<GValueId, GValueId> backward(
            GOp op,
            GValueId gradOut,
            GradContext ctx) {
        GValueRef a = op.inputs().get(0);
        GValueRef b = op.inputs().get(1);
        GValueId da = new GValueId(op.name() + "_da");
        GValueId db = new GValueId(op.name() + "_db");

        // da = gradOut / b
        ctx.addOp(new GOp("div", op.name() + "_gradA", 
                List.of(new GValueRef(gradOut), b), List.of(da), Map.of()));
        // db = -gradOut * a / b^2
        // simplified placeholder for complex ops
        return Map.of(a.id(), da);
    }
}
