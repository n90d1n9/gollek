package tech.kayys.gollek.autograd;

import tech.kayys.gollek.ir.*;
import java.util.Map;

public interface GradFn {
    /**
     *
     * Compute gradients for inputs of an op
     * 
     * @param op      forward op
     * @param gradOut gradient of output
     * @param ctx     grad context (for emitting ops)
     * @return map: input → gradient
     */
    Map<GValueId, GValueId> backward(
            GOp op,
            GValueId gradOut,
            GradContext ctx);
}