package tech.kayys.gollek.ir.schema;
import tech.kayys.aljabr.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.aljabr.core.tensor.*;


import java.util.*;

public interface TypeInfer {
    GType[] infer(
            List<GValue> inputs,
            Map<String, GAttrValue> attrs);
}