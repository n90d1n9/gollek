package tech.kayys.gollek.ir.validate;
import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.model.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.model.*;


import tech.kayys.gollek.ir.*;
import java.util.*;

public interface Validator {
    void validate(List<GValue> inputs, Map<String, Object> attrs);
}