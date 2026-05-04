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


public interface GollekNode {
    String op(); // "matmul", "attention", "conv", etc.

    String name();

    List<GollekNode> inputs();

    Map<String, Object> attrs();
}