package tech.kayys.gollek.ir;
import tech.kayys.aljabr.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.aljabr.core.tensor.*;

import tech.kayys.aljabr.core.tensor.Tensor;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;


import java.util.List;

public interface OpKernel<In, Out, Attrs> {
    Out compute(List<In> inputs, Attrs attrs);
}