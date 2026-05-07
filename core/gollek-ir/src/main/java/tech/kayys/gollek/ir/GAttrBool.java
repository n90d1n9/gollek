package tech.kayys.gollek.ir;
import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.gollek.core.tensor.*;

import tech.kayys.gollek.core.tensor.Tensor;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;


public record GAttrBool(boolean value) implements GAttrValue {
}