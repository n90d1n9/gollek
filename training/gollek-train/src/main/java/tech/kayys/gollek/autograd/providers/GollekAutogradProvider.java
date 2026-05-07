package tech.kayys.gollek.autograd.providers;

import tech.kayys.gollek.autograd.*;
import tech.kayys.gollek.autograd.spi.GradFnProvider;

public class GollekAutogradProvider implements GradFnProvider {
    @Override
    public void registerGradients(GradRegistry registry) {
        registry.register("gelu", new GeluGrad());
        registry.register("add", new AddGrad());
        registry.register("mul", new MulGrad());
        registry.register("sub", new SubGrad());
        registry.register("div", new DivGrad());
        registry.register("relu", new ReluGrad());
        registry.register("sigmoid", new SigmoidGrad());
    }
}