
package tech.kayys.gollek.autograd.spi;

import tech.kayys.gollek.autograd.GradFn;
import tech.kayys.gollek.autograd.GradRegistry;

public interface GradFnProvider {
    void registerGradients(GradRegistry registry);
}
