
package tech.kayys.gollek.autograd.spi;

import tech.kayys.gollek.autograd.GradFn;

public interface GradFnProvider {
    void registerGradients(GradRegistry registry);
}
