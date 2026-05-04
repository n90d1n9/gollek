package tech.kayys.gollek.runtime.kernel;

import tech.kayys.gollek.runtime.Device;

public final class KernelCandidate {
    public final String id;
    public final Device device;

    public KernelCandidate(String id, Device device) {
        this.id = id;
        this.device = device;
    }
}