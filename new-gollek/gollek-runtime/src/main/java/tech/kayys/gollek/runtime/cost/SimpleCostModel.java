package tech.kayys.gollek.runtime.cost;

public final class SimpleCostModel implements CostModel {
    @Override
    public double estimate(GOp op,
            KernelCandidate kernel,
            Device device) {
        // naive baseline
        double base = switch (op.opType()) {
            case "matmul" -> 10;
            case "flash_attention_v3" -> 5;
            default -> 1;
        };
        double deviceFactor = switch (device) {
            case CPU -> 2.0;
            case GPU -> 0.5;
            case METAL -> 0.7;
            case REMOTE -> 1.5;
        };
        return base * deviceFactor;
    }
}