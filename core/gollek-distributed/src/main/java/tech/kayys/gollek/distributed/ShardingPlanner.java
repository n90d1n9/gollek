package tech.kayys.gollek.distributed;

public final class ShardingPlanner {
    public enum Strategy {
        DDP,
        ZERO,
        TENSOR_PARALLEL,
        PIPELINE
    }

    private final Strategy strategy;

    public ShardingPlanner(Strategy strategy) {
        this.strategy = strategy;
    }

    public Strategy strategy() {
        return strategy;
    }

    public int assignParamShard(String key, int worldSize) {
        return Math.abs(key.hashCode()) % worldSize;
    }
}