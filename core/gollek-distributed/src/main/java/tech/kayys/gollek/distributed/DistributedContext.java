package tech.kayys.gollek.distributed;

public final class DistributedContext {
    private final int worldSize;
    private final int rank;

    public DistributedContext(int worldSize, int rank) {
        this.worldSize = worldSize;
        this.rank = rank;
    }

    public int worldSize() {
        return worldSize;
    }

    public int rank() {
        return rank;
    }

    public boolean isMaster() {
        return rank == 0;
    }
}