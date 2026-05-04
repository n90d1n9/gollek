package tech.kayys.gollek.ml.runner;

/**
 * Memory usage statistics for a model runner.
 *
 * @since 0.3.0
 */
public record MemoryStats(
    /** Total device memory in bytes */
    long totalMemory,

    /** Currently allocated memory in bytes */
    long allocatedMemory,

    /** Peak memory usage in bytes */
    long peakMemory,

    /** Number of active allocations */
    long allocationCount,

    /** Number of active tensor allocations */
    long tensorCount
) {
    /**
     * Gets memory utilization as a percentage.
     */
    public double utilizationPercent() {
        return totalMemory > 0 ? (double) allocatedMemory / totalMemory * 100.0 : 0.0;
    }

    /**
     * Gets available memory in bytes.
     */
    public long availableMemory() {
        return totalMemory - allocatedMemory;
    }

    /**
     * Formats allocated memory in human-readable form.
     */
    public String formattedAllocated() {
        return formatBytes(allocatedMemory);
    }

    /**
     * Formats total memory in human-readable form.
     */
    public String formattedTotal() {
        return formatBytes(totalMemory);
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024.0));
        return String.format("%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }

    /**
     * Empty stats for testing/stubbing.
     */
    public static MemoryStats empty() {
        return new MemoryStats(0, 0, 0, 0, 0);
    }
}
