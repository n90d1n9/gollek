package tech.kayys.gollek.spi.provider;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Provider health status, including region and cluster metadata for
 * multi-cluster federation and region-aware routing.
 *
 * <h3>Multi-Cluster Routing</h3>
 * The {@link #region} and {@link #clusterId} fields allow the
 * {@code ModelRouterService} to prefer providers in the same AWS/GCP/Azure
 * region as the request origin, and to failover to the
 * {@link #failoverTarget} cluster when {@link #isUnhealthy()}.
 */
public final class ProviderHealth {

    public enum Status {
        HEALTHY,
        DEGRADED,
        UNHEALTHY,
        UNKNOWN
    }

    private final Status status;
    private final String message;
    private final Instant timestamp;
    private final Map<String, Object> details;

    /**
     * Geographic region identifier (e.g. {@code "us-east-1"}, {@code "eu-west-2"}).
     * Null means region is unknown / not applicable.
     */
    private final String region;

    /**
     * Logical cluster identifier within a region (e.g. {@code "k8s-prod-1"}).
     * Null for single-cluster deployments.
     */
    private final String clusterId;

    /**
     * ID of the failover cluster/provider to route to when this provider is
     * {@link Status#UNHEALTHY}.  Null means no automated failover is configured.
     */
    private final String failoverTarget;

    @JsonCreator
    public ProviderHealth(
            @JsonProperty("status")        Status status,
            @JsonProperty("message")       String message,
            @JsonProperty("timestamp")     Instant timestamp,
            @JsonProperty("details")       Map<String, Object> details,
            @JsonProperty("region")        String region,
            @JsonProperty("clusterId")     String clusterId,
            @JsonProperty("failoverTarget") String failoverTarget) {
        this.status        = Objects.requireNonNull(status, "status");
        this.message       = message;
        this.timestamp     = timestamp != null ? timestamp : Instant.now();
        this.details       = details != null
                ? Collections.unmodifiableMap(new HashMap<>(details))
                : Collections.emptyMap();
        this.region        = region;
        this.clusterId     = clusterId;
        this.failoverTarget = failoverTarget;
    }

    /** Backward-compatible constructor (no region/cluster). */
    public ProviderHealth(
            @JsonProperty("status")    Status status,
            @JsonProperty("message")   String message,
            @JsonProperty("timestamp") Instant timestamp,
            @JsonProperty("details")   Map<String, Object> details) {
        this(status, message, timestamp, details, null, null, null);
    }

    public Status status()             { return status; }
    public String message()            { return message; }
    public Instant timestamp()         { return timestamp; }
    public Map<String, Object> details(){ return details; }

    /** Geographic region of this provider, or null if not configured. */
    public String region()             { return region; }

    /** Logical cluster identifier, or null for single-cluster deployments. */
    public String clusterId()          { return clusterId; }

    /**
     * Returns the failover target cluster/provider ID to route to when
     * this provider is unhealthy, or null if failover is not configured.
     */
    public String failoverTarget()     { return failoverTarget; }

    /**
     * Returns {@code true} if this provider is in the given region.
     * Useful for region-affinity routing.
     */
    public boolean isInRegion(String targetRegion) {
        return targetRegion != null && targetRegion.equals(region);
    }

    public boolean isHealthy() {
        return status == Status.HEALTHY;
    }

    public boolean isDegraded() {
        return status == Status.DEGRADED;
    }

    public boolean isUnhealthy() {
        return status == Status.UNHEALTHY;
    }

    public static ProviderHealth healthy() {
        return new ProviderHealth(Status.HEALTHY, "Provider is healthy", null, null);
    }

    public static ProviderHealth healthy(String message) {
        return new ProviderHealth(Status.HEALTHY, message, null, null);
    }

    public static ProviderHealth degraded(String message) {
        return new ProviderHealth(Status.DEGRADED, message, null, null);
    }

    public static ProviderHealth unhealthy(String message) {
        return new ProviderHealth(Status.UNHEALTHY, message, null, null);
    }

    public static ProviderHealth unknown() {
        return new ProviderHealth(Status.UNKNOWN, "Health status unknown", null, null);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Status status = Status.UNKNOWN;
        private String message;
        private Instant timestamp = Instant.now();
        private final Map<String, Object> details = new HashMap<>();
        private String region;
        private String clusterId;
        private String failoverTarget;

        public Builder status(Status status) { this.status = status; return this; }
        public Builder message(String message) { this.message = message; return this; }
        public Builder timestamp(Instant timestamp) { this.timestamp = timestamp; return this; }
        public Builder detail(String key, Object value) { details.put(key, value); return this; }
        public Builder details(Map<String, Object> details) { this.details.putAll(details); return this; }

        /** Sets the geographic region for multi-cluster routing. */
        public Builder region(String region) { this.region = region; return this; }

        /** Sets the cluster ID for multi-cluster federation. */
        public Builder clusterId(String clusterId) { this.clusterId = clusterId; return this; }

        /**
         * Sets the failover target provider/cluster ID to route to when this
         * provider transitions to UNHEALTHY.
         */
        public Builder failoverTarget(String failoverTarget) {
            this.failoverTarget = failoverTarget; return this;
        }

        public ProviderHealth build() {
            return new ProviderHealth(status, message, timestamp, details,
                                      region, clusterId, failoverTarget);
        }
    }

    @Override
    public String toString() {
        return "ProviderHealth{" +
                "status=" + status +
                ", message='" + message + '\'' +
                ", timestamp=" + timestamp +
                '}';
    }
}