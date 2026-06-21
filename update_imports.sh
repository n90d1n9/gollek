#!/bin/bash
set -e

# Generic SDK Classes (commands)
for class in RouteBenchmarkCacheReports RouteArtifactCache RunnerRouteBenchmarkCache RouteReportPayloads RoutePreflightReport RoutePreflightProblem RoutePreflightAction RunnerRoutePerformanceProfile RunnerRoutePolicy RunnerRoutePolicyContract RunnerRouteReport; do
    find . -name "*.java" -exec sed -i '' "s/tech.kayys.gollek.cli.commands.${class}/tech.kayys.gollek.sdk.route.${class}/g" {} +
done

# Generic SDK Classes (util)
for class in RouteBenchmarkCacheReportContract RoutePreflightDiagnosticContract RoutePreflightDiagnosticFields RouteReportPayloadContract RouteReportPayloadFields RunnerRoutePolicyFields RunnerRouteReportContract RunnerRouteReportFields; do
    find . -name "*.java" -exec sed -i '' "s/tech.kayys.gollek.cli.util.${class}/tech.kayys.gollek.sdk.route.${class}/g" {} +
done

# Safetensor Engine Classes
for class in DirectSafetensorRoutePolicy DirectSafetensorRoutePreflight DirectSafetensorRunProfile DirectSafetensorTextPolicy Gemma4SafetensorExecutionProfile Gemma4UnifiedSafetensorPreflight; do
    find . -name "*.java" -exec sed -i '' "s/tech.kayys.gollek.cli.commands.${class}/tech.kayys.gollek.safetensor.engine.route.${class}/g" {} +
done

# Make all moved classes public
find runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/route -name "*.java" -exec sed -i '' 's/^final class /public final class /g; s/^class /public class /g; s/^record /public record /g; s/^interface /public interface /g; s/^enum /public enum /g; s/ static final class / public static final class /g' {} +
find sdk/gollek-sdk-core/src/main/java/tech/kayys/gollek/sdk/route -name "*.java" -exec sed -i '' 's/^final class /public final class /g; s/^class /public class /g; s/^record /public record /g; s/^interface /public interface /g; s/^enum /public enum /g; s/ static final class / public static final class /g' {} +

# Deduplicate repeated 'public'
find runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/route -name "*.java" -exec sed -i '' 's/public public /public /g' {} +
find sdk/gollek-sdk-core/src/main/java/tech/kayys/gollek/sdk/route -name "*.java" -exec sed -i '' 's/public public /public /g' {} +

echo "Imports updated and classes made public."
