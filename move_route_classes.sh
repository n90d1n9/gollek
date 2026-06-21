#!/bin/bash
set -e

CLI_CMD_DIR="ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands"
CLI_UTIL_DIR="ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/util"
SDK_ROUTE_DIR="sdk/gollek-sdk-core/src/main/java/tech/kayys/gollek/sdk/route"
ENGINE_ROUTE_DIR="runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/route"

mkdir -p "$SDK_ROUTE_DIR"
mkdir -p "$ENGINE_ROUTE_DIR"

# 1. Move Safetensor specific route classes to gollek-safetensor-engine
for file in DirectSafetensorRoutePolicy.java \
            DirectSafetensorRoutePreflight.java \
            DirectSafetensorRunProfile.java \
            DirectSafetensorTextPolicy.java \
            Gemma4SafetensorExecutionProfile.java \
            Gemma4UnifiedSafetensorPreflight.java; do
    if [ -f "$CLI_CMD_DIR/$file" ]; then
        mv "$CLI_CMD_DIR/$file" "$ENGINE_ROUTE_DIR/"
        sed -i '' 's/package tech.kayys.gollek.cli.commands;/package tech.kayys.gollek.safetensor.engine.route;/g' "$ENGINE_ROUTE_DIR/$file"
    fi
done

# 2. Move Generic Route POJOs from commands to sdk-core
for file in RouteBenchmarkCacheReports.java \
            RouteArtifactCache.java \
            RunnerRouteBenchmarkCache.java \
            RouteReportPayloads.java \
            RoutePreflightReport.java \
            RoutePreflightProblem.java \
            RoutePreflightAction.java \
            RunnerRoutePerformanceProfile.java \
            RunnerRoutePolicy.java \
            RunnerRoutePolicyContract.java \
            RunnerRouteReport.java; do
    if [ -f "$CLI_CMD_DIR/$file" ]; then
        mv "$CLI_CMD_DIR/$file" "$SDK_ROUTE_DIR/"
        sed -i '' 's/package tech.kayys.gollek.cli.commands;/package tech.kayys.gollek.sdk.route;/g' "$SDK_ROUTE_DIR/$file"
    fi
done

# 3. Move Generic Route util POJOs from util to sdk-core
for file in RouteBenchmarkCacheReportContract.java \
            RoutePreflightDiagnosticContract.java \
            RoutePreflightDiagnosticFields.java \
            RouteReportPayloadContract.java \
            RouteReportPayloadFields.java \
            RunnerRoutePolicyFields.java \
            RunnerRouteReportContract.java \
            RunnerRouteReportFields.java; do
    if [ -f "$CLI_UTIL_DIR/$file" ]; then
        mv "$CLI_UTIL_DIR/$file" "$SDK_ROUTE_DIR/"
        sed -i '' 's/package tech.kayys.gollek.cli.util;/package tech.kayys.gollek.sdk.route;/g' "$SDK_ROUTE_DIR/$file"
    fi
done

echo "Move script completed successfully."
