package tech.kayys.gollek.runtime.execution;

import tech.kayys.gollek.runtime.data.GData;

public final class ExecutionRequest {
    public final String tenantId;
    public final ExecutionPlan plan;
    public final ExecutionContext context;
    public final ExecutionSession session;

    public ExecutionRequest(String tenantId,
            ExecutionPlan plan,
            ExecutionContext context,
            ExecutionSession session) {
        this.tenantId = tenantId;
        this.plan = plan;
        this.context = context;
        this.session = session;
    }
}