package tech.kayys.gollek.runtime.client;

public interface RemoteClient {
    Map<String, Tensor> execute(
            ExecutionPlan plan,
            Map<String, Tensor> inputs,
            ExecutionSession session);
}