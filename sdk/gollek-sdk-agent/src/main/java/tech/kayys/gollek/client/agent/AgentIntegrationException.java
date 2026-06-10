package tech.kayys.gollek.client.agent;

/**
 * Exception raised by the lightweight agent integration client.
 */
public class AgentIntegrationException extends Exception {
    private final String errorCode;

    public AgentIntegrationException(String message) {
        super(message);
        this.errorCode = "AGENT_INTEGRATION_ERROR";
    }

    public AgentIntegrationException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "AGENT_INTEGRATION_ERROR";
    }

    public AgentIntegrationException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public AgentIntegrationException(String errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public String getErrorCode() {
        return errorCode;
    }
}
