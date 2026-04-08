package tech.kayys.gollek.runtime.unified.websocket;

import jakarta.websocket.*;
import jakarta.websocket.server.ServerEndpoint;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

/**
 * WebSocket endpoint for real-time agent execution streaming.
 * Clients can subscribe to execution progress and receive live updates.
 */
@ServerEndpoint("/ws/agents/{agentName}")
public class AgentExecutionWebSocket {
    private static final Logger LOGGER = Logger.getLogger(AgentExecutionWebSocket.class);
    private static final ConcurrentHashMap<String, Set<Session>> sessionsByAgent = new ConcurrentHashMap<>();
    
    private String agentName;
    private String sessionId;

    @OnOpen
    public void onOpen(Session session, @jakarta.websocket.server.PathParam("agentName") String agentName) {
        this.agentName = agentName;
        this.sessionId = session.getId();
        
        sessionsByAgent.computeIfAbsent(agentName, k -> ConcurrentHashMap.newKeySet()).add(session);
        
        LOGGER.infof("WebSocket connection opened for agent %s (session %s)", agentName, sessionId);
        
        try {
            sendMessage(session, new ExecutionMessage(
                    ExecutionMessageType.CONNECTED,
                    "Connected to agent: " + agentName,
                    null,
                    null
            ));
        } catch (IOException e) {
            LOGGER.errorf(e, "Error sending connection message");
        }
    }

    @OnMessage
    public void onMessage(Session session, String message) {
        try {
            // Parse incoming message
            ExecutionRequest request = parseRequest(message);
            LOGGER.infof("Received execution request from %s: %s", sessionId, request.query);
            
            // Send acknowledgment
            sendMessage(session, new ExecutionMessage(
                    ExecutionMessageType.ACKNOWLEDGED,
                    "Query received: " + request.query,
                    request.requestId,
                    null
            ));
            
            // Execute agent asynchronously
            executeAgentAsync(session, request);
            
        } catch (Exception e) {
            LOGGER.errorf(e, "Error processing message: %s", e.getMessage());
            try {
                sendMessage(session, new ExecutionMessage(
                        ExecutionMessageType.ERROR,
                        "Error: " + e.getMessage(),
                        null,
                        null
                ));
            } catch (IOException ioe) {
                LOGGER.errorf(ioe, "Error sending error message");
            }
        }
    }

    @OnClose
    public void onClose(Session session) {
        Set<Session> sessions = sessionsByAgent.get(agentName);
        if (sessions != null) {
            sessions.remove(session);
            if (sessions.isEmpty()) {
                sessionsByAgent.remove(agentName);
            }
        }
        LOGGER.infof("WebSocket connection closed for agent %s (session %s)", agentName, sessionId);
    }

    @OnError
    public void onError(Session session, Throwable throwable) {
        LOGGER.errorf(throwable, "WebSocket error for agent %s: %s", agentName, throwable.getMessage());
    }

    private void executeAgentAsync(Session session, ExecutionRequest request) {
        CompletableFuture.runAsync(() -> {
            try {
                // Send execution started
                sendMessage(session, new ExecutionMessage(
                        ExecutionMessageType.EXECUTING,
                        "Starting execution...",
                        request.requestId,
                        null
                ));

                // Execute agent (simulated - replace with actual agent execution)
                long startTime = System.currentTimeMillis();
                
                // Send progress update
                sendMessage(session, new ExecutionMessage(
                        ExecutionMessageType.PROGRESS,
                        "Analyzing query...",
                        request.requestId,
                        Map.of("progress", "25%")
                ));
                
                Thread.sleep(500);
                
                sendMessage(session, new ExecutionMessage(
                        ExecutionMessageType.PROGRESS,
                        "Selecting tools...",
                        request.requestId,
                        Map.of("progress", "50%")
                ));
                
                Thread.sleep(500);
                
                sendMessage(session, new ExecutionMessage(
                        ExecutionMessageType.PROGRESS,
                        "Executing skills...",
                        request.requestId,
                        Map.of("progress", "75%")
                ));
                
                Thread.sleep(500);
                
                long executionTime = System.currentTimeMillis() - startTime;
                
                // Send result
                String result = "Execution completed successfully for query: " + request.query;
                sendMessage(session, new ExecutionMessage(
                        ExecutionMessageType.COMPLETED,
                        result,
                        request.requestId,
                        Map.of(
                            "executionTimeMs", executionTime,
                            "status", "success"
                        )
                ));
                
            } catch (Exception e) {
                LOGGER.errorf(e, "Error executing agent async: %s", e.getMessage());
                try {
                    sendMessage(session, new ExecutionMessage(
                            ExecutionMessageType.ERROR,
                            "Execution failed: " + e.getMessage(),
                            request.requestId,
                            null
                    ));
                } catch (IOException ioe) {
                    LOGGER.errorf(ioe, "Error sending error message");
                }
            }
        });
    }

    private void sendMessage(Session session, ExecutionMessage message) throws IOException {
        if (session.isOpen()) {
            String json = serializeMessage(message);
            session.getBasicRemote().sendText(json);
        }
    }

    private ExecutionRequest parseRequest(String json) {
        // Parse JSON - in production, use Jackson
        String query = extractField(json, "query");
        String requestId = extractField(json, "requestId");
        if (requestId == null || requestId.isEmpty()) {
            requestId = UUID.randomUUID().toString();
        }
        return new ExecutionRequest(query, requestId);
    }

    private String serializeMessage(ExecutionMessage message) {
        // Serialize to JSON - in production, use Jackson
        return String.format(
            "{\"type\":\"%s\",\"message\":\"%s\",\"requestId\":\"%s\",\"data\":%s}",
            message.type,
            escapeJson(message.message),
            message.requestId != null ? message.requestId : "",
            message.data != null ? serializeMap(message.data) : "null"
        );
    }

    private String serializeMap(Map<String, Object> map) {
        StringBuilder sb = new StringBuilder("{");
        int i = 0;
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            if (i++ > 0) sb.append(",");
            sb.append("\"").append(entry.getKey()).append("\":\"").append(entry.getValue()).append("\"");
        }
        sb.append("}");
        return sb.toString();
    }

    private String extractField(String json, String field) {
        String pattern = "\"" + field + "\":\"([^\"]*)\"";
        java.util.regex.Matcher matcher = java.util.regex.Pattern.compile(pattern).matcher(json);
        return matcher.find() ? matcher.group(1) : null;
    }

    private String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r");
    }

    /**
     * Broadcast message to all clients listening to an agent.
     */
    public static void broadcastToAgent(String agentName, ExecutionMessage message) throws IOException {
        Set<Session> sessions = sessionsByAgent.get(agentName);
        if (sessions != null) {
            String json = new AgentExecutionWebSocket().serializeMessage(message);
            for (Session session : sessions) {
                if (session.isOpen()) {
                    session.getBasicRemote().sendText(json);
                }
            }
        }
    }

    /**
     * Get connected client count for an agent.
     */
    public static int getConnectedClientCount(String agentName) {
        Set<Session> sessions = sessionsByAgent.get(agentName);
        return sessions != null ? sessions.size() : 0;
    }

    // DTO classes
    public enum ExecutionMessageType {
        CONNECTED,
        ACKNOWLEDGED,
        EXECUTING,
        PROGRESS,
        COMPLETED,
        ERROR,
        CANCELLED
    }

    public record ExecutionMessage(
        ExecutionMessageType type,
        String message,
        String requestId,
        Map<String, Object> data
    ) {}

    public record ExecutionRequest(
        String query,
        String requestId
    ) {}
}
