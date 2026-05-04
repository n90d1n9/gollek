package tech.kayys.gollek.multimodal;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalRequest;
import tech.kayys.gollek.spi.model.MultimodalResponse;

import jakarta.inject.Inject;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Stateful multi-turn conversation manager for multimodal sessions.
 *
 * <p>Maintains an in-memory conversation history (turns of user and assistant
 * messages, each potentially containing multiple modality parts) and injects
 * the accumulated history into each new inference request.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 *   String sessionId = conversationManager.createSession("gpt-4o", "user-123");
 *
 *   // Turn 1: user uploads an image and asks a question
 *   conversationManager.chat(sessionId,
 *       List.of(MultimodalContent.ofText("What's in this picture?"),
 *               MultimodalContent.ofBase64Image(imgBytes, "image/jpeg")))
 *       .subscribe().with(resp -> System.out.println(resp.firstText()));
 *
 *   // Turn 2: follow-up referencing the same image (image is in history)
 *   conversationManager.chat(sessionId,
 *       List.of(MultimodalContent.ofText("Now describe the background.")))
 *       .subscribe().with(resp -> System.out.println(resp.firstText()));
 * }</pre>
 *
 * <h3>Persistence</h3>
 * <p>The default implementation is in-memory.  For production, replace
 * {@link #sessionStore} with a Redis or database-backed implementation by
 * implementing {@link ConversationStore} and registering it as a CDI bean.
 */
@ApplicationScoped
public class ConversationManager {

    private static final Logger LOG = Logger.getLogger(ConversationManager.class);

    /** Maximum turns retained before old turns are evicted (FIFO). */
    public static final int DEFAULT_MAX_TURNS = 20;

    /** Maximum total characters across all text parts in history. */
    public static final int DEFAULT_MAX_CONTEXT_CHARS = 80_000;

    @Inject
    MultimodalRouter router;

    // In-memory store — swap for Redis-backed ConversationStore for production
    private final Map<String, ConversationSession> sessionStore = new ConcurrentHashMap<>();

    // -------------------------------------------------------------------------
    // Session lifecycle
    // -------------------------------------------------------------------------

    /**
     * Creates a new conversation session and returns its ID.
     *
     * @param defaultModel  default model for this session (can be overridden per turn)
     * @param userId        user or tenant identifier for audit
     */
    public String createSession(String defaultModel, String userId) {
        String sessionId = UUID.randomUUID().toString();
        sessionStore.put(sessionId, new ConversationSession(
                sessionId, defaultModel, userId,
                new ArrayList<>(), Instant.now(), Instant.now()));
        LOG.infof("Session created: %s (model=%s, user=%s)", sessionId, defaultModel, userId);
        return sessionId;
    }

    /** Returns the session metadata, or empty if not found. */
    public Optional<ConversationSession> getSession(String sessionId) {
        return Optional.ofNullable(sessionStore.get(sessionId));
    }

    /** Clears all turns from a session without deleting it. */
    public void clearHistory(String sessionId) {
        getSession(sessionId).ifPresent(s -> s.turns().clear());
        LOG.infof("History cleared for session: %s", sessionId);
    }

    /** Deletes a session entirely. */
    public void deleteSession(String sessionId) {
        sessionStore.remove(sessionId);
        LOG.infof("Session deleted: %s", sessionId);
    }

    // -------------------------------------------------------------------------
    // Chat
    // -------------------------------------------------------------------------

    /**
     * Send a user turn in an existing session and receive the assistant reply.
     *
     * <p>The full conversation history is injected into the request before routing.
     *
     * @param sessionId  session ID obtained from {@link #createSession}
     * @param userParts  content parts from the user for this turn
     */
    public Uni<MultimodalResponse> chat(String sessionId,
                                         List<MultimodalContent> userParts) {
        return chat(sessionId, userParts, null, null);
    }

    /**
     * Send a turn with an optional model override and extra parameters.
     */
    public Uni<MultimodalResponse> chat(String sessionId,
                                         List<MultimodalContent> userParts,
                                         String modelOverride,
                                         Map<String, Object> params) {
        ConversationSession session = sessionStore.get(sessionId);
        if (session == null) {
            return Uni.createFrom().item(
                    MultimodalResponse.error(UUID.randomUUID().toString(), "unknown",
                            "CONV-001", "Session not found: " + sessionId));
        }

        // Build flattened parts list: history + new user turn
        List<MultimodalContent> allParts = buildContextParts(session, userParts);

        String model = modelOverride != null ? modelOverride : session.defaultModel();
        MultimodalRequest.Builder reqBuilder = MultimodalRequest.builder()
                .model(model)
                .inputs(allParts.toArray(new MultimodalContent[0]));
        if (params != null) reqBuilder.parameters(params);
        MultimodalRequest request = reqBuilder.build();

        return router.route(request)
                .onItem().invoke(response -> {
                    // Record turn in history
                    session.turns().add(new ConversationTurn(
                            ConversationTurn.Role.USER, new ArrayList<>(userParts),
                            Instant.now()));
                    if (response.getStatus() == MultimodalResponse.ResponseStatus.SUCCESS
                            || response.getStatus() == MultimodalResponse.ResponseStatus.FALLBACK) {
                        session.turns().add(new ConversationTurn(
                                ConversationTurn.Role.ASSISTANT,
                                new ArrayList<>(Arrays.asList(response.getOutputs())),
                                Instant.now()));
                        evictOldTurns(session);
                    }
                    // Update last-active timestamp
                    sessionStore.put(sessionId, new ConversationSession(
                            session.sessionId(), session.defaultModel(), session.userId(),
                            session.turns(), session.createdAt(), Instant.now()));
                });
    }

    // -------------------------------------------------------------------------
    // History construction
    // -------------------------------------------------------------------------

    /**
     * Builds the flat parts list that represents the full conversation context.
     *
     * <p>Text parts from history are included verbatim.  Binary parts (images,
     * audio) from older turns are elided after {@link #DEFAULT_MAX_TURNS}/2 turns
     * to avoid blowing the token budget.
     */
    private List<MultimodalContent> buildContextParts(ConversationSession session,
                                                        List<MultimodalContent> newUserParts) {
        List<MultimodalContent> parts = new ArrayList<>();
        List<ConversationTurn> turns  = session.turns();

        // Determine how far back to include binary content
        int binaryThresholdTurn = Math.max(0, turns.size() - (DEFAULT_MAX_TURNS / 2));
        int totalChars = 0;

        for (int i = 0; i < turns.size(); i++) {
            ConversationTurn turn = turns.get(i);
            // Role separator as a text marker (providers parse role from context)
            String rolePrefix = turn.role() == ConversationTurn.Role.USER
                    ? "User: " : "Assistant: ";

            for (MultimodalContent part : turn.parts()) {
                if (part.getModality().isBinary() && i < binaryThresholdTurn) {
                    // Elide old binary content to save context budget
                    parts.add(MultimodalContent.ofText(
                            rolePrefix + "[" + part.getModality() + " omitted from history]"));
                    continue;
                }
                if (part.getModality().name().equals("TEXT") && part.getText() != null) {
                    int addedChars = part.getText().length();
                    if (totalChars + addedChars > DEFAULT_MAX_CONTEXT_CHARS) {
                        LOG.debugf("History context limit reached at turn %d", i);
                        break;
                    }
                    parts.add(MultimodalContent.ofText(rolePrefix + part.getText()));
                    totalChars += addedChars;
                } else {
                    parts.add(part);
                }
            }
        }

        // Append the new user turn
        parts.addAll(newUserParts);
        return parts;
    }

    private void evictOldTurns(ConversationSession session) {
        List<ConversationTurn> turns = session.turns();
        while (turns.size() > DEFAULT_MAX_TURNS * 2) { // *2 for user+assistant pairs
            turns.remove(0);
        }
    }

    // =========================================================================
    // Domain models
    // =========================================================================

    @JsonInclude(Include.NON_NULL)
    public record ConversationSession(
        String                  sessionId,
        String                  defaultModel,
        String                  userId,
        List<ConversationTurn>  turns,
        Instant                 createdAt,
        Instant                 lastActiveAt
    ) {}

    @JsonInclude(Include.NON_NULL)
    public record ConversationTurn(
        Role                    role,
        List<MultimodalContent> parts,
        Instant                 timestamp
    ) {
        public enum Role { USER, ASSISTANT, SYSTEM }
    }
}
