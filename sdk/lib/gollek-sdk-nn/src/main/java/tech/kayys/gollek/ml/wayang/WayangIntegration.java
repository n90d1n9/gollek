package tech.kayys.gollek.ml.wayang;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.sdk.GollekClient;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Wayang Platform integration — bridges Gollek SDK with the Wayang agent,
 * RAG pipeline, and Gamelan workflow systems.
 *
 * <p>Provides adapters that expose Gollek inference as:
 * <ul>
 *   <li>Agent skills (callable by the Wayang agent framework)</li>
 *   <li>RAG embedder (for vector store population and retrieval)</li>
 *   <li>Gamelan workflow nodes (for AI-powered workflow steps)</li>
 * </ul>
 *
 * <h3>Example — agent skill</h3>
 * <pre>{@code
 * GollekClient client = GollekClient.builder().model("qwen2-7b.gguf").build();
 * WayangIntegration integration = new WayangIntegration(client);
 *
 * // Register as agent skill
 * integration.asSkill("generate", prompt -> client.generate(prompt).text());
 *
 * // Use as RAG embedder
 * WayangIntegration.Embedder embedder = integration.asEmbedder();
 * float[] vec = embedder.embed("What is attention?");
 * }</pre>
 */
public final class WayangIntegration {

    private final GollekClient client;

    /**
     * Creates a Wayang integration wrapper around a Gollek client.
     *
     * @param client initialized {@link GollekClient}
     */
    public WayangIntegration(GollekClient client) {
        this.client = client;
    }

    // ── Agent skill adapter ───────────────────────────────────────────────

    /**
     * Wraps the client as a callable agent skill.
     *
     * <p>The returned {@link Skill} can be registered with the Wayang agent
     * framework to expose Gollek inference as a named capability.
     *
     * @param name    skill name (e.g. "generate", "summarize")
     * @param handler function mapping skill input to output string
     * @return {@link Skill} ready for agent registration
     */
    public Skill asSkill(String name, Function<String, String> handler) {
        return new Skill(name, handler);
    }

    /**
     * Creates a generation skill that calls the client directly.
     *
     * @param name skill name
     * @return generation skill
     */
    public Skill generationSkill(String name) {
        return asSkill(name, prompt -> client.generate(prompt).text());
    }

    // ── RAG embedder adapter ──────────────────────────────────────────────

    /**
     * Returns an {@link Embedder} that uses this client for dense embeddings.
     *
     * <p>Compatible with the Wayang RAG pipeline's embedder interface.
     *
     * @return embedder backed by this client
     */
    public Embedder asEmbedder() {
        return new Embedder() {
            @Override public float[] embed(String text)           { return client.embed(text); }
            @Override public List<float[]> embedBatch(List<String> texts) { return client.embedBatch(texts); }
        };
    }

    // ── Gamelan workflow node adapter ─────────────────────────────────────

    /**
     * Creates a Gamelan workflow node that runs inference on its input.
     *
     * <p>The node reads {@code "prompt"} from the context, runs generation,
     * and writes the result to {@code "output"}.
     *
     * @param nodeName workflow node name
     * @return {@link WorkflowNode} for Gamelan registration
     */
    public WorkflowNode asWorkflowNode(String nodeName) {
        return new WorkflowNode(nodeName, ctx -> {
            String prompt = (String) ctx.getOrDefault("prompt", "");
            String result = client.generate(prompt).text();
            ctx.put("output", result);
            return ctx;
        });
    }

    // ── Nested types ──────────────────────────────────────────────────────

    /**
     * Agent skill — a named callable capability.
     *
     * @param name    skill identifier
     * @param handler input → output function
     */
    public record Skill(String name, Function<String, String> handler) {

        /**
         * Invokes the skill with the given input.
         *
         * @param input skill input string
         * @return skill output string
         */
        public String invoke(String input) { return handler.apply(input); }
    }

    /**
     * RAG embedder interface — converts text to dense vectors.
     */
    public interface Embedder {

        /**
         * Embeds a single text string.
         *
         * @param text input text
         * @return dense embedding vector
         */
        float[] embed(String text);

        /**
         * Embeds multiple texts in a batch.
         *
         * @param texts list of input texts
         * @return list of embedding vectors
         */
        List<float[]> embedBatch(List<String> texts);
    }

    /**
     * Gamelan workflow node — processes a context map and returns updated context.
     *
     * @param name    node name in the workflow graph
     * @param handler context processor function
     */
    public record WorkflowNode(String name, Function<Map<String, Object>, Map<String, Object>> handler) {

        /**
         * Executes the node with the given context.
         *
         * @param context workflow execution context
         * @return updated context
         */
        public Map<String, Object> execute(Map<String, Object> context) {
            return handler.apply(context);
        }
    }
}
