
## Quarkus Metrics


In Quarkus, the standard way to do this is by injecting the `MeterRegistry` and manually recording the events during your stream processing.

### 1. Metric Strategy
You should use two different types of Micrometer meters:
* **TTFT (Time To First Token):** Use a **Timer**. This tracks the latency distribution (p50, p95, p99), which is vital for UX.
* **TPOT (Time Per Output Token):** Use a **DistributionSummary**. Since this is a "per-unit" calculation ($\frac{\text{total\_decode\_time}}{\text{token\_count}}$), a summary is better for tracking the average "weight" of token generation.

---

### 2. Implementation in your Inference Service
Assuming your engine uses an asynchronous or streaming response (like SSE), here is how to instrument it.

```java
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.DistributionSummary;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import java.util.concurrent.TimeUnit;

@ApplicationScoped
public class InferenceMetricsService {

    @Inject
    MeterRegistry registry;

    // Track latency of the very first token
    private Timer ttftTimer;
    // Track the speed of subsequent tokens
    private DistributionSummary tpotSummary;

    public void init() {
        this.ttftTimer = Timer.builder("gollek.inference.ttft")
            .description("Time to first token")
            .publishPercentileHistogram() // Important for Grafana heatmaps
            .register(registry);

        this.tpotSummary = DistributionSummary.builder("gollek.inference.tpot")
            .description("Time per output token")
            .baseUnit("milliseconds")
            .register(registry);
    }

    public void recordMetrics(long startTimeNanos, long firstTokenNanos, long endTimeNanos, int tokenCount) {
        // 1. Record TTFT
        long ttftDuration = firstTokenNanos - startTimeNanos;
        ttftTimer.record(ttftDuration, TimeUnit.NANOSECONDS);

        // 2. Record TPOT
        if (tokenCount > 1) {
            long decodeDurationMs = TimeUnit.NANOSECONDS.toMillis(endTimeNanos - firstTokenNanos);
            double tpot = (double) decodeDurationMs / (tokenCount - 1);
            tpotSummary.record(tpot);
        }
    }
}
```

---

### 3. Correlating with Native Memory
Since you are using the **FFM API**, you should add a **Gauge** to monitor your native memory footprint alongside these performance metrics. This allows you to see if a spike in TTFT correlates with a spike in off-heap memory pressure.

```java
// Inside your service or a specialized MemoryManager
registry.gauge("gollek.native.memory.active_bytes", myNativeArena, 
               arena -> arena.byteSize());
```

---

### 4. Why this matters for "gollek"
By injecting these into your Prometheus endpoint (`/q/metrics`), you can build a Grafana dashboard that shows:
1.  **The "Slow First Token" Alert:** If TTFT spikes but TPOT is stable, your **Prefill** phase (prompt processing) is the bottleneck.
2.  **The "Memory Pressure" Alert:** If TPOT increases over time, your **Decode** phase is likely hitting memory bandwidth limits or your FFM `MemorySegment` allocations are causing fragmentation.


While **TTFT** and **TPOT** are the "Gold Standard" for user experience, they only tell half the story for an enterprise-grade engine like **gollek**. To move beyond a hobbyist setup and into production-ready infrastructure, you need to track **Efficiency**, **Reliability**, and **Quality** metrics.

### 1. The "Enterprise Five" Performance Metrics
In addition to TTFT and TPOT, you should implement these to measure the system's "saturation" and "smoothness."

| Metric | Definition | Why it matters for Enterprise |
| :--- | :--- | :--- |
| **ITL (Inter-Token Latency)** | The time between each individual token. | If ITL is inconsistent, the text "stutters" for the user. High ITL usually means memory bandwidth or KV-cache bottlenecks. |
| **TPS (Tokens Per Second)** | Total output tokens / Total duration. | This is your raw **Throughput**. Essential for calculating how many GPUs (or M4 nodes) you need to support 1,000 users. |
| **Goodput** | % of requests meeting your SLO (e.g., TTFT < 500ms). | In enterprise, "Fast enough" is a binary state. Goodput tracks how many users actually had a "good" experience. |
| **Error Rate** | % of 4xx/5xx or timeout responses. | Critical for SLAs. You should track if errors correlate with high concurrency. |
| **KV-Cache Hit Rate** | % of reused prefix tokens. | If you implement "Context Caching," this tells you how much compute/latency you are saving. |

---

### 2. Implementation: The Extended Metric Collector
You can update your `InferenceMetricsService` to capture these higher-level signals.

```java
public void recordEnterpriseMetrics(String requestId, int inputTokens, int outputTokens, 
                                   long startNanos, List<Long> tokenTimestampsNanos) {
    long endNanos = tokenTimestampsNanos.get(tokenTimestampsNanos.size() - 1);
    double totalTimeSec = (endNanos - startNanos) / 1_000_000_000.0;

    // 1. Throughput (System-wide)
    registry.counter("gollek.inference.tokens.total", "type", "output").increment(outputTokens);
    registry.counter("gollek.inference.tokens.total", "type", "input").increment(inputTokens);

    // 2. ITL Jitter (Stability)
    for (int i = 1; i < tokenTimestampsNanos.size(); i++) {
        long gap = tokenTimestampsNanos.get(i) - tokenTimestampsNanos.get(i-1);
        registry.timer("gollek.inference.itl").record(gap, TimeUnit.NANOSECONDS);
    }

    // 3. Request Success/Failure
    registry.counter("gollek.inference.requests", "status", "success").increment();
}
```

---

### 3. The "Silent" Pillar: Quality & Safety
Performance is useless if the model is hallucinating or leaking data. Enterprise engines typically include **Guardrail Metrics**:

* **Hallucination Score:** Use a smaller "Judge" model (like a 1B-3B parameter model) to check if the output matches the input context (Grounding).
* **PII Leakage Rate:** Monitor if your `Suren` (Security Vault) blocks any output containing sensitive patterns (emails, keys).
* **Toxicity/Bias:** Track how often your safety filters are triggered.

---

### 4. Hardware-Specific  Metrics
We should track **Unified Memory Bandwidth**. If your TPOT drops, it’s rarely CPU—it’s usually the memory bus. and add other metrcis for each hardware/kernel (Metal/Apple Silicon, Cuda,)

> **Tip:** Use `powermetrics` on macOS to see if your `gollek` process is hitting the "Memory Controller" power ceiling.

```bash
# Watch memory bandwidth while your Java benchmark runs
sudo powermetrics -i 1000 --samplers cpu_power,gpu_power
```

