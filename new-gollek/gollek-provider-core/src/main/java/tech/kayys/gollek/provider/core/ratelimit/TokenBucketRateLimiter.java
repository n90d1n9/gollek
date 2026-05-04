package tech.kayys.gollek.provider.core.ratelimit;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Token bucket rate limiter implementation.
 *
 * Allows bursting up to a maximum capacity, then refills at a constant rate.
 * Thread-safe implementation using locks.
 */
public class TokenBucketRateLimiter implements RateLimiter {

    private final long maxTokens;
    private final long refillTokens;
    private final Duration refillPeriod;
    
    private final AtomicLong tokens;
    private volatile Instant lastRefill;
    private final ReentrantLock lock;

    public TokenBucketRateLimiter(long maxTokens, Duration refillPeriod) {
        this(maxTokens, maxTokens, refillPeriod);
    }

    public TokenBucketRateLimiter(long maxTokens, long refillTokens, Duration refillPeriod) {
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens must be positive");
        }
        if (refillTokens <= 0) {
            throw new IllegalArgumentException("refillTokens must be positive");
        }
        if (refillPeriod == null || refillPeriod.isNegative() || refillPeriod.isZero()) {
            throw new IllegalArgumentException("refillPeriod must be positive");
        }

        this.maxTokens = maxTokens;
        this.refillTokens = refillTokens;
        this.refillPeriod = refillPeriod;
        this.tokens = new AtomicLong(maxTokens);
        this.lastRefill = Instant.now();
        this.lock = new ReentrantLock();
    }

    @Override
    public boolean tryAcquire() {
        return tryAcquire(1);
    }

    @Override
    public boolean tryAcquire(int permits) {
        lock.lock();
        try {
            refill();
            
            long currentTokens = tokens.get();
            if (currentTokens >= permits) {
                tokens.addAndGet(-permits);
                return true;
            }
            return false;
        } finally {
            lock.unlock();
        }
    }

    @Override
    public int availablePermits() {
        refill();
        return (int) Math.min(tokens.get(), Integer.MAX_VALUE);
    }

    @Override
    public void reset() {
        lock.lock();
        try {
            tokens.set(maxTokens);
            lastRefill = Instant.now();
        } finally {
            lock.unlock();
        }
    }

    /**
     * Refill tokens based on elapsed time
     */
    private void refill() {
        Instant now = Instant.now();
        Duration elapsed = Duration.between(lastRefill, now);
        
        if (elapsed.compareTo(refillPeriod) >= 0) {
            long tokensToAdd = (elapsed.toNanos() / refillPeriod.toNanos()) * refillTokens;
            long newTokens = Math.min(maxTokens, tokens.get() + tokensToAdd);
            tokens.set(newTokens);
            lastRefill = now;
        }
    }

    /**
     * Get current token count (for testing/monitoring)
     */
    public long getTokenCount() {
        return tokens.get();
    }

    /**
     * Get maximum token capacity
     */
    public long getMaxTokens() {
        return maxTokens;
    }
}
