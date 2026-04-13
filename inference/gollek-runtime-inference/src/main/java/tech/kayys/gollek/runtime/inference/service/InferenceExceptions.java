package tech.kayys.gollek.runtime.inference.service;

import tech.kayys.gollek.runtime.inference.ratelimit.RateLimiter.RateLimitResult;

/**
 * Exception thrown when rate limit is exceeded.
 */
class RateLimitExceededException extends RuntimeException {

    private final RateLimitResult rateResult;

    public RateLimitExceededException(RateLimitResult rateResult) {
        super(rateResult.rejectionReason());
        this.rateResult = rateResult;
    }

    public RateLimitResult getRateResult() {
        return rateResult;
    }

    public long getRetryAfterSeconds() {
        return rateResult.retryAfterSeconds();
    }
}

/**
 * Exception thrown when service is unavailable.
 */
class ServiceUnavailableException extends RuntimeException {

    public ServiceUnavailableException(String message) {
        super(message);
    }
}
