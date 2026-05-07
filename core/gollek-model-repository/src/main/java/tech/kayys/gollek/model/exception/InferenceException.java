package tech.kayys.gollek.model.exception;

import tech.kayys.gollek.error.ErrorCode;

/**
 * Compatibility wrapper for InferenceException in model repository package.
 * Extends the canonical SPI exception.
 */
public class InferenceException extends tech.kayys.gollek.spi.exception.InferenceException {

    public InferenceException(String message) {
        super(ErrorCode.INTERNAL_ERROR, message);
    }

    public InferenceException(String message, Throwable cause) {
        super(ErrorCode.INTERNAL_ERROR, message, cause);
    }

    public InferenceException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public InferenceException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}
