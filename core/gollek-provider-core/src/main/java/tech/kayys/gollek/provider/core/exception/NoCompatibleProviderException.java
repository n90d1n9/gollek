package tech.kayys.gollek.provider.core.exception;

import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.ProviderException;

/**
 * Exception thrown when no compatible provider found
 */
public class NoCompatibleProviderException extends ProviderException {
    public NoCompatibleProviderException(String message) {
        super(null, message, null, ErrorCode.ROUTING_NO_COMPATIBLE_PROVIDER, true);
    }
}
