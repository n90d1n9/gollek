package tech.kayys.gollek.hub;

/**
 * @deprecated Use standard I/O exceptions or {@link tech.kayys.gollek.ml.hub.ModelHub}
 *             directly.
 */
@Deprecated(since = "0.1.1", forRemoval = true)
public class HubException extends RuntimeException {

    public HubException(String message) {
        super(message);
    }

    public HubException(String message, Throwable cause) {
        super(message, cause);
    }
}
