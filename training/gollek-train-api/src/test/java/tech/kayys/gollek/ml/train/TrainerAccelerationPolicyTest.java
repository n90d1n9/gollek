package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.Acceleration;

class TrainerAccelerationPolicyTest {

    @Test
    void cpuAndAutoDoNotCountAsFallback() {
        Acceleration.BackendStatus cpu = new Acceleration.BackendStatus("cpu", "CPU", false, true, 0);

        assertFalse(TrainerAccelerationPolicy.fallback("cpu", cpu));
        assertFalse(TrainerAccelerationPolicy.fallback("auto", cpu));
        assertDoesNotThrow(() -> TrainerAccelerationPolicy.requireNoFallback("cpu", cpu, true));
    }

    @Test
    void requireAcceleratorFailsWhenAutoResolvesToCpu() {
        Acceleration.BackendStatus cpu = new Acceleration.BackendStatus("cpu", "CPU", false, true, 0);

        assertFalse(TrainerAccelerationPolicy.fallback("auto", cpu));
        assertThrows(IllegalStateException.class, () ->
                TrainerAccelerationPolicy.requireNoFallback("auto", cpu, true));
        assertDoesNotThrow(() -> TrainerAccelerationPolicy.requireNoFallback("auto", cpu, false));
    }

    @Test
    void requestedAcceleratorFallbackCanFailFast() {
        Acceleration.BackendStatus cpuFallback = new Acceleration.BackendStatus("cpu", "CPU", false, false, 0);

        assertTrue(TrainerAccelerationPolicy.fallback("mps", cpuFallback));
        assertThrows(IllegalStateException.class, () ->
                TrainerAccelerationPolicy.requireNoFallback("mps", cpuFallback, true));
        assertDoesNotThrow(() -> TrainerAccelerationPolicy.requireNoFallback("mps", cpuFallback, false));
    }
}
