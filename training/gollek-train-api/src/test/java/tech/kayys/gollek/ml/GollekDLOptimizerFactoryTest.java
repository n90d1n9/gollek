package tech.kayys.gollek.ml;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.optim.Adam;
import tech.kayys.gollek.ml.optim.AdamW;
import tech.kayys.gollek.ml.optim.Adadelta;
import tech.kayys.gollek.ml.optim.Adagrad;
import tech.kayys.gollek.ml.optim.LAMB;
import tech.kayys.gollek.ml.optim.Lion;
import tech.kayys.gollek.ml.optim.Lookahead;
import tech.kayys.gollek.ml.optim.NAdam;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.optim.RAdam;
import tech.kayys.gollek.ml.optim.RMSprop;
import tech.kayys.gollek.ml.optim.SAM;
import tech.kayys.gollek.ml.optim.SGD;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GollekDLOptimizerFactoryTest {

    @Test
    void exposesCoreOptimizerFactoriesOnGollekFacade() {
        assertInstanceOf(SGD.class, Gollek.DL.sgd(parameters(), 0.1f));
        assertInstanceOf(SGD.class, Gollek.DL.sgd(parameters(), 0.1f, 0.9f));
        assertInstanceOf(SGD.class, Gollek.DL.sgd(parameters(), 0.1f, 0.9f, 0.01f, true));
    }

    @Test
    void exposesAdaptiveOptimizerFactoriesOnGollekFacade() {
        assertInstanceOf(Adam.class, Gollek.DL.adam(parameters(), 0.001f));
        assertInstanceOf(Adam.class, Gollek.DL.adam(parameters(), 0.001f, 0.01f));
        assertInstanceOf(Adam.class, Gollek.DL.adam(parameters(), 0.001f, 0.8f, 0.99f, 1e-6f, 0.01f, true));
        assertInstanceOf(AdamW.class, Gollek.DL.adamW(parameters(), 0.001f));
        assertInstanceOf(AdamW.class, Gollek.DL.adamW(parameters(), 0.001f, 0.05f));
        assertInstanceOf(AdamW.class, Gollek.DL.adamW(parameters(), 0.001f, 0.8f, 0.99f, 1e-6f, 0.05f, true));
        assertInstanceOf(RMSprop.class, Gollek.DL.rmsprop(parameters(), 0.01f));
        assertInstanceOf(RMSprop.class, Gollek.DL.rmsprop(parameters(), 0.01f, 0.95f, 1e-6f, 0.01f, 0.9f));
        assertInstanceOf(NAdam.class, Gollek.DL.nadam(parameters(), 0.001f));
        assertInstanceOf(NAdam.class, Gollek.DL.nadam(parameters(), 0.001f, 0.01f));
        assertInstanceOf(NAdam.class, Gollek.DL.nadamW(parameters(), 0.001f, 0.01f));
    }

    @Test
    void exposesAdvancedOptimizerFactoriesOnGollekFacade() {
        assertInstanceOf(Adagrad.class, Gollek.DL.adagrad(parameters(), 0.01f));
        assertInstanceOf(Adagrad.class, Gollek.DL.adagrad(parameters(), 0.01f, 1e-8f, 0.01f));
        assertInstanceOf(Adadelta.class, Gollek.DL.adadelta(parameters()));
        assertInstanceOf(Adadelta.class, Gollek.DL.adadelta(parameters(), 1.0f, 0.95f, 1e-6f));
        assertInstanceOf(LAMB.class, Gollek.DL.lamb(parameters(), 0.001f));
        assertInstanceOf(LAMB.class, Gollek.DL.lamb(parameters(), 0.001f, 0.9f, 0.999f, 1e-6f, 0.01f));
        assertInstanceOf(Lion.class, Gollek.DL.lion(parameters(), 0.0001f));
        assertInstanceOf(Lion.class, Gollek.DL.lion(parameters(), 0.0001f, 0.9f, 0.99f, 0.01f));
        assertInstanceOf(RAdam.class, Gollek.DL.radam(parameters(), 0.001f));
        assertInstanceOf(RAdam.class, Gollek.DL.radam(parameters(), 0.001f, 0.01f));

        List<Parameter> lookaheadParams = parameters();
        Optimizer lookaheadBase = Gollek.DL.sgd(lookaheadParams, 0.1f);
        assertInstanceOf(Lookahead.class, Gollek.DL.lookahead(lookaheadBase));
        assertInstanceOf(Lookahead.class, Gollek.DL.lookahead(Gollek.DL.sgd(parameters(), 0.1f), 2, 0.6f));

        List<Parameter> samParams = parameters();
        Optimizer samBase = Gollek.DL.sgd(samParams, 0.1f);
        assertInstanceOf(SAM.class, Gollek.DL.sam(samParams, samBase));
        assertInstanceOf(SAM.class, Gollek.DL.sam(samParams, samBase, 0.05f));
    }

    @Test
    @SuppressWarnings("removal")
    void deprecatedGollekMlBridgeDelegatesToCoreOptimizerFactories() {
        assertInstanceOf(SGD.class, GollekML.DL.sgd(parameters(), 0.1f, 0.8f));
        assertInstanceOf(Adam.class, GollekML.DL.adam(parameters(), 0.001f, 0.01f));
        assertInstanceOf(AdamW.class, GollekML.DL.adamW(parameters(), 0.001f, 0.02f));
        assertInstanceOf(RMSprop.class, GollekML.DL.rmsprop(parameters(), 0.01f, 0.95f, 1e-6f, 0.01f, 0.8f));
    }

    @Test
    void factoryOptimizersCanStepAndExposeState() {
        Parameter parameter = parameter(1f, -2f);
        Optimizer optimizer = Gollek.DL.rmsprop(List.of(parameter), 0.01f, 0.95f, 1e-6f, 0.01f, 0.9f);

        parameter.data().backward(GradTensor.of(new float[] {0.5f, -0.25f}, 2));
        optimizer.step();

        assertNotEquals(1f, parameter.data().data()[0], 1e-6f);
        assertNotEquals(-2f, parameter.data().data()[1], 1e-6f);
        assertTrue(optimizer.supportsStateDict());

        Map<String, Object> state = optimizer.stateDict();
        assertEquals("RMSprop", state.get("optimizer"));
        assertEquals(0.95f, ((Number) state.get("alpha")).floatValue(), 1e-7f);
        assertEquals(0.01f, ((Number) state.get("weightDecay")).floatValue(), 1e-7f);
        assertEquals(0.9f, ((Number) state.get("momentum")).floatValue(), 1e-7f);
    }

    private static List<Parameter> parameters() {
        return List.of(parameter(1f, -2f));
    }

    private static Parameter parameter(float... values) {
        return new Parameter(GradTensor.of(values, values.length));
    }
}
