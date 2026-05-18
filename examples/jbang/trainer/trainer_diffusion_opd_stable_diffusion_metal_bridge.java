///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-diffusion-opd:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-backend-metal:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-runner-stable-diffusion:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-safetensor-core:0.1.0-SNAPSHOT

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DefaultTensor;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.runner.sd.PNDMScheduler;
import tech.kayys.gollek.train.diffusion.opd.adapter.StableDiffusionRunnerAdapters;

/**
 * JBang walkthrough for the Metal-aware Stable Diffusion bridge used by the
 * Java-first DiffusionOPD stack.
 *
 * <p>Reference:
 * Quanhao Li et al., "DiffusionOPD: A Unified Perspective of On-Policy
 * Distillation in Diffusion Models", arXiv:2605.15055, 2026.
 */
public class trainer_diffusion_opd_stable_diffusion_metal_bridge {

    public static void main(String[] args) {
        MetalBinding.initialize();
        MetalBinding metalBinding = MetalBinding.getInstance();
        metalBinding.init();

        var bridge = StableDiffusionRunnerAdapters.metalDefaultTensorFloat32Bridge(metalBinding);
        PNDMScheduler scheduler = new PNDMScheduler(4);
        var diffusionScheduler = StableDiffusionRunnerAdapters.scheduler(scheduler, bridge);

        try (AccelTensor accelLatents = AccelTensor.fromFloatArray(
                new float[] {
                        0.10f, 0.20f, 0.30f, 0.40f,
                        0.50f, 0.60f, 0.70f, 0.80f
                },
                1, 1, 2, 4)) {
            Tensor coreTensor = bridge.toCoreTensor(accelLatents);
            try {
                try (AccelTensor accelRoundTrip = bridge.toAccelTensor(coreTensor)) {
                    System.out.println("====================================================");
                    System.out.println(" Gollek JBang Stable Diffusion Metal Bridge");
                    System.out.println("====================================================");
                    System.out.println("reference=arXiv:2605.15055");
                    System.out.println("acceleratorMetadata=" + bridge.acceleratorMetadata());
                    System.out.println("schedulerMetadata="
                            + StableDiffusionRunnerAdapters.schedulerMetadata(scheduler));
                    System.out.println("coreTensor.device=" + coreTensor.device());
                    System.out.println("coreTensor.dtype=" + coreTensor.dtype());
                    System.out.println("coreTensor.shape=" + Arrays.toString(coreTensor.shape().dims()));
                    System.out.println("coreTensor.values=" + Arrays.toString(readFloat32(coreTensor)));
                    System.out.println("roundTrip.values=" + Arrays.toString(accelRoundTrip.toFloatArray()));
                    try (AccelTensor accelPrediction = AccelTensor.fromFloatArray(
                            new float[] {
                                    0.01f, 0.02f, 0.03f, 0.04f,
                                    0.05f, 0.06f, 0.07f, 0.08f
                            },
                            1, 1, 2, 4)) {
                        Tensor modelPrediction = bridge.toCoreTensor(accelPrediction);
                        try {
                            Tensor stepped = diffusionScheduler.step(coreTensor, modelPrediction, 0);
                            try {
                                System.out.println("schedulerStep.values=" + Arrays.toString(readFloat32(stepped)));
                            } finally {
                                stepped.release();
                            }
                        } finally {
                            modelPrediction.release();
                        }
                    }
                    System.out.println(
                            "note=training can now reuse the stable-diffusion-native PNDM step through the Metal-aware bridge");
                }
            } finally {
                coreTensor.release();
            }
        }
    }

    private static float[] readFloat32(Tensor tensor) {
        if (!(tensor instanceof DefaultTensor defaultTensor) || defaultTensor.dtype() != DType.F32) {
            throw new IllegalArgumentException(
                    "Expected float32 DefaultTensor, got " + tensor.getClass().getName());
        }
        int elementCount = (int) defaultTensor.shape().numel();
        float[] values = new float[elementCount];
        MemorySegment.copy(
                defaultTensor.buffer().segment(),
                0,
                MemorySegment.ofArray(values),
                0,
                (long) elementCount * Float.BYTES);
        return values;
    }
}
