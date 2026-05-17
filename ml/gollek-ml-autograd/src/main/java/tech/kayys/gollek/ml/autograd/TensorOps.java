package tech.kayys.gollek.ml.autograd;

import java.util.List;
import java.util.Objects;

/**
 * Narrow tensor helper surface used by the Gradle-migrated ML stack.
 */
public final class TensorOps {

    private TensorOps() {
    }

    public static GradTensor permute(GradTensor input, int... dims) {
        return input.permute(dims);
    }

    public static GradTensor cat(List<GradTensor> tensors, int dim) {
        Objects.requireNonNull(tensors, "tensors");
        return GradTensor.cat(dim, tensors.toArray(GradTensor[]::new));
    }

    public static GradTensor einsum(String equation, GradTensor left, GradTensor right) {
        return switch (equation) {
            case "bhid,bhjd->bhij" -> attentionScores(left, right);
            case "bhij,bhjd->bhid" -> attentionApply(left, right);
            default -> throw new UnsupportedOperationException(
                    "Unsupported compatibility einsum equation: " + equation);
        };
    }

    private static GradTensor attentionScores(GradTensor query, GradTensor key) {
        long[] qs = query.shape();
        long[] ks = key.shape();
        if (qs.length != 4 || ks.length != 4) {
            throw new IllegalArgumentException("attention score einsum expects rank-4 tensors");
        }
        int b = Math.toIntExact(qs[0]);
        int h = Math.toIntExact(qs[1]);
        int i = Math.toIntExact(qs[2]);
        int d = Math.toIntExact(qs[3]);
        int j = Math.toIntExact(ks[2]);
        if (ks[0] != qs[0] || ks[1] != qs[1] || ks[3] != qs[3]) {
            throw new IllegalArgumentException("attention score einsum shape mismatch");
        }

        float[] qd = query.data();
        float[] kd = key.data();
        float[] out = new float[b * h * i * j];
        for (int batch = 0; batch < b; batch++) {
            for (int head = 0; head < h; head++) {
                for (int qi = 0; qi < i; qi++) {
                    for (int kj = 0; kj < j; kj++) {
                        float acc = 0f;
                        for (int dim = 0; dim < d; dim++) {
                            int qIndex = (((batch * h + head) * i + qi) * d) + dim;
                            int kIndex = (((batch * h + head) * j + kj) * d) + dim;
                            acc += qd[qIndex] * kd[kIndex];
                        }
                        out[(((batch * h + head) * i + qi) * j) + kj] = acc;
                    }
                }
            }
        }
        GradTensor result = GradTensor.of(out, b, h, i, j)
                .requiresGrad(query.requiresGrad() || key.requiresGrad());
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("EinsumAttentionScores") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] upstreamData = upstream.data();
                    if (query.requiresGrad()) {
                        float[] queryGrad = new float[qd.length];
                        for (int batch = 0; batch < b; batch++) {
                            for (int head = 0; head < h; head++) {
                                for (int qi = 0; qi < i; qi++) {
                                    for (int dim = 0; dim < d; dim++) {
                                        float acc = 0f;
                                        for (int kj = 0; kj < j; kj++) {
                                            int upstreamIndex = (((batch * h + head) * i + qi) * j) + kj;
                                            int keyIndex = (((batch * h + head) * j + kj) * d) + dim;
                                            acc += upstreamData[upstreamIndex] * kd[keyIndex];
                                        }
                                        int queryIndex = (((batch * h + head) * i + qi) * d) + dim;
                                        queryGrad[queryIndex] += acc;
                                    }
                                }
                            }
                        }
                        query.backward(GradTensor.of(queryGrad, query.shape()));
                    }
                    if (key.requiresGrad()) {
                        float[] keyGrad = new float[kd.length];
                        for (int batch = 0; batch < b; batch++) {
                            for (int head = 0; head < h; head++) {
                                for (int kj = 0; kj < j; kj++) {
                                    for (int dim = 0; dim < d; dim++) {
                                        float acc = 0f;
                                        for (int qi = 0; qi < i; qi++) {
                                            int upstreamIndex = (((batch * h + head) * i + qi) * j) + kj;
                                            int queryIndex = (((batch * h + head) * i + qi) * d) + dim;
                                            acc += upstreamData[upstreamIndex] * qd[queryIndex];
                                        }
                                        int keyIndex = (((batch * h + head) * j + kj) * d) + dim;
                                        keyGrad[keyIndex] += acc;
                                    }
                                }
                            }
                        }
                        key.backward(GradTensor.of(keyGrad, key.shape()));
                    }
                }
            });
        }
        return result;
    }

    private static GradTensor attentionApply(GradTensor weights, GradTensor value) {
        long[] ws = weights.shape();
        long[] vs = value.shape();
        if (ws.length != 4 || vs.length != 4) {
            throw new IllegalArgumentException("attention apply einsum expects rank-4 tensors");
        }
        int b = Math.toIntExact(ws[0]);
        int h = Math.toIntExact(ws[1]);
        int i = Math.toIntExact(ws[2]);
        int j = Math.toIntExact(ws[3]);
        int d = Math.toIntExact(vs[3]);
        if (vs[0] != ws[0] || vs[1] != ws[1] || vs[2] != ws[3]) {
            throw new IllegalArgumentException("attention apply einsum shape mismatch");
        }

        float[] wd = weights.data();
        float[] vd = value.data();
        float[] out = new float[b * h * i * d];
        for (int batch = 0; batch < b; batch++) {
            for (int head = 0; head < h; head++) {
                for (int qi = 0; qi < i; qi++) {
                    for (int dim = 0; dim < d; dim++) {
                        float acc = 0f;
                        for (int kj = 0; kj < j; kj++) {
                            int wIndex = (((batch * h + head) * i + qi) * j) + kj;
                            int vIndex = (((batch * h + head) * j + kj) * d) + dim;
                            acc += wd[wIndex] * vd[vIndex];
                        }
                        out[(((batch * h + head) * i + qi) * d) + dim] = acc;
                    }
                }
            }
        }
        GradTensor result = GradTensor.of(out, b, h, i, d)
                .requiresGrad(weights.requiresGrad() || value.requiresGrad());
        if (result.requiresGrad() && !NoGrad.active()) {
            result.setGradFn(new Function.Context("EinsumAttentionApply") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] upstreamData = upstream.data();
                    if (weights.requiresGrad()) {
                        float[] weightsGrad = new float[wd.length];
                        for (int batch = 0; batch < b; batch++) {
                            for (int head = 0; head < h; head++) {
                                for (int qi = 0; qi < i; qi++) {
                                    for (int kj = 0; kj < j; kj++) {
                                        float acc = 0f;
                                        for (int dim = 0; dim < d; dim++) {
                                            int upstreamIndex = (((batch * h + head) * i + qi) * d) + dim;
                                            int valueIndex = (((batch * h + head) * j + kj) * d) + dim;
                                            acc += upstreamData[upstreamIndex] * vd[valueIndex];
                                        }
                                        int weightsIndex = (((batch * h + head) * i + qi) * j) + kj;
                                        weightsGrad[weightsIndex] += acc;
                                    }
                                }
                            }
                        }
                        weights.backward(GradTensor.of(weightsGrad, weights.shape()));
                    }
                    if (value.requiresGrad()) {
                        float[] valueGrad = new float[vd.length];
                        for (int batch = 0; batch < b; batch++) {
                            for (int head = 0; head < h; head++) {
                                for (int kj = 0; kj < j; kj++) {
                                    for (int dim = 0; dim < d; dim++) {
                                        float acc = 0f;
                                        for (int qi = 0; qi < i; qi++) {
                                            int upstreamIndex = (((batch * h + head) * i + qi) * d) + dim;
                                            int weightsIndex = (((batch * h + head) * i + qi) * j) + kj;
                                            acc += upstreamData[upstreamIndex] * wd[weightsIndex];
                                        }
                                        int valueIndex = (((batch * h + head) * j + kj) * d) + dim;
                                        valueGrad[valueIndex] += acc;
                                    }
                                }
                            }
                        }
                        value.backward(GradTensor.of(valueGrad, value.shape()));
                    }
                }
            });
        }
        return result;
    }
}
