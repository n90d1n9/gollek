package tech.kayys.gollek.ml.optimize;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.data.DataLoader;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.optim.Optimizer;

import java.util.Arrays;
import java.util.Objects;

/**
 * Knowledge Distillation trainer — trains a small student model to mimic
 * a large teacher model's soft probability distributions.
 *
 * <p>Based on <em>"Distilling the Knowledge in a Neural Network"</em> (Hinton et al., 2015).
 *
 * <p>The combined loss blends soft-target KL divergence with hard-label cross-entropy:
 * <pre>
 *   L = α · T² · KL(softmax(teacher/T) || softmax(student/T)) + (1-α) · CE(student, labels)
 * </pre>
 * where {@code T} is the temperature (higher = softer distributions) and
 * {@code α} is the soft-loss weight.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var distiller = KnowledgeDistillation.builder()
 *     .teacher(teacherModel)
 *     .student(studentModel)
 *     .optimizer(Adam.create(studentModel.parameters(), 1e-3f))
 *     .temperature(4.0f)
 *     .alpha(0.7f)
 *     .epochs(50)
 *     .build();
 *
 * distiller.fit(trainLoader);
 * }</pre>
 */
public final class KnowledgeDistillation {

    private final NNModule  teacher;
    private final NNModule  student;
    private final Optimizer optimizer;
    private final float   temperature;
    private final float   alpha;       // weight for soft loss
    private final int     epochs;

    private KnowledgeDistillation(Builder b) {
        this.teacher     = Objects.requireNonNull(b.teacher, "teacher model must not be null");
        this.student     = Objects.requireNonNull(b.student, "student model must not be null");
        this.optimizer   = Objects.requireNonNull(b.optimizer, "optimizer must not be null");
        this.temperature = requirePositiveFinite(b.temperature, "temperature");
        this.alpha       = requireProbability(b.alpha, "alpha");
        this.epochs      = requireNonNegative(b.epochs, "epochs");
    }

    /**
     * Computes the distillation loss for one batch.
     *
     * @param inputs  input batch {@code [N, ...]}
     * @param labels  hard labels {@code [N]} (class indices as float)
     * @return combined distillation loss scalar
     */
    public GradTensor distillationLoss(GradTensor inputs, GradTensor labels) {
        // Teacher forward (no grad)
        teacher.eval();
        GradTensor teacherLogits = teacher.forward(inputs).detach();

        // Student forward
        student.train();
        GradTensor studentLogits = student.forward(inputs);

        // Soft loss: KL(softmax(teacher/T) || softmax(student/T)).
        GradTensor softLoss = teacherStudentKl(studentLogits, teacherLogits, temperature)
                .mul(temperature * temperature);

        // Hard loss: cross-entropy(student, labels)
        GradTensor hardLoss = crossEntropy(studentLogits, labels);

        // Combined: α·soft + (1-α)·hard
        return softLoss.mul(alpha).add(hardLoss.mul(1f - alpha));
    }

    /**
     * Runs the full distillation training loop.
     *
     * @param loader iterable of {@code [inputs, labels]} batches
     */
    public void fit(Iterable<DataLoader.Batch> loader) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0f;
            int steps = 0;
            for (var batch : loader) {
                student.zeroGrad();
                GradTensor loss = distillationLoss(batch.inputs(), batch.labels());
                loss.backward();
                optimizer.step();
                epochLoss += loss.item();
                steps++;
            }
            System.out.printf("Epoch %d/%d  distill_loss=%.4f%n",
                epoch + 1, epochs, epochLoss / Math.max(1, steps));
        }
    }

    // ── Loss helpers ──────────────────────────────────────────────────────

    /**
     * Teacher-to-student KL divergence averaged over batch.
     *
     * <p>The teacher branch is treated as a fixed target distribution. The
     * backward path for student logits is the stable closed form
     * {@code (softmax(student/T) - softmax(teacher/T)) / (N*T)}.
     */
    private static GradTensor teacherStudentKl(
            GradTensor studentLogits,
            GradTensor teacherLogits,
            float temperature) {
        long[] studentShape = studentLogits.shape();
        long[] teacherShape = teacherLogits.shape();
        if (!Arrays.equals(studentShape, teacherShape)) {
            throw new IllegalArgumentException(
                    "student and teacher logits must have identical shape, got "
                            + Arrays.toString(studentShape) + " vs " + Arrays.toString(teacherShape));
        }
        if (studentShape.length != 2) {
            throw new IllegalArgumentException(
                    "distillation logits must be 2D [batch, classes], got "
                            + Arrays.toString(studentShape));
        }
        int batch = Math.toIntExact(studentShape[0]);
        int classes = Math.toIntExact(studentShape[1]);
        if (batch <= 0 || classes <= 0) {
            throw new IllegalArgumentException(
                    "distillation logits must have positive batch and class dimensions, got "
                            + Arrays.toString(studentShape));
        }

        float[] studentData = studentLogits.data();
        float[] teacherData = teacherLogits.data();
        float[] studentProb = new float[studentData.length];
        float[] teacherProb = new float[teacherData.length];
        float total = 0.0f;

        for (int row = 0; row < batch; row++) {
            int base = row * classes;
            float studentMax = Float.NEGATIVE_INFINITY;
            float teacherMax = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < classes; c++) {
                studentMax = Math.max(studentMax, studentData[base + c] / temperature);
                teacherMax = Math.max(teacherMax, teacherData[base + c] / temperature);
            }

            float studentSum = 0.0f;
            float teacherSum = 0.0f;
            for (int c = 0; c < classes; c++) {
                float studentExp = (float) Math.exp(studentData[base + c] / temperature - studentMax);
                float teacherExp = (float) Math.exp(teacherData[base + c] / temperature - teacherMax);
                studentProb[base + c] = studentExp;
                teacherProb[base + c] = teacherExp;
                studentSum += studentExp;
                teacherSum += teacherExp;
            }

            float logStudentDenom = studentMax + (float) Math.log(studentSum);
            float logTeacherDenom = teacherMax + (float) Math.log(teacherSum);
            for (int c = 0; c < classes; c++) {
                int idx = base + c;
                studentProb[idx] /= studentSum;
                teacherProb[idx] /= teacherSum;
                float logStudent = studentData[idx] / temperature - logStudentDenom;
                float logTeacher = teacherData[idx] / temperature - logTeacherDenom;
                total += teacherProb[idx] * (logTeacher - logStudent);
            }
        }

        GradTensor out = GradTensor.scalar(total / batch);
        if (studentLogits.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("KnowledgeDistillationKL") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / (batch * temperature);
                    float[] grad = new float[studentData.length];
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] = (studentProb[i] - teacherProb[i]) * scale;
                    }
                    studentLogits.backward(GradTensor.of(grad, studentLogits.shape()));
                }
            });
        }
        return out;
    }

    /**
     * Cross-entropy loss from logits and integer class labels.
     *
     * @param logits raw logits {@code [N, C]}
     * @param labels class indices {@code [N]}
     * @return scalar cross-entropy loss
     */
    private static GradTensor crossEntropy(GradTensor logits, GradTensor labels) {
        return new CrossEntropyLoss().compute(logits, labels);
    }

    private static float requirePositiveFinite(float value, String name) {
        if (!Float.isFinite(value) || value <= 0.0f) {
            throw new IllegalArgumentException(name + " must be finite and > 0, got: " + value);
        }
        return value;
    }

    private static float requireProbability(float value, String name) {
        if (!Float.isFinite(value) || value < 0.0f || value > 1.0f) {
            throw new IllegalArgumentException(name + " must be finite and within [0, 1], got: " + value);
        }
        return value;
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative, got: " + value);
        }
        return value;
    }

    /** @return a new builder */
    public static Builder builder() { return new Builder(); }

    /**
     * Builder for {@link KnowledgeDistillation}.
     */
    public static final class Builder {
        private NNModule   teacher, student;
        private Optimizer optimizer;
        private float    temperature = 4.0f;
        private float    alpha       = 0.7f;
        private int      epochs      = 10;

        /** @param teacher large pre-trained teacher model */
        public Builder teacher(NNModule m)       { this.teacher = m; return this; }
        /** @param student small student model to train */
        public Builder student(NNModule m)       { this.student = m; return this; }
        /** @param opt optimizer for student parameters */
        public Builder optimizer(Optimizer o)  { this.optimizer = o; return this; }
        /** @param T temperature for softening distributions (default 4.0) */
        public Builder temperature(float T)    { this.temperature = T; return this; }
        /** @param a weight for soft loss (default 0.7); hard loss weight = 1-a */
        public Builder alpha(float a)          { this.alpha = a; return this; }
        /** @param e number of training epochs */
        public Builder epochs(int e)           { this.epochs = e; return this; }

        /**
         * Builds the {@link KnowledgeDistillation} trainer.
         *
         * @return configured trainer
         */
        public KnowledgeDistillation build()   { return new KnowledgeDistillation(this); }
    }
}
