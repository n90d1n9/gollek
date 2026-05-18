package tech.kayys.gollek.ml.optimize;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.data.DataLoader;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.optim.SGD;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class KnowledgeDistillationTest {

    @Test
    void hardLabelBranchBackpropagatesIntoStudentLogits() {
        float[] studentData = {
                2.0f, -1.0f, 0.5f,
                -0.5f, 1.2f, 0.0f
        };
        LogitModule student = new LogitModule(studentData, 2, 3);
        LogitModule teacher = new LogitModule(new float[] {
                0.1f, 1.1f, -0.4f,
                1.0f, -0.2f, 0.3f
        }, 2, 3);

        KnowledgeDistillation distiller = KnowledgeDistillation.builder()
                .teacher(teacher)
                .student(student)
                .optimizer(SGD.builder(student.parameters(), 0.1f).build())
                .alpha(0.0f)
                .temperature(2.0f)
                .epochs(1)
                .build();

        distiller.distillationLoss(dummyInputs(), GradTensor.of(new float[] {0.0f, 2.0f}, 2)).backward();

        assertArrayEquals(
                crossEntropyGradient(studentData, new int[] {0, 2}, 2, 3),
                student.logits().grad().data(),
                1e-6f);
        assertNull(teacher.logits().grad(), "teacher logits must stay detached during distillation");
    }

    @Test
    void softTeacherBranchUsesTeacherToStudentKlGradient() {
        float temperature = 2.5f;
        float[] studentData = {
                0.4f, -0.2f, 1.0f,
                -0.8f, 0.7f, 0.2f
        };
        float[] teacherData = {
                -0.1f, 1.3f, 0.6f,
                1.1f, -0.4f, 0.0f
        };
        LogitModule student = new LogitModule(studentData, 2, 3);
        LogitModule teacher = new LogitModule(teacherData, 2, 3);

        KnowledgeDistillation distiller = KnowledgeDistillation.builder()
                .teacher(teacher)
                .student(student)
                .optimizer(SGD.builder(student.parameters(), 0.1f).build())
                .alpha(1.0f)
                .temperature(temperature)
                .epochs(1)
                .build();

        distiller.distillationLoss(dummyInputs(), GradTensor.of(new float[] {1.0f, 0.0f}, 2)).backward();

        assertArrayEquals(
                distillationGradient(studentData, teacherData, 2, 3, temperature),
                student.logits().grad().data(),
                1e-6f);
        assertNull(teacher.logits().grad(), "teacher logits must stay detached during soft KL");
    }

    @Test
    void fitRunsOptimizerStepWithDistillationLoss() {
        LogitModule student = new LogitModule(new float[] {
                0.2f, -0.3f,
                -0.4f, 0.1f
        }, 2, 2);
        LogitModule teacher = new LogitModule(new float[] {
                1.0f, -1.0f,
                -1.0f, 1.0f
        }, 2, 2);
        float before = student.logits().data().data()[0];

        KnowledgeDistillation.builder()
                .teacher(teacher)
                .student(student)
                .optimizer(SGD.builder(student.parameters(), 0.1f).build())
                .alpha(0.5f)
                .temperature(2.0f)
                .epochs(1)
                .build()
                .fit(List.of(new DataLoader.Batch(
                        dummyInputs(),
                        GradTensor.of(new float[] {0.0f, 1.0f}, 2))));

        assertNotEquals(before, student.logits().data().data()[0], 1e-7f);
    }

    @Test
    void builderRejectsInvalidDistillationConfiguration() {
        LogitModule module = new LogitModule(new float[] {0.0f, 1.0f}, 1, 2);

        assertThrows(NullPointerException.class, () -> KnowledgeDistillation.builder()
                .student(module)
                .optimizer(SGD.builder(module.parameters(), 0.1f).build())
                .build());
        assertThrows(IllegalArgumentException.class, () -> KnowledgeDistillation.builder()
                .teacher(module)
                .student(module)
                .optimizer(SGD.builder(module.parameters(), 0.1f).build())
                .temperature(0.0f)
                .build());
        assertThrows(IllegalArgumentException.class, () -> KnowledgeDistillation.builder()
                .teacher(module)
                .student(module)
                .optimizer(SGD.builder(module.parameters(), 0.1f).build())
                .alpha(1.2f)
                .build());
        assertThrows(IllegalArgumentException.class, () -> KnowledgeDistillation.builder()
                .teacher(module)
                .student(module)
                .optimizer(SGD.builder(module.parameters(), 0.1f).build())
                .epochs(-1)
                .build());
    }

    @Test
    void lossRejectsMismatchedTeacherAndStudentLogitShapes() {
        LogitModule student = new LogitModule(new float[] {0.0f, 1.0f}, 1, 2);
        LogitModule teacher = new LogitModule(new float[] {0.0f, 1.0f, 2.0f}, 1, 3);

        KnowledgeDistillation distiller = KnowledgeDistillation.builder()
                .teacher(teacher)
                .student(student)
                .optimizer(SGD.builder(student.parameters(), 0.1f).build())
                .build();

        assertThrows(IllegalArgumentException.class,
                () -> distiller.distillationLoss(dummyInputs(), GradTensor.of(new float[] {0.0f}, 1)));
    }

    private static GradTensor dummyInputs() {
        return GradTensor.of(new float[] {0.0f, 0.0f}, 2, 1);
    }

    private static float[] crossEntropyGradient(float[] logits, int[] labels, int batch, int classes) {
        float[] grad = new float[logits.length];
        for (int row = 0; row < batch; row++) {
            float[] probs = softmax(logits, row * classes, classes, 1.0f);
            for (int c = 0; c < classes; c++) {
                grad[row * classes + c] = probs[c] / batch;
            }
            grad[row * classes + labels[row]] -= 1.0f / batch;
        }
        return grad;
    }

    private static float[] distillationGradient(
            float[] student,
            float[] teacher,
            int batch,
            int classes,
            float temperature) {
        float[] grad = new float[student.length];
        for (int row = 0; row < batch; row++) {
            float[] studentProb = softmax(student, row * classes, classes, temperature);
            float[] teacherProb = softmax(teacher, row * classes, classes, temperature);
            for (int c = 0; c < classes; c++) {
                grad[row * classes + c] = temperature * (studentProb[c] - teacherProb[c]) / batch;
            }
        }
        return grad;
    }

    private static float[] softmax(float[] logits, int offset, int classes, float temperature) {
        float max = Float.NEGATIVE_INFINITY;
        for (int c = 0; c < classes; c++) {
            max = Math.max(max, logits[offset + c] / temperature);
        }
        float sum = 0.0f;
        float[] probabilities = new float[classes];
        for (int c = 0; c < classes; c++) {
            probabilities[c] = (float) Math.exp(logits[offset + c] / temperature - max);
            sum += probabilities[c];
        }
        for (int c = 0; c < classes; c++) {
            probabilities[c] /= sum;
        }
        return probabilities;
    }

    private static final class LogitModule extends NNModule {
        private final Parameter logits;

        private LogitModule(float[] logitsData, int batch, int classes) {
            this.logits = registerParameter("logits", GradTensor.of(logitsData.clone(), batch, classes));
        }

        @Override
        public GradTensor forward(GradTensor input) {
            return logits.data();
        }

        private Parameter logits() {
            return logits;
        }
    }
}
