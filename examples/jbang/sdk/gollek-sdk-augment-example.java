///usr/bin/env jbang "$0" "$@" ; exit $?
// Legacy file name retained for compatibility during module migration.
// This script provides a lightweight augmentation flow that runs today.
//
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-sdk-autograd:0.1.0-SNAPSHOT
//COMPILE_OPTIONS --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED

import java.util.Locale;
import java.util.Random;
import tech.kayys.gollek.ml.autograd.GradTensor;

class GollekSdkAugmentExample {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        Random random = new Random(42);

        System.out.println("==============================================");
        System.out.println(" Gollek Augment Example (Compatibility Entry)");
        System.out.println("==============================================");

        GradTensor image = GradTensor.randn(3, 4, 4);
        GradTensor flipped = horizontalFlip(image);
        GradTensor noised = addGaussianNoise(flipped, 0.10f, random);

        System.out.println("Input shape: [3,4,4]");
        System.out.println("Channel-0 before:");
        printChannel(image, 0);
        System.out.println("Channel-0 after flip+noise:");
        printChannel(noised, 0);
    }

    private static GradTensor horizontalFlip(GradTensor chw) {
        long[] shape = chw.shape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("Expected CHW tensor with 3 dimensions.");
        }
        int c = (int) shape[0];
        int h = (int) shape[1];
        int w = (int) shape[2];

        float[] src = chw.data();
        float[] dst = new float[src.length];

        for (int ch = 0; ch < c; ch++) {
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                    int srcIdx = index(ch, row, col, h, w);
                    int dstIdx = index(ch, row, w - 1 - col, h, w);
                    dst[dstIdx] = src[srcIdx];
                }
            }
        }
        return GradTensor.of(dst, shape);
    }

    private static GradTensor addGaussianNoise(GradTensor chw, float sigma, Random random) {
        float[] src = chw.data();
        float[] dst = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            float noise = (float) (random.nextGaussian() * sigma);
            dst[i] = src[i] + noise;
        }
        return GradTensor.of(dst, chw.shape());
    }

    private static int index(int c, int h, int w, int height, int width) {
        return c * height * width + h * width + w;
    }

    private static void printChannel(GradTensor chw, int channel) {
        long[] shape = chw.shape();
        int c = (int) shape[0];
        int h = (int) shape[1];
        int w = (int) shape[2];
        if (channel < 0 || channel >= c) {
            throw new IllegalArgumentException("Invalid channel index: " + channel);
        }

        float[] data = chw.data();
        for (int row = 0; row < h; row++) {
            StringBuilder line = new StringBuilder();
            for (int col = 0; col < w; col++) {
                if (col > 0) {
                    line.append(" ");
                }
                line.append(String.format("% .4f", data[index(channel, row, col, h, w)]));
            }
            System.out.println("  " + line);
        }
    }
}
