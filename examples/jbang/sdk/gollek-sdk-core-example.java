///usr/bin/env jbang "$0" "$@" ; exit $?
// Legacy file name retained for compatibility during module migration.
//
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-sdk-autograd:0.1.0-SNAPSHOT
//COMPILE_OPTIONS --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED

import java.util.Arrays;
import java.util.Locale;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Linear;
import tech.kayys.gollek.ml.nn.ReLU;
import tech.kayys.gollek.ml.nn.Sequential;

class GollekSdkCoreExample {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        System.out.println("==============================================");
        System.out.println(" Gollek Core Tensor Example (Compatibility)");
        System.out.println("==============================================");

        GradTensor a = GradTensor.randn(2, 3);
        GradTensor b = GradTensor.ones(2, 3);
        GradTensor c = a.add(b);
        GradTensor d = c.relu();
        GradTensor e = d.reshape(3, 2);

        System.out.println("a shape: " + Arrays.toString(a.shape()));
        System.out.println("b shape: " + Arrays.toString(b.shape()));
        System.out.println("c=a+b shape: " + Arrays.toString(c.shape()));
        System.out.println("d=relu(c) shape: " + Arrays.toString(d.shape()));
        System.out.println("e=reshape(d,3,2) shape: " + Arrays.toString(e.shape()));
        System.out.printf("Sample value e[0]=%.6f%n", e.item(0));

        Sequential mlp = new Sequential(
                new Linear(3, 8),
                new ReLU(),
                new Linear(8, 2));

        GradTensor logits = mlp.forward(GradTensor.randn(4, 3));
        System.out.println("MLP forward output shape: " + Arrays.toString(logits.shape()));
        System.out.println("Parameter count: " + mlp.parameterCountFormatted());
    }
}
