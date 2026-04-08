//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.autograd.GradTensor;

public class my_script {
    public static void main(String[] args) {
        System.out.println("=== Gollek Neural Network ===");
        
        // TODO: Add your code here
        
        Sequential model = new Sequential(
            new Linear(784, 128),
            new ReLU(),
            new Linear(128, 10)
        );
        
        System.out.println("Model ready for training!");
    }
}
