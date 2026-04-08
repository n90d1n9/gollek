//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.*;
import tech.kayys.gollek.ml.nn.optim.*;
import tech.kayys.gollek.ml.autograd.GradTensor;

public class train_model {
    public static void main(String[] args) {
        // Model
        Sequential model = new Sequential(
            new Linear(28 * 28, 128),
            new ReLU(),
            new Linear(128, 10)
        );
        
        // Loss and optimizer
        CrossEntropyLoss loss = new CrossEntropyLoss();
        Adam optimizer = new Adam(model.parameters(), 0.001f);
        
        // Training (pseudo-code)
        for (int epoch = 1; epoch <= 10; epoch++) {
            System.out.println("Epoch " + epoch);
        }
        
        System.out.println("Training complete!");
    }
}
