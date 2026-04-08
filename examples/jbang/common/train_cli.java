//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.optim.*;

public class train_cli {
    public static void main(String[] args) {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 10;
        float lr = args.length > 1 ? Float.parseFloat(args[1]) : 0.001f;
        
        System.out.println("Training for " + epochs + " epochs");
        System.out.println("Learning rate: " + lr);
        
        Sequential model = new Sequential(
            new Linear(784, 128),
            new ReLU(),
            new Linear(128, 10)
        );
        
        Adam optimizer = new Adam(lr);
        
        for (int e = 1; e <= epochs; e++) {
            System.out.println("Epoch " + e + "...");
        }
        
        System.out.println("Done!");
    }
}
