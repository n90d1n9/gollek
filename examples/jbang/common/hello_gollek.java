//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.nn.*;

public class hello_gollek {
    public static void main(String[] args) {
        System.out.println("Hello from Gollek!");
        
        // Create a simple model
        Sequential model = new Sequential(
            new Linear(10, 5),
            new ReLU(),
            new Linear(5, 1)
        );
        
        System.out.println("Model created successfully!");
    }
}
