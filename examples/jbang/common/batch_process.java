//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.nn.*;
import java.util.*;

public class batch_process {
    public static void main(String[] args) {
        Sequential model = new Sequential(
            new Linear(10, 5),
            new ReLU(),
            new Linear(5, 1)
        );
        
        // Batch data
        List<String> files = Arrays.asList("file1.txt", "file2.txt", "file3.txt");
        
        for (String file : files) {
            System.out.println("Processing " + file);
            // Process...
        }
        
        System.out.println("Batch processing complete!");
    }
}
