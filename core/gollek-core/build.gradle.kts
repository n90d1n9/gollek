plugins {
    java
}

// Read Feature Flags from gradle.properties or command line
val backends: List<String> = project.findProperty("gollek.backend")?.toString()?.split(",") ?: listOf("cpu")

val enableInference: Boolean = project.findProperty("gollek.inference")?.toString()?.toBoolean() ?: true

println("⚙️ Configuring gollek-core build:")
println("   - Backends: $backends")
println("   - Inference Enabled: $enableInference")

dependencies {
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-ir"))
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-model"))
    // Depend on published aljabr core aggregator for shared math/tensor/tokenizer foundations
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
}

// Dynamically include/exclude packages based on feature flags
sourceSets {
    main {
        java {


            // Exclude inference/runner modules if not needed
            if (!enableInference) {
                println("   [Optimizer] Excluded 'runtime' and 'diffusion' packages.")
                exclude("tech/kayys/gollek/runtime/**")
                exclude("tech/kayys/gollek/diffusion/**")
            }

            // Exclude backend implementations that are not requested
            if (!backends.contains("cuda")) {
                println("   [Optimizer] Excluded 'backend/cuda' package.")
                exclude("tech/kayys/gollek/backend/cuda/**")
            }
            if (!backends.contains("metal")) {
                println("   [Optimizer] Excluded 'backend/metal' package.")
                exclude("tech/kayys/gollek/backend/metal/**")
            }
            if (!backends.contains("cpu")) {
                println("   [Optimizer] Excluded 'backend/cpu' package.")
                exclude("tech/kayys/gollek/backend/cpu/**")
            }
        }
    }
}
