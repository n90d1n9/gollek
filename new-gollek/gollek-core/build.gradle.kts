plugins {
    java
}

// Read Feature Flags from gradle.properties or command line
val backends: List<String> = project.findProperty("gollek.backend")?.toString()?.split(",") ?: listOf("cpu")
val enableTraining: Boolean = project.findProperty("gollek.training")?.toString()?.toBoolean() ?: false
val enableInference: Boolean = project.findProperty("gollek.inference")?.toString()?.toBoolean() ?: true

println("⚙️ Configuring gollek-core build:")
println("   - Backends: $backends")
println("   - Training Enabled: $enableTraining")
println("   - Inference Enabled: $enableInference")

dependencies {
    implementation(project(":gollek-common"))
    implementation(project(":gollek-ir"))
}

// Dynamically include/exclude packages based on feature flags
sourceSets {
    main {
        java {
            // Exclude training modules if not needed
            if (!enableTraining) {
                println("   [Optimizer] Excluded 'train' and 'autograd' packages.")
                exclude("tech/kayys/gollek/train/**")
                exclude("tech/kayys/gollek/autograd/**")
            }

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
