plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-tensor"))
    implementation(project(":sdk:gollek-sdk-api"))
    implementation(project(":core:gollek-model-repository"))
    
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
}
