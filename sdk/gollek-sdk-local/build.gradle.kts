plugins {
    java
}

dependencies {
    implementation(project(":sdk:gollek-sdk-core"))
    implementation(project(":core:gollek-core"))
    implementation(project(":core:gollek-model-repository"))
    implementation(project(":models:gollek-model-repo-hf"))
    implementation(project(":models:gollek-model-repo-local"))
    
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("io.opentelemetry:opentelemetry-api:1.34.1")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.inject:jakarta.inject-api:2.0.1")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.16.1")
    implementation("org.apache.commons:commons-lang3:3.14.0")
    
    compileOnly("org.graalvm.sdk:nativeimage:24.1.2")
}
