plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-tensor"))
    implementation(project(":core:gollek-model-repository"))
    implementation("io.smallrye.config:smallrye-config:3.4.4")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
}
