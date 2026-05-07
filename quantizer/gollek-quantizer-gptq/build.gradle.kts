plugins {
    java
}

dependencies {
    implementation(project(":runner:safetensor:gollek-safetensor-loader"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("org.slf4j:slf4j-api:2.0.12")
}
