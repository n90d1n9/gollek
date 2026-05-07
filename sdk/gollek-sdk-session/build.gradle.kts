plugins {
    java
}

dependencies {
    implementation(project(":sdk:gollek-sdk-core"))
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("org.slf4j:slf4j-api:2.0.11")
}
