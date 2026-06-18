plugins {
    java
}

dependencies {
    implementation(project(":sdk:gollek-sdk-core"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    compileOnly("tech.kayys.gollek:gollek-engine:0.1.0-SNAPSHOT")
}
