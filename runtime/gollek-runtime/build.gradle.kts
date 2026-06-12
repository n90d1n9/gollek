plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-tensor"))
    implementation(project(":core:gollek-ir"))
    implementation(project(":compiler:gollek-compiler"))
    implementation(project(":core:gollek-core"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
}
