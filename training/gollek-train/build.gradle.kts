plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-tensor"))
    implementation(project(":core:gollek-ir"))
    implementation(project(":core:gollek-core"))
    implementation(project(":compiler:gollek-compiler"))
    implementation(project(":runtime:gollek-runtime"))
}
