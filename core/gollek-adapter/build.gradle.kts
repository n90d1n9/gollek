plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-tensor"))
    implementation(project(":core:gollek-core"))
    // IR packages provide model representations used by adapters
    implementation(project(":core:gollek-ir"))
    implementation(project(":core:gollek-ir-schema"))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
