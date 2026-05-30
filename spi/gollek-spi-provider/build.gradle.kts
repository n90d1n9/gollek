plugins {
    java
}

dependencies {
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-plugin"))
    implementation(project(":core:gollek-error-code"))
    implementation(project(":core:gollek-tensor"))
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":spi:gollek-spi-runtime"))
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    implementation("commons-codec:commons-codec:1.16.1")
    implementation("org.jetbrains:annotations:24.0.1")
    implementation("io.smallrye.reactive:mutiny:2.5.5")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("org.assertj:assertj-core:3.25.3")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
}
