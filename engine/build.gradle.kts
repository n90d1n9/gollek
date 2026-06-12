plugins {
    java
}

dependencies {
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-provider"))
    implementation(project(":core:plugin:gollek-plugin-runner-core"))
    implementation(project(":plugins:gollek-plugin-model-router"))
    implementation(project(":core:gollek-provider-core"))
    implementation(project(":plugins:gollek-plugin-semantic-cache"))
    implementation("io.smallrye.reactive:mutiny")
    
    // GraalVM annotations for NativeImageFeature
    compileOnly("org.graalvm.sdk:nativeimage:24.1.2")
    implementation("io.quarkus:quarkus-core")

    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
