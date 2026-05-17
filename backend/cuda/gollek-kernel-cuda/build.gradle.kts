plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.gollek"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    implementation(project(":spi:gollek-spi-provider"))
    implementation(project(":core:gollek-model-runner"))
    implementation(project(":core:plugin:gollek-plugin-runner-core"))
    implementation(group = "tech.kayys.gollek", name = "gollek-engine")
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":optimization:gollek-plugin-kv-cache"))
    implementation(group = "tech.kayys.gollek", name = "gollek-spi-tensor")
    implementation(project(":optimization:gollek-plugin-fa4"))
    implementation(project(":optimization:gollek-plugin-fa3"))
    implementation(group = "io.quarkus", name = "quarkus-arc")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.assertj", name = "assertj-core")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
        }
    }
    repositories {
        mavenLocal()
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
