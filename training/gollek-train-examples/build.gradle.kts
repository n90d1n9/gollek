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
    implementation(project(":ml:gollek-ml-api"))
    implementation(project(":ml:gollek-ml-autograd"))
    implementation(project(":ml:gollek-ml-diffusion-opd"))
    implementation(project(":ml:gollek-ml-nn"))
    implementation(project(":ml:gollek-ml-cnn"))
    implementation(project(":ml:gollek-ml-optimize"))
    implementation(project(":runner:gollek-diffusion"))
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
