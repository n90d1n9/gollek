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

sourceSets {
    named("main") {
        java.exclude("tech/kayys/gollek/ml/runner/**")
    }
}

dependencies {
    api(project(":ml:gollek-ml-runner-api"))
    api(project(":trainer:gollek-trainer-api"))
    implementation(project(":trainer:gollek-trainer"))
    api(project(":ml:gollek-ml-autograd"))
    api(project(":ml:gollek-ml-core"))
    api(project(":ml:gollek-ml-data"))
    api(project(":ml:gollek-ml-diffusion-api"))
    api(project(":ml:gollek-ml-diffusion-opd"))
    api(project(":ml:gollek-ml-nn"))
    api(project(":ml:gollek-ml-estimator"))
    api(project(":ml:gollek-ml-preprocessing"))
    api(project(":ml:gollek-ml-selection"))
    api(project(":ml:gollek-ml-optimize"))
    api(project(":ml:gollek-ml-hub"))
    api(project(":ml:gollek-ml-export"))
    implementation(project(":sdk:gollek-sdk-api"))
    api(project(":ml:gollek-ml-multimodal"))
    api(project(":ml:gollek-ml-cnn"))
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.assertj", name = "assertj-core", version = "3.26.3")
    testRuntimeOnly(group = "org.junit.platform", name = "junit-platform-launcher", version = "1.10.2")
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

tasks.jar {
    manifest {
        attributes(
            mapOf(
                "Automatic-Module-Name" to "tech.kayys.gollek.ml"
            )
        )
    }
}

tasks.named<JavaCompile>("compileJava") {
    // The estimator project is mounted under an aliased physical directory.
    // Make the clean-build jar edge explicit so Gradle never races this compile
    // against the jar materialization required by the compile classpath.
    dependsOn(":ml:gollek-ml-estimator:jar")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
