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
    api(project(":ml:gollek-ml-diffusion-api"))
    implementation(project(":core:gollek-core"))
    implementation(project(":runner:gollek-diffusion"))
    implementation(project(":runner:safetensor:gollek-runner-stable-diffusion"))
    implementation(project(":runner:safetensor:gollek-safetensor-core"))
    implementation(project(":backend:metal:gollek-backend-metal"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
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
                "Automatic-Module-Name" to "tech.kayys.gollek.ml.diffusion.opd"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
