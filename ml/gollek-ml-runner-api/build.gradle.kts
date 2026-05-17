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

sourceSets {
    named("main") {
        java.srcDir(project.file("../../training/gollek-train-api/src/main/java"))
        java.include("tech/kayys/gollek/ml/runner/**")
    }
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    implementation(project(":core:gollek-tensor"))
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
                "Automatic-Module-Name" to "tech.kayys.gollek.ml.runner"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
