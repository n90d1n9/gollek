plugins {
    `java-library`
    `maven-publish`
}

import org.gradle.api.file.DuplicatesStrategy

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
    implementation(group = "org.dflib.jjava", name = "jjava", version = "1.0-a4")
    // The kernel itself only needs Gollek ML jars at runtime so notebook
    // snippets can import them; source compilation uses reflection.
    runtimeOnly(project(":core:gollek-tensor"))
    runtimeOnly(project(":ml:gollek-ml-autograd"))
    runtimeOnly(project(":ml:gollek-ml-nn"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation(project(":core:gollek-tensor"))
    testImplementation(project(":core:gollek-core"))
    testImplementation(project(":ml:gollek-ml-nn"))
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs("--add-opens=jdk.jshell/jdk.jshell=ALL-UNNAMED")
}

tasks.jar {
    manifest {
        attributes(
            mapOf(
                "Main-Class" to "tech.kayys.gollek.jupyter.KernelLauncher",
                "Automatic-Module-Name" to "tech.kayys.gollek.jupyter.kernel"
            )
        )
    }
}

val fatJar by tasks.registering(Jar::class) {
    group = "build"
    description = "Assemble a runnable standalone Gollek Jupyter kernel jar."
    archiveBaseName.set("gollek-kernel")
    archiveClassifier.set("")
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE

    manifest {
        attributes(
            mapOf(
                "Main-Class" to "tech.kayys.gollek.jupyter.KernelLauncher",
                "Automatic-Module-Name" to "tech.kayys.gollek.jupyter.kernel"
            )
        )
    }

    from(sourceSets.main.get().output)
    dependsOn(configurations.runtimeClasspath)
    from({
        configurations.runtimeClasspath.get()
            .filter { it.name.endsWith(".jar") }
            .map { zipTree(it) }
    })
}

tasks.assemble {
    dependsOn(fatJar)
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
