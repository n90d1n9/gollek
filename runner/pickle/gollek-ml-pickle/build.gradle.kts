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
    implementation(project(":ml:gollek-ml-autograd"))
    implementation(project(":ml:gollek-ml-nn"))
    implementation(project(":core:gollek-tensor"))
    implementation(project(":ml:gollek-ml-core"))
    implementation(project(":ml:gollek-ml-estimator"))
    implementation(project(":ml:gollek-ml-preprocessing"))
    implementation(project(":ml:gollek-ml-persistence"))
    implementation(group = "org.slf4j", name = "slf4j-api", version = "2.0.12")
    compileOnly(group = "io.quarkus", name = "quarkus-core")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter", version = "5.10.2")
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
                "Implementation-Title" to "Gollek Pickle SDK",
                "Implementation-Version" to "0.1.0-SNAPSHOT"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
