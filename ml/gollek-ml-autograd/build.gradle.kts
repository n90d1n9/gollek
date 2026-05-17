plugins {
    `java-library`
}

dependencies {
    api(project(":core:gollek-tensor"))
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter", version = "5.10.2")
    testRuntimeOnly(group = "org.junit.platform", name = "junit-platform-launcher", version = "1.10.2")
}

tasks.jar {
    manifest {
        attributes(
            mapOf(
                "Automatic-Module-Name" to "tech.kayys.gollek.ml.autograd"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
