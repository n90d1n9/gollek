plugins {
    `java-library`
}

dependencies {
    api("io.smallrye.config:smallrye-config:3.10.1")
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(21)
    options.compilerArgs.remove("--enable-preview")
    options.compilerArgs.remove("--add-modules=jdk.incubator.vector")
}
