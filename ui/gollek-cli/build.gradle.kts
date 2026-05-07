plugins {
    java
    application
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

group = "tech.kayys.gollek"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("info.picocli:picocli:4.7.5")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation(files(
        "../../sdk/gollek-sdk/target/gollek-sdk-0.1.0-SNAPSHOT.jar",
        "../../sdk/gollek-sdk-core/target/gollek-sdk-core-0.1.0-SNAPSHOT.jar",
        "../../sdk/gollek-sdk-session/target/gollek-sdk-session-0.1.0-SNAPSHOT.jar",
    ))
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-model"))
    annotationProcessor("info.picocli:picocli-codegen:4.7.5")

    runtimeOnly(fileTree("../..") {
        include("sdk/gollek-sdk-local/target/*.jar")
        include("runner/**/target/*.jar")
        include("plugin/**/target/*.jar")
        include("plugins/**/target/*.jar")
        include("provider/**/target/*.jar")
        include("integration/**/target/*.jar")
        include("models/**/target/*.jar")
        include("backend/**/target/*.jar")
        exclude("**/original-*.jar")
        exclude("**/quarkus-app/**")
        exclude("**/lib/**")
        exclude("**/generated-bytecode.jar")
        exclude("**/transformed-bytecode.jar")
    })
}

sourceSets {
    main {
        java {
            setSrcDirs(listOf("src/main/java"))
            include("tech/kayys/gollek/cli/NewGollekCLI.java")
        }
    }
}

application {
    mainClass.set("tech.kayys.gollek.cli.NewGollekCLI")
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "--enable-preview")
}
