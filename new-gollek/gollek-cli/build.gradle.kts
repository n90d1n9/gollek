plugins {
    java
    application
}

group = "tech.kayys.gollek"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":gollek-core"))
    implementation(project(":gollek-ir"))
    implementation(project(":gollek-runtime"))
    
    // CLI specific dependencies (Picocli, Jackson, etc. - inferred from existing pom.xml)
    implementation("info.picocli:picocli:4.7.5")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("org.jline:jline:3.25.1")
    
    implementation(project(":gollek-backend-metal"))
    annotationProcessor("info.picocli:picocli-codegen:4.7.5")
}

application {
    mainClass.set("tech.kayys.gollek.cli.NewGollekCLI")
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "--enable-preview")
}
