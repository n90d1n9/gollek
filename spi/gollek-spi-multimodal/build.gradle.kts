plugins {
    `java-library`
}

dependencies {
    implementation(project(":core:gollek-tensor"))
    api("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    api("io.smallrye.reactive:mutiny:2.5.5")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")
}
