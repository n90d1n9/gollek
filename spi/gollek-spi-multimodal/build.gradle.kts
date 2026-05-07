plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-tensor"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")
}
