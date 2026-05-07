plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-tensor"))
    implementation(project(":core:gollek-ir"))
    implementation(project(":core:gollek-core"))
    implementation(platform("io.quarkus.platform:quarkus-bom:3.15.1"))
    implementation("io.quarkus:quarkus-resteasy-reactive")
    implementation("io.quarkus:quarkus-resteasy-reactive-jackson")
    implementation("io.quarkus:quarkus-smallrye-openapi")
    implementation("io.quarkus:quarkus-mutiny")
    
    implementation(project(":sdk:gollek-sdk-core"))
    implementation(project(":spi:gollek-spi-inference"))
    
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("io.quarkus:quarkus-junit5")
}

tasks.test {
    useJUnitPlatform()
}
