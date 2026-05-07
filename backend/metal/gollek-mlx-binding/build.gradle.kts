plugins {
    java
}

dependencies {
    implementation(project(":runner:safetensor:gollek-safetensor-core"))
    implementation(project(":spi:gollek-spi"))
    implementation("io.quarkus:quarkus-arc")
    
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}

