plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-tensor"))
    implementation(project(":core:gollek-ir"))
    implementation(project(":core:gollek-core"))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
