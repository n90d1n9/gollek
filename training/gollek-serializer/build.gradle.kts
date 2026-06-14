plugins {
    java
}

dependencies {

    implementation(project(":core:gollek-core"))
    implementation(project(":ml:gollek-ml-core"))
    implementation(project(":ml:gollek-ml-persistence"))
    implementation(project(":ml:gollek-ml-estimator"))
    implementation("com.google.code.gson:gson:2.10.1")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
