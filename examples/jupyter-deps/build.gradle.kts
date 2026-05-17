plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.gollek"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    implementation(group = "tech.kayys.gollek", name = "gollek-ml-autograd")
    implementation(project(":ml:gollek-ml-nn"))
    implementation(group = "tech.kayys.gollek", name = "gollek-ml-tensor")
    implementation(project(":ml:gollek-ml-cnn"))
    implementation(group = "tech.kayys.gollek", name = "gollek-transformer")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
        }
    }
    repositories {
        mavenLocal()
    }
}
