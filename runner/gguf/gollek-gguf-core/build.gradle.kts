plugins {
    java
}

sourceSets {
    main {
        java {
            // Keep the actively used GGUF metadata/parser/tokenizer surface.
            // The module also contains older exporter/runner/training trees that
            // do not match today's package layout and pull in stale dependencies.
            include(
                "tech/kayys/gollek/gguf/core/GGUFConstants.java",
                "tech/kayys/gollek/gguf/core/GGUFTensorInfo.java",
                "tech/kayys/gollek/gguf/core/GgmlType.java",
                "tech/kayys/gollek/gguf/core/GgufMetaType.java",
                "tech/kayys/gollek/gguf/core/GgufMetaValue.java",
                "tech/kayys/gollek/gguf/core/GgufExporter.java",
                "tech/kayys/gollek/gguf/core/GgufModel.java",
                "tech/kayys/gollek/gguf/loader/GGUFModel.java",
                "tech/kayys/gollek/gguf/loader/GGUFTensorInfo.java",
                "tech/kayys/gollek/gguf/loader/GGUFParser.java",
                "tech/kayys/gollek/gguf/loader/GGUFReader.java",
                "tech/kayys/gollek/gguf/loader/gguf/*.java",
                "tech/kayys/gollek/gguf/loader/quant/*.java",
                "tech/kayys/gollek/gguf/runtime/*.java",
                "tech/kayys/gollek/gguf/tokenizer/GGUFTokenizer.java",
                "tech/kayys/gollek/gguf/writer/GGUFWriter.java",
            )
        }
    }
}

dependencies {
    implementation(project(":ml:gollek-ml-autograd"))
    implementation(project(":ml:gollek-ml-nn"))
    implementation(project(":spi:gollek-spi-model"))
    implementation("com.google.code.gson:gson:2.10.1")
    implementation(project(":core:gollek-tokenizer-core"))
    implementation("io.quarkus:quarkus-core")
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
