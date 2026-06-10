import org.gradle.api.GradleException
import org.gradle.api.tasks.Exec
import org.w3c.dom.Element
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory

val jbangTrainerPublicationModules = listOf(
    ":core:gollek-tensor",
    ":core:gollek-error-code",
    ":core:gollek-runtime-config",
    ":core:gollek-ir",
    ":core:gollek-tokenizer-core",
    ":core:gollek-provider-core",
    ":core:gollek-model-repository",
    ":core:gollek-model-repo-local",
    ":core:gollek-model-repo-hf",
    ":spi:gollek-spi",
    ":spi:gollek-spi-model",
    ":spi:gollek-spi-inference",
    ":spi:gollek-spi-multimodal",
    ":spi:gollek-spi-plugin",
    ":spi:gollek-spi-provider",
    ":spi:gollek-spi-runtime",
    ":trainer:gollek-trainer-api",
    ":trainer:gollek-trainer",
    ":ml:gollek-ml-runner-api",
    ":ml:gollek-ml-autograd",
    ":ml:gollek-ml-core",
    ":ml:gollek-ml-data",
    ":ml:gollek-ml-nn",
    ":ml:gollek-ml-estimator",
    ":ml:gollek-ml-preprocessing",
    ":ml:gollek-ml-selection",
    ":ml:gollek-ml-optimize",
    ":ml:gollek-ml-hub",
    ":ml:gollek-ml-export",
    ":ml:gollek-ml-multimodal",
    ":ml:gollek-ml-cnn",
    ":runner:onnx:gollek-ml-export-onnx",
    ":runner:gguf:gollek-gguf-core",
    ":runner:litert:gollek-litert-core",
    ":runner:litert:gollek-runner-litert",
    ":runner:safetensor:gollek-safetensor-spi",
    ":runner:safetensor:gollek-safetensor-api",
    ":runner:safetensor:gollek-safetensor-loader",
    ":runner:safetensor:gollek-safetensor-core",
    ":runner:safetensor:gollek-safetensor-quantization",
    ":quantizer:gollek-quantizer-gptq",
    ":quantizer:gollek-quantizer-awq",
    ":quantizer:gollek-quantizer-autoround",
    ":quantizer:gollek-quantizer-turboquant",
    ":sdk:gollek-sdk-api",
    ":backend:metal:gollek-backend-metal",
    ":ml:gollek-ml-api",
)

val publishJbangTrainerExamplesToMavenLocal = tasks.register("publishJbangTrainerExamplesToMavenLocal") {
    group = "publishing"
    description = "Publishes the Gradle-built runtime graph needed by trainer JBang examples."
    dependsOn(jbangTrainerPublicationModules.map { "$it:publishToMavenLocal" })
}

val jbangAgentBridgePublicationModules = listOf(
    ":sdk:gollek-sdk-agent",
)

val publishJbangAgentBridgeExamplesToMavenLocal =
    tasks.register("publishJbangAgentBridgeExamplesToMavenLocal") {
        group = "publishing"
        description = "Publishes the local artifact graph needed by agent bridge JBang examples."
        dependsOn(jbangAgentBridgePublicationModules.map { "$it:publishToMavenLocal" })
    }

val jbangTrainerDevice = providers.gradleProperty("gollek.jbang.trainer.device").orElse("auto")
val jbangTrainerQualityProfile = providers.gradleProperty("gollek.jbang.trainer.qualityProfile").orElse("local-experiment")
val jbangTrainerOffline = providers.gradleProperty("gollek.jbang.trainer.offline")
    .map { it.toBooleanStrictOrNull() ?: true }
    .orElse(true)
val jbangAgentBridgeOffline = providers.gradleProperty("gollek.jbang.agentBridge.offline")
    .map { it.toBooleanStrictOrNull() ?: false }
    .orElse(false)
val jbangTrainerHomeDir = layout.buildDirectory.dir("jbang/.jbang")
val wayangGollekBridgeOutputFile =
    layout.buildDirectory.file("jbang/integration/wayang-gollek-serving-bridge.out")
val qualityProfileEvidenceOutputDir =
    layout.buildDirectory.dir("jbang/trainer/quality-profile-ci-gate-evidence")
val qualityProfileEvidenceSummaryFile =
    layout.buildDirectory.file("jbang/trainer/quality-profile-ci-gate-evidence/quality-profile-ci-gate.summary.txt")
val qualityProfileEvidenceRequiredFiles = listOf(
    "runs/baseline/canonical-report.json",
    "runs/candidate/canonical-report.json",
    "quality-profile-ci-gate/quality-profile-ci-gate-manifest.json",
    "quality-profile-ci-gate/quality-profile-ci-gate-manifest.md",
    "quality-profile-ci-gate/manifest-verification-report/quality-profile-ci-gate-manifest-verification.json",
    "quality-profile-ci-gate/manifest-verification-report/quality-profile-ci-gate-manifest-verification.md",
    "quality-profile-ci-gate/manifest-verification-report/quality-profile-ci-gate-manifest-verification.junit.xml",
    "quality-profile-ci-gate/manifest-verification-report/quality-profile-ci-gate-manifest-verification.receipt.json",
)
val runtimeProfileBudgetGateOutputDir =
    layout.buildDirectory.dir("jbang/trainer/runtime-profile-budget-gate")
val runtimeProfileBudgetGateSummaryFile =
    layout.buildDirectory.file("jbang/trainer/runtime-profile-budget-gate/runtime-profile-budget-gate.summary.txt")
val trainerEvidenceIndexJsonFile =
    layout.buildDirectory.file("jbang/trainer/trainer-evidence-index.json")
val trainerEvidenceIndexSummaryFile =
    layout.buildDirectory.file("jbang/trainer/trainer-evidence-index.summary.txt")
val runtimeProfileBudgetGateRequiredFiles = listOf(
    "canonical-runtime-profile-report.json",
    "runtime-profile-budget-gate/runtime-profile-budget-gate.json",
    "runtime-profile-budget-gate/runtime-profile-budget-gate.md",
    "runtime-profile-budget-gate/runtime-profile-budget-gate.junit.xml",
    "runtime-input-profile-gate/runtime-input-profile-gate.json",
    "runtime-input-profile-gate/runtime-input-profile-gate.md",
    "runtime-input-profile-gate/runtime-input-profile-gate.junit.xml",
)

fun requiredNonEmptyFile(root: File, relativePath: String, label: String) =
    root.resolve(relativePath).also { file ->
        if (!file.isFile) {
            throw GradleException("Missing $label file: ${file.absolutePath}")
        }
        if (file.length() <= 0L) {
            throw GradleException("$label file is empty: ${file.absolutePath}")
        }
    }

fun requireFileContains(file: File, marker: String, label: String) {
    if (!file.readText().contains(marker)) {
        throw GradleException("$label file ${file.absolutePath} does not contain marker: $marker")
    }
}

fun requireFileNotContains(file: File, marker: String, label: String) {
    if (file.readText().contains(marker)) {
        throw GradleException("$label file ${file.absolutePath} should not contain marker: $marker")
    }
}

fun requireArtifactExists(root: File, relativePath: String, label: String) {
    requiredNonEmptyFile(root, relativePath, "$label referenced artifact")
}

fun writeKeyValueSummary(file: File, entries: Map<String, String>) {
    file.parentFile.mkdirs()
    file.writeText(entries.entries.joinToString(separator = "\n", postfix = "\n") { (key, value) ->
        "$key=$value"
    })
    if (file.length() <= 0L) {
        throw GradleException("Summary file is empty after write: ${file.absolutePath}")
    }
}

fun readKeyValueSummary(file: File, label: String): Map<String, String> {
    if (!file.isFile) {
        throw GradleException("Missing $label summary file: ${file.absolutePath}")
    }
    return file.readLines()
        .filter { it.isNotBlank() }
        .associate { line ->
            val separator = line.indexOf('=')
            if (separator <= 0) {
                throw GradleException("$label summary contains malformed line: $line")
            }
            line.substring(0, separator) to line.substring(separator + 1)
        }
}

fun verifyKeyValueSummary(file: File, expectedEntries: Map<String, String>, label: String) {
    val actual = readKeyValueSummary(file, label)
    expectedEntries.forEach { (key, expectedValue) ->
        val actualValue = actual[key]
        if (actualValue != expectedValue) {
            throw GradleException(
                "$label summary expected $key=$expectedValue but got ${actualValue ?: "<missing>"}"
            )
        }
    }
}

fun requireSummaryEntry(summary: Map<String, String>, key: String, expectedValue: String, label: String) {
    val actualValue = summary[key]
    if (actualValue != expectedValue) {
        throw GradleException("$label summary expected $key=$expectedValue but got ${actualValue ?: "<missing>"}")
    }
}

fun jsonString(value: String) = buildString {
    append('"')
    value.forEach { character ->
        when (character) {
            '\\' -> append("\\\\")
            '"' -> append("\\\"")
            '\b' -> append("\\b")
            '\u000C' -> append("\\f")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> {
                if (character < ' ') {
                    append("\\u%04x".format(character.code))
                } else {
                    append(character)
                }
            }
        }
    }
    append('"')
}

fun secureXmlDocument(file: File) = DocumentBuilderFactory.newInstance().apply {
    isNamespaceAware = false
    setFeature("http://apache.org/xml/features/disallow-doctype-decl", true)
    setFeature("http://xml.org/sax/features/external-general-entities", false)
    setFeature("http://xml.org/sax/features/external-parameter-entities", false)
}.newDocumentBuilder().parse(file).also { document ->
    document.documentElement.normalize()
}

fun requireXmlAttribute(element: Element, attributeName: String, expectedValue: String) {
    val actualValue = element.getAttribute(attributeName)
    if (actualValue != expectedValue) {
        throw GradleException("Expected XML attribute $attributeName=$expectedValue but got $actualValue")
    }
}

fun requireXmlTagName(element: Element, expectedTagName: String) {
    if (element.tagName != expectedTagName) {
        throw GradleException("Expected XML root element $expectedTagName but got ${element.tagName}")
    }
}

fun requireXmlElementCount(parent: Element, tagName: String, expectedCount: Int, label: String) {
    val actualCount = parent.getElementsByTagName(tagName).length
    if (actualCount != expectedCount) {
        throw GradleException("Expected $expectedCount $label XML elements but found $actualCount")
    }
}

fun Exec.configureTrainerJbangSmoke(
    scriptPath: String,
    outputDirectory: File,
    vararg scriptArgs: String,
) {
    workingDir = layout.projectDirectory.dir("examples/jbang").asFile
    outputs.upToDateWhen { false }

    doFirst {
        delete(outputDirectory)
        val jbangHome = jbangTrainerHomeDir.get().asFile
        jbangHome.mkdirs()
        environment("JBANG_DIR", jbangHome.absolutePath)

        val args = mutableListOf("jbang")
        if (jbangTrainerOffline.get()) {
            args += "--offline"
        }
        args += scriptPath
        args += scriptArgs
        commandLine(args)
    }
}

val smokeJbangTrainerQualityProfileCiGateEvidence =
    tasks.register<Exec>("smokeJbangTrainerQualityProfileCiGateEvidence") {
        group = "verification"
        description = "Runs the trainer quality-profile CI gate JBang evidence example end-to-end."
        dependsOn(publishJbangTrainerExamplesToMavenLocal)

        inputs.file(layout.projectDirectory.file("examples/jbang/trainer/trainer_quality_profile_ci_gate_evidence.java"))
        outputs.dir(qualityProfileEvidenceOutputDir)
        configureTrainerJbangSmoke(
            "trainer/trainer_quality_profile_ci_gate_evidence.java",
            qualityProfileEvidenceOutputDir.get().asFile,
            "--out",
            qualityProfileEvidenceOutputDir.get().asFile.absolutePath,
            "--device",
            jbangTrainerDevice.get(),
            "--profile",
            jbangTrainerQualityProfile.get(),
            "--fail-on-gate",
        )
    }

val verifyJbangTrainerQualityProfileCiGateEvidenceOutput =
    tasks.register("verifyJbangTrainerQualityProfileCiGateEvidenceOutput") {
        group = "verification"
        description = "Verifies the files produced by the trainer quality-profile CI gate JBang smoke."
        dependsOn(smokeJbangTrainerQualityProfileCiGateEvidence)

        inputs.dir(qualityProfileEvidenceOutputDir)
        outputs.file(qualityProfileEvidenceSummaryFile)

        doLast {
            val outputRoot = qualityProfileEvidenceOutputDir.get().asFile
            val label = "Trainer JBang evidence"
            fun requiredFile(relativePath: String) = requiredNonEmptyFile(outputRoot, relativePath, label)
            fun requireContains(file: File, marker: String) = requireFileContains(file, marker, label)
            fun requireNotContains(file: File, marker: String) = requireFileNotContains(file, marker, label)

            qualityProfileEvidenceRequiredFiles.forEach(::requiredFile)

            val verificationJson = requiredFile(
                "quality-profile-ci-gate/manifest-verification-report/quality-profile-ci-gate-manifest-verification.json"
            )
            val verificationMarkdown = requiredFile(
                "quality-profile-ci-gate/manifest-verification-report/quality-profile-ci-gate-manifest-verification.md"
            )
            val verificationJunitXml = requiredFile(
                "quality-profile-ci-gate/manifest-verification-report/quality-profile-ci-gate-manifest-verification.junit.xml"
            )
            val verificationReceipt = requiredFile(
                "quality-profile-ci-gate/manifest-verification-report/quality-profile-ci-gate-manifest-verification.receipt.json"
            )
            val manifestJson = requiredFile("quality-profile-ci-gate/quality-profile-ci-gate-manifest.json")
            val manifestRoot = outputRoot.resolve("quality-profile-ci-gate")

            listOf(
                "\"format\":\"gollek.training.quality-profile.ci-gate.manifest.verification.v1\"",
                "\"passed\":true",
                "\"readyForRelease\":true",
                "\"failureCount\":0",
                "\"artifactsMatch\":true",
                "\"markdownMatchesJson\":true",
                "\"junitXmlContract\":",
                "\"contractValid\":true",
                "\"testCountValid\":true",
                "\"propertiesValid\":true",
                "\"declaredTestCount\":6",
                "\"expectedTestCount\":6",
                "\"observedTestcaseCount\":6",
                "\"propertyCount\":17",
                "\"manifestStatus\":\"passed\"",
            ).forEach { marker -> requireContains(verificationJson, marker) }
            listOf(
                "**Status:** `PASS`",
                "**Ready for release:** `true`",
                "**Failures:** `0`",
                "## JUnit XML Contract",
                "**Contract valid:** `true`",
                "**Expected tests:** `6`",
                "**Observed testcases:** `6`",
                "**Property count:** `17`",
                "**Manifest status:** `passed`",
            ).forEach { marker -> requireContains(verificationMarkdown, marker) }
            listOf(
                "failures=\"0\"",
                "errors=\"0\"",
                "name=\"gollek.training.quality-profile.ci-gate.manifest\"",
                "name=\"manifest.readyForRelease\" value=\"true\"",
            ).forEach { marker -> requireContains(verificationJunitXml, marker) }
            listOf(
                "\"format\":\"gollek.training.quality-profile.ci-gate.manifest.verification.receipt.v1\"",
                "\"passed\":true",
                "\"reportVerified\":true",
                "\"jsonMatchesVerification\":true",
                "\"junitXmlContractValid\":true",
                "\"verificationSummary\":",
                "\"artifactFingerprints\":",
                "\"jsonBytes\":${verificationJson.length()}",
                "\"markdownBytes\":${verificationMarkdown.length()}",
                "\"junitXmlBytes\":${verificationJunitXml.length()}",
                "\"json\":{\"",
                "\"markdown\":{\"",
                "\"junitXml\":{\"",
                "\"file\":\"${verificationJson.absolutePath}\"",
                "\"file\":\"${verificationMarkdown.absolutePath}\"",
                "\"file\":\"${verificationJunitXml.absolutePath}\"",
                "\"bytes\":${verificationJson.length()}",
                "\"bytes\":${verificationMarkdown.length()}",
                "\"bytes\":${verificationJunitXml.length()}",
                "\"testcaseCount\":6",
                "\"propertyCount\":17",
                "\"manifestStatus\":\"passed\"",
            ).forEach { marker -> requireContains(verificationReceipt, marker) }
            requireNotContains(verificationReceipt, "\"verification\":{")
            listOf(
                "\"format\":\"gollek.training.quality-profile.ci-gate.manifest.v1\"",
                "\"passed\":true",
                "\"promotionPassed\":true",
                "\"validationPassed\":true",
            ).forEach { marker -> requireContains(manifestJson, marker) }

            val manifestText = manifestJson.readText()
            val artifactFiles = Regex("\"file\":\"([^\"]+)\"")
                .findAll(manifestText)
                .map { it.groupValues[1] }
                .toList()
            if (artifactFiles.size != 14) {
                throw GradleException(
                    "Expected quality-profile CI manifest to reference 14 artifacts but found ${artifactFiles.size}"
                )
            }
            artifactFiles.forEach { relativePath ->
                requireArtifactExists(manifestRoot, relativePath, "Quality-profile CI manifest")
            }

            val junitDocument = secureXmlDocument(verificationJunitXml)
            val testSuite = junitDocument.documentElement
            requireXmlTagName(testSuite, "testsuite")
            requireXmlAttribute(testSuite, "name", "gollek.training.quality-profile.ci-gate.manifest")
            requireXmlAttribute(testSuite, "tests", "6")
            requireXmlAttribute(testSuite, "failures", "0")
            requireXmlAttribute(testSuite, "errors", "0")
            requireXmlAttribute(testSuite, "skipped", "0")
            requireXmlElementCount(testSuite, "testcase", 6, "JUnit testcase")

            val properties = testSuite.getElementsByTagName("property")
            val propertyValues = (0 until properties.length)
                .map { properties.item(it) as Element }
                .associate { element -> element.getAttribute("name") to element.getAttribute("value") }
            val requiredProperties = mapOf(
                "manifest.passed" to "true",
                "manifest.readyForRelease" to "true",
                "manifest.structureValid" to "true",
                "manifest.artifactsMatch" to "true",
                "manifest.markdownMatchesJson" to "true",
                "manifest.failureCount" to "0",
                "manifest.failedCategoryCount" to "0",
            )
            requiredProperties.forEach { (name, expectedValue) ->
                val actualValue = propertyValues[name]
                if (actualValue != expectedValue) {
                    throw GradleException(
                        "Expected JUnit XML property $name=$expectedValue but got $actualValue"
                    )
                }
            }

            val summaryEntries = linkedMapOf(
                "status" to "verified",
                "gatePassed" to "true",
                "readyForRelease" to "true",
                "failureCount" to "0",
                "profile" to jbangTrainerQualityProfile.get(),
                "baselineReport" to outputRoot.resolve("runs/baseline/canonical-report.json").absolutePath,
                "candidateReport" to outputRoot.resolve("runs/candidate/canonical-report.json").absolutePath,
                "manifestJson" to manifestJson.absolutePath,
                "verificationJson" to verificationJson.absolutePath,
                "verificationMarkdown" to verificationMarkdown.absolutePath,
                "verificationJunitXml" to verificationJunitXml.absolutePath,
                "verificationReceipt" to verificationReceipt.absolutePath,
                "nextAction" to "Upload the manifest, verification report, receipt, and this summary as CI release evidence.",
            )
            val summaryFile = qualityProfileEvidenceSummaryFile.get().asFile
            writeKeyValueSummary(summaryFile, summaryEntries)
            verifyKeyValueSummary(summaryFile, summaryEntries, "Quality-profile CI gate")
        }
    }

val smokeJbangTrainerRuntimeProfileBudgetGate =
    tasks.register<Exec>("smokeJbangTrainerRuntimeProfileBudgetGate") {
        group = "verification"
        description = "Runs the trainer runtime profile budget gate JBang example end-to-end."
        dependsOn(publishJbangTrainerExamplesToMavenLocal)

        inputs.file(layout.projectDirectory.file("examples/jbang/trainer/trainer_runtime_profile_budget_gate.java"))
        outputs.dir(runtimeProfileBudgetGateOutputDir)
        configureTrainerJbangSmoke(
            "trainer/trainer_runtime_profile_budget_gate.java",
            runtimeProfileBudgetGateOutputDir.get().asFile,
            "--out",
            runtimeProfileBudgetGateOutputDir.get().asFile.absolutePath,
            "--policy",
            "strict",
        )
    }

val verifyJbangTrainerRuntimeProfileBudgetGateOutput =
    tasks.register("verifyJbangTrainerRuntimeProfileBudgetGateOutput") {
        group = "verification"
        description = "Verifies the files produced by the trainer runtime profile budget gate JBang smoke."
        dependsOn(smokeJbangTrainerRuntimeProfileBudgetGate)

        inputs.dir(runtimeProfileBudgetGateOutputDir)
        outputs.file(runtimeProfileBudgetGateSummaryFile)

        doLast {
            val outputRoot = runtimeProfileBudgetGateOutputDir.get().asFile
            val label = "Runtime profile budget gate"
            fun requiredFile(relativePath: String) = requiredNonEmptyFile(outputRoot, relativePath, label)
            fun requireContains(file: File, marker: String) = requireFileContains(file, marker, label)

            runtimeProfileBudgetGateRequiredFiles.forEach(::requiredFile)

            val reportJson = requiredFile("canonical-runtime-profile-report.json")
            val gateJson = requiredFile("runtime-profile-budget-gate/runtime-profile-budget-gate.json")
            val gateMarkdown = requiredFile("runtime-profile-budget-gate/runtime-profile-budget-gate.md")
            val gateJunitXml = requiredFile("runtime-profile-budget-gate/runtime-profile-budget-gate.junit.xml")
            val inputGateJson = requiredFile("runtime-input-profile-gate/runtime-input-profile-gate.json")
            val inputGateMarkdown = requiredFile("runtime-input-profile-gate/runtime-input-profile-gate.md")
            val inputGateJunitXml = requiredFile("runtime-input-profile-gate/runtime-input-profile-gate.junit.xml")

            listOf(
                "\"schema\":\"gollek.training.report.v1\"",
                "\"runtimeProfile.primaryGroup.name\":\"input\"",
                "\"runtimeProfile.primaryHotspot.phase\":\"input.train.next\"",
                "\"runtimeProfile.input.train.next.totalMillis\":320.0",
                "\"trainLoaderPlan.prefetch.enabled\":false",
            ).forEach { marker -> requireContains(reportJson, marker) }
            listOf(
                "\"available\":true",
                "\"passed\":false",
                "\"findingCount\":3",
                "\"runtime-profile-primary-group-budget\"",
                "\"runtime-profile-primary-hotspot-percent-budget\"",
                "\"runtime-profile-primary-hotspot-millis-budget\"",
                "\"maxPrimaryGroupPercent\":70.0",
                "\"maxPrimaryHotspotPercent\":45.0",
                "\"maxPrimaryHotspotTotalMillis\":250.0",
                "\"input.train.next\"",
            ).forEach { marker -> requireContains(gateJson, marker) }
            listOf(
                "# Runtime Profile Budget Gate",
                "- Available: `true`",
                "- Passed: `false`",
                "- Findings: `3`",
                "| Primary group | `70.000%` |",
                "| Primary hotspot | `45.000%` |",
                "| Primary hotspot total | `250.000 ms` |",
                "`runtime-profile-primary-hotspot-millis-budget`",
            ).forEach { marker -> requireContains(gateMarkdown, marker) }
            listOf(
                "name=\"gollek.training.runtime.profile\"",
                "tests=\"1\"",
                "failures=\"1\"",
                "errors=\"0\"",
                "name=\"gate.passed\" value=\"false\"",
                "name=\"gate.findingCount\" value=\"3\"",
            ).forEach { marker -> requireContains(gateJunitXml, marker) }
            listOf(
                "\"available\":true",
                "\"passed\":false",
                "\"findingCount\":4",
                "\"runtime-input-dominant-scope\"",
                "\"runtime-input-dominant-stage\"",
                "\"runtime-input-train-validation-skew\"",
                "\"runtime-input-train-prefetch-disabled\"",
                "\"recommendedPrefetchBufferSize\":2",
                "\"trainLoaderPlan.prefetch.enabled\":false",
                "\"trainLoaderPlan.prefetch.summary\":\"prefetch[enabled=false]\"",
                "\"maxDominantScopePercent\":80.0",
                "\"maxDominantStagePercent\":70.0",
                "\"maxTrainToValidationTotalRatio\":5.0",
            ).forEach { marker -> requireContains(inputGateJson, marker) }
            listOf(
                "# Runtime Input Profile Gate",
                "- Passed: `false`",
                "- Findings: `4`",
                "`runtime-input-train-prefetch-disabled`",
                "Wrap the train loader with `DataLoader.prefetch(2)`",
            ).forEach { marker -> requireContains(inputGateMarkdown, marker) }
            listOf(
                "name=\"gollek.training.runtime.input\"",
                "tests=\"1\"",
                "failures=\"1\"",
                "name=\"gate.findingCodes\" value=\"runtime-input-dominant-scope,runtime-input-dominant-stage,runtime-input-train-validation-skew,runtime-input-train-prefetch-disabled\"",
            ).forEach { marker -> requireContains(inputGateJunitXml, marker) }

            val junitDocument = secureXmlDocument(gateJunitXml)
            val testSuite = junitDocument.documentElement
            requireXmlTagName(testSuite, "testsuite")
            requireXmlAttribute(testSuite, "name", "gollek.training.runtime.profile")
            requireXmlAttribute(testSuite, "tests", "1")
            requireXmlAttribute(testSuite, "failures", "1")
            requireXmlAttribute(testSuite, "errors", "0")
            requireXmlElementCount(testSuite, "testcase", 1, "runtime budget JUnit testcase")
            val inputJunitDocument = secureXmlDocument(inputGateJunitXml)
            val inputTestSuite = inputJunitDocument.documentElement
            requireXmlTagName(inputTestSuite, "testsuite")
            requireXmlAttribute(inputTestSuite, "name", "gollek.training.runtime.input")
            requireXmlAttribute(inputTestSuite, "tests", "1")
            requireXmlAttribute(inputTestSuite, "failures", "1")
            requireXmlAttribute(inputTestSuite, "errors", "0")
            requireXmlElementCount(inputTestSuite, "testcase", 1, "runtime input JUnit testcase")

            val summaryEntries = linkedMapOf(
                "status" to "verified",
                "gatePassed" to "false",
                "findingCount" to "3",
                "inputGatePassed" to "false",
                "inputGateFindingCount" to "4",
                "primaryHotspot" to "input.train.next",
                "reportFile" to reportJson.absolutePath,
                "gateJson" to gateJson.absolutePath,
                "gateMarkdown" to gateMarkdown.absolutePath,
                "gateJunitXml" to gateJunitXml.absolutePath,
                "inputGateJson" to inputGateJson.absolutePath,
                "inputGateMarkdown" to inputGateMarkdown.absolutePath,
                "inputGateJunitXml" to inputGateJunitXml.absolutePath,
                "nextAction" to "Inspect runtime-profile-budget-gate.md and prioritize DataLoader next() time before trainer compute rewrites.",
            )
            val summaryFile = runtimeProfileBudgetGateSummaryFile.get().asFile
            writeKeyValueSummary(summaryFile, summaryEntries)
            verifyKeyValueSummary(summaryFile, summaryEntries, "Runtime profile budget gate")
        }
    }

val verifyJbangTrainerEvidenceIndex =
    tasks.register("verifyJbangTrainerEvidenceIndex") {
        group = "verification"
        description = "Builds and verifies a compact aggregate index for trainer JBang CI evidence."
        dependsOn(
            verifyJbangTrainerQualityProfileCiGateEvidenceOutput,
            verifyJbangTrainerRuntimeProfileBudgetGateOutput,
        )

        inputs.file(qualityProfileEvidenceSummaryFile)
        inputs.file(runtimeProfileBudgetGateSummaryFile)
        outputs.file(trainerEvidenceIndexJsonFile)
        outputs.file(trainerEvidenceIndexSummaryFile)

        doLast {
            val qualitySummaryFile = qualityProfileEvidenceSummaryFile.get().asFile
            val runtimeSummaryFile = runtimeProfileBudgetGateSummaryFile.get().asFile
            val quality = readKeyValueSummary(qualitySummaryFile, "Quality-profile CI gate")
            val runtime = readKeyValueSummary(runtimeSummaryFile, "Runtime profile budget gate")

            requireSummaryEntry(quality, "status", "verified", "Quality-profile CI gate")
            requireSummaryEntry(quality, "gatePassed", "true", "Quality-profile CI gate")
            requireSummaryEntry(quality, "readyForRelease", "true", "Quality-profile CI gate")
            requireSummaryEntry(runtime, "status", "verified", "Runtime profile budget gate")
            requireSummaryEntry(runtime, "gatePassed", "false", "Runtime profile budget gate")
            requireSummaryEntry(runtime, "findingCount", "3", "Runtime profile budget gate")
            requireSummaryEntry(runtime, "inputGatePassed", "false", "Runtime profile budget gate")
            requireSummaryEntry(runtime, "inputGateFindingCount", "4", "Runtime profile budget gate")

            val qualityArtifacts = listOf(
                "baselineReport" to quality.getValue("baselineReport"),
                "candidateReport" to quality.getValue("candidateReport"),
                "manifestJson" to quality.getValue("manifestJson"),
                "verificationJson" to quality.getValue("verificationJson"),
                "verificationMarkdown" to quality.getValue("verificationMarkdown"),
                "verificationJunitXml" to quality.getValue("verificationJunitXml"),
                "verificationReceipt" to quality.getValue("verificationReceipt"),
            )
            val runtimeArtifacts = listOf(
                "reportFile" to runtime.getValue("reportFile"),
                "gateJson" to runtime.getValue("gateJson"),
                "gateMarkdown" to runtime.getValue("gateMarkdown"),
                "gateJunitXml" to runtime.getValue("gateJunitXml"),
                "inputGateJson" to runtime.getValue("inputGateJson"),
                "inputGateMarkdown" to runtime.getValue("inputGateMarkdown"),
                "inputGateJunitXml" to runtime.getValue("inputGateJunitXml"),
            )
            (qualityArtifacts + runtimeArtifacts).forEach { (name, path) ->
                val artifact = File(path)
                if (!artifact.isFile || artifact.length() <= 0L) {
                    throw GradleException("Trainer evidence index references missing or empty $name artifact: $path")
                }
            }

            fun artifactJson(name: String, path: String) =
                "      {\"name\":${jsonString(name)},\"file\":${jsonString(path)},\"bytes\":${File(path).length()}}"

            val indexJson = buildString {
                appendLine("{")
                appendLine("  \"format\":\"gollek.training.jbang.evidence-index.v1\",")
                appendLine("  \"status\":\"verified\",")
                appendLine("  \"readyForRelease\":true,")
                appendLine("  \"lanes\":[")
                appendLine("    {")
                appendLine("      \"name\":\"quality-profile-ci-gate\",")
                appendLine("      \"status\":${jsonString(quality.getValue("status"))},")
                appendLine("      \"gatePassed\":true,")
                appendLine("      \"readyForRelease\":true,")
                appendLine("      \"profile\":${jsonString(quality.getValue("profile"))},")
                appendLine("      \"summary\":${jsonString(qualitySummaryFile.absolutePath)},")
                appendLine("      \"artifacts\":[")
                appendLine(qualityArtifacts.joinToString(",\n") { (name, path) -> artifactJson(name, path) })
                appendLine("      ]")
                appendLine("    },")
                appendLine("    {")
                appendLine("      \"name\":\"runtime-profile-budget-gate\",")
                appendLine("      \"status\":${jsonString(runtime.getValue("status"))},")
                appendLine("      \"gatePassed\":false,")
                appendLine("      \"findingCount\":${runtime.getValue("findingCount")},")
                appendLine("      \"inputGatePassed\":false,")
                appendLine("      \"inputGateFindingCount\":${runtime.getValue("inputGateFindingCount")},")
                appendLine("      \"primaryHotspot\":${jsonString(runtime.getValue("primaryHotspot"))},")
                appendLine("      \"summary\":${jsonString(runtimeSummaryFile.absolutePath)},")
                appendLine("      \"artifacts\":[")
                appendLine(runtimeArtifacts.joinToString(",\n") { (name, path) -> artifactJson(name, path) })
                appendLine("      ]")
                appendLine("    }")
                appendLine("  ],")
                appendLine("  \"nextAction\":\"Upload this index plus all referenced artifacts as trainer CI evidence.\"")
                appendLine("}")
            }

            val indexFile = trainerEvidenceIndexJsonFile.get().asFile
            indexFile.parentFile.mkdirs()
            indexFile.writeText(indexJson)
            if (indexFile.length() <= 0L) {
                throw GradleException("Trainer evidence index JSON is empty: ${indexFile.absolutePath}")
            }

            listOf(
                "\"format\":\"gollek.training.jbang.evidence-index.v1\"",
                "\"name\":\"quality-profile-ci-gate\"",
                "\"name\":\"runtime-profile-budget-gate\"",
                "\"readyForRelease\":true",
                "\"primaryHotspot\":\"input.train.next\"",
            ).forEach { marker -> requireFileContains(indexFile, marker, "Trainer evidence index") }

            val summaryEntries = linkedMapOf(
                "status" to "verified",
                "readyForRelease" to "true",
                "laneCount" to "2",
                "qualityGatePassed" to "true",
                "runtimeGatePassed" to "false",
                "runtimeFindingCount" to runtime.getValue("findingCount"),
                "runtimeInputGatePassed" to "false",
                "runtimeInputGateFindingCount" to runtime.getValue("inputGateFindingCount"),
                "indexJson" to indexFile.absolutePath,
                "qualitySummary" to qualitySummaryFile.absolutePath,
                "runtimeSummary" to runtimeSummaryFile.absolutePath,
                "nextAction" to "Upload trainer-evidence-index.json and referenced artifacts as CI release evidence.",
            )
            val summaryFile = trainerEvidenceIndexSummaryFile.get().asFile
            writeKeyValueSummary(summaryFile, summaryEntries)
            verifyKeyValueSummary(summaryFile, summaryEntries, "Trainer evidence index")
        }
    }

val smokeJbangWayangGollekServingBridge =
    tasks.register("smokeJbangWayangGollekServingBridge") {
        group = "verification"
        description = "Runs the Wayang-Gollek serving bridge JBang example in mock mode."
        dependsOn(publishJbangAgentBridgeExamplesToMavenLocal)

        inputs.file(layout.projectDirectory.file("examples/jbang/integration/wayang_gollek_serving_bridge.java"))
        inputs.dir(layout.projectDirectory.dir("sdk/gollek-sdk-agent/src/main/java"))
        inputs.file(layout.projectDirectory.file("sdk/gollek-sdk-agent/build.gradle.kts"))
        outputs.file(wayangGollekBridgeOutputFile)
        outputs.upToDateWhen { false }

        doLast {
            val jbangHome = jbangTrainerHomeDir.get().asFile
            jbangHome.mkdirs()

            val outputFile = wayangGollekBridgeOutputFile.get().asFile
            outputFile.parentFile.mkdirs()

            val command = mutableListOf(
                "jbang",
                "run",
                "--no-integrations",
                "--fresh",
            )
            if (jbangAgentBridgeOffline.get()) {
                command += "--offline"
            }
            command += listOf(
                "integration/wayang_gollek_serving_bridge.java",
                "--mock",
            )

            val process = ProcessBuilder(command)
                .directory(layout.projectDirectory.dir("examples/jbang").asFile)
                .redirectErrorStream(true)
                .apply {
                    environment()["JBANG_DIR"] = jbangHome.absolutePath
                }
                .start()
            val text = process.inputStream.use { stream ->
                String(stream.readBytes(), Charsets.UTF_8)
            }
            val exitValue = process.waitFor()

            outputFile.writeText(text)
            if (exitValue != 0) {
                throw GradleException(
                    "Wayang-Gollek bridge JBang smoke failed with exit $exitValue. "
                        + "See ${outputFile.absolutePath}."
                )
            }

            fun requireMarker(marker: String) {
                if (!text.contains(marker)) {
                    throw GradleException(
                        "Wayang-Gollek bridge smoke output is missing marker '$marker'. "
                            + "See ${outputFile.absolutePath}."
                    )
                }
            }

            listOf(
                "== Wayang-Gollek on Gollek",
                "service_role: inference_serving_engine",
                "tool_names: [search_context]",
                "embedding_dimensions: 6",
                "stream_delta: Gollek can answer",
                "stream_tool_preview: [search_context]",
                "finish_reason: tool_calls",
                "wayang_executes: search_context through Wayang policy, credentials, and audit log",
                "wayang-gollek: planning, tool execution, RAG store, approvals, memory, workflow state, follow-up loops",
            ).forEach(::requireMarker)
        }
    }

tasks.register("smokeJbangAgentBridgeExamples") {
    group = "verification"
    description = "Runs focused smoke checks for agent bridge JBang examples."
    dependsOn(smokeJbangWayangGollekServingBridge)
}

tasks.register("smokeJbangTrainerExamples") {
    group = "verification"
    description = "Runs focused smoke checks for trainer JBang examples."
    dependsOn(verifyJbangTrainerEvidenceIndex)
}

tasks.register("smokeJbangExamples") {
    group = "verification"
    description = "Runs focused smoke checks for JBang trainer and agent bridge examples."
    dependsOn("smokeJbangTrainerExamples", "smokeJbangAgentBridgeExamples")
}
