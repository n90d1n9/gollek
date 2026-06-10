import tech.kayys.gollek.buildlogic.DirectFixtureCoverageGuards
import tech.kayys.gollek.buildlogic.DirectFixtureCoverageReportSmoke
import tech.kayys.gollek.buildlogic.DirectFixtureCoverageReportSpec
import tech.kayys.gollek.buildlogic.DirectFixtureCoverageReportWriter
import tech.kayys.gollek.buildlogic.DirectFixtureCoverageScanner
import tech.kayys.gollek.buildlogic.DirectFixtureCoverageTaskMessages
import org.gradle.api.tasks.PathSensitivity

val validateModelFamilyDirectFixtures = tasks.register("validateModelFamilyDirectFixtures") {
    group = "verification"
    description = "Checks direct-safetensor model-family modules have fixture coverage."

    val modelModulesDir = layout.projectDirectory.dir("models")
    val reportFile = layout.buildDirectory.file("reports/model-family-direct-fixtures.txt")
    val jsonReportFile = layout.buildDirectory.file("reports/model-family-direct-fixtures.json")
    val jsonSchemaFile = layout.buildDirectory.file(
        "reports/model-family-direct-fixtures.v${DirectFixtureCoverageReportSpec.schemaVersion}.schema.json"
    )
    inputs.files(fileTree(modelModulesDir) {
        include(DirectFixtureCoverageReportSpec.fixtureInputIncludes)
    })
        .withPropertyName("directFixtureInputs")
        .withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.property("directFixtureInputIncludes", DirectFixtureCoverageReportSpec.fixtureInputIncludes)
    inputs.property("directFixtureModuleNamePrefix", DirectFixtureCoverageReportSpec.moduleNamePrefix)
    inputs.property("directFixtureMainJavaSourcePath", DirectFixtureCoverageReportSpec.mainJavaSourcePath)
    inputs.property("directFixtureTestJavaSourcePath", DirectFixtureCoverageReportSpec.testJavaSourcePath)
    inputs.property("directFixtureRootPath", DirectFixtureCoverageReportSpec.fixtureRootPath)
    inputs.property("directFixtureSchemaVersion", DirectFixtureCoverageReportSpec.schemaVersion)
    inputs.property("directFixtureMarker", DirectFixtureCoverageReportSpec.directInferenceMarker)
    inputs.property("directFixtureValidatorCall", DirectFixtureCoverageReportSpec.fixtureValidatorCall)
    inputs.property("directFixtureTokenizerMarkers", DirectFixtureCoverageReportSpec.tokenizerMarkers)
    inputs.property("directFixtureRootFields", DirectFixtureCoverageReportSpec.rootFields)
    inputs.property("directFixtureModuleFields", DirectFixtureCoverageReportSpec.moduleFields)
    inputs.property("directFixtureDetailFields", DirectFixtureCoverageReportSpec.detailFields)
    outputs.file(reportFile).withPropertyName("directFixtureTextReport")
    outputs.file(jsonReportFile).withPropertyName("directFixtureJsonReport")
    outputs.file(jsonSchemaFile).withPropertyName("directFixtureJsonSchema")

    doLast {
        val logTimings = providers.gradleProperty("directFixtureScanTimings")
            .map { it.toBoolean() }
            .getOrElse(false)
        val startedAt = System.nanoTime()
        val scan = DirectFixtureCoverageScanner.scan(modelModulesDir.asFile)
        val scannedAt = System.nanoTime()
        DirectFixtureCoverageGuards.requireProblemsClear(scan.problems)
        val fixtureCount = DirectFixtureCoverageGuards.requireReportEntries(
            directModuleCount = scan.directModuleCount,
            uniqueFixtureCount = scan.uniqueFixtureCount,
            reportEntries = scan.reportEntries,
        )
        val guardedAt = System.nanoTime()
        val writtenReports = DirectFixtureCoverageReportWriter.writeReports(
            directModuleCount = scan.directModuleCount,
            fixtureCount = fixtureCount,
            reportEntries = scan.reportEntries,
            textReportFile = reportFile.get().asFile,
            jsonReportFile = jsonReportFile.get().asFile,
            schemaFile = jsonSchemaFile.get().asFile,
        )
        val reportedAt = System.nanoTime()
        logger.lifecycle(DirectFixtureCoverageTaskMessages.validationSummary(scan, writtenReports, projectDir))
        if (logTimings) {
            logger.lifecycle(
                DirectFixtureCoverageTaskMessages.timingSummary(
                    scan = scan,
                    startedAt = startedAt,
                    scannedAt = scannedAt,
                    guardedAt = guardedAt,
                    reportedAt = reportedAt,
                )
            )
        }
    }
}

val verifyModelFamilyDirectFixtureReportContract = tasks.register("verifyModelFamilyDirectFixtureReportContract") {
    group = "verification"
    description = "Smoke-tests the direct model-family fixture report/schema contract helpers."

    doLast {
        DirectFixtureCoverageReportSmoke.requireSmokeContract()
    }
}

tasks.named("check") {
    dependsOn(verifyModelFamilyDirectFixtureReportContract)
    dependsOn(validateModelFamilyDirectFixtures)
}
