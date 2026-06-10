package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * Semantic contract checks for quality-profile CI manifest JUnit XML.
 */
public final class TrainingReportQualityProfileCiGateManifestJUnitXmlContract {
    private TrainingReportQualityProfileCiGateManifestJUnitXmlContract() {
    }

    public record Counts(
            int expectedTestCount,
            int declaredTestCount,
            int observedTestcaseCount,
            int expectedFailureCount,
            int declaredFailureCount,
            int declaredErrorCount,
            int declaredSkippedCount,
            int propertyCount) {
        static Counts unavailable(int expectedTestCount, int expectedFailureCount) {
            return new Counts(
                    expectedTestCount,
                    -1,
                    0,
                    expectedFailureCount,
                    -1,
                    -1,
                    -1,
                    0);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("declaredErrorCount", declaredErrorCount);
            map.put("declaredFailureCount", declaredFailureCount);
            map.put("declaredSkippedCount", declaredSkippedCount);
            map.put("declaredTestCount", declaredTestCount);
            map.put("expectedFailureCount", expectedFailureCount);
            map.put("expectedTestCount", expectedTestCount);
            map.put("observedTestcaseCount", observedTestcaseCount);
            map.put("propertyCount", propertyCount);
            return Map.copyOf(map);
        }
    }

    public record Inspection(
            boolean wellFormed,
            boolean suiteNameValid,
            boolean testCountValid,
            boolean failureCountValid,
            boolean errorCountValid,
            boolean skippedCountValid,
            boolean testcaseNamesValid,
            boolean testcaseFailuresValid,
            boolean propertiesValid,
            Counts counts,
            List<String> testcaseNames,
            Map<String, String> properties,
            List<String> failures) {
        public Inspection {
            counts = counts == null ? Counts.unavailable(0, 0) : counts;
            testcaseNames = testcaseNames == null ? List.of() : List.copyOf(testcaseNames);
            properties = properties == null ? Map.of() : Map.copyOf(properties);
            failures = failures == null ? List.of() : List.copyOf(failures);
        }

        public boolean contractValid() {
            return failures.isEmpty()
                    && wellFormed
                    && suiteNameValid
                    && testCountValid
                    && failureCountValid
                    && errorCountValid
                    && skippedCountValid
                    && testcaseNamesValid
                    && testcaseFailuresValid
                    && propertiesValid;
        }

        public int testcaseCount() {
            return counts.observedTestcaseCount();
        }

        public int propertyCount() {
            return counts.propertyCount();
        }

        public String manifestStatus() {
            return properties.getOrDefault("manifest.status", "");
        }

        public String manifestReadyForRelease() {
            return properties.getOrDefault("manifest.readyForRelease", "");
        }

        public String manifestFailureCount() {
            return properties.getOrDefault("manifest.failureCount", "");
        }

        public String manifestFailedCategoryCount() {
            return properties.getOrDefault("manifest.failedCategoryCount", "");
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("contractValid", contractValid());
            map.put("counts", counts.toMap());
            map.put("wellFormed", wellFormed);
            map.put("suiteNameValid", suiteNameValid);
            map.put("testCountValid", testCountValid);
            map.put("failureCountValid", failureCountValid);
            map.put("errorCountValid", errorCountValid);
            map.put("skippedCountValid", skippedCountValid);
            map.put("testcaseNamesValid", testcaseNamesValid);
            map.put("testcaseFailuresValid", testcaseFailuresValid);
            map.put("propertiesValid", propertiesValid);
            map.put("testcaseCount", testcaseCount());
            map.put("propertyCount", propertyCount());
            map.put("manifestStatus", manifestStatus());
            map.put("manifestReadyForRelease", manifestReadyForRelease());
            map.put("manifestFailureCount", manifestFailureCount());
            map.put("manifestFailedCategoryCount", manifestFailedCategoryCount());
            map.put("testcaseNames", testcaseNames);
            map.put("properties", properties);
            map.put("failures", failures);
            return Map.copyOf(map);
        }
    }

    public static Inspection inspect(
            String junitXml,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        String resolvedJunitXml = Objects.requireNonNull(junitXml, "junitXml must not be null");
        TrainingReportQualityProfileCiGateManifest.ManifestVerification resolvedVerification =
                Objects.requireNonNull(verification, "verification must not be null");

        Map<String, Boolean> expectedChecks =
                TrainingReportQualityProfileCiGateManifestJUnitXml.checkOutcomes(resolvedVerification);
        List<String> expectedNames = new ArrayList<>(expectedChecks.keySet());
        int expectedFailedChecks = (int) expectedChecks.values().stream()
                .filter(passed -> !passed)
                .count();

        Document document = TrainingReportXml.parseDocument(resolvedJunitXml);
        if (document == null) {
            return new Inspection(
                    false,
                    false,
                    false,
                    false,
                    false,
                    false,
                    false,
                    false,
                    false,
                    Counts.unavailable(expectedNames.size(), expectedFailedChecks),
                    List.of(),
                    Map.of(),
                    List.of("JUnit XML is not well-formed."));
        }

        List<String> failures = new ArrayList<>();
        Element suite = document.getDocumentElement();
        boolean suiteTagValid = suite != null && "testsuite".equals(suite.getTagName());
        if (!suiteTagValid) {
            String actualTag = suite == null ? "<missing>" : suite.getTagName();
            failures.add("Expected JUnit XML root element `testsuite` but got `" + actualTag + "`.");
        }

        String suiteName = suiteTagValid ? suite.getAttribute("name") : "";
        boolean suiteNameValid = TrainingReportQualityProfileCiGateManifestJUnitXml.SUITE_NAME.equals(suiteName);
        if (!suiteNameValid) {
            failures.add("Expected JUnit XML suite name `"
                    + TrainingReportQualityProfileCiGateManifestJUnitXml.SUITE_NAME
                    + "` but got `" + suiteName + "`.");
        }

        int tests = intAttribute(suite, "tests", failures);
        int failed = intAttribute(suite, "failures", failures);
        int errors = intAttribute(suite, "errors", failures);
        int skipped = intAttribute(suite, "skipped", failures);

        List<Element> testcases = childElements(suite, "testcase");
        List<String> testcaseNames = testcases.stream()
                .map(testcase -> testcase.getAttribute("name"))
                .toList();
        Map<String, Boolean> testcaseFailureElements = testcaseFailureElements(testcases);
        Map<String, String> properties = properties(suite);
        Counts counts = new Counts(
                expectedNames.size(),
                tests,
                testcases.size(),
                expectedFailedChecks,
                failed,
                errors,
                skipped,
                properties.size());

        boolean testCountValid = tests == expectedNames.size() && testcases.size() == expectedNames.size();
        if (!testCountValid) {
            failures.add("Expected " + expectedNames.size()
                    + " JUnit XML testcases but got attribute tests=" + tests
                    + " and testcase elements=" + testcases.size() + ".");
        }

        boolean failureCountValid = failed == expectedFailedChecks;
        if (!failureCountValid) {
            failures.add("Expected JUnit XML failures=" + expectedFailedChecks + " but got " + failed + ".");
        }

        boolean errorCountValid = errors == 0;
        if (!errorCountValid) {
            failures.add("Expected JUnit XML errors=0 but got " + errors + ".");
        }

        boolean skippedCountValid = skipped == 0;
        if (!skippedCountValid) {
            failures.add("Expected JUnit XML skipped=0 but got " + skipped + ".");
        }

        boolean testcaseNamesValid = expectedNames.equals(testcaseNames);
        if (!testcaseNamesValid) {
            failures.add("Expected JUnit XML testcase names " + expectedNames + " but got " + testcaseNames + ".");
        }

        boolean testcaseFailuresValid = testcaseFailuresMatch(expectedChecks, testcaseFailureElements, failures);
        boolean propertiesValid = propertiesMatch(resolvedVerification, properties, failures);

        return new Inspection(
                true,
                suiteNameValid,
                testCountValid,
                failureCountValid,
                errorCountValid,
                skippedCountValid,
                testcaseNamesValid,
                testcaseFailuresValid,
                propertiesValid,
                counts,
                testcaseNames,
                properties,
                failures);
    }

    private static boolean testcaseFailuresMatch(
            Map<String, Boolean> expectedChecks,
            Map<String, Boolean> testcaseFailureElements,
            List<String> failures) {
        boolean valid = true;
        for (Map.Entry<String, Boolean> entry : expectedChecks.entrySet()) {
            boolean expectedFailureElement = !entry.getValue();
            boolean actualFailureElement = testcaseFailureElements.getOrDefault(entry.getKey(), false);
            if (actualFailureElement != expectedFailureElement) {
                valid = false;
                failures.add("Expected JUnit XML testcase `" + entry.getKey()
                        + "` failure element=" + expectedFailureElement
                        + " but got " + actualFailureElement + ".");
            }
        }
        return valid;
    }

    private static boolean propertiesMatch(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification,
            Map<String, String> properties,
            List<String> failures) {
        Map<String, String> expectedProperties = expectedProperties(verification);
        boolean valid = true;
        for (Map.Entry<String, String> entry : expectedProperties.entrySet()) {
            String actualValue = properties.get(entry.getKey());
            if (!entry.getValue().equals(actualValue)) {
                valid = false;
                failures.add("Expected JUnit XML property `" + entry.getKey()
                        + "` to be `" + entry.getValue() + "` but got `" + actualValue + "`.");
            }
        }
        if (verification.summary().primaryFailureCategory().isEmpty()
                && properties.containsKey("manifest.primaryFailureCategory")) {
            valid = false;
            failures.add("JUnit XML property `manifest.primaryFailureCategory` is present for a passing manifest.");
        }
        return valid;
    }

    private static Map<String, String> expectedProperties(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        TrainingReportQualityProfileCiGateManifestSummary summary = verification.summary();
        Map<String, String> expected = new LinkedHashMap<>();
        expected.put("manifest.passed", Boolean.toString(verification.passed()));
        expected.put("manifest.status", summary.status());
        expected.put("manifest.readyForRelease", Boolean.toString(summary.readyForRelease()));
        expected.put("manifest.structureValid", Boolean.toString(verification.structureValid()));
        expected.put("manifest.artifactsMatch", Boolean.toString(verification.artifactsMatch()));
        expected.put("manifest.markdownMatchesJson", Boolean.toString(verification.markdownMatchesJson()));
        expected.put("manifest.formatValid", Boolean.toString(verification.formatValid()));
        expected.put("manifest.profileKnown", Boolean.toString(verification.profileKnown()));
        expected.put("manifest.failureCount", Integer.toString(summary.failureCount()));
        expected.put("manifest.failedCategoryCount", Integer.toString(summary.failedCategories().size()));
        summary.primaryFailureCategory()
                .ifPresent(value -> expected.put("manifest.primaryFailureCategory", value));
        Map<TrainingReportQualityProfileCiGateManifestFailureCategory, Integer> counts =
                summary.failureCountsByCategory();
        for (TrainingReportQualityProfileCiGateManifestFailureCategory category
                : TrainingReportQualityProfileCiGateManifestFailureCategory.values()) {
            expected.put("manifest.failures." + category.id(), Integer.toString(counts.getOrDefault(category, 0)));
        }
        return expected;
    }

    private static Map<String, Boolean> testcaseFailureElements(List<Element> testcases) {
        Map<String, Boolean> failureElements = new LinkedHashMap<>();
        for (Element testcase : testcases) {
            failureElements.put(testcase.getAttribute("name"), hasDirectChild(testcase, "failure"));
        }
        return Map.copyOf(failureElements);
    }

    private static Map<String, String> properties(Element suite) {
        Map<String, String> properties = new LinkedHashMap<>();
        if (suite == null) {
            return Map.of();
        }
        for (Element property : childElements(firstChildElement(suite, "properties"), "property")) {
            properties.put(property.getAttribute("name"), property.getAttribute("value"));
        }
        return Map.copyOf(properties);
    }

    private static List<Element> childElements(Element parent, String tagName) {
        if (parent == null) {
            return List.of();
        }
        List<Element> elements = new ArrayList<>();
        NodeList children = parent.getChildNodes();
        for (int index = 0; index < children.getLength(); index++) {
            Node child = children.item(index);
            if (child instanceof Element element && tagName.equals(element.getTagName())) {
                elements.add(element);
            }
        }
        return List.copyOf(elements);
    }

    private static Element firstChildElement(Element parent, String tagName) {
        if (parent == null) {
            return null;
        }
        NodeList children = parent.getChildNodes();
        for (int index = 0; index < children.getLength(); index++) {
            Node child = children.item(index);
            if (child instanceof Element element && tagName.equals(element.getTagName())) {
                return element;
            }
        }
        return null;
    }

    private static boolean hasDirectChild(Element parent, String tagName) {
        return firstChildElement(parent, tagName) != null;
    }

    private static int intAttribute(Element element, String attributeName, List<String> failures) {
        if (element == null || !element.hasAttribute(attributeName)) {
            failures.add("Missing JUnit XML attribute `" + attributeName + "`.");
            return Integer.MIN_VALUE;
        }
        try {
            return Integer.parseInt(element.getAttribute(attributeName));
        } catch (NumberFormatException error) {
            failures.add("Invalid JUnit XML integer attribute `" + attributeName
                    + "`: `" + element.getAttribute(attributeName) + "`.");
            return Integer.MIN_VALUE;
        }
    }
}
