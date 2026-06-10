package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.longValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.pathValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.stringListValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateReceiptValues.stringValue;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Shared stale-field comparisons for promotion-gate receipt revalidation.
 */
final class TrainingReportPromotionGateReceiptStaleness {
    private TrainingReportPromotionGateReceiptStaleness() {
    }

    static void verifyPassStatus(
            Path receiptFile,
            String owner,
            boolean recorded,
            boolean actual,
            List<String> failures) {
        if (recorded != actual) {
            failures.add(owner + " pass status is stale for " + receiptFile
                    + ": receipt says " + recorded
                    + " but revalidation says " + actual);
        }
    }

    static void verifyBoolean(
            Path receiptFile,
            Map<String, ?> values,
            String owner,
            String key,
            boolean actual,
            List<String> failures) {
        Optional<Boolean> recorded = booleanValue(values, key);
        if (recorded.isPresent() && recorded.get().booleanValue() != actual) {
            failures.add(owner + " " + key + " is stale for " + receiptFile
                    + ": receipt says " + recorded.get()
                    + " but revalidation says " + actual);
        }
    }

    static void verifyString(
            Path receiptFile,
            Map<String, ?> values,
            String owner,
            String key,
            String actual,
            List<String> failures) {
        Optional<String> recorded = stringValue(values, key);
        if (recorded.isPresent() && !recorded.get().equalsIgnoreCase(actual)) {
            failures.add(owner + " " + key + " is stale for " + receiptFile
                    + ": receipt says " + recorded.get()
                    + " but revalidation says " + actual);
        }
    }

    static void verifyPath(
            Path receiptFile,
            Map<String, ?> values,
            String owner,
            String key,
            Path actual,
            List<String> failures) {
        Optional<Path> recorded = pathValue(values, key);
        Path normalizedActual = actual == null ? null : actual.toAbsolutePath().normalize();
        if (recorded.isPresent() && !recorded.get().equals(normalizedActual)) {
            failures.add(owner + " " + key + " is stale for " + receiptFile
                    + ": receipt says " + recorded.get()
                    + " but revalidation says " + normalizedActual);
        }
    }

    static void verifyNumber(
            Path receiptFile,
            Map<String, ?> values,
            String owner,
            String key,
            long actual,
            List<String> failures) {
        Optional<Long> recorded = longValue(values, key);
        if (recorded.isPresent() && recorded.get().longValue() != actual) {
            failures.add(owner + " " + key + " is stale for " + receiptFile
                    + ": receipt says " + recorded.get()
                    + " but revalidation says " + actual);
        }
    }

    static void verifyFailures(
            Path receiptFile,
            Map<String, ?> values,
            String owner,
            List<String> actual,
            List<String> failures) {
        Optional<List<String>> recorded = stringListValue(values, "failures", owner, failures);
        if (recorded.isPresent() && !recorded.orElseThrow().equals(actual)) {
            failures.add(owner + " failures are stale for " + receiptFile
                    + ": receipt says " + recorded.orElseThrow()
                    + " but revalidation says " + actual);
        }
    }
}
