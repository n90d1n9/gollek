package tech.kayys.gollek.train.data.multimodal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.Dataset;

class MultimodalDataTest {
    @TempDir
    Path tempDir;

    @Test
    void multimodalDatasetDefensivelyCopiesSamples() {
        List<MultimodalContent> first = new ArrayList<>(List.of(
                MultimodalContent.ofText("describe"),
                MultimodalContent.ofImageUri("file:///tmp/cat.png", "image/png")));
        MultimodalDataset dataset = new MultimodalDataset(new ArrayList<>(List.of(first)));
        first.clear();

        assertEquals(1, dataset.size());
        assertEquals(2, dataset.get(0).size());
        assertThrows(UnsupportedOperationException.class, () -> dataset.get(0).add(MultimodalContent.ofText("x")));
        assertThrows(IllegalArgumentException.class, () -> new MultimodalDataset(List.of(List.of())));
    }

    @Test
    void multimodalCollatorSummarizesAndRequiresModalities() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                List.of(
                        MultimodalContent.ofText("caption one"),
                        MultimodalContent.ofImageUri("file:///tmp/one.png", "image/png")),
                List.of(
                        MultimodalContent.ofText("caption two"),
                        MultimodalContent.ofImageUri("file:///tmp/two.png", "image/png"))));

        DataLoader.CollatingDataLoader<List<MultimodalContent>, MultimodalBatch> loader =
                DataLoader.builder(dataset)
                        .batchSize(2)
                        .collate(MultimodalCollators.requireModalities(ModalityType.TEXT, ModalityType.IMAGE));

        MultimodalBatch batch = loader.iterator().next();
        assertEquals(2, batch.sampleCount());
        assertEquals(4, batch.partCount());
        assertEquals(2, batch.modalityCounts().get(ModalityType.TEXT));
        assertEquals(2, batch.modalityCounts().get(ModalityType.IMAGE));
        assertTrue(batch.sampleContains(0, ModalityType.TEXT));

        DataLoader.CollatingDataLoader<List<MultimodalContent>, MultimodalBatch> invalid =
                DataLoader.builder(new MultimodalDataset(List.of(List.of(MultimodalContent.ofText("text only")))))
                        .collate(MultimodalCollators.requireModalities(ModalityType.TEXT, ModalityType.IMAGE));
        assertThrows(IllegalArgumentException.class, () -> invalid.iterator().next());
    }

    @Test
    void imageTextCollatorStacksImagesAndKeepsTexts() {
        List<ImageTextDataset.Sample> samples = List.of(
                new ImageTextDataset.Sample(GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 2), " first "),
                new ImageTextDataset.Sample(GradTensor.of(new float[] {5f, 6f, 7f, 8f}, 1, 2, 2), "second"));

        MultimodalCollators.ImageTextBatch batch =
                MultimodalCollators.imageTextBatch().apply(samples);

        assertEquals(2, batch.size());
        assertArrayEquals(new long[] {2, 1, 2, 2}, batch.images().shape());
        assertArrayEquals(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, batch.images().data(), 1e-6f);
        assertEquals(List.of("first", "second"), batch.texts());
        assertThrows(UnsupportedOperationException.class, () -> batch.texts().add("third"));
        assertThrows(IllegalArgumentException.class, () -> MultimodalCollators.imageTextBatch()
                .apply(List.of(new ImageTextDataset.Sample(GradTensor.of(new float[] {1f}, 1), " "))));
    }

    @Test
    void audioTextDatasetPairsSidecarsDeterministicallyAndSkipsUnpaired() throws Exception {
        Path nested = Files.createDirectories(tempDir.resolve("nested"));
        Files.write(tempDir.resolve("z.wav"), new byte[] {9, 8});
        Files.writeString(tempDir.resolve("z.txt"), " last ");
        Files.write(tempDir.resolve("a.mp3"), new byte[] {1, 2, 3});
        Files.writeString(tempDir.resolve("a.txt"), "first");
        Files.write(nested.resolve("m.flac"), new byte[] {4, 5, 6, 7});
        Files.writeString(nested.resolve("m.txt"), "middle");
        Files.write(tempDir.resolve("unpaired.wav"), new byte[] {0});
        Files.writeString(tempDir.resolve("orphan.txt"), "not paired with supported media");

        AudioTextDataset dataset = new AudioTextDataset(tempDir);

        assertEquals(3, dataset.size());
        assertEquals(tempDir.resolve("a.mp3").toAbsolutePath().normalize(), dataset.getAudioPath(0));
        assertEquals(nested.resolve("m.flac").toAbsolutePath().normalize(), dataset.getAudioPath(1));
        assertEquals(tempDir.resolve("z.wav").toAbsolutePath().normalize(), dataset.getAudioPath(2));
        assertEquals("first", dataset.get(0).text());
        assertEquals("audio/mpeg", dataset.get(0).mimeType());
        assertArrayEquals(new byte[] {4, 5, 6, 7}, dataset.get(1).audio());
        assertThrows(UnsupportedOperationException.class, () -> dataset.audioPaths().add(tempDir.resolve("x.wav")));
    }

    @Test
    void audioTextCollatorConcatenatesBytesWithOffsetsAndLengths() {
        List<AudioTextDataset.Sample> samples = List.of(
                new AudioTextDataset.Sample(new byte[] {1, 2, 3}, " hello ", "audio/wav", tempDir.resolve("a.wav")),
                new AudioTextDataset.Sample(new byte[] {4, 5}, "world", "audio/flac", tempDir.resolve("b.flac")));

        MultimodalCollators.AudioTextBatch batch =
                MultimodalCollators.audioTextBatch().apply(samples);

        assertEquals(2, batch.size());
        assertEquals(5, batch.totalBytes());
        assertArrayEquals(new byte[] {1, 2, 3, 4, 5}, batch.audioBytes());
        assertArrayEquals(new int[] {0, 3}, batch.offsets());
        assertArrayEquals(new int[] {3, 2}, batch.lengths());
        assertArrayEquals(new byte[] {4, 5}, batch.sampleAudio(1));
        assertEquals(List.of("hello", "world"), batch.texts());
        assertEquals(List.of("audio/wav", "audio/flac"), batch.mimeTypes());
        assertThrows(UnsupportedOperationException.class, () -> batch.texts().add("third"));
        assertThrows(IllegalArgumentException.class, () -> MultimodalCollators.audioTextBatch()
                .apply(List.of(new AudioTextDataset.Sample(new byte[] {1}, " ", "audio/wav", tempDir.resolve("x.wav")))));
    }

    @Test
    void manifestDatasetLoadsShorthandRowsWithSafeRelativeAssetUris() throws Exception {
        Path images = Files.createDirectories(tempDir.resolve("assets/images"));
        Path audio = Files.createDirectories(tempDir.resolve("assets/audio"));
        Path imagePath = images.resolve("cat.png");
        Path audioPath = audio.resolve("clip.wav");
        Files.write(imagePath, new byte[] {1, 2, 3});
        Files.write(audioPath, new byte[] {4, 5});
        Path manifest = tempDir.resolve("manifest.jsonl");
        Files.writeString(manifest, """
                {"text":"Describe the cat","image":"assets/images/cat.png","metadata":{"split":"train"}}
                {"caption":"Transcribe this","audio":{"path":"assets/audio/clip.wav","mimeType":"audio/custom"}}
                """);

        MultimodalManifestDataset dataset = new MultimodalManifestDataset(manifest);

        assertEquals(2, dataset.size());
        assertEquals(List.of(ModalityType.TEXT, ModalityType.IMAGE), dataset.get(0).stream()
                .map(MultimodalContent::getModality)
                .toList());
        MultimodalContent image = dataset.get(0).get(1);
        assertEquals(imagePath.toUri().toString(), image.getUri());
        assertEquals("image/png", image.getMimeType());
        assertEquals("train", image.getMetadata().get("split"));
        assertEquals(imagePath.toAbsolutePath().normalize().toString(), image.getMetadata().get("sourcePath"));
        assertEquals(audioPath.toUri().toString(), dataset.get(1).get(1).getUri());
        assertEquals("audio/custom", dataset.get(1).get(1).getMimeType());
        assertThrows(UnsupportedOperationException.class, () -> dataset.get(0).add(MultimodalContent.ofText("x")));
    }

    @Test
    void textAssetBatchProjectsManifestSamplesForTrainerLoops() throws Exception {
        Path images = Files.createDirectories(tempDir.resolve("images"));
        Path cat = images.resolve("cat.png");
        Path dog = images.resolve("dog.png");
        Files.write(cat, new byte[] {1, 2, 3});
        Files.write(dog, new byte[] {4, 5, 6});
        Path manifest = tempDir.resolve("vision.jsonl");
        Files.writeString(manifest, """
                {"text":" cat caption ","image":"images/cat.png"}
                {"caption":"dog caption","image":"images/dog.png"}
                """);

        MultimodalManifestDataset dataset = new MultimodalManifestDataset(manifest);
        DataLoader.CollatingDataLoader<List<MultimodalContent>, TextAssetBatch> loader =
                DataLoader.builder(dataset)
                        .batchSize(2)
                        .collate(MultimodalCollators.textAssetBatch(ModalityType.IMAGE));

        TextAssetBatch batch = loader.iterator().next();

        assertEquals(2, batch.size());
        assertEquals(ModalityType.IMAGE, batch.assetModality());
        assertEquals(List.of("cat caption", "dog caption"), batch.texts());
        assertEquals("cat caption", batch.text(0));
        assertEquals(cat.toUri().toString(), batch.assetUri(0));
        assertEquals("image/png", batch.assetMimeType(0));
        assertEquals(List.of("image/png", "image/png"), batch.assetMimeTypes());
        assertEquals(cat.toAbsolutePath().normalize().toString(), batch.assetSourcePath(0));
        assertEquals(dog.toAbsolutePath().normalize().toString(), batch.assetSourcePaths().get(1));
        assertThrows(UnsupportedOperationException.class, () -> batch.texts().add("third"));
        assertThrows(UnsupportedOperationException.class, () -> batch.assets().clear());
        assertThrows(UnsupportedOperationException.class, () -> batch.assetUris().add("file:///x.png"));
    }

    @Test
    void textAssetBatchRejectsInvalidProjectionRequests() {
        List<MultimodalContent> blankText = List.of(
                MultimodalContent.ofText(" "),
                MultimodalContent.ofImageUri("file:///tmp/cat.png", "image/png"));
        List<MultimodalContent> textOnly = List.of(MultimodalContent.ofText("caption"));

        assertThrows(IllegalArgumentException.class, () -> MultimodalCollators.textAssetBatch(ModalityType.TEXT));
        assertThrows(IllegalArgumentException.class, () -> MultimodalCollators.textAssetBatch(ModalityType.IMAGE)
                .apply(List.of(blankText)));
        assertThrows(IllegalArgumentException.class, () -> MultimodalCollators.textAssetBatch(ModalityType.IMAGE)
                .apply(List.of(textOnly)));
    }

    @Test
    void multimodalDiagnosticsReportsManifestHealthAndReadiness() throws Exception {
        Path images = Files.createDirectories(tempDir.resolve("images"));
        Path cat = images.resolve("cat.png");
        Files.write(cat, new byte[] {1, 2, 3});
        Path manifest = tempDir.resolve("diagnostics.jsonl");
        Files.writeString(manifest, """
                {"text":"caption one","image":"images/cat.png"}
                {"caption":"caption two","image":"images/cat.png"}
                """);

        MultimodalManifestDataset dataset = new MultimodalManifestDataset(manifest);
        MultimodalDatasetReport report = MultimodalDatasetDiagnostics.inspect(dataset);

        assertEquals(2, report.sampleCount());
        assertEquals(4, report.partCount());
        assertEquals(2, report.partCount(ModalityType.TEXT));
        assertEquals(2, report.partCount(ModalityType.IMAGE));
        assertEquals(2, report.sampleCount(ModalityType.IMAGE));
        assertEquals(2, report.sampleSignatureCounts().get("IMAGE+TEXT"));
        assertEquals(2, report.mimeTypeCounts().get("image/png"));
        assertEquals(2, report.remoteAssetCount());
        assertEquals(0, report.inlinedAssetCount());
        assertEquals(0, report.unresolvedAssetCount());
        assertEquals(2, report.sourcePathCount());
        assertEquals(List.of(cat.toAbsolutePath().normalize().toString()), report.duplicateSourcePaths());
        assertTrue(report.isReadyForTextAssetTraining(ModalityType.IMAGE));
        assertFalse(report.isReadyForTextAssetTraining(ModalityType.AUDIO));
        assertThrows(UnsupportedOperationException.class, () -> report.duplicateSourcePaths().add("x"));
        assertThrows(UnsupportedOperationException.class, () -> report.mimeTypeCounts().put("image/jpeg", 1));
    }

    @Test
    void multimodalDiagnosticsFindsBlankTextMissingTextAndUnresolvedAssets() {
        MultimodalContent unresolvedImage = MultimodalContent.builder(ModalityType.IMAGE)
                .mimeType("image/png")
                .build();
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                List.of(MultimodalContent.ofText(" "), MultimodalContent.ofImageUri("file:///tmp/a.png", "image/png")),
                List.of(unresolvedImage),
                List.of(MultimodalContent.ofText("text only"))));

        MultimodalDatasetReport report = MultimodalDatasetDiagnostics.inspect(dataset);

        assertEquals(3, report.sampleCount());
        assertEquals(4, report.partCount());
        assertEquals(2, report.textPartCount());
        assertEquals(1, report.blankTextPartCount());
        assertEquals(1, report.unresolvedAssetCount());
        assertEquals(List.of(0), report.samplesWithBlankText());
        assertEquals(List.of(1), report.samplesWithoutText());
        assertEquals(List.of(2), report.samplesWithoutNonTextAsset());
        assertEquals(1, report.sampleSignatureCounts().get("IMAGE"));
        assertTrue(report.hasBlankText());
        assertTrue(report.hasUnresolvedAssets());
        assertFalse(report.isReadyForTextAssetTraining(ModalityType.IMAGE));
        assertThrows(IllegalArgumentException.class, () -> report.isReadyForTextAssetTraining(ModalityType.TEXT));
    }

    @Test
    void multimodalValidatorAllowsReadyTextAssetDatasetWithLeakageWarning() throws Exception {
        Path images = Files.createDirectories(tempDir.resolve("validated/images"));
        Path cat = images.resolve("cat.png");
        Files.write(cat, new byte[] {1, 2, 3});
        Path manifest = tempDir.resolve("validated/manifest.jsonl");
        Files.writeString(manifest, """
                {"text":"caption one","image":"images/cat.png"}
                {"text":"caption two","image":"images/cat.png"}
                """);

        MultimodalManifestDataset dataset = new MultimodalManifestDataset(manifest);
        MultimodalValidationResult result =
                MultimodalDatasetValidator.textAssetTraining(ModalityType.IMAGE).validate(dataset);

        assertTrue(result.isValid());
        assertFalse(result.hasErrors());
        assertTrue(result.hasWarnings());
        assertTrue(result.hasIssue(MultimodalDatasetValidator.CODE_DUPLICATE_SOURCE_PATH));
        assertEquals(1, result.warnings().size());
        assertEquals(List.of(0, 1), result.warnings().get(0).sampleIndices());
        assertEquals(List.of(cat.toAbsolutePath().normalize().toString()),
                result.warnings().get(0).details().get("sourcePaths"));
        assertDoesNotThrow(result::throwIfInvalid);
        assertThrows(UnsupportedOperationException.class, () -> result.issues().clear());
    }

    @Test
    void multimodalValidatorReportsTrainerBlockingIssues() {
        MultimodalContent unresolvedImage = MultimodalContent.builder(ModalityType.IMAGE)
                .mimeType("image/png")
                .build();
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                List.of(MultimodalContent.ofText(" "), MultimodalContent.ofImageUri("file:///tmp/a.png", "image/png")),
                List.of(MultimodalContent.ofText("text only")),
                List.of(unresolvedImage)));

        MultimodalValidationResult result =
                MultimodalDatasetValidator.textAssetTraining(ModalityType.IMAGE).validate(dataset);

        assertFalse(result.isValid());
        assertTrue(result.hasErrors());
        assertTrue(result.hasIssue(MultimodalDatasetValidator.CODE_BLANK_TEXT));
        assertTrue(result.hasIssue(MultimodalDatasetValidator.CODE_MISSING_REQUIRED_MODALITY));
        assertTrue(result.hasIssue(MultimodalDatasetValidator.CODE_UNRESOLVED_ASSET));
        assertEquals(List.of(0), result.issues(MultimodalDatasetValidator.CODE_BLANK_TEXT).get(0).sampleIndices());
        assertEquals(List.of(1), result.issues(MultimodalDatasetValidator.CODE_MISSING_REQUIRED_MODALITY).stream()
                .filter(issue -> "IMAGE".equals(issue.details().get("modality")))
                .findFirst()
                .orElseThrow()
                .sampleIndices());
        assertEquals(List.of(2), result.issues(MultimodalDatasetValidator.CODE_UNRESOLVED_ASSET).get(0).sampleIndices());
        IllegalStateException failure = assertThrows(IllegalStateException.class, result::throwIfInvalid);
        assertTrue(failure.getMessage().contains(MultimodalDatasetValidator.CODE_UNRESOLVED_ASSET));
    }

    @Test
    void multimodalValidatorReportsEmptyDatasetWithoutRawDiagnosticsFailure() {
        MultimodalValidationResult result = MultimodalDatasetValidator.builder()
                .minSamples(2)
                .build()
                .validate(List.of());

        assertFalse(result.isValid());
        assertEquals(0, result.report().sampleCount());
        assertTrue(result.hasIssue("dataset_too_small"));
        assertEquals(List.of(), result.issues(MultimodalDatasetValidator.CODE_DATASET_TOO_SMALL).get(0).sampleIndices());
        assertThrows(IllegalStateException.class, result::throwIfInvalid);
    }

    @Test
    void multimodalStratifiedSplitPreservesModalitySignatures() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("i0"), imageText("i1"), imageText("i2"), imageText("i3"),
                audioText("a0"), audioText("a1"), audioText("a2"), audioText("a3")));

        Dataset.Split<List<MultimodalContent>> split =
                MultimodalDatasetSplits.stratifiedBySignature(dataset, 0.5, 123L);

        assertEquals(4, split.train().size());
        assertEquals(4, split.validation().size());
        assertEquals(Map.of("IMAGE+TEXT", 2, "AUDIO+TEXT", 2), signatureCounts(split.train()));
        assertEquals(Map.of("IMAGE+TEXT", 2, "AUDIO+TEXT", 2), signatureCounts(split.validation()));
        assertThrows(IllegalArgumentException.class,
                () -> MultimodalDatasetSplits.stratifiedBySignature(dataset, 1.0, 1L));
    }

    @Test
    void multimodalSourceGroupedSplitPreventsDuplicateAssetLeakage() throws Exception {
        Path images = Files.createDirectories(tempDir.resolve("grouped/images"));
        Path cat = images.resolve("cat.png");
        Path dog = images.resolve("dog.png");
        Path bird = images.resolve("bird.png");
        Files.write(cat, new byte[] {1});
        Files.write(dog, new byte[] {2});
        Files.write(bird, new byte[] {3});
        Path manifest = tempDir.resolve("grouped/manifest.jsonl");
        Files.writeString(manifest, """
                {"text":"cat one","image":"images/cat.png"}
                {"text":"cat two","image":"images/cat.png"}
                {"text":"dog","image":"images/dog.png"}
                {"text":"bird","image":"images/bird.png"}
                """);
        MultimodalManifestDataset dataset = new MultimodalManifestDataset(manifest);

        Dataset.Split<List<MultimodalContent>> split =
                MultimodalDatasetSplits.groupedBySourcePath(dataset, 0.5, 777L);

        assertEquals(4, split.train().size() + split.validation().size());
        assertTrue(split.train().size() > 0);
        assertTrue(split.validation().size() > 0);
        assertEquals(0, MultimodalDatasetSplits.overlappingSourcePaths(split).size());
        assertThrows(IllegalArgumentException.class,
                () -> MultimodalDatasetSplits.groupedBySourcePath(
                        new MultimodalDataset(List.of(imageText("same"), imageText("same"))),
                        0.5,
                        1L));
    }

    @Test
    void multimodalStratifiedGroupedSplitBalancesSignaturesWithoutLeakage() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c"),
                audioText("clip-d"), audioText("clip-d")));

        Dataset.Split<List<MultimodalContent>> split =
                MultimodalDatasetSplits.stratifiedGroupedBySourcePath(dataset, 0.5, 888L);
        MultimodalSplitReport report = MultimodalSplitDiagnostics.inspect(split);

        assertEquals(8, split.train().size());
        assertEquals(8, split.validation().size());
        assertEquals(Map.of("IMAGE+TEXT", 4, "AUDIO+TEXT", 4), signatureCounts(split.train()));
        assertEquals(Map.of("IMAGE+TEXT", 4, "AUDIO+TEXT", 4), signatureCounts(split.validation()));
        assertFalse(report.hasSourceLeakage());
        assertEquals(0.0, report.maxSampleSignatureShareDelta(), 1e-9);
        assertDoesNotThrow(() -> report.throwIfInvalid(0.0, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> MultimodalDatasetSplits.stratifiedGroupedBySourcePath(
                        new MultimodalDataset(List.of(imageText("same"), imageText("same"))),
                        0.5,
                        1L));
    }

    @Test
    void multimodalStratifiedGroupedThreeWaySplitKeepsFinalTestLeakageFree() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c"),
                audioText("clip-d"), audioText("clip-d")));

        Dataset.ThreeWaySplit<List<MultimodalContent>> split =
                MultimodalDatasetSplits.stratifiedGroupedThreeWayBySourcePath(dataset, 0.5, 0.25, 999L);
        MultimodalThreeWaySplitReport report = MultimodalThreeWaySplitDiagnostics.inspect(split);

        assertEquals(8, split.train().size());
        assertEquals(4, split.validation().size());
        assertEquals(4, split.test().size());
        assertEquals(16, report.totalSampleCount());
        assertTrue(report.isLeakageFree());
        assertEquals(Set.of(), report.leakingSourcePaths());
        assertTrue(report.isSignatureBalanced(0.25));
        assertDoesNotThrow(() -> report.throwIfInvalid(0.25, 0.25));
        assertThrows(IllegalArgumentException.class,
                () -> MultimodalDatasetSplits.stratifiedGroupedThreeWayBySourcePath(
                        new MultimodalDataset(List.of(imageText("same"), imageText("same"), imageText("same"))),
                        0.5,
                        0.25,
                        1L));
    }

    @Test
    void multimodalSplitDiagnosticsReportsBalancedSplitWithoutLeakage() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("i0"), imageText("i1"), imageText("i2"), imageText("i3"),
                audioText("a0"), audioText("a1"), audioText("a2"), audioText("a3")));
        Dataset.Split<List<MultimodalContent>> split =
                MultimodalDatasetSplits.stratifiedBySignature(dataset, 0.5, 123L);

        MultimodalSplitReport report = MultimodalSplitDiagnostics.inspect(split);

        assertEquals(8, report.totalSampleCount());
        assertEquals(0.5, report.trainSampleFraction(), 1e-9);
        assertEquals(0.5, report.validationSampleFraction(), 1e-9);
        assertFalse(report.hasSourceLeakage());
        assertEquals(0.0, report.maxSampleSignatureShareDelta(), 1e-9);
        assertEquals(0.0, report.maxMimeTypeShareDelta(), 1e-9);
        assertTrue(report.isSignatureBalanced(0.0));
        assertTrue(report.isMimeTypeBalanced(0.0));
        assertDoesNotThrow(report::throwIfSourceLeakage);
        assertThrows(UnsupportedOperationException.class, () -> report.overlappingSourcePaths().add("x"));
        assertThrows(UnsupportedOperationException.class, () -> report.sampleSignatureShareDelta().put("x", 0.0));
    }

    @Test
    void multimodalSplitDiagnosticsDetectsLeakageAndDistributionDrift() {
        Dataset.Split<List<MultimodalContent>> split = new Dataset.Split<>(
                Dataset.from(List.of(imageText("shared"), imageText("train-only"))),
                Dataset.from(List.of(imageText("shared"), audioText("validation-only"))));

        MultimodalSplitReport report = MultimodalSplitDiagnostics.inspect(split);

        assertTrue(report.hasSourceLeakage());
        assertTrue(report.overlappingSourcePaths().contains("file:///tmp/shared.png"));
        assertEquals(0.5, report.maxSampleSignatureShareDelta(), 1e-9);
        assertEquals(0.5, report.maxMimeTypeShareDelta(), 1e-9);
        assertFalse(report.isSignatureBalanced(0.25));
        assertFalse(report.isMimeTypeBalanced(0.25));
        IllegalStateException failure = assertThrows(IllegalStateException.class, report::throwIfSourceLeakage);
        assertTrue(failure.getMessage().contains("file:///tmp/shared.png"));
    }

    @Test
    void multimodalStratifiedKFoldBySignatureKeepsValidationFoldsBalanced() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("i0"), imageText("i1"), imageText("i2"), imageText("i3"), imageText("i4"), imageText("i5"),
                audioText("a0"), audioText("a1"), audioText("a2"), audioText("a3"), audioText("a4"), audioText("a5")));

        List<Dataset.Fold<List<MultimodalContent>>> folds =
                MultimodalDatasetSplits.stratifiedKFoldBySignature(dataset, 3, 111L);

        assertEquals(3, folds.size());
        for (int index = 0; index < folds.size(); index++) {
            Dataset.Fold<List<MultimodalContent>> fold = folds.get(index);
            assertEquals(index, fold.foldIndex());
            assertEquals(3, fold.foldCount());
            assertEquals(8, fold.train().size());
            assertEquals(4, fold.validation().size());
            assertEquals(Map.of("IMAGE+TEXT", 2, "AUDIO+TEXT", 2), signatureCounts(fold.validation()));
            assertFalse(MultimodalSplitDiagnostics.inspect(splitOf(fold)).hasSourceLeakage());
        }
        assertThrows(IllegalArgumentException.class,
                () -> MultimodalDatasetSplits.stratifiedKFoldBySignature(
                        new MultimodalDataset(List.of(imageText("i0"), audioText("a0"))),
                        3,
                        1L));
    }

    @Test
    void multimodalGroupedKFoldBySourcePathPreventsLeakageInEveryFold() throws Exception {
        Path images = Files.createDirectories(tempDir.resolve("folds/images"));
        Path cat = images.resolve("cat.png");
        Path dog = images.resolve("dog.png");
        Path bird = images.resolve("bird.png");
        Files.write(cat, new byte[] {1});
        Files.write(dog, new byte[] {2});
        Files.write(bird, new byte[] {3});
        Path manifest = tempDir.resolve("folds/manifest.jsonl");
        Files.writeString(manifest, """
                {"text":"cat one","image":"images/cat.png"}
                {"text":"cat two","image":"images/cat.png"}
                {"text":"dog one","image":"images/dog.png"}
                {"text":"dog two","image":"images/dog.png"}
                {"text":"bird one","image":"images/bird.png"}
                {"text":"bird two","image":"images/bird.png"}
                """);
        MultimodalManifestDataset dataset = new MultimodalManifestDataset(manifest);

        List<Dataset.Fold<List<MultimodalContent>>> folds =
                MultimodalDatasetSplits.groupedKFoldBySourcePath(dataset, 3, 222L);

        assertEquals(3, folds.size());
        for (Dataset.Fold<List<MultimodalContent>> fold : folds) {
            assertEquals(4, fold.train().size());
            assertEquals(2, fold.validation().size());
            MultimodalSplitReport report = MultimodalSplitDiagnostics.inspect(splitOf(fold));
            assertFalse(report.hasSourceLeakage());
            assertDoesNotThrow(report::throwIfSourceLeakage);
        }
        assertThrows(IllegalArgumentException.class,
                () -> MultimodalDatasetSplits.groupedKFoldBySourcePath(
                        new MultimodalDataset(List.of(imageText("same"), imageText("same"))),
                        2,
                        1L));
    }

    @Test
    void multimodalStratifiedGroupedKFoldBySourcePathBalancesSignaturesWithoutLeakage() throws Exception {
        Path root = Files.createDirectories(tempDir.resolve("stratified-grouped"));
        Path images = Files.createDirectories(root.resolve("images"));
        Path audio = Files.createDirectories(root.resolve("audio"));
        for (String id : List.of("cat", "dog", "bird")) {
            Files.write(images.resolve(id + ".png"), new byte[] {1, 2, 3});
        }
        for (String id : List.of("clip-a", "clip-b", "clip-c")) {
            Files.write(audio.resolve(id + ".wav"), new byte[] {4, 5, 6});
        }
        Path manifest = root.resolve("manifest.jsonl");
        Files.writeString(manifest, """
                {"text":"cat one","image":"images/cat.png"}
                {"text":"cat two","image":"images/cat.png"}
                {"text":"dog one","image":"images/dog.png"}
                {"text":"dog two","image":"images/dog.png"}
                {"text":"bird one","image":"images/bird.png"}
                {"text":"bird two","image":"images/bird.png"}
                {"text":"clip a one","audio":"audio/clip-a.wav"}
                {"text":"clip a two","audio":"audio/clip-a.wav"}
                {"text":"clip b one","audio":"audio/clip-b.wav"}
                {"text":"clip b two","audio":"audio/clip-b.wav"}
                {"text":"clip c one","audio":"audio/clip-c.wav"}
                {"text":"clip c two","audio":"audio/clip-c.wav"}
                """);
        MultimodalManifestDataset dataset = new MultimodalManifestDataset(manifest);

        List<Dataset.Fold<List<MultimodalContent>>> folds =
                MultimodalDatasetSplits.stratifiedGroupedKFoldBySourcePath(dataset, 3, 333L);

        assertEquals(3, folds.size());
        for (Dataset.Fold<List<MultimodalContent>> fold : folds) {
            assertEquals(8, fold.train().size());
            assertEquals(4, fold.validation().size());
            assertEquals(Map.of("IMAGE+TEXT", 2, "AUDIO+TEXT", 2), signatureCounts(fold.validation()));
            MultimodalSplitReport report = MultimodalSplitDiagnostics.inspect(splitOf(fold));
            assertFalse(report.hasSourceLeakage());
            assertEquals(0.0, report.maxSampleSignatureShareDelta(), 1e-9);
        }
        assertThrows(IllegalArgumentException.class,
                () -> MultimodalDatasetSplits.stratifiedGroupedKFoldBySourcePath(
                        new MultimodalDataset(List.of(imageText("same"), imageText("same"))),
                        2,
                        1L));
    }

    @Test
    void multimodalCrossValidationDiagnosticsAuditsAllFolds() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("i0"), imageText("i1"), imageText("i2"), imageText("i3"), imageText("i4"), imageText("i5"),
                audioText("a0"), audioText("a1"), audioText("a2"), audioText("a3"), audioText("a4"), audioText("a5")));
        List<Dataset.Fold<List<MultimodalContent>>> folds =
                MultimodalDatasetSplits.stratifiedKFoldBySignature(dataset, 3, 444L);

        MultimodalCrossValidationReport report = MultimodalCrossValidationDiagnostics.inspect(folds);

        assertEquals(3, report.foldCount());
        assertEquals(12, report.totalValidationSamples());
        assertTrue(report.isLeakageFree());
        assertEquals(List.of(), report.foldsWithSourceLeakage());
        assertEquals(0.0, report.maxSampleSignatureShareDelta(), 1e-9);
        assertEquals(0.0, report.maxMimeTypeShareDelta(), 1e-9);
        assertTrue(report.isSignatureBalanced(0.0));
        assertTrue(report.isMimeTypeBalanced(0.0));
        assertDoesNotThrow(() -> report.throwIfInvalid(0.0, 0.0));
        assertThrows(UnsupportedOperationException.class, () -> report.foldReports().clear());
        assertThrows(IllegalArgumentException.class, () -> MultimodalCrossValidationDiagnostics.inspect(List.of()));
    }

    @Test
    void multimodalCrossValidationDiagnosticsReportsLeakageAndDrift() {
        List<Dataset.Fold<List<MultimodalContent>>> folds = List.of(new Dataset.Fold<>(
                0,
                1,
                Dataset.from(List.of(imageText("shared"), imageText("train-only"))),
                Dataset.from(List.of(imageText("shared"), audioText("validation-only")))));

        MultimodalCrossValidationReport report = MultimodalCrossValidationDiagnostics.inspect(folds);

        assertFalse(report.isLeakageFree());
        assertEquals(List.of(0), report.foldsWithSourceLeakage());
        assertTrue(report.leakingSourcePaths().contains("file:///tmp/shared.png"));
        assertEquals(0.5, report.maxSampleSignatureShareDelta(), 1e-9);
        assertEquals(0.5, report.maxMimeTypeShareDelta(), 1e-9);
        assertFalse(report.isSignatureBalanced(0.25));
        assertFalse(report.isMimeTypeBalanced(0.25));
        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> report.throwIfInvalid(0.25, 0.25));
        assertTrue(failure.getMessage().contains("source leakage"));
        assertTrue(failure.getMessage().contains("signature drift"));
    }

    @Test
    void multimodalTrainValidationPlannerBuildsReadyGroupedStratifiedPlan() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c"),
                audioText("clip-d"), audioText("clip-d")));

        MultimodalTrainValidationPlan plan = MultimodalTrainValidationPlanner.builder(dataset)
                .trainFraction(0.5)
                .seed(888L)
                .balanceTolerance(0.0)
                .build();

        assertEquals(MultimodalTrainValidationPlanner.SplitStrategy.STRATIFIED_GROUPED_BY_SOURCE_PATH,
                plan.strategy());
        assertEquals(8, plan.train().size());
        assertEquals(8, plan.validation().size());
        assertTrue(plan.validationResult().isValid());
        assertTrue(plan.validationResult().hasWarnings());
        assertTrue(plan.isReady());
        assertFalse(plan.splitReport().hasSourceLeakage());
        assertEquals(0.0, plan.splitReport().maxSampleSignatureShareDelta(), 1e-9);
        assertDoesNotThrow(plan::throwIfInvalid);
        assertTrue(plan.summary().contains("STRATIFIED_GROUPED_BY_SOURCE_PATH"));
    }

    @Test
    void multimodalTrainValidationPlanCreatesTrainerLoaders() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c"),
                audioText("clip-d"), audioText("clip-d")));
        MultimodalTrainValidationPlan plan = MultimodalTrainValidationPlanner.builder(dataset)
                .trainFraction(0.5)
                .seed(888L)
                .balanceTolerance(0.0)
                .build();

        DataLoader<List<MultimodalContent>> trainLoader = plan.trainLoader(4);
        DataLoader<List<MultimodalContent>> validationLoader = plan.validationLoader(4);
        DataLoader.CollatingDataLoader<List<MultimodalContent>, MultimodalBatch> trainBatches =
                plan.trainLoader(4, MultimodalCollators.batch());
        DataLoader.CollatingDataLoader<List<MultimodalContent>, MultimodalBatch> shuffledValidationBatches =
                plan.validationLoaderBuilder()
                        .batchSize(4)
                        .shuffle(123L)
                        .collate(MultimodalCollators.batch());

        assertEquals(8, trainLoader.sampleCount());
        assertEquals(8, validationLoader.sampleCount());
        assertEquals(2, trainLoader.numBatches());
        assertEquals(2, validationLoader.numBatches());
        assertEquals(4, trainBatches.iterator().next().sampleCount());
        assertEquals(4, shuffledValidationBatches.iterator().next().sampleCount());
        assertEquals(123L, shuffledValidationBatches.shuffleSeed().orElseThrow());
    }

    @Test
    void multimodalTrainValidationSplitManifestRoundTripsAndReplaysPlan() throws Exception {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c"),
                audioText("clip-d"), audioText("clip-d")));
        MultimodalTrainValidationPlan plan = MultimodalTrainValidationPlanner.builder(dataset)
                .trainFraction(0.5)
                .seed(888L)
                .balanceTolerance(0.0)
                .build();

        MultimodalTrainValidationManifest manifest = plan.splitManifest();
        Path manifestPath = tempDir.resolve("train-validation-manifest.json");
        plan.writeSplitManifest(manifestPath);
        MultimodalTrainValidationManifest reloaded = MultimodalTrainValidationManifest.read(manifestPath);
        Dataset.Split<List<MultimodalContent>> replayed = reloaded.applyTo(dataset);
        MultimodalTrainValidationPlan replayedPlan = MultimodalTrainValidationPlanner
                .replay(dataset, reloaded)
                .balanceTolerance(0.0)
                .build();
        MultimodalTrainValidationPlan pathReplayedPlan = MultimodalTrainValidationPlanner
                .replay(dataset, manifestPath)
                .balanceTolerance(0.0)
                .build();
        MultimodalManifestAuditReport audit = reloaded.audit(dataset);

        assertEquals(manifest, reloaded);
        assertEquals(16, reloaded.sampleCount());
        assertEquals(8, reloaded.trainIndices().size());
        assertEquals(8, reloaded.validationIndices().size());
        assertTrue(reloaded.matches(dataset));
        assertTrue(audit.matches());
        assertEquals(16, audit.checkedSampleCount());
        assertEquals(0, audit.sampleMismatches().size());
        assertTrue(reloaded.summary().contains("validation=8"));
        assertEquals(MultimodalTrainValidationPlanner.SplitStrategy.REPLAYED_MANIFEST, replayedPlan.strategy());
        assertTrue(replayedPlan.isReady());
        assertEquals(firstSourcePaths(plan.train()), firstSourcePaths(replayed.train()));
        assertEquals(firstSourcePaths(plan.validation()), firstSourcePaths(replayed.validation()));
        assertEquals(firstSourcePaths(plan.train()), firstSourcePaths(replayedPlan.train()));
        assertEquals(firstSourcePaths(plan.validation()), firstSourcePaths(replayedPlan.validation()));
        assertEquals(firstSourcePaths(plan.train()), firstSourcePaths(pathReplayedPlan.train()));
        assertEquals(firstSourcePaths(plan.validation()), firstSourcePaths(pathReplayedPlan.validation()));
        assertEquals(2, replayedPlan.validationLoader(4).numBatches());

        MultimodalDataset changed = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c"),
                audioText("clip-d"), audioText("clip-z")));
        MultimodalManifestAuditReport changedAudit = reloaded.audit(changed);
        assertFalse(reloaded.matches(changed));
        assertFalse(changedAudit.matches());
        assertFalse(changedAudit.hasSampleCountMismatch());
        assertEquals(1, changedAudit.sampleMismatches().size());
        assertEquals(15, changedAudit.firstMismatch().orElseThrow().index());
        assertTrue(changedAudit.firstMismatch().orElseThrow().contentChanged());
        assertTrue(changedAudit.firstMismatch().orElseThrow().sourcePathsChanged());
        assertTrue(changedAudit.summary().contains("fingerprintMismatches=1"));
        assertThrows(IllegalArgumentException.class, changedAudit::throwIfInvalid);
        assertThrows(IllegalArgumentException.class, () -> MultimodalTrainValidationPlanner
                .replay(changed, reloaded)
                .balanceTolerance(0.0)
                .build());

        MultimodalDataset shorter = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c"),
                audioText("clip-d")));
        MultimodalManifestAuditReport shorterAudit = reloaded.audit(shorter);
        assertFalse(shorterAudit.matches());
        assertTrue(shorterAudit.hasSampleCountMismatch());
        assertEquals(1, shorterAudit.missingSampleCount());
        assertEquals(0, shorterAudit.extraSampleCount());
    }

    @Test
    void multimodalTrainValidationTextAssetProfileCreatesTypedLoaders() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish")));

        MultimodalTrainValidationPlan plan = MultimodalTrainValidationPlanner
                .textAssetTraining(dataset, ModalityType.IMAGE)
                .trainFraction(0.5)
                .seed(888L)
                .balanceTolerance(0.0)
                .build();

        TextAssetBatch trainBatch = plan.trainTextAssetLoader(4, ModalityType.IMAGE).iterator().next();
        TextAssetBatch validationBatch = plan.validationTextAssetLoader(4, ModalityType.IMAGE).iterator().next();

        assertEquals(ModalityType.IMAGE, trainBatch.assetModality());
        assertEquals(ModalityType.IMAGE, validationBatch.assetModality());
        assertEquals(4, trainBatch.size());
        assertEquals(4, validationBatch.size());
        assertTrue(trainBatch.texts().stream().allMatch(text -> text.startsWith("caption ")));
        assertTrue(validationBatch.assetMimeTypes().stream().allMatch("image/png"::equals));
        assertThrows(IllegalArgumentException.class,
                () -> plan.trainTextAssetLoader(4, ModalityType.AUDIO).iterator().next());
        assertThrows(IllegalArgumentException.class,
                () -> MultimodalTrainValidationPlanner.textAssetTraining(dataset, ModalityType.TEXT));
    }

    @Test
    void multimodalTrainValidationTestPlannerBuildsReadyPlanWithFinalTestLoaders() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c"),
                audioText("clip-d"), audioText("clip-d")));

        MultimodalTrainValidationTestPlan plan = MultimodalTrainValidationTestPlanner.builder(dataset)
                .fractions(0.5, 0.25)
                .seed(999L)
                .balanceTolerance(0.25)
                .build();

        assertEquals(8, plan.train().size());
        assertEquals(4, plan.validation().size());
        assertEquals(4, plan.test().size());
        assertTrue(plan.isReady());
        assertEquals(2, plan.trainLoader(4).numBatches());
        assertEquals(1, plan.validationLoader(4, MultimodalCollators.batch()).numBatches());
        assertEquals(1, plan.testLoader(4).numBatches());
        assertTrue(plan.summary().contains("testSamples=4"));
        assertDoesNotThrow(plan::throwIfInvalid);
    }

    @Test
    void multimodalTrainValidationTestTextAssetProfileCreatesTypedTestLoader() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish")));

        MultimodalTrainValidationTestPlan plan = MultimodalTrainValidationTestPlanner
                .textAssetTraining(dataset, ModalityType.IMAGE)
                .fractions(0.5, 0.25)
                .seed(999L)
                .balanceTolerance(0.0)
                .build();

        TextAssetBatch testBatch = plan.testTextAssetLoader(2, ModalityType.IMAGE).iterator().next();

        assertEquals(2, testBatch.size());
        assertEquals(ModalityType.IMAGE, testBatch.assetModality());
        assertTrue(testBatch.assetMimeTypes().stream().allMatch("image/png"::equals));
        assertThrows(IllegalArgumentException.class,
                () -> plan.testTextAssetLoader(2, ModalityType.AUDIO).iterator().next());
        assertThrows(IllegalArgumentException.class,
                () -> MultimodalTrainValidationTestPlanner.textAssetTraining(dataset, ModalityType.TEXT));
    }

    @Test
    void multimodalTrainValidationTestSplitManifestRoundTripsAndReplaysMembership() throws Exception {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c"),
                audioText("clip-d"), audioText("clip-d")));
        MultimodalTrainValidationTestPlan plan = MultimodalTrainValidationTestPlanner.builder(dataset)
                .fractions(0.5, 0.25)
                .seed(999L)
                .balanceTolerance(0.25)
                .build();

        MultimodalSplitManifest manifest = plan.splitManifest();
        Path manifestPath = tempDir.resolve("split-manifest.json");
        plan.writeSplitManifest(manifestPath);
        MultimodalSplitManifest reloaded = MultimodalSplitManifest.read(manifestPath);
        Dataset.ThreeWaySplit<List<MultimodalContent>> replayed = reloaded.applyTo(dataset);
        MultimodalTrainValidationTestPlan replayedPlan = MultimodalTrainValidationTestPlanner
                .replay(dataset, reloaded)
                .balanceTolerance(0.25)
                .build();
        MultimodalTrainValidationTestPlan pathReplayedPlan = MultimodalTrainValidationTestPlanner
                .replay(dataset, manifestPath)
                .balanceTolerance(0.25)
                .build();

        assertEquals(manifest, reloaded);
        assertEquals(16, reloaded.sampleCount());
        assertEquals(8, reloaded.trainIndices().size());
        assertEquals(4, reloaded.validationIndices().size());
        assertEquals(4, reloaded.testIndices().size());
        assertTrue(reloaded.matches(dataset));
        assertTrue(reloaded.audit(dataset).matches());
        assertTrue(reloaded.summary().contains("test=4"));
        assertEquals(firstSourcePaths(plan.train()), firstSourcePaths(replayed.train()));
        assertEquals(firstSourcePaths(plan.validation()), firstSourcePaths(replayed.validation()));
        assertEquals(firstSourcePaths(plan.test()), firstSourcePaths(replayed.test()));
        assertTrue(replayedPlan.isReady());
        assertEquals(firstSourcePaths(plan.train()), firstSourcePaths(replayedPlan.train()));
        assertEquals(firstSourcePaths(plan.validation()), firstSourcePaths(replayedPlan.validation()));
        assertEquals(firstSourcePaths(plan.test()), firstSourcePaths(replayedPlan.test()));
        assertEquals(firstSourcePaths(plan.test()), firstSourcePaths(pathReplayedPlan.test()));
        assertEquals(1, replayedPlan.testLoader(4).numBatches());

        MultimodalDataset changed = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                imageText("fish"), imageText("fish"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c"),
                audioText("clip-d"), audioText("clip-z")));
        assertFalse(reloaded.matches(changed));
        assertEquals(1, reloaded.audit(changed).sampleMismatches().size());
        assertThrows(IllegalArgumentException.class, () -> reloaded.applyTo(changed));
    }

    @Test
    void multimodalTrainValidationPlannerCanExposeUnsafePlanForInspection() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("shared"), imageText("shared"), imageText("shared"), imageText("shared"),
                audioText("a0"), audioText("a1"), audioText("a2"), audioText("a3")));

        MultimodalTrainValidationPlan unsafe = MultimodalTrainValidationPlanner.builder(dataset)
                .strategy(MultimodalTrainValidationPlanner.SplitStrategy.STRATIFIED_BY_SIGNATURE)
                .trainFraction(0.5)
                .seed(7L)
                .failOnAuditFailure(false)
                .build();

        assertFalse(unsafe.isReady());
        assertTrue(unsafe.splitReport().hasSourceLeakage());
        assertTrue(unsafe.splitReport().overlappingSourcePaths().contains("file:///tmp/shared.png"));
        assertThrows(IllegalStateException.class, unsafe::throwIfInvalid);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> MultimodalTrainValidationPlanner.builder(dataset)
                        .strategy(MultimodalTrainValidationPlanner.SplitStrategy.STRATIFIED_BY_SIGNATURE)
                        .trainFraction(0.5)
                        .seed(7L)
                        .build());
        assertTrue(failure.getMessage().contains("multimodal split audit failed"));
    }

    @Test
    void multimodalCrossValidationPlannerBuildsReadyGroupedStratifiedPlan() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c")));

        MultimodalCrossValidationPlan plan = MultimodalCrossValidationPlanner.builder(dataset)
                .folds(3)
                .seed(333L)
                .balanceTolerance(0.0)
                .build();

        assertEquals(MultimodalCrossValidationPlanner.SplitStrategy.STRATIFIED_GROUPED_BY_SOURCE_PATH,
                plan.strategy());
        assertEquals(3, plan.foldCount());
        assertEquals(8, plan.train(0).size());
        assertEquals(4, plan.validation(0).size());
        assertTrue(plan.validationResult().isValid());
        assertTrue(plan.validationResult().hasWarnings());
        assertTrue(plan.isReady());
        assertDoesNotThrow(plan::throwIfInvalid);
        assertTrue(plan.summary().contains("STRATIFIED_GROUPED_BY_SOURCE_PATH"));
        assertThrows(UnsupportedOperationException.class, () -> plan.folds().clear());
        assertThrows(IndexOutOfBoundsException.class, () -> plan.fold(3));
        for (Dataset.Fold<List<MultimodalContent>> fold : plan.folds()) {
            assertFalse(MultimodalSplitDiagnostics.inspect(splitOf(fold)).hasSourceLeakage());
        }
    }

    @Test
    void multimodalCrossValidationPlanCreatesFoldLoaders() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c")));
        MultimodalCrossValidationPlan plan = MultimodalCrossValidationPlanner.builder(dataset)
                .folds(3)
                .seed(333L)
                .balanceTolerance(0.0)
                .build();

        DataLoader<List<MultimodalContent>> foldTrain = plan.trainLoader(0, 4);
        DataLoader<List<MultimodalContent>> foldValidation = plan.validationLoader(0, 4);
        DataLoader.CollatingDataLoader<List<MultimodalContent>, MultimodalBatch> validationBatches =
                plan.validationLoader(0, 4, MultimodalCollators.batch());
        DataLoader.CollatingDataLoader<List<MultimodalContent>, MultimodalBatch> shuffledTrainBatches =
                plan.trainLoaderBuilder(0)
                        .batchSize(4)
                        .shuffle(321L)
                        .collate(MultimodalCollators.batch());

        assertEquals(8, foldTrain.sampleCount());
        assertEquals(4, foldValidation.sampleCount());
        assertEquals(2, foldTrain.numBatches());
        assertEquals(1, foldValidation.numBatches());
        assertEquals(4, validationBatches.iterator().next().sampleCount());
        assertEquals(4, shuffledTrainBatches.iterator().next().sampleCount());
        assertEquals(321L, shuffledTrainBatches.shuffleSeed().orElseThrow());
        assertThrows(IndexOutOfBoundsException.class, () -> plan.trainLoader(3, 4));
    }

    @Test
    void multimodalCrossValidationTextAssetProfileCreatesFoldLoaders() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird")));

        MultimodalCrossValidationPlan plan = MultimodalCrossValidationPlanner
                .textAssetTraining(dataset, ModalityType.IMAGE)
                .folds(3)
                .seed(333L)
                .balanceTolerance(0.0)
                .build();

        DataLoader.CollatingDataLoader<List<MultimodalContent>, TextAssetBatch> trainLoader =
                plan.trainTextAssetLoader(0, 2, ModalityType.IMAGE);
        DataLoader.CollatingDataLoader<List<MultimodalContent>, TextAssetBatch> validationLoader =
                plan.validationTextAssetLoader(0, 2, ModalityType.IMAGE);
        TextAssetBatch trainBatch = trainLoader.iterator().next();
        TextAssetBatch validationBatch = validationLoader.iterator().next();

        assertEquals(4, trainLoader.sampleCount());
        assertEquals(2, validationLoader.sampleCount());
        assertEquals(2, trainBatch.size());
        assertEquals(2, validationBatch.size());
        assertEquals(ModalityType.IMAGE, validationBatch.assetModality());
        assertTrue(validationBatch.assetUris().stream().allMatch(uri -> uri.startsWith("file:///tmp/")));
        assertThrows(IllegalArgumentException.class,
                () -> plan.validationTextAssetLoader(0, 2, ModalityType.AUDIO).iterator().next());
        assertThrows(IndexOutOfBoundsException.class,
                () -> plan.trainTextAssetLoader(3, 2, ModalityType.IMAGE));
    }

    @Test
    void multimodalCrossValidationManifestRoundTripsAndReplaysFoldMembership() throws Exception {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-c")));
        MultimodalCrossValidationPlan plan = MultimodalCrossValidationPlanner.builder(dataset)
                .folds(3)
                .seed(333L)
                .balanceTolerance(0.0)
                .build();

        MultimodalCrossValidationManifest manifest = plan.foldManifest();
        Path manifestPath = tempDir.resolve("fold-manifest.json");
        plan.writeFoldManifest(manifestPath);
        MultimodalCrossValidationManifest reloaded = MultimodalCrossValidationManifest.read(manifestPath);
        List<Dataset.Fold<List<MultimodalContent>>> replayed = reloaded.applyTo(dataset);
        MultimodalCrossValidationPlan replayedPlan = MultimodalCrossValidationPlanner
                .replay(dataset, reloaded)
                .balanceTolerance(0.0)
                .build();
        MultimodalCrossValidationPlan pathReplayedPlan = MultimodalCrossValidationPlanner
                .replay(dataset, manifestPath)
                .balanceTolerance(0.0)
                .build();

        assertEquals(manifest, reloaded);
        assertEquals(12, reloaded.sampleCount());
        assertEquals(3, reloaded.foldCount());
        assertEquals(12, reloaded.totalValidationSamples());
        assertTrue(reloaded.matches(dataset));
        assertTrue(reloaded.audit(dataset).matches());
        assertTrue(reloaded.summary().contains("folds=3"));
        assertEquals(MultimodalCrossValidationPlanner.SplitStrategy.REPLAYED_MANIFEST, replayedPlan.strategy());
        assertTrue(replayedPlan.isReady());
        assertTrue(pathReplayedPlan.isReady());
        assertEquals(3, replayedPlan.foldCount());
        assertEquals(3, pathReplayedPlan.foldCount());
        assertEquals(1, replayedPlan.validationLoader(0, 4).numBatches());
        for (int foldIndex = 0; foldIndex < plan.foldCount(); foldIndex++) {
            assertEquals(firstSourcePaths(plan.train(foldIndex)), firstSourcePaths(replayed.get(foldIndex).train()));
            assertEquals(firstSourcePaths(plan.validation(foldIndex)),
                    firstSourcePaths(replayed.get(foldIndex).validation()));
            assertEquals(firstSourcePaths(plan.train(foldIndex)), firstSourcePaths(replayedPlan.train(foldIndex)));
            assertEquals(firstSourcePaths(plan.validation(foldIndex)),
                    firstSourcePaths(replayedPlan.validation(foldIndex)));
            assertEquals(firstSourcePaths(plan.validation(foldIndex)),
                    firstSourcePaths(pathReplayedPlan.validation(foldIndex)));
            assertFalse(MultimodalSplitDiagnostics.inspect(splitOf(replayed.get(foldIndex))).hasSourceLeakage());
        }

        MultimodalDataset changed = new MultimodalDataset(List.of(
                imageText("cat"), imageText("cat"),
                imageText("dog"), imageText("dog"),
                imageText("bird"), imageText("bird"),
                audioText("clip-a"), audioText("clip-a"),
                audioText("clip-b"), audioText("clip-b"),
                audioText("clip-c"), audioText("clip-z")));
        assertFalse(reloaded.matches(changed));
        assertEquals(1, reloaded.audit(changed).sampleMismatches().size());
        assertThrows(IllegalArgumentException.class, () -> reloaded.applyTo(changed));
    }

    @Test
    void multimodalCrossValidationPlannerCanExposeUnsafePlanForInspection() {
        MultimodalDataset dataset = new MultimodalDataset(List.of(
                imageText("shared"), imageText("shared"), imageText("shared"), imageText("image-only"),
                audioText("a0"), audioText("a1"), audioText("a2"), audioText("a3")));

        MultimodalCrossValidationPlan unsafe = MultimodalCrossValidationPlanner.builder(dataset)
                .strategy(MultimodalCrossValidationPlanner.SplitStrategy.STRATIFIED_BY_SIGNATURE)
                .folds(2)
                .seed(7L)
                .failOnAuditFailure(false)
                .build();

        assertFalse(unsafe.isReady());
        assertFalse(unsafe.crossValidationReport().isLeakageFree());
        assertTrue(unsafe.crossValidationReport().leakingSourcePaths().contains("file:///tmp/shared.png"));
        assertThrows(IllegalStateException.class, unsafe::throwIfInvalid);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> MultimodalCrossValidationPlanner.builder(dataset)
                        .strategy(MultimodalCrossValidationPlanner.SplitStrategy.STRATIFIED_BY_SIGNATURE)
                        .folds(2)
                        .seed(7L)
                        .build());
        assertTrue(failure.getMessage().contains("multimodal cross-validation audit failed"));
    }

    @Test
    void manifestDatasetLoadsPartsAndCanInlineBinaryAssets() throws Exception {
        Path docs = Files.createDirectories(tempDir.resolve("assets/docs"));
        Path docPath = docs.resolve("guide.pdf");
        byte[] docBytes = new byte[] {37, 80, 68, 70};
        Files.write(docPath, docBytes);
        Path manifest = tempDir.resolve("manifest.jsonl");
        Files.writeString(manifest, "{\"parts\":["
                + "{\"modality\":\"text\",\"text\":\"Read the guide\"},"
                + "{\"modality\":\"document\",\"path\":\"docs/guide.pdf\"},"
                + "{\"type\":\"embedding\",\"embedding\":[1.0,2.5,3.0]}"
                + "]}");

        MultimodalManifestDataset.Options options = MultimodalManifestDataset.Options.forManifest(manifest)
                .withAssetRoot(tempDir.resolve("assets"))
                .withInlineBinaryAssets(true);
        MultimodalManifestDataset dataset = new MultimodalManifestDataset(manifest, options);

        assertEquals(1, dataset.size());
        MultimodalContent document = dataset.get(0).get(1);
        assertEquals(ModalityType.DOCUMENT, document.getModality());
        assertEquals("pdf", document.getDocumentFormat());
        assertEquals("application/pdf", document.getMimeType());
        assertEquals(Base64.getEncoder().encodeToString(docBytes), document.getBase64Data());
        assertNull(document.getUri());
        assertArrayEquals(new float[] {1f, 2.5f, 3f}, dataset.get(0).get(2).getEmbedding(), 1e-6f);
    }

    @Test
    void manifestDatasetRejectsRelativePathTraversalOutsideAssetRoot() throws Exception {
        Path assets = Files.createDirectories(tempDir.resolve("assets"));
        Path manifest = tempDir.resolve("manifest.jsonl");
        Files.writeString(manifest, "{\"image\":\"../secret.png\"}");

        MultimodalManifestDataset.Options options = MultimodalManifestDataset.Options.forManifest(manifest)
                .withAssetRoot(assets);
        assertThrows(IllegalArgumentException.class, () -> new MultimodalManifestDataset(manifest, options));
    }

    private static List<MultimodalContent> imageText(String id) {
        return List.of(
                MultimodalContent.ofText("caption " + id),
                MultimodalContent.ofImageUri("file:///tmp/" + id + ".png", "image/png"));
    }

    private static List<MultimodalContent> audioText(String id) {
        return List.of(
                MultimodalContent.ofText("transcript " + id),
                MultimodalContent.ofAudioUri("file:///tmp/" + id + ".wav"));
    }

    private static Map<String, Integer> signatureCounts(Dataset<List<MultimodalContent>> dataset) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (int i = 0; i < dataset.size(); i++) {
            counts.merge(MultimodalDatasetSplits.signature(dataset.get(i)), 1, Integer::sum);
        }
        return counts;
    }

    private static Dataset.Split<List<MultimodalContent>> splitOf(Dataset.Fold<List<MultimodalContent>> fold) {
        return new Dataset.Split<>(fold.train(), fold.validation());
    }

    private static List<String> firstSourcePaths(Dataset<List<MultimodalContent>> dataset) {
        List<String> paths = new ArrayList<>();
        for (int i = 0; i < dataset.size(); i++) {
            paths.add(MultimodalDatasetSplits.sourcePaths(dataset.get(i)).stream()
                    .findFirst()
                    .orElse(""));
        }
        return paths;
    }
}
