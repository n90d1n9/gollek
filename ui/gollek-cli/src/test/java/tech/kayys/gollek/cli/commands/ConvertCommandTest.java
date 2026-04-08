package tech.kayys.gollek.cli.commands;

import io.quarkus.test.InjectMock;
import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.Test;
import jakarta.inject.Inject;
import tech.kayys.gollek.converter.GGUFConverter;
import tech.kayys.gollek.converter.model.ConversionResult;
import tech.kayys.gollek.converter.model.GGUFConversionParams;
import tech.kayys.gollek.converter.model.ModelMetadata;
import tech.kayys.gollek.converter.model.QuantizationType;

import java.nio.file.Path;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@QuarkusTest
class ConvertCommandTest {

    @Inject
    ConvertCommand command;

    @InjectMock
    GGUFConverter converter;

    @Test
    void dryRunResolvesPaths() {
        GGUFConversionParams resolved = GGUFConversionParams.builder()
                .inputPath(Path.of("/tmp/input"))
                .outputPath(Path.of("/tmp/output.gguf"))
                .quantization(QuantizationType.Q4_K_M)
                .build();
        when(converter.resolveParams(any())).thenReturn(resolved);

        command.inputPath = "/tmp/input";
        command.outputPath = "/tmp/output.gguf";
        command.dryRun = true;
        command.jsonOutput = true;
        command.run();
    }

    @Test
    void convertRunsConversion() {
        ConversionResult result = ConversionResult.builder()
                .conversionId(1L)
                .success(true)
                .inputInfo(ModelMetadata.builder().build())
                .outputPath(Path.of("/tmp/output.gguf"))
                .outputSize(1024)
                .durationMs(1000)
                .compressionRatio(1.0)
                .build();
        when(converter.convert(any(GGUFConversionParams.class), any())).thenReturn(result);

        command.inputPath = "/tmp/input";
        command.outputPath = "/tmp/output.gguf";
        command.dryRun = false;
        command.jsonOutput = true;
        command.run();
    }
}
