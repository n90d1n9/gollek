package tech.kayys.gollek.jupyter;

import org.dflib.jjava.JavaKernel;
import org.dflib.jjava.jupyter.kernel.LanguageInfo;
import org.dflib.jjava.jupyter.kernel.ReplacementOptions;
import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GollekKernelJshellCoverageTest {

    private GollekKernel kernel;

    @AfterEach
    void tearDown() {
        if (kernel != null) {
            kernel.onShutdown(false);
        }
    }

    @Test
    void evalReturnsPlainTextForSimpleExpression() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("1 + 2");

        assertEquals("3", result.getData(MIMEType.TEXT_PLAIN));
    }

    @Test
    void evalReturnsEmptyDisplayForDeclarationOnlySnippet() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("int declaredOnly = 7;");

        assertSame(DisplayData.EMPTY, result);
    }

    @Test
    void evalPreservesJshellStateAcrossCells() throws Exception {
        kernel = new GollekKernel();

        kernel.eval("int retained = 40;");
        DisplayData result = kernel.eval("retained + 2");

        assertEquals("42", result.getData(MIMEType.TEXT_PLAIN));
    }

    @Test
    void evalHandlesMultipleCompleteSnippetsInOneCell() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("int left = 1; int right = 2; left + right;");

        assertEquals("3", result.getData(MIMEType.TEXT_PLAIN));
    }

    @Test
    void evalSupportsPreloadedGradTensorFromFreshKernelSession() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("""
                var t = GradTensor.ones(2, 2);
                t.shape()[0] + t.shape()[1];
                """);

        assertEquals("4", result.getData(MIMEType.TEXT_PLAIN));
    }

    @Test
    void evalSupportsPreloadedNnLayersFromFreshKernelSession() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("""
                new Sequential(new Linear(2, 3), new ReLU()).size();
                """);

        assertEquals("2", result.getData(MIMEType.TEXT_PLAIN));
    }

    @Test
    void evalSupportsTimeMagicForSimpleExpressions() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%time 1 + 2");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("elapsedMs="));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("result=3"));
    }

    @Test
    void evalSupportsTimeMagicForRichNotebookValues() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%time gollekLossCurveDemo()");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("elapsedMs="));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("LineChart(title=Training Loss, points=5"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("elapsedMs="));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
    }

    @Test
    void evalSupportsTimeitMagicForSimpleExpressions() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%timeit -n 3 1 + 2");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("runs=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("avgMs="));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("minMs="));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("maxMs="));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("result=3"));
    }

    @Test
    void evalSupportsTimeitMagicForRichNotebookValues() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%timeit -n 2 gollekLossCurveDemo()");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("runs=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("LineChart(title=Training Loss, points=5"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("avgMs="));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
    }

    @Test
    void evalSupportsClasspathMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%classpath");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Classpath(entries="));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("gollek-jupyter-kernel"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<ul"));
    }

    @Test
    void evalSupportsMagicsDiscovery() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%magics");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%magics - list notebook magics"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%reset - clear notebook state and reload default Gollek helpers"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%cd <PATH> - change the notebook working directory"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%tree [PATH] - show a shallow directory tree"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%du [PATH] - summarize file or directory disk usage"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%cat <PATH> - preview a text file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%head [-n N] <PATH> - preview the first lines of a text file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%tail [-n N] <PATH> - preview the last lines of a text file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%grep <TEXT> [PATH] - search text in one file or a shallow directory tree"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%findfile <TEXT> [PATH] - search file and directory names"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%wc <PATH> - show file size and line stats"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%diff <LEFT_PATH> <RIGHT_PATH> - show a lightweight line diff for two text files"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%zipls <PATH> - list entries inside a zip archive"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%zipcat <ZIP_PATH> <ENTRY_PATH> - preview one text entry from a zip archive"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%tarls <PATH> - list entries inside a tar or tar.gz archive"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%tarcat <TAR_PATH> <ENTRY_PATH> - preview one text entry from a tar or tar.gz archive"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH] - extract one archive entry into the notebook directory"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%sha256 <PATH> - compute a file SHA-256 digest"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%json <PATH> - pretty-print a JSON file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%csv [-n N] [--profile] <PATH> - preview a CSV file as a dataframe-style table"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%tsv [-n N] [--profile] <PATH> - preview a TSV file as a dataframe-style table"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%sample [--tsv] [-n N] [--seed S] <PATH> - sample random rows from a CSV or TSV file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%sort [--tsv] [-n N] [--desc] <PATH> <COLUMN> - preview rows sorted by a CSV or TSV column"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE] - preview CSV or TSV rows matching a column predicate"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%describe [--tsv] <PATH> - summarize numeric columns from a CSV or TSV file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%schema [--tsv] <PATH> - infer lightweight column types from a CSV or TSV file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%missing [--tsv] <PATH> - summarize missing values by CSV or TSV column"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%valuecounts [--tsv] [--top N] <PATH> <COLUMN> - count distinct values in a CSV or TSV column"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [AGG] - summarize grouped CSV or TSV data"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%corr [--tsv] [--heatmap] <PATH> - show a numeric Pearson correlation matrix from a CSV or TSV file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN> - render a quick scatter plot from numeric CSV or TSV columns"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN> - render a quick line plot from a CSV or TSV file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%hist [--tsv] [--bins N] <PATH> <COLUMN> - render a quick histogram from a numeric CSV or TSV column"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%md <PATH> - preview a Markdown file as rich text"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%html <PATH> - preview an HTML file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%yaml <PATH> - preview a YAML file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%toml <PATH> - preview a TOML file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%xml <PATH> - preview an XML file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%ini <PATH> - preview an INI file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%properties <PATH> - preview a .properties file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%envfile <PATH> - preview a .env-style file"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%timeit [-n N] <expr>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%maven [--allow-remote] [--fetch] [--explain] <G:A:V>"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Notebook Magics"));
    }

    @Test
    void evalSupportsHelpForKnownMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %maven");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%maven [--allow-remote] [--fetch] [--explain] <groupId:artifactId:version>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("--fetch requires GOLLEK_JUPYTER_ENABLE_REMOTE_MAVEN=true"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsResetMagic() throws Exception {
        kernel = new GollekKernel();

        kernel.eval("int retained = 40;");
        DisplayData reset = kernel.eval("%reset");

        assertTrue(reset.getData(MIMEType.TEXT_PLAIN).toString().contains("Notebook session reset"));
        assertThrows(Throwable.class, () -> kernel.eval("retained"));

        DisplayData helper = kernel.eval("gollekTensorDemo().shape()[0]");
        DisplayData deps = kernel.eval("%deps");

        assertEquals("2", helper.getData(MIMEType.TEXT_PLAIN));
        assertEquals("Dependencies(dynamic=0)", deps.getData(MIMEType.TEXT_PLAIN));
    }

    @Test
    void evalSupportsHelpForResetMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %reset");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%reset"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Clears notebook variables, declarations, and dynamic dependency loads."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForCdMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %cd");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%cd <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Changes the notebook working directory"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForTreeMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %tree");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%tree [PATH]"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Shows a shallow directory tree"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForCatMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %cat");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%cat <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Shows a UTF-8 text preview"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForHeadMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %head");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%head [-n N] <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Default line count is 10."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForTailMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %tail");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%tail [-n N] <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Default line count is 10."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForGrepMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %grep");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%grep <TEXT> [PATH]"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Searches for plain text"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForFindFileMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %findfile");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%findfile <TEXT> [PATH]"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Searches file and directory names"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForWordCountMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %wc");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%wc <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Shows line, word, character, and byte counts"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForDiffMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %diff");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%diff <LEFT_PATH> <RIGHT_PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Shows a lightweight line diff for two UTF-8 text files."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForZipLsMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %zipls");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%zipls <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Lists entries inside one zip archive"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForZipCatMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %zipcat");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%zipcat <ZIP_PATH> <ENTRY_PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one UTF-8 text entry from a zip archive."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForTarLsMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %tarls");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%tarls <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Lists entries inside one tar or tar.gz archive"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForTarCatMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %tarcat");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%tarcat <TAR_PATH> <ENTRY_PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one UTF-8 text entry from a tar or tar.gz archive."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForExtractMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %extract");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Use --dry-run to inspect the planned output path before writing."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForSha256Magic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %sha256");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%sha256 <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Computes the SHA-256 digest for one file"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForDiskUsageMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %du");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%du [PATH]"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Summarizes file count, directory count, and byte usage"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForJsonMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %json");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%json <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Pretty-prints one JSON file"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForCsvMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %csv");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%csv [-n N] [--profile] <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one CSV file as a dataframe-style notebook table."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Use -n N to control preview rows, and --profile for column stats."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForTsvMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %tsv");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%tsv [-n N] [--profile] <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one TSV file as a dataframe-style notebook table."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Use -n N to control preview rows, and --profile for column stats."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForLinePlotMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %lineplot");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Renders a quick inline SVG line plot"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForValueCountsMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %valuecounts");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%valuecounts [--tsv] [--top N] <PATH> <COLUMN>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Counts distinct values in one CSV or TSV column."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Blank cells are shown as (blank), and output includes count and percent."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForSampleMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %sample");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%sample [--tsv] [-n N] [--seed S] <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Samples random rows from one CSV or TSV file."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Use --seed S for reproducible sampling."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForSortMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %sort");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews rows sorted by one CSV or TSV column."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("blank values are kept at the end"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForFilterMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %filter");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE]"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews rows matching one CSV or TSV column predicate."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Supported operators: ==, !=, >, >=, <, <=, contains, starts, ends, blank, notblank."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForMissingMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %missing");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%missing [--tsv] <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Summarizes blank cells by CSV or TSV column."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Reports missing count, present count, and missing percent per column."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForGroupByMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %groupby");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [count|sum|mean|min|max]"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Groups CSV or TSV rows by GROUP_COLUMN."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Without VALUE_COLUMN, reports row counts per group."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForSchemaMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %schema");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%schema [--tsv] <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Infers lightweight column types from one CSV or TSV file."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Types are empty, boolean, integer, decimal, date, datetime, or text."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForScatterMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %scatter");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Renders a quick inline SVG scatter plot"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Rows with blank or non-numeric values in either column are skipped."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForDescribeMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %describe");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%describe [--tsv] <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Reports count, missing, mean, std, min, p25, median, p75, and max."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForCorrelationMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %corr");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%corr [--tsv] [--heatmap] <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Shows a Pearson correlation matrix"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Use --heatmap to add an inline SVG heatmap above the matrix table."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Blank cells mean insufficient data or zero variance."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForHistogramMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %hist");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%hist [--tsv] [--bins N] <PATH> <COLUMN>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Renders a quick inline SVG histogram"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Non-numeric and blank values are skipped and reported."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForMarkdownMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %md");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%md <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one Markdown file as rich notebook HTML."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForHtmlMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %html");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%html <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one HTML file directly in notebook HTML output."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForYamlMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %yaml");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%yaml <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one YAML file with notebook-friendly formatting."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForTomlMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %toml");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%toml <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one TOML file with notebook-friendly formatting."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForXmlMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %xml");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%xml <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one XML file with notebook-friendly formatting."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForIniMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %ini");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%ini <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one INI file with notebook-friendly formatting."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForPropertiesMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %properties");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%properties <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one Java .properties file as key/value entries."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsHelpForEnvFileMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %envfile");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%envfile <PATH>"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Previews one .env-style file as key/value entries."));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalRejectsUnknownHelpTopic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%help %definitelyMissing");

        assertEquals("Unknown help topic: %definitelyMissing\nTry %magics", result.getData(MIMEType.TEXT_PLAIN));
    }

    @Test
    void evalSupportsPwdMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%pwd");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("/"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Working Directory"));
    }

    @Test
    void evalSupportsCdMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-cd");
        Files.writeString(dir.resolve("visible.txt"), "ok", StandardCharsets.UTF_8);

        DisplayData cdResult = kernel.eval("%cd " + dir);
        DisplayData pwdResult = kernel.eval("%pwd");
        DisplayData lsResult = kernel.eval("%ls");

        assertTrue(cdResult.getData(MIMEType.TEXT_PLAIN).toString().contains("Working directory changed to"));
        assertTrue(cdResult.getData(MIMEType.TEXT_PLAIN).toString().contains(dir.toString()));
        assertEquals(dir.toAbsolutePath().normalize().toString(), pwdResult.getData(MIMEType.TEXT_PLAIN));
        assertTrue(lsResult.getData(MIMEType.TEXT_PLAIN).toString().contains("visible.txt"));
        assertTrue(lsResult.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsLsMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%ls");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Directory("));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("gollek/"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<ul"));
    }

    @Test
    void evalSupportsTreeMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-tree");
        Files.createDirectories(dir.resolve("nested/deeper"));
        Files.writeString(dir.resolve("root.txt"), "ok", StandardCharsets.UTF_8);
        Files.writeString(dir.resolve("nested/child.txt"), "ok", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%tree " + dir);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Tree(" + dir.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("root.txt"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("nested/"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("child.txt"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<pre"));
    }

    @Test
    void evalSupportsCatMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-cat");
        Path file = dir.resolve("notes.txt");
        Files.writeString(file, "alpha\nbeta\ngamma\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%cat " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("File(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("alpha"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("beta"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<pre"));
    }

    @Test
    void evalSupportsHeadMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-head");
        Path file = dir.resolve("notes.txt");
        Files.writeString(file, "line1\nline2\nline3\nline4\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%head -n 2 " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Head(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("lines=2/4"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("line1"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("line2"));
        assertFalse(result.getData(MIMEType.TEXT_PLAIN).toString().contains("line3"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<pre"));
    }

    @Test
    void evalSupportsTailMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-tail");
        Path file = dir.resolve("notes.txt");
        Files.writeString(file, "line1\nline2\nline3\nline4\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%tail -n 2 " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Tail(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("lines=2/4"));
        assertFalse(result.getData(MIMEType.TEXT_PLAIN).toString().contains("line1"));
        assertFalse(result.getData(MIMEType.TEXT_PLAIN).toString().contains("line2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("line3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("line4"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<pre"));
    }

    @Test
    void evalSupportsGrepMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-grep");
        Files.writeString(dir.resolve("a.txt"), "hello\nneedle here\n", StandardCharsets.UTF_8);
        Files.createDirectories(dir.resolve("nested"));
        Files.writeString(dir.resolve("nested/b.txt"), "other\nneedle there\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%grep needle " + dir);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Grep(pattern=needle, matches=2)"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("a.txt:2: needle here"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("nested/b.txt:2: needle there"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<pre"));
    }

    @Test
    void evalSupportsFindFileMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-findfile");
        Files.writeString(dir.resolve("alpha-notes.txt"), "ok", StandardCharsets.UTF_8);
        Files.createDirectories(dir.resolve("nested-beta"));
        Files.writeString(dir.resolve("nested-beta/gamma.txt"), "ok", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%findfile beta " + dir);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("FindFile(pattern=beta, matches=2)"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("nested-beta/"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("nested-beta/gamma.txt"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<pre"));
    }

    @Test
    void evalSupportsWordCountMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-wc");
        Path file = dir.resolve("stats.txt");
        Files.writeString(file, "one two\nthree\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%wc " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("WC(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("lines=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("words=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("chars=14"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<pre"));
    }

    @Test
    void evalSupportsDiskUsageMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-du");
        Path nested = Files.createDirectories(dir.resolve("nested"));
        Files.writeString(dir.resolve("a.txt"), "abc", StandardCharsets.UTF_8);
        Files.writeString(nested.resolve("b.txt"), "hello", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%du " + dir);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("DU(" + dir.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("type=directory"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("files=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("directories=1"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("bytes=8"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Disk Usage"));
    }

    @Test
    void evalSupportsSha256Magic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-sha256");
        Path file = dir.resolve("artifact.txt");
        Files.writeString(file, "abc", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%sha256 " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("SHA256(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("bytes=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("SHA-256"));
    }

    @Test
    void evalSupportsDiffMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-diff");
        Path left = dir.resolve("left.txt");
        Path right = dir.resolve("right.txt");
        Files.writeString(left, "alpha\nbeta\nsame\n", StandardCharsets.UTF_8);
        Files.writeString(right, "alpha\ngamma\nsame\nextra\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%diff " + left + " " + right);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Diff(left=" + left.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("changed=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("- 2: beta"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("+ 2: gamma"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("+ 4: extra"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#b31d28"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#22863a"));
    }

    @Test
    void evalSupportsZipLsMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-zipls");
        Path zip = dir.resolve("bundle.zip");
        try (ZipOutputStream out = new ZipOutputStream(Files.newOutputStream(zip))) {
            out.putNextEntry(new ZipEntry("manifest.json"));
            out.write("{\"ok\":true}".getBytes(StandardCharsets.UTF_8));
            out.closeEntry();
            out.putNextEntry(new ZipEntry("nested/"));
            out.closeEntry();
            out.putNextEntry(new ZipEntry("nested/report.txt"));
            out.write("hello".getBytes(StandardCharsets.UTF_8));
            out.closeEntry();
        }

        DisplayData result = kernel.eval("%zipls " + zip);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Zip(" + zip.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("entries=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("manifest.json"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("nested/"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("nested/report.txt"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<pre"));
    }

    @Test
    void evalSupportsZipCatMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-zipcat");
        Path zip = dir.resolve("bundle.zip");
        try (ZipOutputStream out = new ZipOutputStream(Files.newOutputStream(zip))) {
            out.putNextEntry(new ZipEntry("manifest.json"));
            out.write("{\"ok\":true}\n".getBytes(StandardCharsets.UTF_8));
            out.closeEntry();
            out.putNextEntry(new ZipEntry("nested/report.txt"));
            out.write("hello from bundle\n".getBytes(StandardCharsets.UTF_8));
            out.closeEntry();
        }

        DisplayData result = kernel.eval("%zipcat " + zip + " nested/report.txt");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("ZipEntry(" + zip.toAbsolutePath().normalize() + "!nested/report.txt"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("hello from bundle"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Zip Entry"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("hello from bundle"));
    }

    @Test
    void evalSupportsTarLsMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-tarls");
        Path tarGz = dir.resolve("bundle.tar.gz");
        writeTarArchive(tarGz, true);

        DisplayData result = kernel.eval("%tarls " + tarGz);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Tar(" + tarGz.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("entries=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("manifest.json"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("nested/"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("nested/report.txt"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<pre"));
    }

    @Test
    void evalSupportsTarCatMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-tarcat");
        Path tarGz = dir.resolve("bundle.tar.gz");
        writeTarArchive(tarGz, true);

        DisplayData result = kernel.eval("%tarcat " + tarGz + " nested/report.txt");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("TarEntry(" + tarGz.toAbsolutePath().normalize() + "!nested/report.txt"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("hello from bundle"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Tar Entry"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("hello from bundle"));
    }

    @Test
    void evalSupportsExtractDryRunForZipMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-extract-zip");
        Path zip = dir.resolve("bundle.zip");
        try (ZipOutputStream out = new ZipOutputStream(Files.newOutputStream(zip))) {
            out.putNextEntry(new ZipEntry("nested/report.txt"));
            out.write("hello from zip\n".getBytes(StandardCharsets.UTF_8));
            out.closeEntry();
        }
        kernel.eval("%cd " + dir);

        DisplayData result = kernel.eval("%extract --dry-run " + zip + " nested/report.txt extracted/report.txt");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Extract(dryRun=true"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("entry=nested/report.txt"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("output=" + dir.resolve("extracted/report.txt").toAbsolutePath().normalize()));
        assertFalse(Files.exists(dir.resolve("extracted/report.txt")));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsExtractForTarMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-extract-tar");
        Path tarGz = dir.resolve("bundle.tar.gz");
        writeTarArchive(tarGz, true);
        kernel.eval("%cd " + dir);

        DisplayData result = kernel.eval("%extract " + tarGz + " nested/report.txt extracted/report.txt");

        Path extracted = dir.resolve("extracted/report.txt");
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Extract(dryRun=false"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Wrote " + extracted.toAbsolutePath().normalize()));
        assertEquals("hello from bundle\n", Files.readString(extracted, StandardCharsets.UTF_8));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalSupportsJsonMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-json");
        Path file = dir.resolve("config.json");
        Files.writeString(file, "{\"name\":\"gollek\",\"items\":[1,2]}", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%json " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("JSON(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("\"name\": \"gollek\""));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("\"items\": ["));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<pre"));
    }

    @Test
    void evalSupportsCsvMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-csv");
        Path file = dir.resolve("table.csv");
        Files.writeString(file, "name,score\nalice,10\nbob,20\ncharlie,30\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%csv -n 2 --profile " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("CSV(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=3, columns=2, previewRows=2, truncated=true, profile=true"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("name | score"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("alice | 10"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("... 1 more rows"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("score | 3 | 0 | 3 | 10 | 30 | 20"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Column Profile"));
    }

    @Test
    void evalSupportsTsvMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-tsv");
        Path file = dir.resolve("table.tsv");
        Files.writeString(file, "name\tscore\nalice\t10\nbob\t20\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%tsv " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("TSV(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=2, columns=2, previewRows=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("name | score"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("alice | 10"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsSampleMagicForCsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-sample");
        Path file = dir.resolve("table.csv");
        Files.writeString(file, "id,label\n1,a\n2,b\n3,c\n4,d\n5,e\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%sample -n 2 --seed 7 " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Sample(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=5"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("sample=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("columns=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("seed=7"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("id | label"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Sample"));
    }

    @Test
    void evalSupportsSampleMagicForTsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-sample-tsv");
        Path file = dir.resolve("table.tsv");
        Files.writeString(file, "name\tscore\nalice\t10\nbob\t20\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%sample --tsv -n 5 " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("sample=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("seed=random"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("alice | 10"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsSortMagicForCsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-sort");
        Path file = dir.resolve("table.csv");
        Files.writeString(file, "id,label,score\n1,a,9\n2,b,20\n3,c,10\n4,d,\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%sort -n 3 --desc " + file + " score");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Sort(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("column=score"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("order=desc"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("mode=numeric"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=4"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("previewRows=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("# | id | label | score"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("1 | 2 | b | 20"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("2 | 3 | c | 10"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Sort"));
    }

    @Test
    void evalSupportsSortMagicForTsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-sort-tsv");
        Path file = dir.resolve("table.tsv");
        Files.writeString(file, "name\tstatus\nzara\tbeta\namy\talpha\nbob\tbeta\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%sort --tsv -n 2 " + file + " name");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("column=name"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("order=asc"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("mode=text"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("previewRows=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("1 | amy | alpha"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("2 | bob | beta"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsFilterMagicForCsvNumericPredicate() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-filter");
        Path file = dir.resolve("table.csv");
        Files.writeString(file, "id,label,score\n1,a,9\n2,b,20\n3,c,10\n4,d,\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%filter -n 2 " + file + " score >= 10");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Filter(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("column=score"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("op=>="));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("value=10"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=4"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("matched=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("previewRows=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("# | id | label | score"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("1 | 2 | b | 20"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("2 | 3 | c | 10"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Filter"));
    }

    @Test
    void evalSupportsFilterMagicForTsvTextPredicate() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-filter-tsv");
        Path file = dir.resolve("table.tsv");
        Files.writeString(file, "name\tstatus\namy\talpha\nbob\tbeta\ncal\tbeta\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%filter --tsv " + file + " status == beta");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("column=status"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("op==="));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("value=beta"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("matched=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("1 | bob | beta"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("2 | cal | beta"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsDescribeMagicForCsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-describe");
        Path file = dir.resolve("metrics.csv");
        Files.writeString(file, "epoch,loss,label\n1,10,a\n2,20,b\n3,30,c\n4,40,d\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%describe " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Describe(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=4"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("columns=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("numericColumns=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("loss | 4 | 0 | 25 | 11.1803 | 10 | 17.5000 | 25 | 32.5000 | 40"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Describe"));
    }

    @Test
    void evalSupportsDescribeMagicForTsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-describe-tsv");
        Path file = dir.resolve("features.tsv");
        Files.writeString(file, "step\taccuracy\tlabel\n1\t0.5\ta\n2\t0.75\tb\n3\t\tc\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%describe --tsv " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("numericColumns=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("accuracy | 2 | 1 | 0.6250"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsCorrelationMagicForCsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-corr");
        Path file = dir.resolve("metrics.csv");
        Files.writeString(file, "x,y,z,label\n1,2,3,a\n2,4,2,b\n3,6,1,c\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%corr " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Correlation(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("numericColumns=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("column | x | y | z"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("x | 1 | 1 | -1"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Correlation"));
    }

    @Test
    void evalSupportsCorrelationMagicForTsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-corr-tsv");
        Path file = dir.resolve("features.tsv");
        Files.writeString(file, "a\tb\tlabel\n1\t3\tx\n2\t2\ty\n3\t1\tz\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%corr --tsv " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("numericColumns=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("a | 1 | -1"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsCorrelationMagicHeatmap() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-corr-heatmap");
        Path file = dir.resolve("metrics.csv");
        Files.writeString(file, "x,y,z,label\n1,2,3,a\n2,4,2,b\n3,6,1,c\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%corr --heatmap " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("heatmap=true"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("x | 1 | 1 | -1"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Correlation heatmap"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("blue = negative"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsLinePlotMagicForCsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-lineplot");
        Path file = dir.resolve("metrics.csv");
        Files.writeString(file, "epoch,loss\n1,0.9\n2,0.7\n3,0.5\nbad,nope\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%lineplot " + file + " epoch loss");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("LinePlot(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("x=epoch"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("y=loss"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("points=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("skipped=1"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("loss by epoch"));
    }

    @Test
    void evalSupportsLinePlotMagicForTsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-lineplot-tsv");
        Path file = dir.resolve("metrics.tsv");
        Files.writeString(file, "step\taccuracy\n1\t0.5\n2\t0.75\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%lineplot --tsv " + file + " step accuracy");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("points=2"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
    }

    @Test
    void evalSupportsScatterMagicForCsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-scatter");
        Path file = dir.resolve("metrics.csv");
        Files.writeString(file, "weight,height,label\n10,20,a\n20,35,b\n30,50,c\nbad,60,d\n40,nope,e\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%scatter " + file + " weight height");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("ScatterPlot(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("x=weight"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("y=height"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("points=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("skipped=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("10 | 20"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("height vs weight"));
    }

    @Test
    void evalSupportsScatterMagicForTsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-scatter-tsv");
        Path file = dir.resolve("metrics.tsv");
        Files.writeString(file, "x\ty\tlabel\n1\t3\ta\n2\t5\tb\n\t6\tc\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%scatter --tsv " + file + " x y");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("points=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("skipped=1"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
    }

    @Test
    void evalSupportsValueCountsMagicForCsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-valuecounts");
        Path file = dir.resolve("labels.csv");
        Files.writeString(file, "label,score\ncat,1\ndog,2\ncat,3\nbird,4\ncat,5\n,6\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%valuecounts --top 2 " + file + " label");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("ValueCounts(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("column=label"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=6"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("unique=4"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("top=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("other=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("cat | 3 | 50.00%"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("(other) | 2 | 33.33%"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Value counts: label"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsValueCountsMagicForTsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-valuecounts-tsv");
        Path file = dir.resolve("labels.tsv");
        Files.writeString(file, "label\tscore\nred\t1\nblue\t2\nred\t3\n\t4\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%valuecounts --tsv --top 5 " + file + " label");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=4"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("unique=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("red | 2 | 50.00%"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("(blank) | 1 | 25.00%"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
    }

    @Test
    void evalSupportsMissingMagicForCsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-missing");
        Path file = dir.resolve("metrics.csv");
        Files.writeString(file, "a,b,c\n1,,x\n,2,y\n3,4,\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%missing " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Missing(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("columns=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("columnsWithMissing=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("a | 1 | 2 | 33.33%"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("c | 1 | 2 | 33.33%"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Missing values"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsMissingMagicForTsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-missing-tsv");
        Path file = dir.resolve("metrics.tsv");
        Files.writeString(file, "a\tb\n1\t\n\t2\n3\t4\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%missing --tsv " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("columnsWithMissing=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("a | 1 | 2 | 33.33%"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("b | 1 | 2 | 33.33%"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
    }

    @Test
    void evalSupportsSchemaMagicForCsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-schema");
        Path file = dir.resolve("metrics.csv");
        Files.writeString(file, "id,price,flag,day,stamp,label,empty\n"
                + "1,1.5,true,2026-05-27,2026-05-27T10:15:30,cat,\n"
                + "2,2,false,2026-05-28,2026-05-28T10:15:30,dog,\n"
                + "3,3.25,true,,2026-05-29T10:15:30,cat,\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%schema " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Schema(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("columns=7"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("id | integer | 3 | 0 | 0.00% | 1, 2, 3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("price | decimal | 3 | 0 | 0.00% | 1.5, 2, 3.25"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("flag | boolean | 3 | 0 | 0.00% | true, false"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("day | date | 2 | 1 | 33.33%"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("stamp | datetime | 3 | 0 | 0.00%"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("label | text | 3 | 0 | 0.00% | cat, dog"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("empty | empty | 0 | 3 | 100.00% | "));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Schema"));
    }

    @Test
    void evalSupportsSchemaMagicForTsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-schema-tsv");
        Path file = dir.resolve("metrics.tsv");
        Files.writeString(file, "id\tactive\tlabel\n1\ttrue\ta\n2\tfalse\tb\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%schema --tsv " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("id | integer | 2 | 0 | 0.00%"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("active | boolean | 2 | 0 | 0.00%"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("label | text | 2 | 0 | 0.00%"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsGroupByMagicForCsvCounts() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-groupby-count");
        Path file = dir.resolve("sales.csv");
        Files.writeString(file, "region,amount\nwest,10\neast,5\nwest,7\n,3\neast,2\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%groupby " + file + " region");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("GroupBy(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("group=region"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("agg=count"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("rows=5"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("groups=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("west | 2 | 2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("(blank) | 1 | 1"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("GroupBy"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsGroupByMagicForCsvMean() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-groupby-mean");
        Path file = dir.resolve("sales.csv");
        Files.writeString(file, "region,amount\nwest,10\neast,5\nwest,6\neast,nope\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%groupby " + file + " region amount mean");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("value=amount"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("agg=mean"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("skipped=1"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("west | 2 | 2 | 8"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("east | 2 | 1 | 5"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
    }

    @Test
    void evalSupportsGroupByMagicForTsvSum() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-groupby-tsv");
        Path file = dir.resolve("sales.tsv");
        Files.writeString(file, "team\tpoints\na\t1\nb\t2\na\t3\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%groupby --tsv " + file + " team points sum");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("agg=sum"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("a | 2 | 2 | 4"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("b | 1 | 1 | 2"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
    }

    @Test
    void evalSupportsHistogramMagicForCsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-hist");
        Path file = dir.resolve("metrics.csv");
        Files.writeString(file, "epoch,loss\n1,0.9\n2,0.7\n3,0.5\n4,0.3\nbad,nope\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%hist --bins 2 " + file + " loss");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Histogram(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=CSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("column=loss"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("values=4"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("skipped=1"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("bins=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("min=0.3000"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("max=0.9000"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Histogram: loss"));
    }

    @Test
    void evalSupportsHistogramMagicForTsv() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-hist-tsv");
        Path file = dir.resolve("features.tsv");
        Files.writeString(file, "step\taccuracy\n1\t0.5\n2\t0.75\n3\t0.75\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%hist --tsv --bins 3 " + file + " accuracy");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("format=TSV"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("column=accuracy"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("values=3"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("bins=3"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
    }

    @Test
    void evalSupportsMarkdownMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-md");
        Path file = dir.resolve("notes.md");
        Files.writeString(file, "# Title\n\n- item one\n- item two\n\n`code`\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%md " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Markdown(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("# Title"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<h1>Title</h1>"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<li>item one</li>"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<code>code</code>"));
    }

    @Test
    void evalSupportsHtmlMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-html");
        Path file = dir.resolve("report.html");
        Files.writeString(file, "<h1>Report</h1><p>hello</p>", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%html " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("HTML(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("<h1>Report</h1>"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<h1>Report</h1>"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<p>hello</p>"));
    }

    @Test
    void evalSupportsYamlMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-yaml");
        Path file = dir.resolve("config.yaml");
        Files.writeString(file, "name: gollek\n# note\nitems:\n  - one\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%yaml " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("YAML(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("name: gollek"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("font-weight:600"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#6a737d"));
    }

    @Test
    void evalSupportsTomlMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-toml");
        Path file = dir.resolve("config.toml");
        Files.writeString(file, "[app]\nname = \"gollek\"\n# note\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%toml " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("TOML(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("[app]"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#d73a49"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#005cc5"));
    }

    @Test
    void evalSupportsXmlMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-xml");
        Path file = dir.resolve("config.xml");
        Files.writeString(file, "<!-- note -->\n<app name=\"gollek\"><mode value=\"demo\"/></app>\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%xml " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("XML(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("<app name=\"gollek\">"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#d73a49"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#005cc5"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#6a737d"));
    }

    @Test
    void evalSupportsIniMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-ini");
        Path file = dir.resolve("config.ini");
        Files.writeString(file, "[app]\nname=gollek\n; note\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%ini " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("INI(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("[app]"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#d73a49"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#005cc5"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("color:#6a737d"));
    }

    @Test
    void evalSupportsPropertiesMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-properties");
        Path file = dir.resolve("app.properties");
        Files.writeString(file, "app.name=gollek\napp.mode=demo\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%properties " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Properties(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("entries=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("app.name=gollek"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("app.mode"));
    }

    @Test
    void evalSupportsEnvFileMagic() throws Exception {
        kernel = new GollekKernel();
        Path dir = Files.createTempDirectory("gollek-kernel-envfile");
        Path file = dir.resolve(".env");
        Files.writeString(file, "# comment\nAPP_NAME=gollek\nAPP_MODE=\"demo\"\n", StandardCharsets.UTF_8);

        DisplayData result = kernel.eval("%envfile " + file);

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("EnvFile(" + file.toAbsolutePath().normalize()));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("entries=2"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("APP_NAME=gollek"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("APP_MODE=demo"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("APP_MODE"));
    }

    @Test
    void evalSupportsEnvMagic() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%env PATH");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().startsWith("PATH="));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("Environment"));
    }

    @Test
    void evalSupportsDepsMagicBeforeDynamicLoads() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%deps");

        assertEquals("Dependencies(dynamic=0)", result.getData(MIMEType.TEXT_PLAIN));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(result.getData(MIMEType.TEXT_HTML).toString().contains("No dynamic notebook dependencies loaded yet."));
    }

    @Test
    void evalSupportsAddJarMagic() throws Exception {
        kernel = new GollekKernel();
        Path jar = buildTempJarWithHelloClass();

        DisplayData addResult = kernel.eval("%addjar " + jar);
        DisplayData evalResult = kernel.eval("demo.MagicAdded.message()");
        DisplayData classpath = kernel.eval("%classpath");

        assertTrue(addResult.getData(MIMEType.TEXT_PLAIN).toString().contains("Added jar to notebook classpath"));
        assertEquals("from-added-jar", evalResult.getData(MIMEType.TEXT_PLAIN));
        assertTrue(classpath.getData(MIMEType.TEXT_PLAIN).toString().contains(jar.toString()));
    }

    @Test
    void evalSupportsMavenMagicFromLocalM2() throws Exception {
        kernel = new GollekKernel();
        String version = "1.0.0-test";
        installTempArtifactToLocalM2(version);

        DisplayData addResult = kernel.eval("%maven demo:magic-added:" + version);
        DisplayData evalResult = kernel.eval("demo.MagicAdded.message()");
        DisplayData classpath = kernel.eval("%classpath");

        assertTrue(addResult.getData(MIMEType.TEXT_PLAIN).toString().contains("Added Maven artifact from local cache: demo:magic-added:" + version));
        assertEquals("from-added-jar", evalResult.getData(MIMEType.TEXT_PLAIN));
        assertTrue(classpath.getData(MIMEType.TEXT_PLAIN).toString().contains("magic-added-" + version + ".jar"));
    }

    @Test
    void evalSupportsDepsAndWhichClassAfterDynamicLoad() throws Exception {
        kernel = new GollekKernel();
        Path jar = buildTempJarWithHelloClass();

        kernel.eval("%addjar " + jar);
        DisplayData deps = kernel.eval("%deps");
        DisplayData which = kernel.eval("%whichclass demo.MagicAdded");

        assertTrue(deps.getData(MIMEType.TEXT_PLAIN).toString().contains("Dependencies(dynamic=1)"));
        assertTrue(deps.getData(MIMEType.TEXT_PLAIN).toString().contains(jar.toString()));
        assertTrue(which.getData(MIMEType.TEXT_PLAIN).toString().contains("Class demo.MagicAdded"));
        assertTrue(which.getData(MIMEType.TEXT_PLAIN).toString().contains(jar.toString()));
        assertTrue(which.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalRejectsMissingWhichClass() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%whichclass definitely.missing.Type");

        assertEquals("Class not found: definitely.missing.Type", result.getData(MIMEType.TEXT_PLAIN));
    }

    @Test
    void evalRejectsMissingMavenArtifact() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%maven definitely.missing:artifact:0.0.1");

        assertEquals("Artifact not found in local Maven cache: definitely.missing:artifact:0.0.1", result.getData(MIMEType.TEXT_PLAIN));
    }

    @Test
    void evalExplainsMissingMavenArtifactSearchPaths() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%maven --explain definitely.missing:artifact:0.0.1");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Artifact not found in local Maven cache: definitely.missing:artifact:0.0.1"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("searched:"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains(".m2/repository"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains(".gradle/caches/modules-2/files-2.1"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalProvidesRemoteIntentGuidanceForMissingMavenArtifact() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%maven --allow-remote definitely.missing:artifact:0.0.1");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("remoteResolution=not-available-in-kernel"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("fetchHint=%maven --allow-remote --fetch definitely.missing:artifact:0.0.1"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("mvn dependency:get -Dartifact=definitely.missing:artifact:0.0.1"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("retry=%maven definitely.missing:artifact:0.0.1"));
        assertTrue(result.hasDataForType(MIMEType.TEXT_HTML));
    }

    @Test
    void evalRejectsRemoteFetchWithoutExplicitEnvironmentOptIn() throws Exception {
        kernel = new GollekKernel();

        DisplayData result = kernel.eval("%maven --allow-remote --fetch definitely.missing:artifact:0.0.1");

        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("Remote Maven fetch is disabled."));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("GOLLEK_JUPYTER_ENABLE_REMOTE_MAVEN=true"));
        assertTrue(result.getData(MIMEType.TEXT_PLAIN).toString().contains("%maven --allow-remote --fetch definitely.missing:artifact:0.0.1"));
    }

    @Test
    void evalSupportsPreloadedNotebookHelpersFromFreshKernelSession() throws Exception {
        kernel = new GollekKernel();

        DisplayData help = kernel.eval("gollekHelp()");
        DisplayData tensorDemo = kernel.eval("gollekTensorDemo().shape()[0]");
        DisplayData previewTensor = kernel.eval("gollekPreviewTensorDemo()");
        DisplayData metricsDemo = kernel.eval("gollekMetricsDemo()");
        DisplayData lossCurve = kernel.eval("gollekLossCurveDemo()");
        DisplayData imageDemo = kernel.eval("gollekImageDemo()");
        DisplayData audioDemo = kernel.eval("gollekAudioDemo()");
        DisplayData modelDemo = kernel.eval("gollekModelDemo().size()");
        DisplayData modelSummary = kernel.eval("gollekModelSummary()");

        assertTrue(help.getData(MIMEType.TEXT_PLAIN).toString().contains("Gollek notebook helpers"));
        assertEquals("2", tensorDemo.getData(MIMEType.TEXT_PLAIN));
        assertTrue(previewTensor.getData(MIMEType.TEXT_PLAIN).toString().contains("GradTensor(shape=[2, 2]"));
        assertTrue(previewTensor.getData(MIMEType.TEXT_PLAIN).toString().contains("size=4"));
        assertTrue(previewTensor.getData(MIMEType.TEXT_PLAIN).toString().contains("min=0"));
        assertTrue(previewTensor.getData(MIMEType.TEXT_PLAIN).toString().contains("max=3"));
        assertTrue(previewTensor.getData(MIMEType.TEXT_PLAIN).toString().contains("sample=[0, 1, 2, 3]"));
        assertTrue(previewTensor.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(previewTensor.getData(MIMEType.TEXT_HTML).toString().contains("size=4"));
        assertTrue(previewTensor.getData(MIMEType.TEXT_HTML).toString().contains("sample=[0, 1, 2, 3]"));
        assertTrue(metricsDemo.getData(MIMEType.TEXT_PLAIN).toString().contains("Table(rows=2, columns=[epoch, loss, accuracy])"));
        assertTrue(metricsDemo.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(metricsDemo.getData(MIMEType.TEXT_HTML).toString().contains("<table"));
        assertTrue(metricsDemo.getData(MIMEType.TEXT_HTML).toString().contains("accuracy"));
        assertTrue(lossCurve.getData(MIMEType.TEXT_PLAIN).toString().contains("LineChart(title=Training Loss, points=5"));
        assertTrue(lossCurve.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(lossCurve.getData(MIMEType.TEXT_HTML).toString().contains("<svg"));
        assertTrue(lossCurve.getData(MIMEType.TEXT_HTML).toString().contains("Training Loss"));
        assertTrue(imageDemo.getData(MIMEType.TEXT_PLAIN).toString().contains("Image(width=4, height=4"));
        assertTrue(imageDemo.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(imageDemo.getData(MIMEType.TEXT_HTML).toString().contains("data:image/png;base64,"));
        assertTrue(imageDemo.hasDataForType(MIMEType.IMAGE_PNG));
        assertTrue(audioDemo.getData(MIMEType.TEXT_PLAIN).toString().contains("Audio(title=Demo Tone, sampleRate=8000, channels=1"));
        assertTrue(audioDemo.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(audioDemo.getData(MIMEType.TEXT_HTML).toString().contains("<audio controls"));
        assertTrue(audioDemo.getData(MIMEType.TEXT_HTML).toString().contains("data:audio/wav;base64,"));
        assertEquals("2", modelDemo.getData(MIMEType.TEXT_PLAIN));
        assertTrue(modelSummary.getData(MIMEType.TEXT_PLAIN).toString().contains("Sequential(layers=2, parameters="));
    }

    @Test
    void completeReturnsJshellSuggestionsAndAnchors() throws Exception {
        kernel = new GollekKernel();

        ReplacementOptions options = kernel.complete("Math.ma", "Math.ma".length());

        assertNotNull(options);
        assertEquals("Math.ma".length(), options.getSourceEnd());
        assertTrue(options.getSourceStart() >= 0);
        assertTrue(options.getSourceStart() < options.getSourceEnd());
        assertFalse(options.getReplacements().isEmpty());
        assertTrue(
                options.getReplacements().stream().allMatch(replacement -> replacement != null && !replacement.isBlank())
        );
    }

    @Test
    void completeReturnsNullWhenJshellHasNoSuggestions() throws Exception {
        kernel = new GollekKernel();

        ReplacementOptions options = kernel.complete("definitelyNoSuchSymbolHere", "definitelyNoSuchSymbolHere".length());

        assertNull(options);
    }

    @Test
    void isCompleteUsesExpectedJjavaSignifiers() {
        kernel = new GollekKernel();

        assertEquals(JavaKernel.completeCodeSignifier(), kernel.isComplete("1 + 2"));
        assertTrue(kernel.isComplete("if (true) {").isBlank());
    }

    @Test
    void lifecycleHooksAreCallableAndShutdownClosesTheShell() {
        kernel = new GollekKernel();

        assertDoesNotThrow(() -> kernel.interrupt());
        assertDoesNotThrow(() -> kernel.onShutdown(false));
        assertThrows(Throwable.class, () -> kernel.eval("1 + 1"));
        kernel = null;
    }

    @Test
    void languageInfoBannerAndHelpLinksAreExposed() {
        kernel = new GollekKernel();

        LanguageInfo info = kernel.getLanguageInfo();
        List<LanguageInfo.Help> helpLinks = kernel.getHelpLinks();

        assertEquals("java", info.getName());
        assertEquals("text/x-java-source", info.getMimetype());
        assertEquals(".jshell", info.getFileExtension());
        assertEquals("java", info.getPygmentsLexer());
        assertEquals("java", info.getCodemirrorMode());
        assertTrue(kernel.getBanner().contains("Gollek Kernel"));
        assertTrue(kernel.getBanner().contains("Preloaded: GradTensor, Linear, ReLU, Sequential"));
        assertTrue(kernel.getBanner().contains("Helpers: gollekHelp(), gollekTensorDemo(), gollekPreviewTensorDemo(), gollekMetricsDemo(), gollekLossCurveDemo(), gollekImageDemo(), gollekAudioDemo(), gollekModelDemo(), gollekModelSummary()"));
        assertTrue(kernel.getBanner().contains("Magics: %magics, %help <magic>, %reset, %time <expr>, %timeit [-n N] <expr>, %classpath, %pwd, %cd <PATH>, %ls, %tree [PATH], %du [PATH], %cat <PATH>, %head [-n N] <PATH>, %tail [-n N] <PATH>, %grep <TEXT> [PATH], %findfile <TEXT> [PATH], %wc <PATH>, %diff <LEFT_PATH> <RIGHT_PATH>, %zipls <PATH>, %zipcat <ZIP_PATH> <ENTRY_PATH>, %tarls <PATH>, %tarcat <TAR_PATH> <ENTRY_PATH>, %extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH], %sha256 <PATH>, %json <PATH>, %csv [-n N] [--profile] <PATH>, %tsv [-n N] [--profile] <PATH>, %sample [--tsv] [-n N] [--seed S] <PATH>, %sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>, %filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE], %describe [--tsv] <PATH>, %schema [--tsv] <PATH>, %missing [--tsv] <PATH>, %valuecounts [--tsv] [--top N] <PATH> <COLUMN>, %groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [AGG], %corr [--tsv] [--heatmap] <PATH>, %scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>, %lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>, %hist [--tsv] [--bins N] <PATH> <COLUMN>, %md <PATH>, %html <PATH>, %yaml <PATH>, %toml <PATH>, %xml <PATH>, %ini <PATH>, %properties <PATH>, %envfile <PATH>, %env <NAME>, %addjar <PATH>, %maven <G:A:V>, %deps, %whichclass <FQCN>"));
        assertEquals(1, helpLinks.size());
        assertEquals("Gollek Docs", helpLinks.getFirst().getText());
        assertEquals("https://gollek-ai.github.io/docs/", helpLinks.getFirst().getUrl());
    }

    private void writeTarArchive(Path file, boolean gzip) throws IOException {
        try (OutputStream fileOut = Files.newOutputStream(file);
             OutputStream out = gzip ? new GZIPOutputStream(fileOut) : fileOut) {
            writeTarEntry(out, "manifest.json", "{\"ok\":true}\n".getBytes(StandardCharsets.UTF_8), (byte) '0');
            writeTarEntry(out, "nested/", new byte[0], (byte) '5');
            writeTarEntry(out, "nested/report.txt", "hello from bundle\n".getBytes(StandardCharsets.UTF_8), (byte) '0');
            out.write(new byte[1024]);
        }
    }

    private void writeTarEntry(OutputStream out, String name, byte[] content, byte typeFlag) throws IOException {
        byte[] header = new byte[512];
        byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
        System.arraycopy(nameBytes, 0, header, 0, Math.min(nameBytes.length, 100));
        writeTarOctal(header, 100, 8, 0644);
        writeTarOctal(header, 108, 8, 0);
        writeTarOctal(header, 116, 8, 0);
        writeTarOctal(header, 124, 12, content.length);
        writeTarOctal(header, 136, 12, 0);
        for (int i = 148; i < 156; i++) {
            header[i] = (byte) ' ';
        }
        header[156] = typeFlag;
        byte[] magic = "ustar".getBytes(StandardCharsets.US_ASCII);
        System.arraycopy(magic, 0, header, 257, magic.length);
        header[262] = 0;
        header[263] = '0';
        header[264] = '0';
        long checksum = 0;
        for (byte b : header) {
            checksum += b & 0xff;
        }
        writeTarOctal(header, 148, 8, checksum);
        out.write(header);
        if (content.length > 0) {
            out.write(content);
            int padding = (int) ((512 - (content.length % 512)) % 512);
            if (padding > 0) {
                out.write(new byte[padding]);
            }
        }
    }

    private void writeTarOctal(byte[] header, int offset, int length, long value) {
        String octal = Long.toOctalString(value);
        int pos = offset + length - 2;
        for (int i = octal.length() - 1; i >= 0 && pos >= offset; i--) {
            header[pos--] = (byte) octal.charAt(i);
        }
        while (pos >= offset) {
            header[pos--] = (byte) '0';
        }
        header[offset + length - 1] = 0;
    }

    @Test
    void inspectReturnsDocumentationForKnownJavaApi() throws Exception {
        kernel = new GollekKernel();

        DisplayData data = kernel.inspect("Math.abs", "Math.abs".length() - 1, false);

        if (data != null) {
            assertNotNull(data.getData(MIMEType.TEXT_PLAIN));
        }
    }

    private static Path buildTempJarWithHelloClass() throws IOException {
        Path root = Files.createTempDirectory("gollek-kernel-addjar");
        Path srcDir = root.resolve("src/demo");
        Path classDir = root.resolve("classes");
        Files.createDirectories(srcDir);
        Files.createDirectories(classDir);

        Path source = srcDir.resolve("MagicAdded.java");
        Files.writeString(source, """
                package demo;
                public class MagicAdded {
                    public static String message() { return "from-added-jar"; }
                }
                """, StandardCharsets.UTF_8);

        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertNotNull(compiler);
        int exit = compiler.run(null, null, null, "-d", classDir.toString(), source.toString());
        assertEquals(0, exit);

        Path jar = root.resolve("magic-added.jar");
        try (JarOutputStream out = new JarOutputStream(Files.newOutputStream(jar))) {
            out.putNextEntry(new JarEntry("demo/MagicAdded.class"));
            out.write(Files.readAllBytes(classDir.resolve("demo/MagicAdded.class")));
            out.closeEntry();
        }
        return jar;
    }

    private static void installTempArtifactToLocalM2(String version) throws IOException {
        Path jar = buildTempJarWithHelloClass();
        Path artifactDir = Path.of(System.getProperty("user.home"), ".m2", "repository", "demo", "magic-added", version);
        Files.createDirectories(artifactDir);
        Files.copy(jar, artifactDir.resolve("magic-added-" + version + ".jar"), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }
}
