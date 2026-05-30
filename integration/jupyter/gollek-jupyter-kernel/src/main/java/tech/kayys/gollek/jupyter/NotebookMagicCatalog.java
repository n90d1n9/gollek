package tech.kayys.gollek.jupyter;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;

import java.util.List;
import java.util.Map;

final class NotebookMagicCatalog {

    private static final List<String> BANNER_USAGES = List.of(
            "%magics",
            "%help <magic>",
            "%reset",
            "%time <expr>",
            "%timeit [-n N] <expr>",
            "%classpath",
            "%pwd",
            "%cd <PATH>",
            "%ls",
            "%tree [PATH]",
            "%du [PATH]",
            "%cat <PATH>",
            "%head [-n N] <PATH>",
            "%tail [-n N] <PATH>",
            "%grep <TEXT> [PATH]",
            "%findfile <TEXT> [PATH]",
            "%wc <PATH>",
            "%diff <LEFT_PATH> <RIGHT_PATH>",
            "%zipls <PATH>",
            "%zipcat <ZIP_PATH> <ENTRY_PATH>",
            "%tarls <PATH>",
            "%tarcat <TAR_PATH> <ENTRY_PATH>",
            "%extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]",
            "%sha256 <PATH>",
            "%json <PATH>",
            "%csv [-n N] [--profile] <PATH>",
            "%tsv [-n N] [--profile] <PATH>",
            "%sample [--tsv] [-n N] [--seed S] <PATH>",
            "%sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>",
            "%filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE]",
            "%describe [--tsv] <PATH>",
            "%schema [--tsv] <PATH>",
            "%missing [--tsv] <PATH>",
            "%valuecounts [--tsv] [--top N] <PATH> <COLUMN>",
            "%groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [AGG]",
            "%corr [--tsv] [--heatmap] <PATH>",
            "%scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>",
            "%lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>",
            "%hist [--tsv] [--bins N] <PATH> <COLUMN>",
            "%md <PATH>",
            "%html <PATH>",
            "%yaml <PATH>",
            "%toml <PATH>",
            "%xml <PATH>",
            "%ini <PATH>",
            "%properties <PATH>",
            "%envfile <PATH>",
            "%env <NAME>",
            "%addjar <PATH>",
            "%maven <G:A:V>",
            "%deps",
            "%whichclass <FQCN>"
    );

    private static final List<String> MAGIC_LINES = List.of(
            "%magics - list notebook magics and short usage hints",
            "%reset - clear notebook state and reload default Gollek helpers",
            "%time <expr> - run once and report elapsed time",
            "%timeit [-n N] <expr> - run multiple times and report avg/min/max timings",
            "%classpath - show the active notebook classpath",
            "%pwd - show the current working directory",
            "%cd <PATH> - change the notebook working directory",
            "%ls - list entries in the current working directory",
            "%tree [PATH] - show a shallow directory tree",
            "%du [PATH] - summarize file or directory disk usage",
            "%cat <PATH> - preview a text file",
            "%head [-n N] <PATH> - preview the first lines of a text file",
            "%tail [-n N] <PATH> - preview the last lines of a text file",
            "%grep <TEXT> [PATH] - search text in one file or a shallow directory tree",
            "%findfile <TEXT> [PATH] - search file and directory names",
            "%wc <PATH> - show file size and line stats",
            "%diff <LEFT_PATH> <RIGHT_PATH> - show a lightweight line diff for two text files",
            "%zipls <PATH> - list entries inside a zip archive",
            "%zipcat <ZIP_PATH> <ENTRY_PATH> - preview one text entry from a zip archive",
            "%tarls <PATH> - list entries inside a tar or tar.gz archive",
            "%tarcat <TAR_PATH> <ENTRY_PATH> - preview one text entry from a tar or tar.gz archive",
            "%extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH] - extract one archive entry into the notebook directory",
            "%sha256 <PATH> - compute a file SHA-256 digest",
            "%json <PATH> - pretty-print a JSON file",
            "%csv [-n N] [--profile] <PATH> - preview a CSV file as a dataframe-style table",
            "%tsv [-n N] [--profile] <PATH> - preview a TSV file as a dataframe-style table",
            "%sample [--tsv] [-n N] [--seed S] <PATH> - sample random rows from a CSV or TSV file",
            "%sort [--tsv] [-n N] [--desc] <PATH> <COLUMN> - preview rows sorted by a CSV or TSV column",
            "%filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE] - preview CSV or TSV rows matching a column predicate",
            "%describe [--tsv] <PATH> - summarize numeric columns from a CSV or TSV file",
            "%schema [--tsv] <PATH> - infer lightweight column types from a CSV or TSV file",
            "%missing [--tsv] <PATH> - summarize missing values by CSV or TSV column",
            "%valuecounts [--tsv] [--top N] <PATH> <COLUMN> - count distinct values in a CSV or TSV column",
            "%groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [AGG] - summarize grouped CSV or TSV data",
            "%corr [--tsv] [--heatmap] <PATH> - show a numeric Pearson correlation matrix from a CSV or TSV file",
            "%scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN> - render a quick scatter plot from numeric CSV or TSV columns",
            "%lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN> - render a quick line plot from a CSV or TSV file",
            "%hist [--tsv] [--bins N] <PATH> <COLUMN> - render a quick histogram from a numeric CSV or TSV column",
            "%md <PATH> - preview a Markdown file as rich text",
            "%html <PATH> - preview an HTML file",
            "%yaml <PATH> - preview a YAML file",
            "%toml <PATH> - preview a TOML file",
            "%xml <PATH> - preview an XML file",
            "%ini <PATH> - preview an INI file",
            "%properties <PATH> - preview a .properties file",
            "%envfile <PATH> - preview a .env-style file",
            "%env <NAME> - show one environment variable",
            "%addjar <PATH> - add one local jar to the live notebook classpath",
            "%maven [--allow-remote] [--fetch] [--explain] <G:A:V> - resolve an artifact from local cache, with optional guidance/fetch intent",
            "%deps - show dynamically added notebook dependencies",
            "%whichclass <FQCN> - show where a class was loaded from"
    );

    private static final Map<String, String> USAGES = Map.ofEntries(
            Map.entry("%help", "Usage: %help <magic>"),
            Map.entry("%cd", "Usage: %cd <PATH>"),
            Map.entry("%cat", "Usage: %cat <PATH>"),
            Map.entry("%head", "Usage: %head [-n N] <PATH>"),
            Map.entry("%tail", "Usage: %tail [-n N] <PATH>"),
            Map.entry("%grep", "Usage: %grep <TEXT> [PATH]"),
            Map.entry("%findfile", "Usage: %findfile <TEXT> [PATH]"),
            Map.entry("%wc", "Usage: %wc <PATH>"),
            Map.entry("%diff", "Usage: %diff <LEFT_PATH> <RIGHT_PATH>"),
            Map.entry("%zipls", "Usage: %zipls <PATH>"),
            Map.entry("%zipcat", "Usage: %zipcat <ZIP_PATH> <ENTRY_PATH>"),
            Map.entry("%tarls", "Usage: %tarls <PATH>"),
            Map.entry("%tarcat", "Usage: %tarcat <TAR_PATH> <ENTRY_PATH>"),
            Map.entry("%extract", "Usage: %extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]"),
            Map.entry("%sha256", "Usage: %sha256 <PATH>"),
            Map.entry("%json", "Usage: %json <PATH>"),
            Map.entry("%csv", "Usage: %csv [-n N] [--profile] <PATH>"),
            Map.entry("%tsv", "Usage: %tsv [-n N] [--profile] <PATH>"),
            Map.entry("%sample", "Usage: %sample [--tsv] [-n N] [--seed S] <PATH>"),
            Map.entry("%sort", "Usage: %sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>"),
            Map.entry("%filter", "Usage: %filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE]"),
            Map.entry("%valuecounts", "Usage: %valuecounts [--tsv] [--top N] <PATH> <COLUMN>"),
            Map.entry("%schema", "Usage: %schema [--tsv] <PATH>"),
            Map.entry("%missing", "Usage: %missing [--tsv] <PATH>"),
            Map.entry("%groupby", "Usage: %groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [count|sum|mean|min|max]"),
            Map.entry("%lineplot", "Usage: %lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>"),
            Map.entry("%scatter", "Usage: %scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>"),
            Map.entry("%hist", "Usage: %hist [--tsv] [--bins N] <PATH> <COLUMN>"),
            Map.entry("%describe", "Usage: %describe [--tsv] <PATH>"),
            Map.entry("%corr", "Usage: %corr [--tsv] [--heatmap] <PATH>"),
            Map.entry("%md", "Usage: %md <PATH>"),
            Map.entry("%html", "Usage: %html <PATH>"),
            Map.entry("%yaml", "Usage: %yaml <PATH>"),
            Map.entry("%toml", "Usage: %toml <PATH>"),
            Map.entry("%xml", "Usage: %xml <PATH>"),
            Map.entry("%ini", "Usage: %ini <PATH>"),
            Map.entry("%properties", "Usage: %properties <PATH>"),
            Map.entry("%envfile", "Usage: %envfile <PATH>"),
            Map.entry("%env", "Usage: %env <NAME>"),
            Map.entry("%addjar", "Usage: %addjar </absolute/or/relative/path/to/file.jar>"),
            Map.entry("%maven", "Usage: %maven [--allow-remote] [--fetch] [--explain] <groupId:artifactId:version>"),
            Map.entry("%whichclass", "Usage: %whichclass <fully.qualified.ClassName>"),
            Map.entry("%time", "Usage: %time <java-expression-or-cell>"),
            Map.entry("%timeit", "Usage: %timeit [-n N] <java-expression-or-cell>")
    );

    private static final Map<String, String> HELP = Map.ofEntries(
            Map.entry("%magics", """
                    %magics
                    Lists the supported notebook magics with short usage hints.
                    Use %help <magic> for one focused command.
                    """),
            Map.entry("%help", """
                    %help <magic>
                    Shows one focused help entry for a notebook magic.
                    Example: %help %maven
                    """),
            Map.entry("%reset", """
                    %reset
                    Clears notebook variables, declarations, and dynamic dependency loads.
                    Reloads the default Gollek imports and helper functions into a fresh JShell session.
                    """),
            Map.entry("%time", """
                    %time <java-expression-or-cell>
                    Runs once and reports elapsedMs.
                    Keeps rich notebook rendering when the expression returns a renderable value.
                    """),
            Map.entry("%timeit", """
                    %timeit [-n N] <java-expression-or-cell>
                    Runs the expression multiple times and reports runs, avgMs, minMs, and maxMs.
                    Default run count is 5.
                    """),
            Map.entry("%classpath", """
                    %classpath
                    Shows the active notebook classpath, including dynamically added entries.
                    """),
            Map.entry("%pwd", """
                    %pwd
                    Shows the current working directory of the notebook kernel process.
                    """),
            Map.entry("%cd", """
                    %cd <PATH>
                    Changes the notebook working directory used by %pwd and %ls.
                    Relative paths resolve from the current notebook working directory.
                    """),
            Map.entry("%ls", """
                    %ls
                    Lists entries in the current working directory.
                    """),
            Map.entry("%tree", """
                    %tree [PATH]
                    Shows a shallow directory tree for the current notebook directory or one target path.
                    Relative paths resolve from the current notebook working directory.
                    """),
            Map.entry("%du", """
                    %du [PATH]
                    Summarizes file count, directory count, and byte usage for the current notebook directory or one target path.
                    Directory scans are capped to keep notebook output responsive.
                    """),
            Map.entry("%cat", """
                    %cat <PATH>
                    Shows a UTF-8 text preview for one file from the current notebook directory or one target path.
                    Large files are truncated to keep notebook output manageable.
                    """),
            Map.entry("%head", """
                    %head [-n N] <PATH>
                    Shows the first lines of a UTF-8 text file.
                    Default line count is 10.
                    """),
            Map.entry("%tail", """
                    %tail [-n N] <PATH>
                    Shows the last lines of a UTF-8 text file.
                    Default line count is 10.
                    """),
            Map.entry("%grep", """
                    %grep <TEXT> [PATH]
                    Searches for plain text in one file or in a shallow directory tree.
                    If PATH is omitted, the current notebook directory is searched.
                    """),
            Map.entry("%findfile", """
                    %findfile <TEXT> [PATH]
                    Searches file and directory names for a plain text fragment.
                    If PATH is omitted, the current notebook directory is searched.
                    """),
            Map.entry("%wc", """
                    %wc <PATH>
                    Shows line, word, character, and byte counts for one UTF-8 text file.
                    """),
            Map.entry("%diff", """
                    %diff <LEFT_PATH> <RIGHT_PATH>
                    Shows a lightweight line diff for two UTF-8 text files.
                    Lines prefixed with - are only in the left file, and lines prefixed with + are only in the right file.
                    """),
            Map.entry("%zipls", """
                    %zipls <PATH>
                    Lists entries inside one zip archive from the notebook working directory.
                    Useful for packaged reports, bundles, and generated artifacts.
                    """),
            Map.entry("%zipcat", """
                    %zipcat <ZIP_PATH> <ENTRY_PATH>
                    Previews one UTF-8 text entry from a zip archive.
                    Useful after %zipls when you want to inspect one bundled file without unpacking the archive.
                    """),
            Map.entry("%tarls", """
                    %tarls <PATH>
                    Lists entries inside one tar or tar.gz archive from the notebook working directory.
                    Useful for packaged reports, bundles, and generated artifacts.
                    """),
            Map.entry("%tarcat", """
                    %tarcat <TAR_PATH> <ENTRY_PATH>
                    Previews one UTF-8 text entry from a tar or tar.gz archive.
                    Useful after %tarls when you want to inspect one bundled file without unpacking the archive.
                    """),
            Map.entry("%extract", """
                    %extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]
                    Extracts one zip, tar, tar.gz, or tgz archive entry into the notebook working directory.
                    Use --dry-run to inspect the planned output path before writing.
                    OUT_PATH is optional; when omitted, the archive entry path is used.
                    """),
            Map.entry("%sha256", """
                    %sha256 <PATH>
                    Computes the SHA-256 digest for one file from the notebook working directory.
                    Useful for verifying archives, extracted artifacts, and generated outputs.
                    """),
            Map.entry("%json", """
                    %json <PATH>
                    Pretty-prints one JSON file from the notebook working directory.
                    Falls back with an error if the file is not readable JSON text.
                    """),
            Map.entry("%csv", """
                    %csv [-n N] [--profile] <PATH>
                    Previews one CSV file as a dataframe-style notebook table.
                    Use -n N to control preview rows, and --profile for column stats.
                    The first row is treated as the header.
                    """),
            Map.entry("%tsv", """
                    %tsv [-n N] [--profile] <PATH>
                    Previews one TSV file as a dataframe-style notebook table.
                    Use -n N to control preview rows, and --profile for column stats.
                    The first row is treated as the header.
                    """),
            Map.entry("%sample", """
                    %sample [--tsv] [-n N] [--seed S] <PATH>
                    Samples random rows from one CSV or TSV file.
                    Use -n N to control sample size; the default is 5.
                    Use --seed S for reproducible sampling.
                    Use --tsv for tab-separated files; CSV is the default.
                    """),
            Map.entry("%sort", """
                    %sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>
                    Previews rows sorted by one CSV or TSV column.
                    Use -n N to control preview size; the default is 20.
                    Numeric columns are sorted numerically when all non-blank values are finite numbers.
                    Use --desc for descending order; blank values are kept at the end.
                    """),
            Map.entry("%filter", """
                    %filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE]
                    Previews rows matching one CSV or TSV column predicate.
                    Use -n N to control preview size; the default is 20.
                    Supported operators: ==, !=, >, >=, <, <=, contains, starts, ends, blank, notblank.
                    Numeric comparison operators only match finite numeric cells.
                    """),
            Map.entry("%describe", """
                    %describe [--tsv] <PATH>
                    Summarizes numeric columns from one CSV or TSV file.
                    Reports count, missing, mean, std, min, p25, median, p75, and max.
                    Use --tsv for tab-separated files; CSV is the default.
                    """),
            Map.entry("%schema", """
                    %schema [--tsv] <PATH>
                    Infers lightweight column types from one CSV or TSV file.
                    Types are empty, boolean, integer, decimal, date, datetime, or text.
                    Reports non-empty count, missing count, missing percent, and example values.
                    Use --tsv for tab-separated files; CSV is the default.
                    """),
            Map.entry("%missing", """
                    %missing [--tsv] <PATH>
                    Summarizes blank cells by CSV or TSV column.
                    Reports missing count, present count, and missing percent per column.
                    Includes an inline SVG bar chart for columns with missing values.
                    Use --tsv for tab-separated files; CSV is the default.
                    """),
            Map.entry("%valuecounts", """
                    %valuecounts [--tsv] [--top N] <PATH> <COLUMN>
                    Counts distinct values in one CSV or TSV column.
                    Use --top N to control how many values are shown; the default is 20.
                    Blank cells are shown as (blank), and output includes count and percent.
                    Use --tsv for tab-separated files; CSV is the default.
                    """),
            Map.entry("%groupby", """
                    %groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [count|sum|mean|min|max]
                    Groups CSV or TSV rows by GROUP_COLUMN.
                    Without VALUE_COLUMN, reports row counts per group.
                    With VALUE_COLUMN, defaults to mean and also supports sum, mean, min, and max.
                    Blank group keys are shown as (blank); non-numeric values are skipped for numeric aggregations.
                    """),
            Map.entry("%corr", """
                    %corr [--tsv] [--heatmap] <PATH>
                    Shows a Pearson correlation matrix for numeric CSV or TSV columns.
                    Correlations are computed pairwise from rows where both columns are numeric.
                    Use --heatmap to add an inline SVG heatmap above the matrix table.
                    Blank cells mean insufficient data or zero variance.
                    """),
            Map.entry("%scatter", """
                    %scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>
                    Renders a quick inline SVG scatter plot from one CSV or TSV file.
                    X_COLUMN and Y_COLUMN must both contain numeric values.
                    Rows with blank or non-numeric values in either column are skipped.
                    Use --tsv for tab-separated files; CSV is the default.
                    """),
            Map.entry("%lineplot", """
                    %lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>
                    Renders a quick inline SVG line plot from one CSV or TSV file.
                    X_COLUMN is used for labels. Y_COLUMN must contain numeric values.
                    Use --tsv for tab-separated files; CSV is the default.
                    """),
            Map.entry("%hist", """
                    %hist [--tsv] [--bins N] <PATH> <COLUMN>
                    Renders a quick inline SVG histogram from one numeric CSV or TSV column.
                    Use --bins N to control bin count. Use --tsv for tab-separated files; CSV is the default.
                    Non-numeric and blank values are skipped and reported.
                    """),
            Map.entry("%md", """
                    %md <PATH>
                    Previews one Markdown file as rich notebook HTML.
                    Also keeps the original markdown text in plain output.
                    """),
            Map.entry("%html", """
                    %html <PATH>
                    Previews one HTML file directly in notebook HTML output.
                    Also keeps the original HTML source in plain output.
                    """),
            Map.entry("%yaml", """
                    %yaml <PATH>
                    Previews one YAML file with notebook-friendly formatting.
                    Also keeps the original YAML source in plain output.
                    """),
            Map.entry("%toml", """
                    %toml <PATH>
                    Previews one TOML file with notebook-friendly formatting.
                    Also keeps the original TOML source in plain output.
                    """),
            Map.entry("%xml", """
                    %xml <PATH>
                    Previews one XML file with notebook-friendly formatting.
                    Also keeps the original XML source in plain output.
                    """),
            Map.entry("%ini", """
                    %ini <PATH>
                    Previews one INI file with notebook-friendly formatting.
                    Also keeps the original INI source in plain output.
                    """),
            Map.entry("%properties", """
                    %properties <PATH>
                    Previews one Java .properties file as key/value entries.
                    Also keeps the original source in plain output.
                    """),
            Map.entry("%envfile", """
                    %envfile <PATH>
                    Previews one .env-style file as key/value entries.
                    Comment and blank lines are ignored in the structured view.
                    """),
            Map.entry("%env", """
                    %env <NAME>
                    Shows one environment variable from the kernel process.
                    Example: %env PATH
                    """),
            Map.entry("%addjar", """
                    %addjar <PATH>
                    Adds one local jar to the live notebook classpath.
                    Example: %addjar /absolute/path/to/lib.jar
                    """),
            Map.entry("%maven", """
                    %maven [--allow-remote] [--fetch] [--explain] <groupId:artifactId:version>
                    Resolves an artifact from local Maven or Gradle caches.
                    --explain shows searched paths.
                    --allow-remote prints or enables remote-fetch intent.
                    --fetch requires GOLLEK_JUPYTER_ENABLE_REMOTE_MAVEN=true.
                    """),
            Map.entry("%deps", """
                    %deps
                    Shows dynamically added notebook dependency entries from %addjar and %maven.
                    """),
            Map.entry("%whichclass", """
                    %whichclass <fully.qualified.ClassName>
                    Shows where a class was loaded from in the live notebook runtime.
                    Example: %whichclass tech.kayys.gollek.ml.autograd.GradTensor
                    """)
    );

    private static final String UNKNOWN_MAGIC_LIST = "%magics, %help, %reset, %time, %timeit, %classpath, %pwd, %cd, %ls, %tree, %du, %cat, %head, %tail, %grep, %findfile, %wc, %diff, %zipls, %zipcat, %tarls, %tarcat, %extract, %sha256, %json, %csv, %tsv, %sample, %sort, %filter, %describe, %schema, %missing, %valuecounts, %groupby, %corr, %scatter, %lineplot, %hist, %md, %html, %yaml, %toml, %xml, %ini, %properties, %envfile, %env, %addjar, %maven, %deps, %whichclass";

    private NotebookMagicCatalog() {
    }

    static String bannerLine() {
        return "Magics: " + String.join(", ", BANNER_USAGES);
    }

    static DisplayData renderMagics() {
        String plain = String.join("\n", MAGIC_LINES);
        StringBuilder html = new StringBuilder()
                .append("<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'>")
                .append("<b>Notebook Magics</b><ul style='margin-top:6px'>");
        for (String line : MAGIC_LINES) {
            int sep = line.indexOf(" - ");
            if (sep > 0) {
                html.append("<li><code>")
                        .append(NotebookHtml.escape(line.substring(0, sep)))
                        .append("</code> - ")
                        .append(NotebookHtml.escape(line.substring(sep + 3)))
                        .append("</li>");
            } else {
                html.append("<li>").append(NotebookHtml.escape(line)).append("</li>");
            }
        }
        html.append("</ul></div>");
        DisplayData data = new DisplayData(plain);
        data.putData(MIMEType.TEXT_HTML, html.toString());
        return data;
    }

    static DisplayData renderUsage(String magic) {
        String usage = USAGES.get(magic);
        return usage == null ? null : new DisplayData(usage);
    }

    static DisplayData renderHelp(String topic) {
        if (topic == null || topic.isBlank()) {
            return new DisplayData("Usage: %help <magic>");
        }
        String normalized = topic.startsWith("%") ? topic : "%" + topic;
        String plain = HELP.get(normalized);
        if (plain == null) {
                return new DisplayData("Unknown help topic: " + topic + "\nTry %magics");
        }
        String html = "<div style='font-family:monospace;border:1px solid #ccc;padding:8px;border-radius:4px'><pre style='margin:0'>"
                + NotebookHtml.escape(plain)
                + "</pre></div>";
        DisplayData data = new DisplayData(plain.strip());
        data.putData(MIMEType.TEXT_HTML, html);
        return data;
    }

    static DisplayData unknownMagic(String magic) {
        return new DisplayData("Unknown magic: " + magic + "\nAvailable magics: " + UNKNOWN_MAGIC_LIST);
    }

}
