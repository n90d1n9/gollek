# gollek-kernel

Standalone Jupyter kernel for the Gollek AI platform.

Extends [JJava](https://github.com/dflib/jjava)'s `BaseKernel` (Jupyter wire
protocol over ZMQ/JeroMQ) and executes notebook cells through JShell. There is
no IJava dependency and no separate server process.

## Architecture

```
Jupyter ←──ZMQ──→ KernelLauncher
                       │
                  GollekKernel (extends BaseKernel)
                  ├── CodeEvaluator (JShell)
                  ├── CompletionProvider (JShell sourceCodeAnalysis)
                  ├── NotebookMagicDispatcher (exact/prefix magic routing)
                  ├── NotebookMagicArgs (shared magic option parsing)
                  ├── NotebookRuntimeMagicOptions (runtime magic option parsing)
                  ├── NotebookTableMagicOptions (table/data magic option records and parsers)
                  ├── NotebookStartupScripts (preloaded helper source)
                  ├── NotebookMagicCatalog (banner, usage, help, discovery)
                  ├── NotebookRuntimeRenderer (reset, timing, environment, and class-location previews)
                  ├── NotebookHtml (shared notebook HTML escaping)
                  ├── NotebookDependencies (classpath helpers, Maven option parsing, and artifact lookup)
                  ├── NotebookDependencyRenderer (classpath and Maven preview envelopes)
                  ├── NotebookDataLoader (shared delimited file validation and loading)
                  ├── NotebookDocuments (JSON formatting, Markdown conversion, and syntax highlighting helpers)
                  ├── NotebookDocumentRenderer (JSON, Markdown, HTML, config, properties, and env file preview envelopes)
                  ├── NotebookFileMagicOptions (filesystem and archive magic option parsing)
                  ├── NotebookFiles (filesystem measurement, text preview, and search helpers)
                  ├── NotebookFileRenderer (filesystem magic preview envelopes)
                  ├── NotebookArchives (ZIP/TAR listing and extraction bytes)
                  ├── NotebookArchiveRenderer (archive listing, entry, and extraction preview envelopes)
                  ├── NotebookTables (CSV/TSV parsing and stat formatting)
                  ├── NotebookPreview (plain text and HTML preview payload)
                  ├── NotebookTableRenderer (table rows, CSV/TSV, sample, sort, and filter previews)
                  ├── NotebookSummaryRenderer (schema, missing, value-count, and group-by previews)
                  ├── NotebookTableOps (sampling, sorting, and filtering helpers)
                  ├── NotebookStats (profiles, schema inference, numeric summaries, correlations, and aggregations)
                  ├── NotebookStatsRenderer (describe and correlation preview envelopes)
                  ├── NotebookCharts (notebook SVG chart rendering)
                  ├── NotebookChartRenderer (line, scatter, and histogram preview envelopes)
                  ├── TensorDisplay (HTML heatmap rendering)
                  └── Gollek ML libraries on the kernel classpath
```

`GollekKernel` owns kernel lifecycle and runtime execution. Keep reusable
routing, argument parsing, runtime option parsing, table/data option parsing, file/archive option parsing, catalog, preload, runtime preview rendering, dependency parsing and lookup, dependency preview rendering, delimited data
loading, document rendering, document preview rendering, file utilities, file preview rendering, archive parsing, table parsing,
archive preview rendering, table rendering, summary rendering, stats preview rendering, chart preview rendering, table operations, profiles, schema inference, numeric stats, chart rendering, and HTML
utilities outside it so new notebook magics can evolve without turning the
kernel class back into a registry plus documentation dump.

## Build & Install

```bash
# 1. Inspect the local rebuild environment
./rebuild_kernel.sh doctor

# 2. Fastest fallback: assemble a fresh standalone jar without Maven or Gradle
./rebuild_kernel.sh local-fat

# 3. Seed a writable local Maven cache from ~/.m2 when needed
./rebuild_kernel.sh seed-home-m2

# 4. Preferred when Gradle is healthy: build the standalone fat jar with Gradle
./rebuild_kernel.sh gradle-fat

# 5. Fallback: use the older Maven shaded build when Gradle is unavailable
./rebuild_kernel.sh maven

# 6. If Maven still fails offline, diagnose exactly what is missing
./rebuild_kernel.sh maven-diagnose

# 7. Inspect the install plan
cd gollek/integration/jupyter/gollek-jupyter-kernel
python3 install.py --dry-run

# 8. Verify the launcher path before opening Jupyter
python3 selftest.py

# 9. Inspect what the installed kernelspec currently points at
python3 doctor.py

# 10. If doctor reports drift, preview a safe active-kernel refresh
python3 sync_active_install.py --dry-run

# 11. Install the kernel
python3 install.py --user

# 12. Verify the kernelspec
jupyter kernelspec list
```

`rebuild_kernel.sh` keeps writable local caches under:

- `.cache/m2`
- `.cache/gradle`

That avoids relying on global cache directories when the local environment or
sandbox makes those unwritable.

The Gradle path now has a dedicated `fatJar` task that writes a standalone
`build/libs/gollek-kernel.jar`, which is also one of the installer's preferred
jar candidates.

For environments where Gradle startup or Maven resolution is the real blocker,
`build_local_standalone.sh` and `./rebuild_kernel.sh local-fat` can assemble a
fresh `build/libs/gollek-kernel.jar` directly from local source plus jars
already present in your Maven repository.

By default, the Maven rebuild path prefers offline mode after seeding the local
cache. Override that with `GOLLEK_JUPYTER_MAVEN_OFFLINE=false` if you explicitly
want Maven to try network resolution.

If the focused seed still misses parent POMs or plugin BOMs in your local
offline cache, retry with:

```bash
GOLLEK_JUPYTER_SEED_MODE=full ./rebuild_kernel.sh seed-home-m2
GOLLEK_JUPYTER_SEED_M2=true GOLLEK_JUPYTER_SEED_MODE=full ./rebuild_kernel.sh maven
```

If you want the helper to tell you whether a missing artifact exists in your
home Maven repository versus only being absent from the local seeded cache, run:

```bash
./rebuild_kernel.sh maven-diagnose
```

The installer resolves the runnable kernel jar from the standalone kernel
module. It currently checks these paths in order:

- `target/gollek-kernel.jar`
- `build/libs/gollek-kernel.jar`
- `target/original-gollek-kernel.jar`
- other `build/libs/*kernel*.jar` candidates

For safety, the installer and self-test reject stale jars by default when the
source tree is newer than the built artifact. Rebuild first if that happens.
Use `--allow-stale-jar` only when you intentionally want to test an older jar.

If you built the jar somewhere else, override it explicitly:

```bash
python3 install.py --user --jar /path/to/gollek-kernel.jar
```

You can run the same override through the startup self-test:

```bash
python3 selftest.py --jar /path/to/gollek-kernel.jar
```

To inspect what Jupyter will actually launch, use:

```bash
python3 doctor.py
```

That helper checks:

- whether the `gollek` kernelspec is installed
- which `kernel.json` Jupyter will resolve
- which jar the installed `argv` points at
- whether that jar is stale relative to current source
- whether the installed kernelspec is still drifting from the freshest local build

`install.py --user` now resolves the install base from `jupyter --data-dir`
instead of assuming a Linux-style `~/.local/share/jupyter`. That matters on
macOS, where the active user kernels typically live under `~/Library/Jupyter`.

If `doctor.py` finds an older copied jar or a mismatched `kernel.json`, you can
repair the active kernelspec in place with:

```bash
python3 sync_active_install.py --dry-run
python3 sync_active_install.py
```

That helper:

- targets the kernelspec Jupyter is actually resolving
- refreshes the copied `gollek-kernel.jar`
- rewrites `kernel.json` to point at the copied install-directory jar
- makes timestamped backups before replacing existing files

When `selftest.py` sees repeated random-port ZMQ bind failures, it now runs a
small localhost TCP bind probe and reports a more specific verdict:

- `environment-localhost-bind-restriction`
  - even plain Python localhost binds failed
  - this points to the current environment or sandbox rather than a normal port collision
- `jeroMQ-or-kernel-bind-boundary`
  - plain Python binds worked, but JJava/JeroMQ startup still failed
  - this points more directly at the Java-side kernel networking stack

That makes the self-test much more truthful than the earlier generic
`Address already in use` loop.

The older `../jupyter-kernel/install.sh` helper is still available, but it now
delegates to this installer.

## Usage

Open any notebook and select **Gollek (Java 25 + AI/ML)** as the kernel.

Cells run as plain Java snippets through JShell, so imports and variables work
the way they do in a JShell session:

```java
import tech.kayys.gollek.ml.nn.Linear;

var layer = new Linear(8, 4);
System.out.println(layer);
```

The kernel now preloads a small default Gollek notebook import set, so a fresh
session can use these symbols without a manual import cell:

- `GradTensor`
- `Linear`
- `ReLU`
- `Sequential`

That means first cells like these work immediately:

```java
var t = GradTensor.ones(2, 2);
new Sequential(new Linear(2, 3), new ReLU()).size();
```

The kernel also preloads a few tiny startup helpers for the first notebook cell:

- `gollekHelp()`
- `gollekTensorDemo()`
- `gollekPreviewTensorDemo()`
- `gollekMetricsDemo()`
- `gollekLossCurveDemo()`
- `gollekImageDemo()`
- `gollekAudioDemo()`
- `gollekModelDemo()`
- `gollekModelSummary()`

Example:

```java
gollekHelp();
gollekTensorDemo().shape()[0];
gollekPreviewTensorDemo();
gollekMetricsDemo();
gollekLossCurveDemo();
gollekImageDemo();
gollekAudioDemo();
gollekModelDemo().size();
gollekModelSummary();
```

The kernel also supports a small starter magic layer:

- `%magics`
- `%help <magic>`
- `%reset`
- `%time <java-expression-or-cell>`
- `%timeit [-n N] <java-expression-or-cell>`
- `%classpath`
- `%pwd`
- `%cd <PATH>`
- `%ls`
- `%tree [PATH]`
- `%du [PATH]`
- `%cat <PATH>`
- `%head [-n N] <PATH>`
- `%tail [-n N] <PATH>`
- `%grep <TEXT> [PATH]`
- `%findfile <TEXT> [PATH]`
- `%wc <PATH>`
- `%diff <LEFT_PATH> <RIGHT_PATH>`
- `%zipls <PATH>`
- `%zipcat <ZIP_PATH> <ENTRY_PATH>`
- `%tarls <PATH>`
- `%tarcat <TAR_PATH> <ENTRY_PATH>`
- `%extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]`
- `%sha256 <PATH>`
- `%json <PATH>`
- `%csv [-n N] [--profile] <PATH>`
- `%tsv [-n N] [--profile] <PATH>`
- `%sample [--tsv] [-n N] [--seed S] <PATH>`
- `%sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>`
- `%filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE]`
- `%describe [--tsv] <PATH>`
- `%schema [--tsv] <PATH>`
- `%missing [--tsv] <PATH>`
- `%valuecounts [--tsv] [--top N] <PATH> <COLUMN>`
- `%groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [AGG]`
- `%corr [--tsv] [--heatmap] <PATH>`
- `%scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>`
- `%lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>`
- `%hist [--tsv] [--bins N] <PATH> <COLUMN>`
- `%md <PATH>`
- `%html <PATH>`
- `%yaml <PATH>`
- `%toml <PATH>`
- `%xml <PATH>`
- `%ini <PATH>`
- `%properties <PATH>`
- `%envfile <PATH>`
- `%env <NAME>`
- `%addjar <PATH>`
- `%maven <G:A:V>`
- `%deps`
- `%whichclass <FQCN>`

Example:

```java
%magics
%help %reset
%help %cd
%help %tree
%help %du
%help %cat
%help %head
%help %tail
%help %grep
%help %findfile
%help %wc
%help %diff
%help %zipls
%help %zipcat
%help %tarls
%help %tarcat
%help %extract
%help %sha256
%help %json
%help %csv
%help %tsv
%help %sample
%help %sort
%help %filter
%help %describe
%help %schema
%help %missing
%help %valuecounts
%help %groupby
%help %corr
%help %scatter
%help %lineplot
%help %hist
%help %md
%help %html
%help %yaml
%help %toml
%help %xml
%help %ini
%help %properties
%help %envfile
%help %maven
%reset
%time gollekLossCurveDemo()
%timeit -n 3 gollekLossCurveDemo()
%classpath
%pwd
%cd ..
%ls
%tree
%du .
%cat README.md
%head -n 5 README.md
%tail -n 5 README.md
%grep Gollek README.md
%findfile README .
%wc README.md
%diff baseline.yaml current.yaml
%zipls bundle.zip
%zipcat bundle.zip nested/report.txt
%tarls bundle.tar.gz
%tarcat bundle.tar.gz nested/report.txt
%extract --dry-run bundle.tar.gz nested/report.txt extracted/report.txt
%extract bundle.tar.gz nested/report.txt extracted/report.txt
%sha256 bundle.tar.gz
%json config.json
%csv -n 20 --profile data.csv
%tsv -n 20 --profile data.tsv
%sample -n 5 --seed 42 data.csv
%sample --tsv -n 5 metrics.tsv
%sort -n 10 --desc data.csv score
%sort --tsv -n 10 metrics.tsv latency
%filter -n 10 data.csv score >= 0.8
%filter --tsv metrics.tsv status == ok
%describe data.csv
%describe --tsv metrics.tsv
%schema data.csv
%schema --tsv metrics.tsv
%missing data.csv
%missing --tsv metrics.tsv
%valuecounts data.csv label
%valuecounts --tsv --top 10 metrics.tsv status
%groupby data.csv label
%groupby data.csv label score mean
%groupby --tsv metrics.tsv status latency sum
%corr data.csv
%corr --tsv metrics.tsv
%corr --heatmap data.csv
%scatter data.csv weight height
%scatter --tsv metrics.tsv step accuracy
%lineplot data.csv epoch loss
%lineplot --tsv metrics.tsv step accuracy
%hist --bins 12 data.csv loss
%hist --tsv --bins 12 metrics.tsv accuracy
%md README.md
%html report.html
%yaml config.yaml
%toml config.toml
%xml config.xml
%ini config.ini
%properties app.properties
%envfile .env
%env PATH
%addjar /absolute/path/to/local-lib.jar
%maven tech.kayys.gollek:gollek-ml-autograd:0.1.0-SNAPSHOT
%deps
%whichclass tech.kayys.gollek.ml.autograd.GradTensor
%maven --explain definitely.missing:artifact:0.0.1
%maven --allow-remote definitely.missing:artifact:0.0.1
%maven --allow-remote --fetch definitely.missing:artifact:0.0.1
```

`%reset` clears notebook variables, declarations, and dynamic dependency loads
from `%addjar` / `%maven`, then reloads the default Gollek imports and helpers
into a fresh JShell session.

`%cd <PATH>` changes the notebook working directory used by `%pwd` and `%ls`.
Relative paths resolve from the current notebook directory.

`%tree [PATH]` shows a shallow directory tree for the current notebook
directory or one target path.

`%du [PATH]` summarizes file count, directory count, and total bytes for the
current notebook directory or one target path. Directory scans are capped to
keep notebook output responsive.

`%cat <PATH>` previews a UTF-8 text file from the current notebook directory or
one target path, with truncation for larger files.

`%head [-n N] <PATH>` previews the first lines of a UTF-8 text file when you
want a more focused view than `%cat`.

`%tail [-n N] <PATH>` previews the last lines of a UTF-8 text file, which is
especially useful for logs and generated outputs.

`%grep <TEXT> [PATH]` searches for plain text in one file or in a shallow
directory tree from the notebook.

`%findfile <TEXT> [PATH]` searches file and directory names by path fragment
from the notebook.

`%wc <PATH>` shows line, word, character, and byte counts for one text file
before you decide how to preview it.

`%diff <LEFT_PATH> <RIGHT_PATH>` shows a lightweight line diff for two UTF-8
text files, which is useful for configs and generated report artifacts.

`%zipls <PATH>` lists entries inside a zip archive so packaged bundles and
generated artifacts can be inspected without unpacking them first.

`%zipcat <ZIP_PATH> <ENTRY_PATH>` previews one UTF-8 text entry from a zip
archive after `%zipls` helps you find the bundled file you want.

`%tarls <PATH>` lists entries inside a tar or tar.gz archive so packaged
outputs can be inspected from the notebook without unpacking them first.

`%tarcat <TAR_PATH> <ENTRY_PATH>` previews one UTF-8 text entry from a tar or
tar.gz archive after `%tarls` helps you find the bundled file you want.

`%extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]` extracts one zip
or tar entry into the notebook working directory. Use `--dry-run` first to
check the planned output path without writing.

`%sha256 <PATH>` computes a file SHA-256 digest for verifying archives,
extracted artifacts, and generated outputs.

`%json <PATH>` pretty-prints one JSON file for easier notebook-side inspection
of configs and report artifacts.

`%csv [-n N] [--profile] <PATH>` previews a CSV file as a dataframe-style
notebook table for quick tabular artifact inspection. Use `-n` to control the
preview row count and `--profile` to add column-level non-empty, missing, and
numeric summary stats.

`%tsv [-n N] [--profile] <PATH>` previews a TSV file with the same
dataframe-style table and optional profile workflow.

`%sample [--tsv] [-n N] [--seed S] <PATH>` samples random rows from a CSV or
TSV file. Use `-n` to control sample size and `--seed` for reproducible
notebook output.

`%sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>` previews rows sorted by one
CSV or TSV column. Numeric columns are sorted numerically when possible, and
blank values stay at the end for quick top/bottom checks.

`%filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE]` previews rows matching
one CSV or TSV column predicate. Supported operators include `==`, `!=`, `>`,
`>=`, `<`, `<=`, `contains`, `starts`, `ends`, `blank`, and `notblank`;
numeric comparisons only match finite numeric cells.

`%describe [--tsv] <PATH>` summarizes numeric columns with count, missing,
mean, standard deviation, min, p25, median, p75, and max.

`%schema [--tsv] <PATH>` infers lightweight column types from CSV or TSV data:
empty, boolean, integer, decimal, date, datetime, or text. Output includes
non-empty count, missing count, missing percent, and example values.

`%missing [--tsv] <PATH>` summarizes blank cells by CSV or TSV column with
missing count, present count, missing percent, and an inline SVG missingness bar
chart.

`%valuecounts [--tsv] [--top N] <PATH> <COLUMN>` counts distinct values in a
CSV or TSV column, including blank cells as `(blank)`. Output includes counts,
percentages, and an inline SVG bar chart for the top values.

`%groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [AGG]` summarizes
groups in CSV or TSV data. Without `VALUE_COLUMN`, it reports row counts per
group; with `VALUE_COLUMN`, it defaults to `mean` and supports `count`, `sum`,
`mean`, `min`, and `max`.

`%corr [--tsv] [--heatmap] <PATH>` shows a Pearson correlation matrix for
numeric CSV or TSV columns. Correlations are computed pairwise from rows where
both columns are numeric. Use `--heatmap` to add an inline SVG heatmap above
the matrix table.

`%scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>` renders a quick inline SVG
scatter plot from paired numeric CSV or TSV columns. Rows with blank or
non-numeric values in either column are skipped and reported.

`%lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>` renders a quick inline SVG
line plot from a CSV or TSV file. The X column is used for labels and the Y
column must contain numeric values.

`%hist [--tsv] [--bins N] <PATH> <COLUMN>` renders a quick inline SVG
histogram from a numeric CSV or TSV column. Non-numeric and blank values are
skipped and reported.

`%md <PATH>` previews a Markdown file as rich notebook HTML while keeping the
original markdown in plain text output.

`%html <PATH>` previews a saved HTML file directly in notebook HTML output
while keeping the original source in plain text.

`%yaml <PATH>` previews a YAML file with notebook-friendly formatting while
keeping the original source in plain text.

`%toml <PATH>` previews a TOML file with notebook-friendly formatting while
keeping the original source in plain text.

`%xml <PATH>` previews an XML file with notebook-friendly formatting while
keeping the original source in plain text.

`%ini <PATH>` previews an INI file with notebook-friendly formatting while
keeping the original source in plain text.

`%properties <PATH>` previews a Java `.properties` file as notebook-side
key/value entries.

`%envfile <PATH>` previews a `.env`-style config file as notebook-side
key/value entries while ignoring comments and blank lines.

`%maven` currently resolves from local caches only:

- `~/.m2/repository`
- Gradle module cache under `~/.gradle/caches/modules-2/files-2.1`

If an artifact is missing:

- `%maven --explain ...` shows the exact local paths the kernel searched
- `%maven --allow-remote ...` prints the recommended `mvn dependency:get` command and the retry command, but does not perform network resolution inside the kernel
- `%maven --allow-remote --fetch ...` is the explicit remote-fetch path, but it is disabled by default unless `GOLLEK_JUPYTER_ENABLE_REMOTE_MAVEN=true` is set in the notebook environment

The notebook renderer now also recognizes simple metrics/data-exploration shapes:

- `GradTensor` and current core tensors with:
  - shape, dtype, device
  - min/max/sample values
  - small 2D heatmap preview
- `Map<?, ?>` as a key/value record table
- `List<Map<?, ?>>` as a lightweight metrics/data table
- `NotebookLineChart` as an inline SVG line chart for notebook metrics like loss curves
- `BufferedImage` as an inline PNG preview
- `NotebookAudioClip` as an inline WAV audio preview

## JVM options

Edit `kernel.json` (in the installed kernelspec directory) to tune memory:

```json
"-Xmx8g",
"-XX:MaxDirectMemorySize=4g"
```

## License

Apache 2.0
