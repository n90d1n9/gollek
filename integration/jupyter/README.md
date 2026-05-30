# Gollek Jupyter Integration

The maintained Jupyter path is the standalone kernel in:

- `gollek-jupyter-kernel/`

That kernel speaks the Jupyter wire protocol directly through JJava and runs
Gollek code in-process through JShell. It does not depend on the older IJava
classpath-based SDK kernelspec.

## Recommended Install Flow

```bash
cd gollek/integration/jupyter/gollek-jupyter-kernel
./rebuild_kernel.sh doctor
python3 install.py --dry-run
python3 selftest.py
python3 doctor.py
python3 sync_active_install.py --dry-run
python3 install.py --user
```

`selftest.py` now distinguishes a real kernel-side ZMQ startup boundary from a
broader localhost socket restriction in the current environment. That matters in
sandboxed setups where repeated `Address already in use` errors can actually be
caused by bind restrictions rather than a broken kernel jar.

If you already use the older helper:

```bash
cd gollek/integration/jupyter/jupyter-kernel
bash install.sh --user
```

That path now delegates to the standalone kernel installer instead of copying
the stale IJava kernelspec.

## Current Layout

- `gollek-jupyter-kernel/`
  - standalone Gollek kernel implementation
  - canonical `kernel.json` template
  - canonical installer
- `jupyter-kernel/`
  - legacy compatibility shim
  - older docs kept for migration context

## First Notebook Cell

After selecting **Gollek (Java 25 + AI/ML)**, the recommended first-cell path is
now the built-in helper flow from the standalone kernel:

```java
gollekHelp();
gollekTensorDemo().shape()[0];
gollekPreviewTensorDemo();
gollekMetricsDemo();
gollekLossCurveDemo();
gollekImageDemo();
gollekAudioDemo();
gollekModelSummary();
```

Notebook magics are available too:

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

Use `%reset` when you want a clean notebook session without restarting the
kernel. It clears JShell state and dynamic `%addjar` / `%maven` loads, then
reloads the default Gollek helpers.

Use `%cd <PATH>` when you want notebook-local directory changes for `%pwd` and
`%ls` without leaving the kernel.

Use `%tree [PATH]` when you want a quick shallow project-structure view from
the notebook.

Use `%du [PATH]` when you want a quick file count, directory count, and byte
summary for the current notebook directory or a target path.

Use `%cat <PATH>` when you want a small text preview right inside the notebook
after navigating with `%cd` or `%tree`.

Use `%head [-n N] <PATH>` when you only want the first few lines of a larger
text file.

Use `%tail [-n N] <PATH>` when you want the end of a log or generated file
without dumping the whole thing.

Use `%grep <TEXT> [PATH]` when you want quick in-notebook text discovery across
one file or a small directory tree.

Use `%findfile <TEXT> [PATH]` when you want to search by filename or path
fragment instead of file contents.

Use `%wc <PATH>` when you want a quick sense of file size and text density
before choosing `%cat`, `%head`, or `%tail`.

Use `%diff <LEFT_PATH> <RIGHT_PATH>` when you want a quick in-notebook line
comparison for configs, reports, or generated text artifacts.

Use `%zipls <PATH>` when you want to inspect packaged zip outputs before
deciding whether to unpack or open them elsewhere.

Use `%zipcat <ZIP_PATH> <ENTRY_PATH>` when you already know which bundled text
file you want to inspect inside the archive.

Use `%tarls <PATH>` when packaged outputs arrive as tar or tar.gz archives and
you want a quick entry listing inside the notebook.

Use `%tarcat <TAR_PATH> <ENTRY_PATH>` when you already know which bundled text
file you want to inspect inside a tar or tar.gz archive.

Use `%extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]` when you want
to unpack one selected archive entry into the notebook working directory after
checking the planned output path.

Use `%sha256 <PATH>` when you want to verify an archive, extracted file, or
generated artifact directly from the notebook.

Use `%json <PATH>` when you want structured config or report files to be
readable without leaving the notebook.

Use `%csv [-n N] [--profile] <PATH>` when you want a dataframe-style preview
for lightweight tabular artifacts without leaving the notebook. `-n` controls
preview row count and `--profile` adds column-level non-empty, missing, and
numeric summary stats.

Use `%tsv [-n N] [--profile] <PATH>` when the same kind of tabular artifact is
tab-separated instead of comma-separated.

Use `%sample [--tsv] [-n N] [--seed S] <PATH>` when you want quick random rows
from a CSV or TSV artifact. `-n` controls sample size, and `--seed` makes the
sample reproducible in the notebook.

Use `%sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>` when you want quick
top/bottom row previews by one CSV or TSV column. Numeric columns are sorted
numerically when possible, and blank values stay at the end.

Use `%filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE]` when you want quick
row subsets by one CSV or TSV column. It supports equality, numeric
comparisons, text predicates, and `blank`/`notblank` checks.

Use `%describe [--tsv] <PATH>` when you want pandas-style numeric summaries
for CSV or TSV artifacts: count, missing, mean, standard deviation, min, p25,
median, p75, and max.

Use `%schema [--tsv] <PATH>` when you want a quick dtype-style view for CSV or
TSV artifacts. It infers empty, boolean, integer, decimal, date, datetime, or
text columns and includes missing counts plus example values.

Use `%missing [--tsv] <PATH>` when you want a pandas-style missingness check
for CSV or TSV artifacts. It reports missing count, present count, missing
percent, and an inline SVG bar chart by column.

Use `%valuecounts [--tsv] [--top N] <PATH> <COLUMN>` when you want
pandas-style categorical counts from a CSV or TSV column. Blank cells are shown
as `(blank)`, and the notebook output includes counts, percentages, and an
inline SVG bar chart.

Use `%groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [AGG]` when you
want quick grouped summaries. Without a value column it counts rows; with a
value column it defaults to `mean` and supports `count`, `sum`, `mean`, `min`,
and `max`.

Use `%corr [--tsv] [--heatmap] <PATH>` when you want a Pearson correlation
matrix for numeric CSV or TSV columns. Correlations are computed pairwise from
rows where both columns are numeric. Add `--heatmap` for an inline SVG
correlation heatmap.

Use `%scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>` when you want a quick
inline SVG relationship check from paired numeric CSV or TSV columns. Rows with
blank or non-numeric values in either column are skipped and reported.

Use `%lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>` when you want a quick
inline SVG trend from a CSV or TSV artifact. The X column is used for labels
and the Y column must contain numeric values.

Use `%hist [--tsv] [--bins N] <PATH> <COLUMN>` when you want a quick inline SVG
distribution view from a numeric CSV or TSV column. Non-numeric and blank
values are skipped and reported.

Use `%md <PATH>` when you want human-readable Markdown reports or notes to be
rendered as rich notebook content instead of plain text.

Use `%html <PATH>` when you want saved HTML reports or generated pages to
render directly in the notebook.

Use `%yaml <PATH>` when you want YAML configs to be more readable than a raw
text dump while staying inside the notebook.

Use `%toml <PATH>` when you want TOML configs to get the same structured,
notebook-friendly preview flow.

Use `%xml <PATH>` when you want XML configs or report artifacts to get the
same structured, notebook-friendly preview flow.

Use `%ini <PATH>` when you want INI-style configs to get the same structured,
notebook-friendly preview flow.

Use `%properties <PATH>` when you want Java-style config files rendered as
structured key/value entries instead of plain text.

Use `%envfile <PATH>` when you want `.env`-style config files rendered as
structured key/value entries instead of plain text.

Those helpers work without a manual import cell because the kernel preloads:

- `GradTensor`
- `Linear`
- `ReLU`
- `Sequential`

The standalone kernel now also renders a few common notebook inspection shapes
more naturally:

- tensors with summary stats plus small heatmaps
- `Map<?, ?>` as a record table
- `List<Map<?, ?>>` as a simple metrics/data table
- inline SVG line charts for notebook metrics like training loss
- inline image preview for `BufferedImage`
- inline WAV audio preview for `NotebookAudioClip`
