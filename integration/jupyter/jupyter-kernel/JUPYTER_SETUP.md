# Jupyter Setup Guide - Gollek SDK

Complete guide for setting up and using Gollek SDK with Jupyter notebooks.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Usage Examples](#usage-examples)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required
- Python 3.6+
- Jupyter 4.0+
- Java 11+ (ideally Java 25)
- Maven 3.6+

### Optional
- Conda (recommended for Python environment management)
- JupyterLab (modern Jupyter interface)

## Installation

> Status note
>
> The old IJava-based `gollek-sdk` kernelspec described in earlier versions of
> this document is no longer the recommended path. The maintained path is the
> standalone kernel in `../gollek-jupyter-kernel/`, which speaks the Jupyter
> wire protocol directly through JJava.

### Fast Path

```bash
cd gollek/integration/jupyter/gollek-jupyter-kernel
./rebuild_kernel.sh doctor
python3 install.py --dry-run
python3 selftest.py
python3 doctor.py
python3 sync_active_install.py --dry-run
python3 install.py --user
jupyter kernelspec list
```

The self-test now retries random connection-file ports and then runs a small
localhost bind probe before it gives up. If it reports
`environment-localhost-bind-restriction`, the remaining issue is the current
environment or sandbox rather than a normal stale-jar or simple port-collision
problem.

If you still enter through `jupyter-kernel/install.sh`, that script now
delegates to the standalone kernel installer.

### Step 1: Install Jupyter

**Using pip:**
```bash
pip install jupyter
```

**Using conda:**
```bash
conda install jupyter
```

### Step 2: Use The Standalone Gollek Kernel

```bash
cd gollek/integration/jupyter/gollek-jupyter-kernel
python3 install.py --dry-run
python3 install.py --user
jupyter kernelspec list
```

### Step 3: Verify Installation

You should see the standalone `gollek` kernelspec in the output.

## Quick Start

### 1. Start Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

### 2. Create New Notebook

- Click "New" button
- Select "Gollek SDK (Java)" from kernel list
- A new notebook opens with Java kernel

### 3. First Code Cell

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

Press Shift+Enter to execute.

Small notebook magics are available too:

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

`%reset` is the quickest in-notebook recovery path. It clears notebook
variables, declarations, and dynamic dependency loads, then reloads the
default Gollek imports and helper functions.

`%cd <PATH>` lets you move the notebook working directory in-session so `%pwd`
and `%ls` follow the new location immediately.

`%tree [PATH]` gives a quick shallow directory tree for the current notebook
directory or a target path.

`%du [PATH]` gives a quick file count, directory count, and byte summary for
the current notebook directory or a target path.

`%cat <PATH>` gives a small in-notebook text preview for one file after you
find it with `%tree`.

`%head [-n N] <PATH>` gives a tighter line-oriented preview when the file is
larger than you want to fully inspect with `%cat`.

`%tail [-n N] <PATH>` gives the matching end-of-file view for logs and longer
generated artifacts.

`%grep <TEXT> [PATH]` lets you search for plain text in one file or a shallow
directory tree without leaving the notebook.

`%findfile <TEXT> [PATH]` lets you search by filename or path fragment across a
small directory tree from inside the notebook.

`%wc <PATH>` gives a quick line/word/character/byte summary before you preview
the file contents in more detail.

`%diff <LEFT_PATH> <RIGHT_PATH>` gives a quick line-oriented comparison for two
text files directly in the notebook.

`%zipls <PATH>` gives a quick archive entry listing for zip bundles and other
packaged notebook-side artifacts.

`%zipcat <ZIP_PATH> <ENTRY_PATH>` gives a direct text preview for one bundled
file inside a zip archive.

`%tarls <PATH>` gives the same archive listing workflow for tar and tar.gz
bundles.

`%tarcat <TAR_PATH> <ENTRY_PATH>` gives a direct text preview for one bundled
file inside a tar or tar.gz archive.

`%extract [--dry-run] <ARCHIVE_PATH> <ENTRY_PATH> [OUT_PATH]` unpacks one
selected archive entry into the notebook working directory after an optional
dry-run check.

`%sha256 <PATH>` computes a file digest for archive, extracted-file, or
generated-artifact verification.

`%json <PATH>` gives a readable pretty-printed view of JSON configs and report
artifacts directly in the notebook.

`%csv [-n N] [--profile] <PATH>` gives a dataframe-style preview for
lightweight CSV artifacts directly in the notebook. `-n` controls preview row
count and `--profile` adds column-level non-empty, missing, and numeric summary
stats.

`%tsv [-n N] [--profile] <PATH>` gives the same style of preview for
tab-separated tabular artifacts.

`%sample [--tsv] [-n N] [--seed S] <PATH>` gives quick random rows from a CSV
or TSV artifact. `-n` controls sample size, and `--seed` makes the sample
reproducible in the notebook.

`%sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>` gives quick top/bottom row
previews by one CSV or TSV column. Numeric columns are sorted numerically when
possible, and blank values stay at the end.

`%filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE]` gives quick row subsets
by one CSV or TSV column. It supports equality, numeric comparisons, text
predicates, and `blank`/`notblank` checks.

`%describe [--tsv] <PATH>` gives pandas-style numeric summaries for CSV or TSV
artifacts: count, missing, mean, standard deviation, min, p25, median, p75, and
max.

`%schema [--tsv] <PATH>` gives a quick dtype-style view for CSV or TSV
artifacts. It infers empty, boolean, integer, decimal, date, datetime, or text
columns and includes missing counts plus example values.

`%missing [--tsv] <PATH>` gives a pandas-style missingness check for CSV or TSV
artifacts. It reports missing count, present count, missing percent, and an
inline SVG bar chart by column.

`%valuecounts [--tsv] [--top N] <PATH> <COLUMN>` gives pandas-style
categorical counts for a CSV or TSV column. Blank cells are shown as `(blank)`,
and the notebook output includes counts, percentages, and an inline SVG bar
chart.

`%groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [AGG]` gives quick
grouped summaries for CSV or TSV data. Without a value column it counts rows;
with a value column it defaults to `mean` and supports `count`, `sum`, `mean`,
`min`, and `max`.

`%corr [--tsv] [--heatmap] <PATH>` gives a Pearson correlation matrix for
numeric CSV or TSV columns. Correlations are computed pairwise from rows where
both columns are numeric. Add `--heatmap` for an inline SVG correlation
heatmap.

`%scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>` renders a quick inline SVG
relationship check from paired numeric CSV or TSV columns. Rows with blank or
non-numeric values in either column are skipped and reported.

`%lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>` renders a quick inline SVG
trend from a CSV or TSV artifact. The X column is used for labels and the Y
column must contain numeric values.

`%hist [--tsv] [--bins N] <PATH> <COLUMN>` renders a quick inline SVG
distribution view from a numeric CSV or TSV column. Non-numeric and blank
values are skipped and reported.

`%md <PATH>` gives a readable rich-text preview for Markdown notes and reports
directly in the notebook.

`%html <PATH>` gives a direct notebook preview for saved HTML reports and
generated pages.

`%yaml <PATH>` gives a more readable notebook-side preview for YAML config
files while preserving the original source in plain output.

`%toml <PATH>` gives the same kind of notebook-side preview for TOML config
files.

`%xml <PATH>` gives the same kind of notebook-side preview for XML config or
report files.

`%ini <PATH>` gives the same kind of notebook-side preview for INI config
files.

`%properties <PATH>` gives a structured notebook-side preview for Java
`.properties` config files.

`%envfile <PATH>` gives a structured notebook-side preview for `.env`-style
config files while ignoring comments and blank lines.

The standalone kernel now preloads these common notebook symbols, so they work
without a manual import cell:

- `GradTensor`
- `Linear`
- `ReLU`
- `Sequential`

The standalone kernel also renders:

- tensors with summary stats and small 2D heatmaps
- `Map<?, ?>` as a record table
- `List<Map<?, ?>>` as a simple metrics/data table
- inline SVG line charts for notebook metrics like training loss
- inline image preview for `BufferedImage`
- inline WAV audio preview for `NotebookAudioClip`

## Usage Examples

### Example 1: Building a Neural Network

```java
// Create model
var model = new Sequential(
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10)
);

System.out.println("Model created successfully!");
```

### Example 2: Tensor Creation

```java
var tensor = GradTensor.ones(2, 2);
System.out.println(tensor);
```

## Tips & Tricks

### 1. Use The Built-In Helper Surface

```java
gollekHelp();
gollekTensorDemo();
gollekPreviewTensorDemo();
gollekModelDemo();
gollekModelSummary();
```

### 2. Print Model Architecture

```java
var model = new Sequential(
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10)
);

System.out.println(model);
```

### 3. Count Parameters

```java
long paramCount = 0;
for (var param : model.parameters()) {
    paramCount += param.data().numel();
}
System.out.println("Total parameters: " + paramCount);
```

### 4. Use Markdown Cells for Documentation

- Click "+ Markdown" to add text cells
- Document your experiments
- Include math formulas with LaTeX: `$equation$`

### 5. Visualize with Print Statements

```java
System.out.println("Epoch: " + epoch);
System.out.println("Loss: " + lossVal);
System.out.println("Accuracy: " + accuracy.compute());
```

## Troubleshooting

### Issue: Kernel not found

**Solution:**
```bash
cd gollek/integration/jupyter/gollek-jupyter-kernel
python3 doctor.py
python3 sync_active_install.py --dry-run
python3 install.py --user
```

### Issue: Import errors (class not found)

**Solution:**
```bash
cd gollek/integration/jupyter/gollek-jupyter-kernel
./rebuild_kernel.sh local-fat
python3 doctor.py
python3 sync_active_install.py --dry-run
python3 install.py --user
```

### Issue: Out of memory (OOM)

**Solution:**
```bash
cd gollek/integration/jupyter/gollek-jupyter-kernel
# Increase Java heap size in the installed kernelspec's kernel.json
"argv": [
    "java",
    "-Xmx2G",  # Increase heap to 2GB
    "-jar",
    ...
]
```

### Issue: Very slow first execution

**Solution:**
- First execution compiles bytecode (normal)
- Subsequent cells run faster
- Be patient with first few imports

## Performance Tips

1. **Batch Operations**: Use batch processing instead of single samples
2. **Avoid Re-imports**: Import once at the start
3. **Use Variables**: Store model references to avoid recreating
4. **Chunk Processing**: For large datasets, process in chunks

## Next Steps

1. Review [example notebooks](../examples/notebooks/)
2. Check [API reference](../API_REFERENCE.md)
3. Try your own experiments!

## Additional Resources

- [IJava GitHub](https://github.com/SpencerPark/IJava)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Gollek SDK Documentation](../README.md)

## Support

For issues or questions:
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review example notebooks
3. Check Gollek SDK documentation

---

**Happy Data Science with Gollek SDK! 🚀**
