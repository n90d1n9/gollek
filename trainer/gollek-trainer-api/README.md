This module contains the canonical trainer-facing contracts for the Gradle
first migration:

- `TrainerSession`
- `TrainerConfig`
- `TrainingListener`
- `TrainingSummary`

It is intentionally small and dependency-light so trainer runtimes, examples,
and integrations can depend on it without pulling the whole ML framework.
