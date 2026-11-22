# Contributing

TeamBench is authored by Kevin Fairley (Provefirst). External pull requests are welcome for bug fixes and new *deterministic* tasks.

## Rules for a new task

1. The grader must be a pure function of the final world state and the transcript. No model-as-judge.
2. The scripted mock must complete or fail the task without network access.
3. Add a unit test for at least one passing world and one failing world.
4. Document the task in `METHODOLOGY.md` and the README suite table.
5. Re-run the committed mock benchmark and update `results/mock.json` plus the generated report.

## Local checks

```bash
python -m pip install -e ".[dev]"
ruff check src tests
ruff format --check src tests
mypy --strict src
pytest --cov=teambench --cov-fail-under=90
teambench run --model mock --repeats 5 --seed 2026 --fallibility 0.1 --output /tmp/mock.json
teambench compare results/mock.json /tmp/mock.json
```

Do not publish this package to PyPI.
