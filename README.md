
## Hypotheses

H1: Optimizer differences diminish with scale (like speedup does)
H2: Optimizer differences amplify with scale (representations diverge more)
H3: Non-monotonic relationship (differences peak at intermediate scales)

```
$ python scripts/run_experiment.py --config config/test_config.yaml --optimizer muon
```