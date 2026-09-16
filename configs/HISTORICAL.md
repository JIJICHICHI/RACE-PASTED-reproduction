# Historical configurations

The default and formal experiment surface is documented in the repository
README. The following retained configuration families are research history and
must not be used as current PASTED-RACE defaults:

- `LE_RACE.json` and `ablations/`: earlier local-evidence experiments.
- `FCE_RACE.json` and `far_race/`: earlier factorized/FAR investigations.
- `tuning/`, `tuning2/`, and `tuning_multiseed/`: exploratory learning-rate,
  batch-size, and checkpoint-selection sweeps.

Their old hyperparameters are intentionally preserved for auditability. Current
formal PASTED-RACE configs live in `configs/pasted_race/` and use the canonical
group-safe `2.9e-5`, batch-16, 20-epoch, patience-5 protocol.
