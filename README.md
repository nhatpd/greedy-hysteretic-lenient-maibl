# Reproducing Results

All the results of multiagent IBL models can be reproduced by running corresponding scripts for each senario. In particular, to run the scenarios with MAIBL agents simply execute the following command and the experiment will start.

```markdown
python3 cmotp_ibl.py --environment [env] --mamethod [type]
```

With argument [env] is replaced by: CMOTP_V3 for Scenario 1, CMOTP_V8 for Scenario 2, CMOTP_V6 for Scenario 3, and CMOTP_V4 for Scenario 4.

With argument [type] is replaced by: greedy for Greedy-MAIBL, hysteretic for Hysteretic-MAIBL, and leniency for Lenient-MAIBL.
