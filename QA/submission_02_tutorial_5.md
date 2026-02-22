# Submission 02 - Structured Answers (Tutorial 5)

## Scope

This document answers Tutorial 5 questions on support polygon references and control modality comparison.

## Q4. Which ground-reference points can exist outside the support polygon?

**Answer:**

- **ZMP:** For stable contact without tipping/slipping assumptions, ZMP is expected to remain inside the support polygon. If it exits, balance/contact constraints are being violated.
- **CMP:** Can move outside the support polygon when nonzero centroidal angular-momentum rate is used (e.g., upper-body/hip strategy).
- **DCM:** Can be outside the support polygon during dynamic motion and disturbances; this is common and informative for recovery planning.

## Q5. Which modality tolerates larger push disturbances in this project: torque interface or position interface?

**Answer (based on this repository's implementation):** The position-interface workflow shows higher tolerance in the provided experiments.

Observed setup:

- `t51.py` (torque interface path) uses lower push magnitude (`f_push_mag = 10.0`).
- `t52.py` (virtual-state integration + position command path) is tested with stronger pushes (`f_push_mag = 40.0`).

Interpretation:

- In this project, the position interface behaves more robustly under larger disturbances.
- This is an empirical result of the current controller design/tuning, not a universal rule for all robots.

## Q6. Are torque and position modalities equivalent in the proposed method?

**Answer:** No, they are not equivalent.

Key differences in this repository:

- **Control path**
  - Torque mode (`t51`): directly applies torque-oriented behavior from TSID output flow.
  - Position mode (`t52`): integrates virtual TSID state and sends position-like commands.

- **Disturbance behavior**
  - With current gains and architecture, position mode is less sensitive to strong pushes.

- **Practical implications**
  - Torque mode can provide direct dynamic control but is sensitive to model/sensing/tuning quality.
  - Position mode can be easier to stabilize in simulation and in many practical stacks.

Conclusion:

- The two modalities have different closed-loop dynamics and robustness profiles; they should not be treated as equivalent.
