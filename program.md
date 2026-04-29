# Research Program

Goal: improve the explainability and interpretability of a coding and shell-use
LLM while preserving useful functionality.

Each candidate should change exactly one of these surfaces:

1. Agent policy prompt or decision trace requirements.
2. Adapter training configuration.
3. Training or evaluation data mix.
4. Explainability metric weighting.
5. Mechanistic probe configuration.

Candidates are accepted only if they pass the baseline-relative functionality
floor and improve total explainability score.

Prefer conservative experiments that can run on CPU and degrade cleanly when
CUDA is not exposed to the process.

