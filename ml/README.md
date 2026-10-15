# ml/

Training, evaluation and ONNX export for MukhaSpectra models (Phase 4).

Phase 1 ships only the interfaces in `mukha_pipeline.wl` and `mukha_pipeline.spectral` plus the
model card template in `model_cards/TEMPLATE.md`. With no weights installed, the white-light
classifier reports `model_unavailable` and the pipeline runs without it.

To install a model later, place `model.onnx` and `model_card.md` (from the template) in a
directory and set `MUKHA_WL_MODEL_DIR` for the backend. Weights are not committed to git.

Rules: no synthetic image may enter an evaluation set, and no metric leaves this directory
without a documented evaluation protocol.
