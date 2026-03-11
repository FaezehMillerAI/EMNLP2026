# C3XR: Concept-Constrained Counterfactual Explanations for X-ray Reports

## Summary
C3XR keeps the dual-path spirit of Nesy-Gen but removes heavy graph reasoning and LTN constraints. It uses a lightweight **concept bottleneck** for explainable report generation and a **counterfactual concept influence** module for attribution. No heatmaps are produced.

## Key Modules
1. **Vision Encoder**: ResNet backbone producing spatial features.
2. **Concept Bottleneck**: Multi-label prediction of clinically meaningful concepts (CheXpert-14). This creates a human-interpretable evidence layer.
3. **Concept-Conditioned Decoder**: Transformer decoder conditioned on both visual features and concept embeddings.
4. **Counterfactual Explainer**: For each concept token, remove it from memory and measure the drop in report log-likelihood to quantify its causal influence on the generated report.

## Explanation Artifacts
- **Concept Evidence Table**: Probabilities of clinically relevant concepts.
- **Counterfactual Influence**: Per-concept influence score (LL drop).
- **Qualitative Report Cards**: Image + concept evidence + counterfactual influence + generated vs reference report.

## Why This Is Explainable Without Heatmaps
- Explanations are discrete, clinical concepts.
- Counterfactual influence gives causal attribution without pixel saliency.
- Outputs are human-readable and auditable for medical validation.

