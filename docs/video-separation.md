# Video-generation boundary

The `main` branch is intentionally text-only. The current repository never
contained a video model, diffusion pipeline, frame encoder, renderer, codec
integration, or video dataset; video was only part of the original project
ambition. Keeping those concerns out of the text package prevents a small
causal language-model experiment from acquiring an unrelated dependency graph.

Future video work should live in a separate `videogen` package, repository, or
branch. A future integration may consume a stable text-side interface such as
prompt plans or generated token sequences, but it must not add frame tensors,
FPS/resolution fields, media imports, or video-specific training logic to
`MoREModel.forward`.

This boundary is deliberate:

- `more` owns tokenization, causal text modeling, recursive depth routing,
  expert routing, checkpoints, and text evaluation.
- `videogen` will own video encoders/decoders, diffusion or video-transformer
  components, media datasets, codecs, and rendering.
- Integration should be an explicit adapter or service contract, with separate
  dependencies and benchmarks.

The main branch therefore makes no claim that MoRE is already a video model.
