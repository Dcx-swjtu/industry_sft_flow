"""Prompt for induce_rrels — SFT variant (no target_surface)."""

from dataflow.prompts.base import InputField, OutputField, PromptSection, PromptTemplate


INDUCE_RRELS_PROMPT = PromptTemplate(
    name="induce_rrels",
    requires_image=True,
    input_fields=[
        InputField("figure_context", "{FIGURE_CONTEXT}", "sample.metadata"),
        InputField("evidence_graph", "{EVIDENCE_GRAPH}", "extract.evidence_graph"),
        InputField("control_layer", "{CONTROL_LAYER}", "extract.control_layer"),
        InputField("context", "{CONTEXT}", "sample.context"),
    ],
    output_key="rrel_output",
    output_fields=[
        OutputField("rrel_bank", "array"),
        OutputField("candidate_paths", "array"),
        OutputField("induction_summary", "object"),
    ],
    sections=[
        PromptSection(
            title="Role",
            title_level=2,
            text="""Induce a compact bank of atomic reasoning assets (`rrel_bank`) from the evidence graph. Stay below the final reasoning path: each rrel should be a small, image-grounded building block, not a finished answer. Discover usable local assets broadly enough to preserve the dominant chain, but do not let them sprawl into several unrelated chains. Respect observability. Do not turn approximate visual reads into fake precision, and do not use text to sharpen an image-only value. Treat `control_layer` as an audit contract: it does not force every claim to be text-dependent, but it tells you where an image-only phrasing would be dishonest.""",
        ),
        PromptSection(
            title="Output Contract",
            title_level=2,
            text="""Return exactly:

```json
{
  "rrel_output": {
    "rrel_bank": [...],
    "candidate_paths": [...],
    "induction_summary": {
      "dominant_theme": "...",
      "best_path_id": "path_1 or empty string",
      "path_summaries": [
        {
          "path_id": "path_1",
          "reasoning_shape": "exact tuple read -> selective comparison -> mechanistic interpretation",
          "focus_anchors": ["vis_1", "vis_3"],
          "why_good": "..."
        }
      ],
      "notes": "..."
    }
  }
}
```

Each `rrel` uses this shape:
- `id`, `claim`, `support = {"evidence_ids": [...], "extract_rel_ids": [...], "control_targets": [...]}`
- `reasoning`, `kind = read|compare|compute|bridge|boundary|synthesis`, `score`
- `knowledge = {"txt_required": "none"|"caption"|"context"|"mixed", "source": "specific quote or reference from text"}`
  - When `txt_required` is not `"none"`, always include a concrete `source`.

`extract_rel_ids` must point only to ids from `EVIDENCE_GRAPH.rels`, i.e. `rel_*`.
Never place `rrel_*` ids inside the support object.

Each `candidate_path` uses:
- `path_id`, `step_rrel_ids`, `missing_hinges`, `current_depth`, `shortcut_risk`

`candidate_paths` are optional weak hints, not a final commitment. They may be empty if the bank is strong enough and an early path commitment would be misleading. Whenever you do emit path hints, every `path_summary.path_id` must refer to a real candidate path. `best_path_id`, if non-empty, must refer to one of those same path IDs and also appear in `path_summaries`.""",
        ),
        PromptSection(
            title="RREL Method",
            title_level=2,
            text="""Build rrels in this order:

1. **Observational roots**
   Each bank must contain at least one direct visual root backed by `support.evidence_ids`. Use `read` for a tuple/value observation or `compare` when the figure's atomic observation is itself a visible contrast (for example dense vs porous morphology or higher vs lower coverage). Do not invent a redundant `read` solely to satisfy a label convention.

2. **Observability-aware reads**
   Distinguish three cases:
   - Direct label in figure: preserve the exact value.
   - Axis/bar-height estimate: keep it approximate, bucketed, or relational.
   - Text-required claim: do not hide it inside a `read`; emit a separate `bridge` rrel and mark the text dependency.

Use `control_layer.dependency_level` carefully:
- `image_sufficient`: image-only `read/compare/boundary` is allowed
- `image_plus_text_clarification`: the image carries the main pattern, but labels or condition names may need text clarification
- `text_required`: do not pretend the claim is image-only; mark `knowledge.txt_required` and include a concrete text `source`

3. **Control / boundary evidence**
   When the figure shows a selective effect, include one `read`, `compare`, or `boundary` rrel for the control or preserved group. Do not omit the comparison branch.

4. **Transformations**
   Add `compare`, `compute`, `boundary`, or `bridge` rrels that operate on earlier reads. These should remain small and local.

5. **Optional path hints**
   If a promising partial chain is obvious, emit 1-2 `candidate_paths` as weak fusion hints. Do not force a brittle path guess when the main job is to surface reusable local assets.

6. **At most one synthesis**
   A single `synthesis` rrel may combine 2-3 earlier rrels into a mechanistic or interpretive conclusion.

Target 5-6 rrels total (maximum 6). RREL ids must be unique. Claims must be non-redundant. Avoid high-level summary assets that belong in later stages.""",
        ),
        PromptSection(
            title="Guardrails",
            title_level=2,
text="""DO:
- Create at least one direct evidence-backed `read` or `compare` observational root
- Use a single tuple `read` when the observation is naturally a paired value read
- Keep control-group evidence when the figure shows selective effects
- Prefer measurement-derived anchors over schematic summaries
- Describe approximate axis reads as ranges, order bins, or relative comparisons
- Keep every `support.evidence_ids` and `support.extract_rel_ids` grounded in the provided extract graph
- Use `control_layer` to decide when a claim is truly image-sufficient versus only image-plus-text-clarification

DO NOT:
- Create 7+ rrels when 5-6 suffice
- Treat `candidate_paths` as a required final path choice
- Write a `read` rrel like "calibrated by text" or "aligned with the reported fold change"
- Use caption/context to sharpen an image-only approximate value
- Use subjective words like `negligible`, `dramatic`, or `significant` unless the figure visibly licenses them
- Produce several unrelated local chains that only meet at the very end
- Repeat the same claim with different wording or recycle the same rrel id
- Put any `rrel_*` id inside `support.evidence_ids`, `support.extract_rel_ids`, or `support.control_targets`
- Invent new `vis_*` or `rel_*` ids that are not present in the extract graph""",
        ),
        PromptSection(
            title="Example",
            title_level=2,
            text="""For a compact labeled structure/function figure, a good bank would be:

1. `read`: "WT distance is 3.9 A and Mutant distance is 5.3 A"
2. `compute`: "The Mutant increases the distance by 1.4 A"
3. `compare`: "C8 shows the largest activity drop among the assayed substrates"
4. `boundary`: "Shorter-chain substrates change much less than C8"
5. `synthesis`: "The pocket shift selectively disrupts the longest-chain interaction rather than globally collapsing activity"

For a chart-only sample with no direct labels:
- good `read`: "Treatment group is about one order of magnitude lower than control"
- bad `read`: "Treatment group is exactly 6.7E+06 because the text reports a 15-fold decrease"

Good `reasoning_shape` examples:
- `exact tuple read -> delta compute -> mechanistic interpretation`
- `magnitude-bin read -> boundary comparison -> reject unsupported mechanism`

For a broad multi-panel electrophysiology figure, a good bank would:
- keep one anatomy/identity anchor if it is needed to license the physiology read
- keep one stimulus or perturbation read
- keep one response-pattern read
- keep one compare/boundary rrel that separates direct mechanism from broader overclaim
- optionally propose path hints that can later collapse into one globally connected main spine""",
        ),
        PromptSection(
            title="Inputs",
            title_level=2,
            text="""The image is attached.

`FIGURE_CONTEXT = {FIGURE_CONTEXT}`
`EVIDENCE_GRAPH = {EVIDENCE_GRAPH}`
`CONTROL_LAYER = {CONTROL_LAYER}`
`CONTEXT = {CONTEXT}`""",
        ),
    ],
)
