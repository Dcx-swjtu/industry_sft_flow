"""Prompt for extract_evidence — SFT variant (simplified control layer guidance)."""

from dataflow.prompts.base import InputField, OutputField, PromptSection, PromptTemplate


EXTRACT_EVIDENCE_PROMPT = PromptTemplate(
    name="extract_evidence",
    requires_image=True,
    input_fields=[
        InputField("figure_context", "{FIGURE_CONTEXT}", "sample.metadata"),
        InputField("caption", "{CAPTION}", "sample.caption"),
        InputField("raw_caption", "{RAW_CAPTION}", "sample.raw_caption"),
        InputField("context", "{CONTEXT}", "sample.context"),
        InputField("title", "{TITLE}", "sample.title"),
        InputField("subfigure_infos", "{SUBFIGURE_INFOS}", "sample.subfigure_infos"),
        InputField("raw_subject", "{RAW_SUBJECT}", "sample.raw_subject"),
    ],
    output_key="extract_output",
    output_fields=[
        OutputField("evidence_graph", "object"),
        OutputField("control_layer", "object"),
        OutputField("extraction_summary", "object"),
    ],
    sections=[
        PromptSection(
            title="Role",
            title_level=2,
            text="""Build the evidence graph for one likely training spine. The main purpose of this stage is to preserve the graph that later reasoning will consume: discover enough local assets to support the dominant spine, but do not index the whole figure indiscriminately and do not yet produce rrels or a reasoning path.""",
        ),
        PromptSection(
            title="Output Contract",
            title_level=2,
            text="""Return exactly:

```json
{
  "extract_output": {
    "evidence_graph": {
      "vis": [...],
      "rels": [...]
    },
    "control_layer": {
      "items": [...]
    },
    "extraction_summary": {
      "dominant_region": "...",
      "vis_count": 6,
      "rel_count": 3,
      "recommended_vis_ids": ["vis_1", "vis_3", "vis_4"],
      "recommended_rel_ids": ["rel_1", "rel_2"],
      "notes": "..."
    }
  }
}
```""",
        ),
        PromptSection(
            title="Indexing Method",
            title_level=2,
            text="""Build the evidence graph using two layers:

1. `vis` — reusable visual units. Each vis item must have:
   - `id`, `label`, `region` (descriptive location in the figure)
   - `has_readable_value`: boolean — true if the visual element contains extractable numerical values
   - `value_type`: string — the type of measurement this vis represents (e.g., "physical distance in Å", "percentage increase", "kinetic profile")

2. `rels` — shallow visible relations between `vis` units, such as comparability, contrast, alignment, temporal adjacency, or structure-to-function linkage

Do not create separate `vis` units for axes, scale ticks, or coordinate labels. These should be mentioned as part of their parent measurement panel's `vis` description or `region` field.

Keep `vis` and `rels` one level below the final biological interpretation. Good `vis` units sound like:
- `SUV420H1 row with AML point and no MDS points`
- `del(7q36) bar at about 2% in MDS-stage BM`
- `del(7q36) bars at about 90% in AML BM/MLP/GMP`
- `paired distance labels 3.9 Å and 5.3 Å`

Bad `vis` units sound like:
- `late AML mutation`
- `progenitor restriction`
- `selective long-chain disruption`

Likewise, a good `rel` says that two `vis` units can be compared or composed. A bad `rel` already states the final answer.

Prefer about 4-10 `vis` items and 2-5 `rels`. For compact figures, fewer is fine. For broad figures, choose the dominant evidence region plus only the supporting branches that will actually feed one fused spine rather than indexing the entire paper figure equally.""",
        ),
        PromptSection(
            title="Preservation Rules",
            title_level=2,
            text="""Critical rules for preserving information that later stages need:

1. **Paired measurements**: When a figure shows paired measurements (e.g., WT vs Mutant distances, MDS vs AML proportions), preserve them as separate `vis` units if later stages need both as explicit operands. Do NOT merge them into a single `vis` that already encodes the full comparison.

2. **Control groups**: Always index control/comparison groups (e.g., shorter-chain substrates, untreated samples, MDS-stage baselines) alongside the main effect group. Omitting the control group destroys the evidence for selective effects.

3. **Value readability**: Mark `has_readable_value=true` only when numerical values can be directly read from the figure (labels, axis ticks, scale bars). Use `value_type` to describe what kind of measurement it represents. Do not set `has_readable_value=true` for qualitative patterns where no number can be extracted.

Prefer measurement panels, quantitative panels, and directly recoverable structures over schematic or clonal-model panels. A schematic or model panel may appear as corroboration, but it should usually not dominate `recommended_vis_ids` when stronger measurement anchors exist.""",
        ),
        PromptSection(
            title="Control Layer",
            title_level=2,
            text="""`control_layer.items` must audit nontrivial evidence units. Each item uses:
- `target_id`
- `dependency_level` = `image_sufficient` | `image_plus_text_clarification` | `text_required`
- `txt_used` (boolean, not string)
- `txt_used_sources`
- `txt_required` (boolean, not string)
- `audit_reason`

`txt_used` and `txt_required` must be boolean values. Do not emit strings, arrays, or prose in those fields.
`dependency_level` is the primary field. Use the booleans only as a backward-compatible mirror:
- `image_sufficient`: the visual evidence is enough on its own
- `image_plus_text_clarification`: the image carries the main pattern, but text helps identify a label, condition, or semantic role
- `text_required`: without text, the evidence unit cannot be stated faithfully

Be conservative with `text_required`. If the figure pattern is visible and text only clarifies labels or condition names, use `image_plus_text_clarification`, not `text_required`.

Focus on evidence that supports the figure's primary reasoning chain. Use `candidate_focus_panels` and `seed_evidence_preferences` from the admission summary to guide which evidence to prioritize. Ensure the extracted graph can support the kind of multi-step visual reasoning the figure naturally affords.

`recommended_vis_ids` and `recommended_rel_ids` are the initial focus contract for the next stage — they are not extra decoration. Prefer marking the small subset that should anchor the main spine, even if the graph contains a few additional reusable items.

Before finalizing, self-check:
- every `vis.id` is unique
- every `rel.id` is unique
- `vis_count` equals the actual number of `vis` items
- `rel_count` equals the actual number of `rels` items
- every recommended id refers to a real graph item
- every `text_required` item includes a minimal supporting text source

If you intentionally deviate from the admission scope, explain that in `notes`. Do not silently drift to a different subproblem.""",
        ),
        PromptSection(
            title="Example",
            title_level=2,
            text="""For a broad multi-panel progression figure, a good extraction should preserve:
- one stage-pattern `vis`
- one persistence-pattern `vis`
- one compartment-pattern `vis`
- one early/low anchor `vis`
- one late/high anchor `vis`
- shallow `rels` linking stage evidence, compartment evidence, and timing-anchor evidence

The early/low anchor and late/high anchor should usually be separate `vis` units, not one merged `vis` that already encodes the full comparison.

For a compact labeled structure-function figure, the graph should preserve:
- separate `vis` units for the paired labeled measurements
- one `vis` for the strongest effect on the focal substrate or condition
- one `vis` for the preserved control or comparison substrates/conditions
- `rels` linking the paired measurements and linking the focal effect to its preserved controls""",
        ),
        PromptSection(
            title="Inputs",
            title_level=2,
            text="""The image is attached.

`FIGURE_CONTEXT = {FIGURE_CONTEXT}`
`CAPTION = {CAPTION}`
`RAW_CAPTION = {RAW_CAPTION}`
`CONTEXT = {CONTEXT}`
`TITLE = {TITLE}`
`SUBFIGURE_INFOS = {SUBFIGURE_INFOS}`
`RAW_SUBJECT = {RAW_SUBJECT}`""",
        ),
    ],
)
