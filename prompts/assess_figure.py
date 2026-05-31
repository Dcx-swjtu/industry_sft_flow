"""Prompt for assess_figure — SFT variant (no RLVR artifacts)."""

from dataflow.prompts.base import InputField, OutputField, PromptSection, PromptTemplate


ASSESS_FIGURE_PROMPT = PromptTemplate(
    name="assess_figure",
    requires_image=True,
    input_fields=[
        InputField("caption", "{CAPTION}", "sample.caption"),
        InputField("raw_caption", "{RAW_CAPTION}", "sample.raw_caption"),
        InputField("context", "{CONTEXT}", "sample.context"),
        InputField("title", "{TITLE}", "sample.title"),
        InputField("subfigure_infos", "{SUBFIGURE_INFOS}", "sample.subfigure_infos"),
        InputField("raw_subject", "{RAW_SUBJECT}", "sample.raw_subject"),
    ],
    output_key="assess_output",
    output_fields=[
        OutputField("figure_profile", "object"),
        OutputField("admission_summary", "object"),
    ],
    sections=[
        PromptSection(
            title="Role",
            title_level=2,
            text="""Decide whether this scientific figure is worth turning into one high-quality SFT training sample. This stage is only about admission: discover the figure's recoverable reasoning assets broadly enough to judge feasibility, but commit to one dominant fused spine rather than several unrelated mini-spines. Do not extract full evidence units yet.""",
        ),
        PromptSection(
            title="Output Contract",
            title_level=2,
            text="""Return exactly:

```json
{
  "assess_output": {
    "figure_profile": {
      "should_generate": true,
      "why": "...",
      "figure_family": "...",
      "shortcut_risk": "low|medium|high",
      "visual_hinge_density": "low|medium|high"
    },
    "admission_summary": {
      "targetable_spine_count": 1,
      "recommended_scope": "broad evidence discovery, one dominant fused spine",
      "expected_chain_shape": "read/anchor -> compare/boundary/bridge -> final interpretation",
      "expected_question_type": "multi-step visual reasoning requiring integration of 2+ pieces of evidence",
      "visual_reasoning_density": "moderate",
      "candidate_focus_panels": ["d", "f", "g"],
      "seed_evidence_preferences": ["50 μm scale bar in panel b", "3.9 Å and 5.3 Å distance labels", "..."],
      "why_not_other_scopes": "..."
    }
  }
}
```""",
        ),
        PromptSection(
            title="Admission Criteria",
            title_level=2,
            text="""Admit a figure only when it supports one real reasoning spine whose later steps must consume earlier outputs. We are not building generic QA, and we are not rewarding panel-hopping for its own sake. Scientific figures differ from math diagrams: the image usually does not already contain a ready-made question. The job here is to judge whether the figure contains enough recoverable hinges that a good chain can be built later.

Good admitted figures support at least one problem with:
- a recoverable visual anchor such as an exact value, a compact tuple, a visible contrast, or a structural cue
- at least one nontrivial transformation such as compare, compute, timing relation, boundary, or selective interpretation
- a free-form answer requiring genuine visual reasoning (the answer must integrate evidence from the image)

Broad figures may contribute multiple local assets or multiple panels, but those assets must be able to fuse into one globally connected main spine. Reject figures that would collapse into one obvious caption paraphrase, a pure summary panel shortcut, or several unrelated reads plus recap.""",
        ),
        PromptSection(
            title="Field Guide",
            title_level=2,
            text="""Interpret each field concretely:

- `shortcut_risk=high` when caption or context would answer the likely question more easily than the image
- `visual_hinge_density=high` when the figure has several reusable hinges and at least one clean spine
- `targetable_spine_count` should be small; one dominant spine is ideal even if the figure contains other local facts
- `recommended_scope` should describe broad evidence discovery but only one dominant fused spine; it should be archetype-shaped, not paper-shaped
- `expected_question_type` should describe the kind of question the figure naturally supports (e.g., "causal mechanism requiring comparison of two conditions", "trend analysis requiring multi-panel integration")
- `visual_reasoning_density` should be `low`, `moderate`, or `high` depending on how much genuine visual evidence the answer must integrate
- `candidate_focus_panels` should name the panels most likely to support the chosen spine
- `seed_evidence_preferences` **must list 4-7 concrete evidence elements visible in the figure** (e.g., "50 μm scale bar in panel b", "3.9 Å and 5.3 Å distance labels in panel a", "reversible calcium trace in panel e"). Do NOT write abstract categories like "spatial pattern" or "temporal pattern" — name the actual visual element and its location.""",
        ),
        PromptSection(
            title="Example",
            title_level=2,
            text="""Example judgment for a compact labeled structure-function figure:
- should_generate = true
- expected_chain_shape = `read paired distances -> compute change -> identify strongest substrate effect -> interpret mechanism`

Example judgment for a broad multi-panel progression figure:
- should_generate = true
- recommended_scope = `broad evidence discovery, one dominant fused spine`
- why_not_other_scopes = `the figure may contain several local facts, but only one connected main spine is recoverable without drifting into paper-level narration`""",
        ),
        PromptSection(
            title="Inputs",
            title_level=2,
            text="""The image is attached.

`CAPTION = {CAPTION}`
`RAW_CAPTION = {RAW_CAPTION}`
`CONTEXT = {CONTEXT}`
`TITLE = {TITLE}`
`SUBFIGURE_INFOS = {SUBFIGURE_INFOS}`
`RAW_SUBJECT = {RAW_SUBJECT}`""",
        ),
    ],
)
