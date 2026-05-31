"""Prompt for compose_path — SFT variant (no RLVR supervision artifacts)."""

from dataflow.prompts.base import InputField, OutputField, PromptSection, PromptTemplate


COMPOSE_PATH_PROMPT = PromptTemplate(
    name="compose_path",
    requires_image=True,
    input_fields=[
        InputField("rrel_bank", "{RREL_BANK}", "rrel.rrel_bank"),
        InputField("candidate_paths", "{CANDIDATE_PATHS}", "rrel.candidate_paths"),
        InputField("induction_summary", "{INDUCTION_SUMMARY}", "rrel.induction_summary"),
        InputField("context", "{CONTEXT}", "sample.context"),
        InputField("validation_feedback", "{VALIDATION_FEEDBACK}", "compose.local_validation"),
    ],
    output_key="path_output",
    output_fields=[
        OutputField("reasoning_path", "object"),
        OutputField("compose_summary", "object"),
        OutputField("question_spec", "object"),
    ],
    sections=[
        PromptSection(
            title="Role",
            title_level=2,
            text="""You are the Deep-Mine stage. Collapse the RREL bank into one tight reasoning spine. Do NOT design questions here. Your job is only to fuse multiple local reasoning assets into one globally connected main spine that later stages can compile into SFT training data. Multiple local assets may feed the spine, but only one dominant fused chain should survive.""",
        ),
        PromptSection(
            title="Output Contract",
            title_level=2,
            text="""Return exactly:

```json
{
  "path_output": {
    "reasoning_path": {
      "latent_target": "narrow description of what this chain establishes",
      "minimal_givens": [],
      "reasoning": "S1 -> S2 -> S3 -> ...",
      "reasoning_path": [
        {
          "step_id": "S1",
          "source_rrels": ["rrel_1"],
          "step_kind": "read",
          "step_claim": "precise claim supported by the figure",
          "depends_on": [],
          "observability": "exact_labeled",
          "txt_dependency": "none"
        }
      ]
    },
    "compose_summary": {
      "selected_path_id": "path_1 or empty string",
      "selected_rrel_ids": ["rrel_1", "rrel_2", "rrel_3"],
      "notes": "..."
    },
    "question_spec": {
      "source_mode": "image_only",
      "must_include": [],
      "must_not_use": []
    }
  }
}
```

Each step must have:
- `step_id`: S1, S2, ...
- `source_rrels`: rrel IDs this step draws from. No step may leave this empty. For a synthesis step, list the rrels directly consumed by that synthesis rather than leaving only `depends_on`.
- `step_kind`: one of `read`, `compare`, `compute`, `bridge`, `boundary`, `synthesis`
- `step_claim`: concrete claim supported by the figure
- `depends_on`: step IDs this step depends on
- `observability`: one of `exact_labeled`, `axis_estimated`, `relational`, `text_dependent`
- `txt_dependency`: one of `none`, `caption`, `context`, `mixed`

If any step has non-`none` `txt_dependency`, populate `minimal_givens` with the smallest text bridge needed downstream.""",
        ),
        PromptSection(
            title="Question Spec Policy",
            title_level=2,
            text="""After composing the reasoning path, output a `question_spec` that guides the next stage's question generation:

1. `source_mode`: What information sources does this path actually require?
   - "image_only": All steps have txt_dependency="none" — the figure alone suffices
   - "image_plus_caption": Some steps need caption labels (txt_dependency="caption")
   - "image_plus_context": Steps require paper context (txt_dependency="context" or "mixed")

2. `must_include`: List specific premises the question MUST state as given. Only include:
   - Identity mappings not visible in figure (e.g., "thick curve = CNF electrode")
   - Abbreviation expansions needed to answer (e.g., "WT = wild-type")
   - Do NOT include obviously visible facts like panel labels or axis units
   - Do NOT include visual findings or comparative conclusions such as "panel X lacks hotspots", "the baseline is balanced", "condition A is higher", or "a dense vertical band is present". These remain evidence the answer must infer from the image.

3. `must_not_use`: List categories of information the answer must NOT use unless stated in question:
   - Method names not visible in figure (e.g., "alignment algorithm names not shown in figure")
   - Biological identities not in labels (e.g., "specific protein names not in figure legend")
   - Causal mechanism claims beyond what the figure shows (e.g., "X causes Y when only co-occurrence is shown")
   - Abbreviation or feature-name expansions that come only from caption/context unless they are provided in `must_include`

Dependency rules are hard constraints:
- A reasoning step may not declare a lower `txt_dependency` than any RREL in its `source_rrels`.
- If a selected RREL requires `context` or `mixed`, the final `source_mode` must be `image_plus_context`.
- When a context-dependent conclusion cannot be stated using a minimal premise in `must_include`, do not select that conclusion; weaken the path to what the image supports.
- When weakening a conclusion to a purely visible pattern, omit any context-only RREL that is no longer needed. Do not carry a stronger context dependency into a weaker visual-only conclusion.
- If the final answer would need to spell out the meaning of a textual label or feature abbreviation, put that exact minimal definition in `must_include`; otherwise the answer must leave the label uninterpreted.""",
        ),
        PromptSection(
            title="Minimal Givens Policy",
            title_level=2,
            text="""`minimal_givens` are not mini-summaries of the figure. They are glossary-style bridges only.

Allowed:
- abbreviation expansions
- cohort / condition identity notes
- label disambiguation that the image alone cannot supply

Not allowed:
- numbers, percentages, or exact values
- trend statements such as increase / decrease / stable
- distribution or absence findings such as `balanced`, `uniform`, or `lacks hotspots`
- patient-specific or panel-specific observations
- mechanistic implications such as "X implies Y"
- any statement that would directly answer a question

Good:
- `MDS-RS refers to SF3B1-mutated patients`
- `MAB stands for mutant allele burden`

Bad:
- `MAB increased in MDS3`
- `MAB increase implies clonal advantage`
- `Panel c shows a lower ratio in the patient group`

If a path needs a stronger text bridge than this, drop that path and choose a more image-grounded spine.""",
        ),
        PromptSection(
            title="Local Validation Feedback",
            title_level=2,
            text="""`VALIDATION_FEEDBACK = {VALIDATION_FEEDBACK}`

When this field is non-empty, the prior draft failed a deterministic contract check. Correct the stated failure in a complete fresh output. In particular, any selected text-dependent step requires glossary-style `minimal_givens` and a corresponding premise in `question_spec.must_include`; do not remove required dependency declarations merely to pass validation.""",
        ),
        PromptSection(
            title="Compression Method",
            title_level=2,
            text="""Collapse the selected RRELs using these ordered rules:

1. **Merge paired reads**
   If the bank has a matched pair, produce one `read` step rather than two isolated leaves.

2. **Keep the control / boundary branch**
   If the figure shows a selective effect, preserve one dedicated `boundary` or `compare` step.

3. **Fuse rather than list**
   Remove redundant one-rrel-one-step expansion. The path should usually be 4-5 steps, maximum 6.
   A good path is not five parallel roots plus one recap. Introduce real intermediate dependencies when one step materially transforms an earlier step.

4. **End with one real synthesis**
   The final step must combine at least two earlier anchors into a mechanistic or interpretive conclusion. It must not be a recap that simply restates every prior value.""",
        ),
        PromptSection(
            title="Reasoning-Step Policy",
            title_level=2,
            text="""Choose only the reasoning-step semantics from what the figure actually supports:

- `read`
  Use for direct observations, paired observations, or stable value statements.

- `compute`
  Use only when the step is a transformation of earlier anchored reads, such as a delta or derived quantity.

- `compare`
  Use for direction, ordering, larger/smaller, stronger/weaker, or matched-condition comparison.

- `boundary`
  Use when the figure supports preserved-vs-disrupted, selective-vs-global, threshold, or exclusion logic.

- `bridge`
  Use when a narrow text bridge or identity clarification is genuinely required.

- `synthesis`
  Use for the final mechanistic or interpretive fusion step.

Observability and text dependence still matter here:
- `exact_labeled`: directly labeled or stably exact visual evidence
- `axis_estimated`: range-like, binned, or approximate visual evidence
- `relational`: ordering, boundary, preserved-vs-disrupted, selective comparison
- `text_dependent`: not honestly recoverable without a narrow text bridge""",
        ),
        PromptSection(
            title="Deep-Mine Rules",
            title_level=2,
text="""DO:
- Search broadly across the bank, but keep only one dominant fused spine
- Use exact values when the figure explicitly labels them
- Use ranges, order bins, or relative language when the figure only supports approximate reading
- Keep final interpretations tied to earlier anchors
- Mark `txt_dependency` accurately
- Keep `source_rrels` explicit even for final synthesis steps

DO NOT:
- Introduce options, answer formats, or any question wording
- Turn approximate chart reads into fake exact numbers
- Write step_claims like "aligned with the reported fold change" or "calibrated by the text"
- Let the final step restate all prior operands inline
- Output more than 6 steps
- Omit the control branch when the conclusion depends on selectivity
- Let every pre-final step stay as an isolated root unless the figure truly offers no intermediate transformation""",
        ),
        PromptSection(
            title="Example",
            title_level=2,
            text="""For a compact labeled structure-function figure, a good path is:

```json
{
  "reasoning_path": {
    "latent_target": "selective distal-pocket disruption rather than global active-site failure",
    "minimal_givens": [],
    "reasoning": "S1 -> S2 -> S3 -> S4 -> S5",
    "reasoning_path": [
      {"step_id": "S1", "source_rrels": ["rrel_1"], "step_kind": "read", "step_claim": "WT distance is 3.9 A and Mutant distance is 5.3 A", "depends_on": [], "observability": "exact_labeled", "txt_dependency": "none"},
      {"step_id": "S2", "source_rrels": ["rrel_2"], "step_kind": "compute", "step_claim": "The pocket recession is 1.4 A", "depends_on": ["S1"], "observability": "exact_labeled", "txt_dependency": "none"},
      {"step_id": "S3", "source_rrels": ["rrel_3"], "step_kind": "compare", "step_claim": "C8 shows the largest activity drop among the assayed substrates", "depends_on": [], "observability": "relational", "txt_dependency": "none"},
      {"step_id": "S4", "source_rrels": ["rrel_4"], "step_kind": "boundary", "step_claim": "Shorter-chain substrates change much less than C8", "depends_on": ["S3"], "observability": "relational", "txt_dependency": "none"},
      {"step_id": "S5", "source_rrels": ["rrel_5"], "step_kind": "synthesis", "step_claim": "The structural shift selectively disrupts the distal long-chain interaction instead of globally collapsing activity", "depends_on": ["S2", "S3", "S4"], "observability": "relational", "txt_dependency": "none"}
    ]
  }
}
```

For a broad multi-panel figure, a good path may draw from more than one panel, but every surviving step should feed the same final conclusion. If a local branch does not change the final synthesis, drop it here. Candidate paths are only hints; if the bank supports a better fused spine than the hinted path, choose the better fused spine.""",
        ),
        PromptSection(
            title="Inputs",
            title_level=2,
            text="""The image is attached.

`RREL_BANK = {RREL_BANK}`
`CANDIDATE_PATHS = {CANDIDATE_PATHS}`
`INDUCTION_SUMMARY = {INDUCTION_SUMMARY}`
`CONTEXT = {CONTEXT}`
`VALIDATION_FEEDBACK = {VALIDATION_FEEDBACK}`""",
        ),
    ],
)
