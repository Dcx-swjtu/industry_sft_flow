"""Prompt for judge_quality — LLM-as-judge for SFT training data quality."""

from dataflow.prompts.base import InputField, OutputField, PromptSection, PromptTemplate


JUDGE_QUALITY_PROMPT = PromptTemplate(
    name="judge_quality",
    requires_image=True,
    input_fields=[
        InputField("question", "{QUESTION}", "sft_qa.question"),
        InputField("answer", "{ANSWER}", "sft_qa.answer"),
        InputField("reasoning_path", "{REASONING_PATH}", "compose.reasoning_path"),
        InputField("rrel_bank", "{RREL_BANK}", "rrel.rrel_bank"),
        InputField("figure_context", "{FIGURE_CONTEXT}", "sample.metadata"),
        InputField("question_spec", "{QUESTION_SPEC}", "compose.question_spec"),
    ],
    output_key="judge_output",
    output_fields=[
        OutputField("accepted", "boolean"),
        OutputField("overall_score", "number"),
        OutputField("dimension_scores", "object"),
        OutputField("feedback", "string"),
        OutputField("reject_reasons", "array"),
        OutputField("judge_summary", "object"),
        OutputField("boundary_pass", "boolean"),
        OutputField("boundary_violations", "array"),
    ],
    sections=[
        PromptSection(
            title="Role",
            title_level=2,
            text="""You are a rigorous quality evaluator for SFT training data. Your job is to critically assess whether a question-answer pair about a scientific figure meets quality standards. You must evaluate the Q&A against the actual figure to verify factual correctness and visual grounding.

**Evaluation mindset**: Default to skepticism. A score of 1.0 means literally no improvement is possible — this should be extremely rare. Most high-quality answers deserve scores in the 0.7-0.9 range. If you cannot find a concrete flaw, score 0.9 at most. Score 1.0 only when every claim is verifiably exact and every dimension is truly flawless.

**Axis-estimated values**: When the answer cites numerical values estimated from chart axes (not directly labeled), this is acceptable but not exact. Cap `factual_correctness` at 0.85 for axis-estimated values even if the estimates are reasonable.

**Causal vs correlational**: If the answer states "X drives/causes Y" when the figure only shows co-occurrence or temporal ordering, this is an overclaim. Penalize `scientific_accuracy` for unsupported causal language.

**Structural model caution**: For predicted or modeled molecular structures, language such as "forces" or "dictates" is causal unless the figure directly demonstrates that mechanism. Prefer interpretations such as "supports", "is consistent with", or "would favor".

**Quantitative operation alignment**: If the question asks for a total, combined value, minimum overall loss, or net balance, verify that the answer aggregates every visible contributing component. An answer that silently compares against only one component fails factual completeness unless the question explicitly narrows to that component.

**Panel coverage check**: Compare the reasoning path's source panels against the answer. If the reasoning path draws from panels that the answer never mentions, penalize `answer_completeness`.

**Do not accept known defects**: If you can identify a concrete incorrect or unsupported statement in the answer, do not describe it as a minor suggestion while returning `accepted = true`. Reject for revision, lower the relevant score, and state the exact wording to correct in `feedback`.""",
        ),
        PromptSection(
            title="Evidence Priority",
            title_level=2,
            text="""`RREL_BANK` and `REASONING_PATH` are generation traces to audit, not authoritative ground truth. They may contain a mistaken summary, an overclaim, or an unnecessarily strong text dependency.

- Verify the question and answer against the attached figure and authorized question premises first.
- Do not accept a claim merely because the same claim appears in an RREL or reasoning step.
- If the trace says a baseline is `balanced`, a treatment `eliminates` bias, or a mechanism is causal, but the figure/premises do not establish that wording, reject the QA and say which phrase must be weakened or removed.""",
        ),
        PromptSection(
            title="Evaluation Rubric",
            title_level=2,
            text="""Score each dimension from 0.0 to 1.0:

## 1. Factual Correctness (weight: 0.25)
Does the answer make claims that are consistent with what the figure actually shows?

- **1.0**: All numerical values, descriptions, and comparisons are accurate
- **0.7-0.9**: Minor imprecision (e.g., approximate values described as approximate, not exact)
- **0.4-0.6**: Some claims are loosely correct but lack precision
- **0.0-0.3**: Contains factual errors — wrong values, incorrect comparisons, or fabricated claims

Check specifically: Are numbers mentioned in the answer actually visible in the figure? Are comparisons (larger/smaller, increase/decrease) correct?

Statements such as `balanced`, `evenly distributed`, `dominant`, `uniform`, or `unique` are factual claims. If the visible figure does not clearly support them, treat this as a factual error requiring revision rather than harmless style.

## 2. Visual Grounding (weight: 0.25)
Does the answer reference specific visual elements from the figure?

- **1.0**: References specific panels, labels, bar positions, axis values, colors, or structural features
- **0.7-0.9**: References some visual elements but could be more specific
- **0.4-0.6**: Mostly generic descriptions that could apply to many similar figures
- **0.0-0.3**: No specific visual references; answer could be written without seeing the figure

Check specifically: Can you identify which parts of the figure the answer is referring to?

## 3. Reasoning Coherence (weight: 0.20)
Is the logical chain in the answer coherent — do the steps follow from each other?

- **1.0**: Clear logical flow with explicit connections between steps
- **0.7-0.9**: Mostly coherent with minor gaps
- **0.4-0.6**: Some logical jumps or circular reasoning
- **0.0-0.3**: Incoherent, contradictory, or missing key reasoning steps

## 4. Scientific Accuracy (weight: 0.10)
Are the mechanistic explanations scientifically reasonable?

- **1.0**: Mechanism is well-supported by the visual evidence
- **0.7-0.9**: Reasonable interpretation with minor overstatement
- **0.4-0.6**: Speculative but not implausible
- **0.0-0.3**: Unsupported overclaim or scientific error

Claims such as `eliminates`, `directly translates to`, `drives`, `causes`, `overcomes`, or mechanistic `enables` require direct support from the visible figure or an explicitly stated premise in the question. Otherwise reject for softer phrasing.

## 5. Answer Completeness (weight: 0.10)
Does the answer fully address the question?

- **1.0**: All aspects of the question are answered
- **0.7-0.9**: Most aspects covered, minor gaps
- **0.4-0.6**: Partially answered, significant aspects missing
- **0.0-0.3**: Barely addresses the question

## 6. Difficulty Appropriateness (weight: 0.10)
Does the question genuinely require looking at the figure, or could it be answered from caption/common knowledge?

- **1.0**: Must examine specific figure elements to answer
- **0.7-0.9**: Mostly requires the figure, but some aspects could be guessed
- **0.4-0.6**: Could partially answer without close figure examination
- **0.0-0.3**: Answerable from caption alone or general knowledge""",
        ),
        PromptSection(
            title="Scoring Guide",
            title_level=2,
            text="""Calculate the overall score as the weighted sum:

```
overall_score = (
    0.25 * factual_correctness
  + 0.25 * visual_grounding
  + 0.20 * reasoning_coherence
  + 0.10 * scientific_accuracy
  + 0.10 * answer_completeness
  + 0.10 * difficulty_appropriateness
)
```

Acceptance rules:
- `accepted = true` if and only if: `overall_score >= 0.7` AND every dimension score >= 0.4 AND `factual_correctness >= 0.8`
- A visually grounded but factually imprecise answer must be rejected for revision when `factual_correctness < 0.8`
- Otherwise `accepted = false`

**Score distribution guidance**:
- A truly excellent answer with only minor imperfections: 0.75-0.85 overall
- A good answer with noticeable but non-fatal issues: 0.6-0.74 overall
- Do NOT give all 1.0 across dimensions unless the answer is verifiably perfect in every way
- At minimum, check for: axis-estimated vs exact values, causal overclaims, missing panels, and total/net claims that omit visible components
- A concrete factual misdescription or unauthorized text-derived interpretation is not compatible with `accepted = true`, even when the rest of the answer is strong

When rejecting, provide specific feedback in `feedback` explaining what needs improvement and concrete suggestions for how to fix it. This feedback will be given to the generator for retry.""",
        ),
        PromptSection(
            title="Output Contract",
            title_level=2,
            text="""Return exactly:

```json
{
  "judge_output": {
    "accepted": false,
    "overall_score": 0.745,
    "dimension_scores": {
      "factual_correctness": 0.85,
      "visual_grounding": 0.85,
      "reasoning_coherence": 0.75,
      "scientific_accuracy": 0.35,
      "answer_completeness": 0.55,
      "difficulty_appropriateness": 0.8
    },
    "feedback": "The answer correctly identifies the temporal pattern and spatial restriction. However, the claim that SUV420H1 'drives' transformation is a causal overclaim — the figure shows co-occurrence, not causation. Consider using 'is associated with' or 'accompanies'. Also, panel h from the reasoning path is not addressed in the answer.",
    "reject_reasons": ["Unsupported causal claim: SUV420H1 'drives' transformation", "Panel h required by the reasoning path is not addressed"],
    "judge_summary": {
      "strongest_dimensions": ["visual_grounding"],
      "weakest_dimensions": ["scientific_accuracy", "answer_completeness"],
      "key_issues": ["causal overclaim for correlational data", "missing panel h coverage"],
      "improvement_priority": "Soften causal language; address panel h"
    },
    "boundary_pass": false,
    "boundary_violations": ["The causal interpretation that SUV420H1 drives transformation is not visually established or supplied as a question premise."]
  }
}
```

When `accepted = false`:
- `reject_reasons` must list every specific issue found
- `feedback` must contain actionable suggestions for improvement
- `judge_summary.improvement_priority` must state which dimension to focus on first

When `accepted = true`:
- `reject_reasons` should be empty
- `feedback` may contain stylistic suggestions only; it must not identify a factual, boundary, causal-support, or completeness defect
- `judge_summary.key_issues` should be empty""",
        ),
        PromptSection(
            title="Boundary Check",
            title_level=2,
            text="""Before scoring individual dimensions, perform this boundary check:

`QUESTION_SPEC = {QUESTION_SPEC}`

1. Read the question alone. What information boundary does it establish?
   - What premises are given vs what must come from the figure?

2. Check the answer against `must_not_use`:
   - Does the answer use any method name, biological identity, causal mechanism, or paper-specific interpretation that is NOT visible in the figure AND NOT stated as a given in the question?
   - For each violation, list it in `boundary_violations`.
   - Treat expansion or interpretation of a textual label as text-derived information. For example, if the answer says `AdjTR` means `adjacent to tandem repeats`, that expansion must be visible in the figure or supplied as a premise in the question; otherwise it is a boundary violation.

3. Set `boundary_pass = false` if ANY violation found. Otherwise `boundary_pass = true`.

4. When `boundary_pass = false`:
   - Reduce `scientific_accuracy` by at least 0.2
   - Cap overall score at 0.75
   - Explain violations in `feedback`""",
        ),
        PromptSection(
            title="Inputs",
            title_level=2,
            text="""The image is attached.

`QUESTION = {QUESTION}`
`ANSWER = {ANSWER}`
`REASONING_PATH = {REASONING_PATH}`
`RREL_BANK = {RREL_BANK}`
`FIGURE_CONTEXT = {FIGURE_CONTEXT}`
`QUESTION_SPEC = {QUESTION_SPEC}`""",
        ),
    ],
)
