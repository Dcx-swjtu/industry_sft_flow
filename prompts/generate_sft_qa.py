"""Prompt for generate_sft_qa — convert reasoning path into free-form Q&A for SFT."""

from dataflow.prompts.base import InputField, OutputField, PromptSection, PromptTemplate


GENERATE_SFT_QA_PROMPT = PromptTemplate(
    name="generate_sft_qa",
    requires_image=True,
    input_fields=[
        InputField("rrel_bank", "{RREL_BANK}", "rrel.rrel_bank"),
        InputField("reasoning_path", "{REASONING_PATH}", "compose.reasoning_path"),
        InputField("figure_context", "{FIGURE_CONTEXT}", "sample.metadata"),
        InputField("judge_feedback", "{JUDGE_FEEDBACK}", "judge.feedback"),
        InputField("question_spec", "{QUESTION_SPEC}", "compose.question_spec"),
    ],
    output_key="sft_qa_output",
    output_fields=[
        OutputField("question", "string"),
        OutputField("answer", "string"),
        OutputField("difficulty_signals", "object"),
        OutputField("internal_path", "object"),
        OutputField("generation_summary", "object"),
    ],
    sections=[
        PromptSection(
            title="Role",
            title_level=2,
            text="""You are converting a structured reasoning path about a scientific figure into a natural language question-answer pair for supervised fine-tuning (SFT). The question must require genuine visual reasoning about the figure — someone must look at the image to answer it. The answer must provide a coherent, step-by-step explanation that references specific visual elements from the figure.""",
        ),
        PromptSection(
            title="Output Contract",
            title_level=2,
            text="""Return exactly:

```json
{
  "sft_qa_output": {
    "question": "A natural language question that requires looking at the figure to answer",
    "answer": "A coherent paragraph (or multi-paragraph) answer with step-by-step reasoning",
    "difficulty_signals": {
      "min_reasoning_steps": 3,
      "requires_comparison": true,
      "requires_quantitative_reasoning": false,
      "panels_needed": ["a", "b"],
      "visual_elements_referenced": ["bar chart", "distance labels"],
      "text_dependency_level": "none"
    },
    "internal_path": {
      "latent_target": "...",
      "step_count": 4,
      "reasoning_chain": "S1 -> S2 -> S3 -> S4"
    },
    "generation_summary": {
      "question_type": "causal mechanism with comparison",
      "answer_length_tokens_estimate": 200,
      "grounding_confidence": "high",
      "question_type_category": "causal_mechanism"
    }
  }
}
```""",
        ),
        PromptSection(
            title="Question Guidelines",
            title_level=2,
            text="""The question must satisfy ALL of these criteria:

1. **Requires the image**: The question cannot be answered from the caption alone or from general scientific knowledge. The reader must examine specific visual elements in the figure.

2. **Multi-step reasoning**: The question should require at least 2-3 steps of reasoning, not a single lookup. Good questions ask about mechanisms, comparisons, trends, or integrative conclusions.

3. **Specific but not leading**: The question should be specific enough that there is a clear correct answer, but should not reveal the reasoning path or final conclusion. It may include minimal necessary premises from `question_spec.must_include`, including identity-to-panel mappings. Do not include visual observations that directly answer the question.

4. **Natural language**: The question should sound like something a scientist would genuinely ask about this figure. Avoid robotic phrasing like "Based on the reasoning chain S1-S4..."

5. **Authorize textual definitions explicitly**: If the answer needs a caption/context-derived abbreviation expansion, feature meaning, cohort identity, or method identity, state that minimal definition in the question first. Otherwise omit the interpretation and describe only the visible labeled pattern.

**Question types** — Vary the question type based on what the figure and reasoning path support. Do NOT always default to the same "integrative analysis" style. Choose from these types:

- **Causal mechanism**: "What structural change explains the selective loss of activity for long-chain substrates?"
- **Comparison / contrast**: "How do the temporal dynamics of mutation X differ from mutation Y across disease stages?"
- **Predictive / counterfactual**: "If the mutation only affected the active site rather than the distal pocket, what would you expect to see in the activity profile?"
- **Diagnostic / identification**: "Which genetic event marks the transition from pre-malignant to malignant state, and what visual evidence distinguishes it from earlier events?"
- **Trend interpretation**: "What pattern in the dose-response curve suggests a threshold effect rather than a gradual transition?"
- **Multi-panel synthesis**: "How do the spatial distribution data in panel g and the temporal tracking in panel f together constrain the timing of the SUV420H1 acquisition?"

Bad questions (too easy or caption-answerable):
- "What color is the bar for condition A?" (single lookup, no reasoning)
- "What does the figure show?" (too vague)
- "How many panels are in the figure?" (no scientific reasoning)

Good questions (require visual reasoning):
- "How does the mutation affect the enzyme's substrate chain-length specificity, and what structural change explains this selectivity?"
- "Which stage of disease progression shows the most dramatic shift in clonal architecture, and how does this relate to the boundary between pre-malignant and malignant states?" """,
        ),
        PromptSection(
            title="Answer Guidelines",
            title_level=2,
            text="""The answer must be a coherent, self-contained explanation:

1. **Step-by-step reasoning**: Walk through the logical chain that leads from visual observations to the conclusion. Each step should build on previous ones.

2. **Visual grounding**: Reference specific visual elements from the figure — panel names, bar positions, axis values, color differences, structural features, labels. Use phrases like:
   - "In panel a, the WT distance is 3.9 Å while the mutant shows 5.3 Å"
   - "The bar chart in panel d shows that condition X has approximately twice the value of the control"
   - "Comparing the two traces in panel e reveals..."

3. **Scientific precision**: Use accurate scientific language. State quantitative observations with appropriate precision (exact when labeled, approximate when estimated from axes).

   For structural modeling figures, prefer "supports", "is consistent with", or "would favor" unless the figure directly demonstrates causality. Do not convert a modeled structural association into claims such as "forces" or "dictates" without direct evidence.

   If a question asks for a total, combined, minimum overall loss, or net balance and the figure provides multiple contributing quantities, explicitly aggregate every included component before drawing the conclusion. If only one dominant component is compared, name that narrower component in both question and answer.

   For observational plots and schematics, avoid absolute mechanistic phrasing such as "eliminates", "directly translates to", "causes", or "overcomes" unless that intervention or mechanism is explicitly established by the question premises. Prefer "shows minimal", "is associated with", "is consistent with", or "supports".

4. **Complete but concise**: Cover all aspects of the question, but avoid unnecessary repetition or tangential information. A good answer is typically 3-8 sentences for a focused question, or 2-3 paragraphs for a broader question.

5. **No meta-language**: Do not say "Looking at the figure, I can see..." or "Based on the evidence graph...". Just state the observations and reasoning directly.""",
        ),
        PromptSection(
            title="Leakage Prevention",
            title_level=2,
            text="""Critical rules to prevent information leakage:

- Do NOT expose internal reasoning path step IDs (S1, S2, etc.) in the question or answer
- Do NOT reference RREL IDs or evidence graph terminology
- The question should not reveal the intermediate steps of the reasoning chain
- The answer should flow as natural scientific reasoning, not as a translation of the structured path
- Do NOT mention "the reasoning path" or "the RREL bank" — these are internal pipeline artifacts

The Q&A should read as if a knowledgeable scientist examined the figure and wrote the question and answer from scratch.""",
        ),
        PromptSection(
            title="Retry Feedback Integration",
            title_level=2,
            text="""When JUDGE_FEEDBACK is non-empty, you are generating a revised version based on quality evaluation feedback. In this case:

1. Read the feedback carefully and identify the specific issues flagged
2. Address each issue directly in the revised question and/or answer
3. Common issues to fix:
   - **Factual incorrectness**: Correct any claims that don't match the figure
   - **Weak visual grounding**: Add more specific visual references (panel names, positions, values)
   - **Reasoning gaps**: Fill in missing logical steps
   - **Question too easy**: Make the question require more visual reasoning
   - **Answer incomplete**: Ensure all aspects of the question are addressed
4. Do not overcorrect — preserve what was already working well

`JUDGE_FEEDBACK = {JUDGE_FEEDBACK}`""",
        ),
        PromptSection(
            title="Question Specification",
            title_level=2,
            text="""You must respect the following question specification:

`QUESTION_SPEC = {QUESTION_SPEC}`

Follow this generation order:

**Step A — Write the question first:**
- If `must_include` lists premises, the question MUST state each one explicitly.
  Example: if must_include says "thick curve = CNF electrode", write: "Assuming the thick curve represents the CNF electrode..."
- If `must_not_use` lists forbidden categories, do NOT use them anywhere in question or answer unless the question explicitly provides the relevant information.

**Step B — Verify the question:**
- Does the question include all `must_include` items? If not, revise.
- Does the question avoid all `must_not_use` items unless explicitly given? If not, revise.

**Step C — Write the answer within the question's boundary:**
- The question defines the information boundary.
- Only use information that is (a) visible in the figure, (b) stated as a given in the question, or (c) general scientific reasoning principles.
- Do NOT introduce method names, biological identities, or causal mechanisms that are not visible and not given in the question.""",
        ),
        PromptSection(
            title="Final Boundary Check",
            title_level=2,
            text="""Before returning the QA pair, reread the answer sentence by sentence:

- Every expansion of a label or abbreviation must either be visible in the image or appear as a premise in the question.
- Every mechanism or causal phrase must either be directly shown or be softened to an interpretation supported by the visible pattern.
- Every qualitative summary such as `balanced`, `uniform`, `dominant`, or `unique` must be plainly supported by the visual comparison; otherwise describe the visible distribution more narrowly.

Revise the existing question or answer when one of these checks fails; do not add extra metadata fields.""",
        ),
        PromptSection(
            title="Inputs",
            title_level=2,
            text="""The image is attached.

`RREL_BANK = {RREL_BANK}`
`REASONING_PATH = {REASONING_PATH}`
`FIGURE_CONTEXT = {FIGURE_CONTEXT}`
`QUESTION_SPEC = {QUESTION_SPEC}`""",
        ),
    ],
)
