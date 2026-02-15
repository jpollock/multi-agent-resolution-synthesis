"""Prompt templates for debate strategies."""

SYSTEM_CONTEXT_TEMPLATE = (
    "You are participating in a structured debate. The user's prompt "
    "includes context that is essential to the task. Treat the context "
    "as primary source material - reference it directly, address its "
    "specifics, and ensure your answer covers every requirement stated "
    "in both the context and prompt.\n\n"
    "CONTEXT:\n{context}"
)

CRITIQUE_INSTRUCTIONS = (
    "\nIMPORTANT: Re-read the original prompt and context above carefully. "
    "For each specific question or requirement in the original prompt, "
    "evaluate whether the other models addressed it adequately.\n\n"
    "1. Identify specific points where other answers are wrong, incomplete, "
    "or miss requirements from the original prompt.\n"
    "2. Identify what they got right that your answer missed.\n"
    "3. Call out where any answer (including yours) replaced concrete data "
    "from the original prompt with vague generalities.\n"
    "4. Provide your COMPLETE improved answer that addresses ALL "
    "requirements from the original prompt, incorporating valid points "
    "from others while correcting errors.\n\n"
    "When the prompt asks for examples, give CONCRETE examples using "
    "real data from the context - not generic placeholders. When it asks "
    "for code, prompts, or schemas, provide complete, usable output. "
    "Do not summarize or shorten - give a full, detailed answer."
)

EVALUATION_RULES = (
    "CRITICAL RULES:\n"
    "- Address EVERY numbered question or requirement in the original prompt.\n"
    "- When the prompt asks for examples, provide CONCRETE examples with "
    "real data, names, numbers, and specifics - not generic placeholders.\n"
    "- When the prompt or context mentions specific data (names, numbers, "
    "scores, versions), use that exact data in your answer.\n"
    "- When the prompt asks for code, prompts, schemas, or configs, "
    "provide complete, copy-pasteable output - not descriptions of what "
    "it would look like.\n"
    "- Prefer the most specific and detailed version of any point across "
    "the models. Never abstract a concrete example into a vague summary.\n"
    "- If models disagree, pick the version with the strongest reasoning "
    "and most specificity.\n\n"
    "Structure your response in two sections:\n\n"
    "## Resolution Analysis\n"
    "For each model, list which specific points you accepted and which "
    "you rejected, with reasoning tied to the original requirements.\n\n"
    "## Final Answer\n"
    "Provide the complete synthesized answer. Match the level of detail "
    "and specificity the original prompt demands."
)

SYNTHESIS_PREAMBLE = (
    "\nSynthesize the best possible answer from all models' responses. "
    "Re-read the original prompt and context above carefully.\n\n"
)

JUDGE_PREAMBLE = (
    "\nYou are the judge. Re-read the original prompt and context above "
    "carefully. Evaluate each response against EVERY specific requirement "
    "in the original prompt.\n\n"
)
