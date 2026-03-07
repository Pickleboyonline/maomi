
## Planning Guidelines

When creating implementation plans, follow these principles:

### Structure
1. **Intent** - State the goal. What are we trying to achieve and why?
2. **Mechanism Overview** - Describe how things work structurally preferring pseudocode over actual code. You can use actual code for examples of whats to be implemented if its small, but overall prefer pseudocode of whats to be implemented. Full code review happens after the plan is understood and approved. This should be high level overview understanding of that the plan is to do, how, and why. 
3. **Steps with What + Why** - Every step must explain both what it does and why it's necessary. No steps without reasoning. If a step involves code, like the mechanisms part, explain the high level changes, preferring pseudocode over actual code. You can use actual code for examples of whats to be implemented if it's small, but overall prefer pseudocode of whats to be implemented.
4. **Concept Clarifications** - When a plan involves unfamiliar or nuanced concepts (protocols, patterns, library internals, etc.), explain them in plain language. Provide sources (docs, references) so the user can verify independently. Do not expect the user to trust AI interpretation at face value.

### Research-Informed Work
When research informs a plan, separate into two phases:
1. **Interpret** - Present what the docs/sources/existing code says, with references. No recommendations yet.
2. **Propose** - Only after the user has reviewed the interpretation, suggest implementation based on it.

Do not mix "what does the source material say" with "what should we do" in one pass.

### Debugging / Problem-Solving
- State what is observed, form a hypothesis, and obtain evidence. Do not fabricate causal explanations. When debugging, try to always look/observe the actual issue and use hypothesis's to determine where and how to search.
- Do not make changes without explaining the concrete mechanism of why they fix the issue.
- If stuck after 2-3 attempts on the same issue, stop and present findings rather than continuing down the same path.
- When going in circles, short-circuit: stop generating fixes, surface relevant source material / logs / docs, and let the user reason about it.

### Completion Criteria
A plan is understood when the user has no remaining questions. The user may rephrase concepts in their own words to verify understanding — confirm or correct as needed.


# General:
- When the user says `use worktrees`, OR YOU ARE MAKING A PLAN, or something similar for a medium/large feature, you will make a separate worktree/feature branch for the feature you are implementing. The folder will be adjacent to the maomi folder. And has a name with the suffix being related to the feature branch.