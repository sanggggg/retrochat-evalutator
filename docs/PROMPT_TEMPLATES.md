# Prompt Templates Specification

This document specifies the prompt templates used in the training and evaluation modules.

## 1. Rubric Extractor Prompts

The system supports multiple rubric extractor prompts based on the score type being used. The appropriate prompt is automatically selected based on the `score_name` configuration.

### 1.1 Default Rubric Extractor

**File**: `prompts/rubric_extractor.txt`

**Purpose**: Extract evaluation rubrics from a high-quality chat session (general/default).

**Template**:
```
You are an expert at analyzing AI agent interactions and identifying patterns that indicate high-quality user behavior.

Analyze the following chat session between a user and an AI coding assistant. This session has been rated as high-quality.

## Chat Session
{chat_session}

## Task
Extract a list of evaluation rubrics that capture what makes this user interaction effective. Focus on:
- How clearly the user communicated their requirements
- How efficiently the user guided the AI toward the solution
- How well the user handled clarifications and corrections
- Any patterns of effective AI agent usage

## Output Format
Provide your response as a JSON array of rubrics. Each rubric should have:
- name: Short descriptive name (2-5 words)
- description: What this rubric measures (1-2 sentences)
- scoring_criteria: How to score from 1 (poor) to 5 (excellent)
- evidence: Specific example from this session demonstrating the rubric

```json
[
  {
    "name": "Example Rubric Name",
    "description": "What this measures...",
    "scoring_criteria": "1: Poor... 5: Excellent...",
    "evidence": "In turn 3, the user..."
  }
]
```

Extract {extraction_min_rubrics}-{extraction_max_rubrics} rubrics that are specific and actionable.
```

### 1.2 Excellence Rubric Extractor

**File**: `prompts/rubric_extractor_excellence.txt`

**Purpose**: Extract rubrics focused on excellence in accomplishing difficult tasks.

**Use Case**: When `score_name` is "excellence" or similar, this prompt focuses on:
- How the user leveraged AI capabilities to solve challenging problems
- How the user effectively communicated complex requirements
- How the user guided the AI through sophisticated tasks
- Patterns of effective collaboration on high-value tasks

### 1.3 Token Efficiency Rubric Extractor

**File**: `prompts/rubric_extractor_token_efficiency.txt`

**Purpose**: Extract rubrics focused on token efficiency (code growth per token used).

**Use Case**: When `score_name` is "token_efficiency" or similar, this prompt focuses on:
- How the user minimized unnecessary back-and-forth
- How the user provided clear, concise instructions
- How the user avoided redundant requests
- Patterns of efficient communication that reduce token waste

### 1.4 User Turn Efficiency Rubric Extractor

**File**: `prompts/rubric_extractor_user_turn_efficiency.txt`

**Purpose**: Extract rubrics focused on user turn efficiency (code growth per user message turn).

**Use Case**: When `score_name` is "user_turn_efficiency" or similar, this prompt focuses on:
- How the user achieved maximum progress with minimal message exchanges
- How the user provided comprehensive requirements upfront
- How the user effectively guided the AI without excessive iteration
- Patterns of effective communication that minimize back-and-forth exchanges

---

## 2. Rubric Summarizer Prompt

**File**: `prompts/rubric_summarizer.txt`

**Purpose**: Consolidate multiple rubric lists into a coherent final set.

**Template**:
```
You are an expert at synthesizing evaluation criteria for AI agent interactions.

You have been given rubrics extracted from multiple high-quality chat sessions. Your task is to consolidate these into a final, coherent set of evaluation rubrics.

## Extracted Rubrics from All Sessions
{all_rubrics}

## Task
Create a final list of 5-10 evaluation rubrics by:
1. Identifying common themes across the extracted rubrics
2. Merging similar or overlapping rubrics
3. Removing redundant or overly specific rubrics
4. Ensuring comprehensive coverage of user efficiency aspects
5. Making criteria clear and consistently scorable

## Output Format
Provide your response as a JSON object with the final rubrics:

```json
{
  "rubrics": [
    {
      "id": "rubric_001",
      "name": "Clear Initial Requirements",
      "description": "The user provides complete, unambiguous requirements in their initial request",
      "scoring_criteria": "1: Vague request requiring multiple clarifications\n2: Partial requirements, missing key details\n3: Adequate requirements with minor gaps\n4: Clear requirements with most details\n5: Comprehensive requirements covering edge cases",
      "weight": 1.0
    }
  ],
  "consolidation_notes": "Brief explanation of how rubrics were merged/prioritized"
}
```

Ensure each rubric:
- Has a unique, descriptive name
- Has clear, objective scoring criteria
- Is applicable across different types of coding tasks
- Focuses on USER behavior, not AI performance
```

---

## 3. LLM-as-a-Judge Template

**File**: `prompts/judge_template.txt`

**Purpose**: Evaluate a chat session against a specific rubric.

**Template**:
```
You are an impartial evaluator assessing the quality of user interaction with an AI coding assistant.

## Evaluation Rubric

### {rubric_name}
{rubric_description}

### Scoring Scale
{scoring_criteria}

---

## Chat Session to Evaluate
{chat_session}

---

## Evaluation Instructions

1. Carefully read the entire chat session
2. Focus ONLY on the user's behavior and communication, not the AI's responses
3. Look for specific evidence related to the rubric above
4. Assign a score based strictly on the scoring criteria

## Required Output Format

SCORE: [integer from 1 to 5]
REASONING: [2-3 sentences with specific evidence from the session]

Example:
SCORE: 4
REASONING: The user provided clear initial requirements including the specific function name and expected behavior. Minor deduction because edge cases were not mentioned until the second turn.

---

Now evaluate the session above:
```

---

## 4. Chat Session Formatting

All prompts receive chat sessions in a standardized format.

**Formatter Output**:
```
=== Chat Session ===
Session ID: {session_id}
Timestamp: {start_timestamp}
Git Branch: {git_branch} (if available)
Working Directory: {cwd}

--- Turn 1 ---
[USER]
{user_message_content}

[ASSISTANT]
{assistant_text_response}

[TOOL_CALLS]
- {tool_name}: {tool_input_summary}
- {tool_name}: {tool_input_summary}

[TOOL_RESULTS]
- {tool_name}: {truncated_result}

--- Turn 2 ---
[USER]
{user_message_content}

[ASSISTANT]
{assistant_text_response}

(continues for all turns...)

=== End Session ===
Total Turns: {turn_count}
```

**Formatting Rules**:
1. User messages: Include full content
2. Assistant text: Include full text responses
3. Tool calls: Summarize with tool name and key input parameters
4. Tool results: Truncate long outputs (> 500 chars) with "... (truncated)"
5. Thinking blocks: Omit from formatted output (internal reasoning)
6. File snapshots: Omit (not relevant for user behavior evaluation)

---

## Prompt Variables Summary

| Prompt | Variable | Description |
|--------|----------|-------------|
| Extractor (all variants) | `{chat_session}` | Formatted chat session |
| Extractor (all variants) | `{extraction_min_rubrics}` | Minimum number of rubrics to extract |
| Extractor (all variants) | `{extraction_max_rubrics}` | Maximum number of rubrics to extract |
| Summarizer | `{all_rubrics}` | JSON array of all extracted rubrics |
| Judge | `{rubric_name}` | Name of rubric being evaluated |
| Judge | `{rubric_description}` | Full rubric description |
| Judge | `{scoring_criteria}` | Scoring scale with descriptions |
| Judge | `{chat_session}` | Formatted chat session |

---

## Response Parsing

### Rubric Extraction Response

Expected format:
```json
[
  {"name": "...", "description": "...", "scoring_criteria": "...", "evidence": "..."},
  ...
]
```

Parsing strategy:
1. Find JSON array in response (may have surrounding text)
2. Validate each rubric has required fields
3. Generate unique IDs if not provided

### Rubric Summarization Response

Expected format:
```json
{
  "rubrics": [...],
  "consolidation_notes": "..."
}
```

Parsing strategy:
1. Extract JSON object from response
2. Validate rubrics array
3. Assign sequential IDs (rubric_001, rubric_002, ...)

### Judge Response

Expected format:
```
SCORE: 4
REASONING: The user demonstrated...
```

Parsing strategy:
1. Use regex to extract score: `SCORE:\s*(\d+)`
2. Use regex to extract reasoning: `REASONING:\s*(.+)`
3. Validate score is within 1-5 range
4. Retry if parsing fails
