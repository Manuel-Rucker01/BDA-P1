# Claude Code Instructions

## Communication Style & Token Efficiency
- Be extremely concise and omit all conversational filler.
- Do not explain code unless explicitly asked to do so.
- Prioritize minimal token usage in all responses.
- Always code in English (including comments and variable names).

## Local Delegation Tools (Token Saving)

**SESSION START — run this first, every session, before any Read or Write:**
```bash
which ask-local write-local
```
If either binary is missing: tell the user and stop. Do NOT silently fall back.

---

### MANDATORY pre-flight check before every Read call

Before invoking the Read tool, answer these two questions out loud in your reasoning:

1. **Will I read more than 1 file in this task (total)?** → If yes, use ask-local for ALL of them.
2. **Is this file over 400 lines?** → If yes, use ask-local.

If either answer is yes and ask-local is available, you MUST use ask-local. Using Read anyway is a violation.

---

### ask-local — bulk reading

```bash
ask-local --paths <file1> <file2>... --question "<question>"
```

**Triggers (ANY one is sufficient):**
- More than 1 file will be read in the current task
- A single file exceeds 400 lines

**Hard rules:**
- NEVER use Read as a fallback when ask-local is available and a trigger applies.
- Do NOT start with Read for "just one file to orient myself" and then switch — decide upfront.
- ask-local ALWAYS runs as a background task. After launching it, immediately call `TaskOutput` with `block=true` to wait for the result before proceeding:
  ```
  TaskOutput(task_id="<id from ask-local output>", block=true, timeout=60000)
  ```

---

### write-local — boilerplate generation

```bash
write-local --spec "<what to generate>" --context <reference_file> --target <output_path>
```

**Triggers (ALL must be true):**
- Output is a new file (not editing an existing one)
- File is a test, config, analysis script, or follows a repetitive pattern

**Hard rules:**
- NEVER use Write to create new files of the above types if write-local is available.
- After write-local runs, review output with Read and apply only the delta with Edit.

---

### When NOT to delegate
- Editing existing files → use Edit directly (never write-local)
- Architectural decisions, debugging, safety-critical numerical code → use Claude directly
- Need exact line numbers for an edit → read that single file with Read (only if no other files needed in the task)

---

### Documentation workflow (MANDATORY)
**NEVER write documentation directly. Always delegate.**
1. User runs `extract-chat` to produce a transcript (e.g., `/tmp/chat.txt`).
2. Delegate reading + diffing:
   `ask-local --paths /tmp/chat.txt docs/architecture.md --question "What doc updates are needed? Give exact edits."`
3. Apply suggestions with surgical Edit calls.