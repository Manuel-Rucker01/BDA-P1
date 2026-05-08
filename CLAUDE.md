# Claude Code Instructions

## Communication Style & Token Efficiency
- Be extremely concise and omit all conversational filler.
- Do not explain code unless explicitly asked to do so.
- Prioritize minimal token usage in all responses.
- Always code in English (including comments and variable names).

## Local Delegation Tools (Token Saving)

**BEFORE using Read or Write tools, check availability once per session:**
```bash
which ask-local write-local
```
If either is missing, tell the user and stop — do NOT silently fall back to Read/Write.

### ask-local — bulk reading (MANDATORY trigger)
**You MUST use this instead of the Read tool when ANY of these are true:**
- You are about to read 3 or more files in the same task
- A single file is over 400 lines

```bash
ask-local --paths <file1> <file2>... --question "<question>"
```
**NEVER use the Read tool as a fallback if ask-local is available.**

### write-local — boilerplate generation (MANDATORY trigger)
**You MUST use this instead of Write when ALL of these are true:**
- The output is a new file (not editing an existing one)
- AND the file is a test, config, analysis script, or follows a repetitive pattern

```bash
write-local --spec "<what to generate>" --context <reference_file> --target <output_path>
```
Then review the output with Read and apply only the delta with Edit.
**NEVER use the Write tool to create new files of the above types if write-local is available.**

### When NOT to delegate
- Tasks under ~2000 tokens total
- Architectural decisions, debugging, safety-critical code
- Anything requiring exact line numbers for editing
- Editing existing files (use Edit directly)

**Decision Logic:**
- Debugging a race condition or numerical stability? **Use Claude.**
- Reading 5 files to understand which ports are in use? **Use ask-local.**
- Writing a new test file matching existing patterns? **Use write-local.**

### Documentation workflow (MANDATORY)
**NEVER write documentation directly. Always delegate.**
1. The user will manually run `extract-chat` to generate a clean transcript (e.g., `/tmp/chat.txt`).
2. Use `ask-local` to read the transcript and existing docs to get update suggestions:
   `ask-local --paths /tmp/chat.txt docs/architecture.md --question "Read the chat. What doc updates are needed? Give exact edits."`
3. Apply the suggestions making surgical edits to save tokens.