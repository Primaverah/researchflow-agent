# Session, Retrieval, and Grounded Answer Quality Design

## Goal

Make persisted research chats resolve follow-up references correctly, retrieve
only relevant evidence, and answer the latest user question without inventing
facts when evidence is absent.

## Scope and constraints

- Work stays in the current repository and branch; no extra worktree, push, or
  merge.
- `fix/session-retrieval-quality` starts from `origin/main` at `55ad63b`.
- A single persistent SQLite database stores multiple sessions and isolates
  state by LangGraph `thread_id == session_id`; `:memory:` is prohibited.
- Research mode is grounded by default. Model-only answers require a separately
  selected, explicitly labelled non-grounded mode; this change does not enable
  that mode.
- Candidate limit is 10, selected-read limit is 5, and no API keys enter traces.

## Data flow and state contract

Each chat turn carries distinct values rather than deriving one from another by
string concatenation:

```
current_input -> conversation_history -> contextualizer -> standalone_query
                 resolved_subject          intent, required_facets
standalone_query + intent + required_facets -> search_queries -> candidates
-> selected_sources -> evidence_status -> summarizer_input -> final_answer
```

- `current_input` is exactly the newest user message and is the answer target.
- `conversation_history` contains appended user and assistant messages only;
  it is context, not query text.
- `resolved_subject` is the most recently explicit entity or entity set, such
  as `C和C++`, `成龙`, or `重返未来1999`.
- `standalone_query` replaces a reference with `resolved_subject`; it never
  copies a full preceding question and never removes an arbitrary character.
- `intent` classifies the request needed for retrieval (for example comparison,
  compiler, representative works, age, or update time).
- `required_facets` declares the facts required by the current intent. Compiler
  queries require compiler evidence, age queries require a birth date, and
  representative-work queries require works evidence.

The deterministic fallback recognises singular/plural people and things
(`它、它们、他、他们、两者、该项目、这个游戏`) by replacing the complete
reference token only. It uses `resolved_subject`; it may return an unresolved
query and request clarification rather than invent a subject. For example,
`C和C++有什么区别` stores `C和C++`, and `它们都用什么编译器` becomes
`C和C++都使用哪些常见编译器`.

## Checkpoint and interruption design

`messages` uses LangGraph's append reducer. A normal invocation submits only
the new user-message delta and current-turn fields; nodes return only fields
they changed. Empty lists and empty strings do not overwrite checkpointed
history. Both user messages and agent replies are checkpointed.

All `invoke`, `Command(resume=...)`, and `get_state` calls use the identical
`{"configurable": {"thread_id": session_id}}` configuration. `interrupt()` is
the first side-effecting boundary on the clarification path: it runs before
research, tools, trace writes, or note writes. A resumed graph continues with
the same thread, preserving its checkpoint. A new runner opens the same SQLite
file and can recover that state.

## Retrieval and evidence design

The query planner emits one query for a simple intent and a small stable set of
facet-specific, complementary queries for complex requests. Search output is
stable-deduplicated and capped at ten candidates. Candidates receive a
traceable score from title, snippet, fetched content, subject coverage, intent
facet coverage, and source quality. Merely naming the subject is insufficient.
The five highest eligible candidates are read.

Each selected source has a stable source id and an accept/reject reason. The
post-read assessor derives one of:

- `SUFFICIENT`: successful, relevant sources cover all required facets.
- `PARTIAL`: successful, relevant sources cover only some facets.
- `INSUFFICIENT`: no successful relevant source or no supported fact.

`INSUFFICIENT` triggers bounded replanning; after retries it returns a direct
evidence-insufficient response and does not call an LLM to fill facts.
`PARTIAL` answers only covered facets and names the gaps. The verifier permits
only claims with source ids from accepted reads and rejects unknown citations,
unsupported claims, or non-empty factual content with no evidence.

## Answer and trace design

The summarizer receives `current_input`, intent, facets, and accepted evidence
only. It is instructed to answer the newest input directly and to bind each
factual conclusion to source ids, rather than summarize source history. The
deterministic fallback follows the same evidence gate.

Structured diagnostic trace records include thread id, turn count, input,
history count, subject, standalone query, intent, facets, generated queries,
candidate scores and reasons, evidence status/gaps, summarizer source ids,
answer policy, and interrupt/resume state. Secret values are excluded.

## Validation matrix

Offline tests must establish: C/C++ compiler follow-up rewriting; the
three-turn Jackie Chan works/age flow; the game-update follow-up; session
isolation; cross-runner SQLite recovery; interruption with no duplicated tool
or file side effect; irrelevant-page rejection; grounded no-source refusal;
GBK and GB18030 decoding; and existing run/chat/offline behaviour.

After offline verification, controlled `.env.local` runs cover the same three
conversations plus two simultaneous session ids. They must show no LLM or
search fallback, correct standalone queries, bounded candidate/read/source
counts, current-input-oriented answers, citations from successful reads only,
and grounded refusal if no source can be read.

## Non-goals

- This does not alter the shared SQLite topology to one file per session.
- This does not hard-code facts, entities, sites, or model knowledge as
  retrieval evidence.
- This does not publish, merge, or push the branch.
