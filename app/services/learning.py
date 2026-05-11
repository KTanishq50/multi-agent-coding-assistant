from __future__ import annotations
import re
from app.services.memory import store_memory
from app.services.session_memory import session_memory


def _extract(text: str, key: str) -> str | None:
    key_lower = key.lower()
    text_lower = text.lower()
    if key_lower not in text_lower:
        return None
    start = text_lower.index(key_lower) + len(key)
    end = text.find("\n", start)
    val = text[start:end].strip() if end != -1 else text[start:].strip()
    return val if len(val) > 8 else None


def _extract_lesson_from_prose(review_text: str, query: str) -> str | None:
    lines = review_text.strip().split("\n")
    for line in lines[1:8]:
        line = line.strip()
        if re.match(r"^[\d\-\*•]\.", line) or re.match(r"^\d+\.", line):
            cleaned = re.sub(r"^[\d\-\*•]+\.\s*", "", line).strip()
            if len(cleaned) > 20:
                return f"For '{query[:40]}': {cleaned[:120]}"
    for line in lines[1:5]:
        line = line.strip()
        if len(line) > 30 and not line.upper().startswith("REJECTED"):
            return f"For '{query[:40]}': {line[:120]}"
    return None


def process_learning(
    review_text: str,
    answer: str = "",
    intent: str = "",
    query: str = "",
) -> None:
    if not review_text:
        return

    review_upper = review_text.upper()

    # determine outcome 
    # covers: "APPROVED", "ACCEPTED", "Accepted after retry."
    is_approved = (
        "APPROVED" in review_upper
        or "ACCEPTED" in review_upper
    )
    is_rejected = "REJECTED" in review_upper

    # learn from rejections
    if is_rejected:
        mistake = _extract(review_text, "Mistake:")
        fix     = _extract(review_text, "Fix:")
        lesson  = _extract(review_text, "Lesson:")

        if mistake and fix:
            store_memory(mistake, fix, tags=[intent] if intent else [])
            print(f"[LEARNING] Stored mistake→fix pattern")

        if lesson:
            session_memory.add(lesson)
            print(f"[LEARNING] Session: {lesson[:60]}")

        if not lesson and query:
            prose_lesson = _extract_lesson_from_prose(review_text, query)
            if prose_lesson:
                session_memory.add(prose_lesson)
                print(f"[LEARNING] Session (prose): {prose_lesson[:60]}")

        if not mistake and not fix:
            numbered = re.findall(r"\d+\.\s+(.+?)(?=\n|$)", review_text)
            for item in numbered[:2]:
                item = item.strip()
                if len(item) > 20:
                    store_memory(
                        mistake=f"{intent} task: {query[:50]}",
                        fix=item[:200],
                        tags=[intent, "reviewer_feedback"]
                    )
                    print(f"[LEARNING] Stored reviewer feedback as pattern")
                    break

    # learn from approvals (including retry-accepted)
    if is_approved and answer and intent in ("write", "fix", "explain", "general"):
        # store function patterns for write/fix
        if intent in ("write", "fix"):
            func_matches = re.findall(
                r"(def \w+\([^)]*\):(?:\n(?:[ \t]+.*))+)",
                answer,
                re.MULTILINE
            )
            for match in func_matches[:2]:
                snippet = match.strip()
                if len(snippet) > 40:
                    name_match = re.match(r"def (\w+)", snippet)
                    func_name = name_match.group(1) if name_match else "function"
                    store_memory(
                        mistake=f"Pattern for {intent}: {func_name}",
                        fix=snippet[:300],
                        tags=[intent, "approved_pattern", func_name]
                    )
                    print(f"[LEARNING] Stored approved pattern: {func_name}")

        # session note for all approved intents
        first_line = answer.strip().split("\n")[0][:80]
        if len(first_line) > 20:
            session_memory.add(f"Previously answered: {first_line}")
            print(f"[LEARNING] Session note: {first_line[:60]}")

    # always: session continuity regardless of outcome 
    # fires for ALL intents including general
    if answer and query:
        first_code_line = ""
        for line in answer.split("\n"):
            if line.strip().startswith(("def ", "class ")):
                first_code_line = line.strip()[:60]
                break

        if first_code_line:
            session_memory.add(f"Wrote: {first_code_line}")
            print(f"[LEARNING] Session wrote: {first_code_line[:50]}")
        elif answer.strip() and len(query) > 5:
            note = f"Answered '{query[:40]}': {answer.strip()[:60]}"
            session_memory.add(note)
            print(f"[LEARNING] Session answered: {note[:60]}")