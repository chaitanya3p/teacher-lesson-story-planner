"""Shared prompt format, used by dataset builder, trainer, pipeline and evaluator."""
import re

SYSTEM = (
    "You are a classroom assistant that writes age-appropriate stories and lesson plans "
    "for school teachers. Reply in exactly this format:\n"
    "STORY: <a short story that naturally uses the keywords>\n"
    "LESSON PLAN:\nTitle, Objectives, Materials, Activities (with minutes), "
    "Discussion Questions, Assessment."
)


def make_user_prompt(grade, subject, topic, keywords):
    kws = ", ".join(keywords) if not isinstance(keywords, str) else keywords
    return f"Grade: {grade}\nSubject: {subject}\nTopic: {topic}\nKeywords: {kws}"


def parse_user_prompt(text):
    out = {}
    for key in ("Grade", "Subject", "Topic", "Keywords"):
        m = re.search(rf"^{key}:\s*(.*)$", text, re.M)
        out[key.lower()] = m.group(1).strip() if m else ""
    try:
        out["grade"] = int(re.sub(r"\D", "", out["grade"]) or 3)
    except ValueError:
        out["grade"] = 3
    out["keywords"] = [k.strip() for k in out["keywords"].split(",") if k.strip()]
    return out


def split_output(text):
    """Return (story, lesson_plan). Either may be '' if the model broke format."""
    m = re.search(r"STORY:\s*(.*?)\s*LESSON PLAN:\s*(.*)", text, re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"STORY:\s*(.*)", text, re.S)
    return (m.group(1).strip() if m else ""), ""
