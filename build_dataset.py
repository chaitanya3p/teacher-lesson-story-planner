"""Build the instruction-tuning dataset (JSONL, chat 'messages' format).

Pipeline: curriculum seeds -> 3 story structures x lesson-plan variants -> de-duplicate -> critic quality
gate (structure, keywords, safety, reading level, length) -> topic-level hold-out split -> data_report.json.
Outputs: train / val / test (seen topics) / test_unseen (topics never used in training).
"""
import argparse, json, random
from curriculum import TOPICS
from common import SYSTEM, make_user_prompt

NAMES = ["Asha", "Ravi", "Meena", "Kiran", "Sana", "Arjun", "Lakshmi", "Imran", "Divya", "Teja"]
PLACES = ["a small village near a river", "a busy town with a big school", "a green hill town", "a quiet seaside town"]
GUIDES = ["Grandmother", "Teacher", "Uncle Raju", "Aunty Latha", "Grandfather"]


def band(grade):
    return "lower" if grade <= 2 else "mid" if grade <= 5 else "upper"


def _join(kws):
    return kws[0] if len(kws) == 1 else ", ".join(kws[:-1]) + " and " + kws[-1]


def _story(t, name, place, guide, kws, grade, style):
    b = band(grade)
    k1 = kws[0]
    s = []
    if style == 0:      # guide explains the facts in dialogue
        if b == "lower":
            s += [f"{name} lived in {place}.", f"One day, {name} saw {k1} and asked, \"What is this?\""]
        elif b == "mid":
            s += [f"{name} lived in {place}.", f"One morning, {name} noticed something curious about {k1} and could not stop wondering about it."]
        else:
            s += [f"{name} lived in {place}, and loved asking difficult questions.",
                  f"While studying {t['topic'].lower()} for school, {name} became curious about how {k1} really works."]
        s.append(f"{name} ran to ask {guide}.")
        for f in t["facts"]:
            s.append(f"{guide} said, \"{f}\"")
            s.append(f"{name} listened carefully and nodded." if b != "lower" else f"{name} nodded.")
        s.append(f"Then they tried a small activity together. They did this: {t['activity'][0].lower() + t['activity'][1:]}.")
    elif style == 1:    # the child investigates first, then learns
        s += [f"{name} lived in {place}.",
              f"At school, {name}'s class began a new lesson about {t['topic'].lower()}."]
        s.append(f"{name} wanted to find out more about {k1}, so {name} tried this: {t['activity'][0].lower() + t['activity'][1:]}.")
        s.append(f"{guide} came to see what {name} was doing and smiled.")
        for i, f in enumerate(t["facts"]):
            lead = ["First", "Next", "Finally"][min(i, 2)]
            s.append(f"{lead}, {guide} explained, \"{f}\"")
        s.append(f"{name} wrote everything down in a notebook." if b != "lower" else f"{name} drew a picture to remember it.")
    else:               # the class solves a small problem together
        s += [f"{name} lived in {place}.",
              f"One day, {guide} gave the class a puzzle about {k1}, and nobody knew the answer."]
        s.append(f"{name} said, \"Let us find out together.\"")
        s.append(f"The children tried an experiment: {t['activity'][0].lower() + t['activity'][1:]}.")
        for f in t["facts"]:
            s.append(f"They discovered this: {f}")
        s.append(f"At last the puzzle was solved, and {guide} clapped for the whole class.")
    if b == "upper":
        s.append(f"That evening, {name} thought about how {_join(kws[:2])} connect to things seen every day.")
    s.append(f"{name} smiled and remembered the words: {_join(kws)}.")
    s.append(f"The next day, {name} shared everything with the class.")
    return " ".join(s)


def render(t, name, place, guide, kws, grade, rng=None):
    """Return the 'STORY: ... LESSON PLAN: ...' target text for topic dict t.
    rng picks one of 3 story structures, a lesson length and a lesson-plan wording variant."""
    style = rng.randrange(3) if rng else 0
    mins = rng.choice([(5, 15, 10, 10), (5, 15, 10, 10), (5, 20, 10, 10), (5, 15, 15, 10)]) if rng else (5, 15, 10, 10)
    hook = rng.choice(["Hook", "Warm-up", "Starter"]) if rng else "Hook"
    assess = rng.randrange(3) if rng else 0
    story = _story(t, name, place, guide, kws, grade, style)
    obj0 = t["objectives"][0][0].lower() + t["objectives"][0][1:]
    assessments = [
        "Assessment: Exit ticket - each student writes two sentences using the key words. Check that they can " + obj0 + ".",
        "Assessment: Quick oral quiz - ask three students to explain one key word each. Check that they can " + obj0 + ".",
        "Assessment: Draw and label - each student draws one thing from the story and labels it with a key word. Check that they can " + obj0 + ".",
    ]
    q = list(t["questions"]) + [f"Can you explain {kws[-1]} in your own words?"]
    plan = [
        f"Title: {t['topic']} - a story-based lesson",
        f"Grade: {grade} | Subject: {t['subject']} | Duration: {sum(mins)} minutes",
        "Objectives:",
        *[f"- {o}" for o in t["objectives"]],
        "Materials: " + ", ".join(t["materials"]),
        "Activities:",
        f"1. {hook} ({mins[0]} min): Read the story aloud and ask students to predict what {name} will learn.",
        f"2. Explore ({mins[1]} min): {t['activity']}.",
        f"3. Discuss ({mins[2]} min): Review the key words - {_join(kws)} - using the questions below.",
        f"4. Practise ({mins[3]} min): Students draw or write one thing they learned and share with a partner.",
        "Discussion Questions:",
        *[f"{i}. {x}" for i, x in enumerate(q, 1)],
        assessments[assess],
    ]
    return "STORY: " + story + "\nLESSON PLAN:\n" + "\n".join(plan)


def make_example(t, rng):
    n_kw = rng.choice([2, 3, 3, 4])
    kws = rng.sample(t["keywords"], min(n_kw, len(t["keywords"])))
    out = render(t, rng.choice(NAMES), rng.choice(PLACES), rng.choice(GUIDES), kws, t["grade"], rng)
    user = make_user_prompt(t["grade"], t["subject"], t["topic"], kws)
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": out}],
            "meta": {"grade": t["grade"], "subject": t["subject"], "topic": t["topic"]}}


def clean(ex):
    """Quality gate: keep only examples that pass the same critic used at inference time."""
    from common import parse_user_prompt
    from pipeline import planner, critic
    req = parse_user_prompt(ex["messages"][1]["content"])
    issues, _ = critic(planner(req["grade"], req["subject"], req["topic"], req["keywords"]), ex["messages"][2]["content"])
    return not issues, issues


def main():
    import collections, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=60)
    ap.add_argument("--heldout-topics", type=int, default=6, help="topics kept out of training for the unseen-topic test set")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # topic-level hold-out: evaluates generalisation to topics the model never saw in training
    by_grade = sorted(TOPICS, key=lambda t: (t["grade"], t["topic"]))
    heldout = {t["topic"] for t in rng.sample(by_grade, a.heldout_topics)}

    seen, rows, rejected = set(), [], collections.Counter()
    n_raw = n_dup = 0
    for t in TOPICS:
        n, tries = 0, 0
        while n < a.per_topic and tries < a.per_topic * 30:
            tries += 1
            ex = make_example(t, rng); n_raw += 1
            key = ex["messages"][2]["content"]
            if key in seen:                       # 1. de-duplicate
                n_dup += 1; continue
            ok, issues = clean(ex)                # 2. validate with the critic
            if not ok:
                for i in issues:
                    cat = ("reading level too high" if "reading level" in i else "keywords missing" if "Keywords" in i
                           else "unsafe words" if "Unsuitable" in i else "length" if "length" in i else "structure")
                    rejected[cat] += 1
                continue
            seen.add(key); rows.append(ex); n += 1

    unseen = [r for r in rows if r["meta"]["topic"] in heldout]
    pool = [r for r in rows if r["meta"]["topic"] not in heldout]
    rng.shuffle(pool)
    n = len(pool); a_end, b_end = int(n * .8), int(n * .9)
    splits = {"train": pool[:a_end], "val": pool[a_end:b_end], "test": pool[b_end:], "test_unseen": unseen}
    os.makedirs(a.out, exist_ok=True)
    for name, part in splits.items():
        with open(f"{a.out}/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "topics": len(TOPICS), "heldout_topics": sorted(heldout),
        "generated_raw": n_raw, "duplicates_removed": n_dup, "rejected_by_critic": dict(rejected),
        "kept": len(rows), "split_sizes": {k: len(v) for k, v in splits.items()},
        "by_grade": dict(sorted(collections.Counter(r["meta"]["grade"] for r in rows).items())),
        "by_subject": dict(collections.Counter(r["meta"]["subject"] for r in rows)),
        "unique_outputs": len({r["messages"][2]["content"] for r in rows}),
        "train_test_topic_overlap": sorted({r["meta"]["topic"] for r in splits["train"]} & {r["meta"]["topic"] for r in splits["test_unseen"]}),
    }
    json.dump(report, open(f"{a.out}/data_report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
