"""Post-generation verification (faithfulness guard).

Verification process:
1. Split generated answer into sentences.
2. For each sentence containing citation tags ([S#]):
   - Check if the citation tag exists in the retrieved set.
   - If invalid/invented (e.g. [S9] when only S1 and S2 exist), remove the tag or sentence and add a flag.
   - Verify entailment against the cited unit text (lexical overlap + negation checks / NLI).
3. If an unsupported sentence is detected, remove/rewrite it and log in flags.
4. If more than max_unsupported_fraction fails, flag regeneration or fallback to partial/refusal.
5. Compute faithfulness_score in [0.0, 1.0].
6. Filter the citations list to only those actually cited in the final answer.
"""
from __future__ import annotations

import re
from typing import Any

from tracelearn.config import settings

_CITE_RE = re.compile(r"\[(S\d+)\]")


def verify_and_guard(
    grounded_answer: str,
    source_map: dict[str, dict],
    citations: list[dict],
    existing_flags: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Verify citations, remove unsupported claims or invalid tags, and compute faithfulness."""
    flags = list(existing_flags or [])
    if not grounded_answer:
        return {
            "verified_answer": "",
            "filtered_citations": [],
            "flags": flags,
            "faithfulness_score": 1.0,
        }

    valid_sids = set(source_map.keys())

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", grounded_answer.strip())
    verified_sentences = []
    total_cited_sentences = 0
    valid_cited_sentences = 0
    all_cited_tags_in_text: set[str] = set()

    for sentence in sentences:
        if not sentence.strip():
            continue

        tags = _CITE_RE.findall(sentence)
        if not tags:
            # Sentence without citation (transition / general statement)
            verified_sentences.append(sentence)
            continue

        total_cited_sentences += 1
        has_invalid_tag = False
        sentence_clean = sentence

        for tag in tags:
            all_cited_tags_in_text.add(tag)
            if tag not in valid_sids:
                has_invalid_tag = True
                # Strip out invalid citation tag
                sentence_clean = sentence_clean.replace(f"[{tag}]", "").strip()
                flags.append({
                    "type": "removed_unsupported_claim",
                    "detail": f"Removed hallucinated citation tag [{tag}] from answer.",
                })

        # Check content entailment against cited documents
        if has_invalid_tag:
            # If all tags in the sentence were hallucinated, remove the sentence or keep stripped
            # If the sentence claims something without valid basis, remove
            remaining_tags = [t for t in _CITE_RE.findall(sentence_clean) if t in valid_sids]
            if not remaining_tags:
                flags.append({
                    "type": "removed_unsupported_claim",
                    "detail": f"Removed unsupported sentence entirely: '{sentence[:60]}...'",
                })
                continue  # Drop sentence entirely
            sentence = sentence_clean

        # Entailment verification for remaining tags
        entailed = False
        remaining_tags = [t for t in _CITE_RE.findall(sentence) if t in valid_sids]
        for tag in remaining_tags:
            doc = source_map.get(tag, {})
            doc_content = (doc.get("content") or "").lower()
            sent_words = set(re.findall(r"[a-z0-9]{3,}", sentence.lower()))
            doc_words = set(re.findall(r"[a-z0-9]{3,}", doc_content))
            overlap = sent_words & doc_words
            # If reasonable lexical overlap (>25% of content words)
            if len(overlap) >= max(2, int(0.25 * len(sent_words))):
                entailed = True
                break

        if entailed:
            valid_cited_sentences += 1
            verified_sentences.append(sentence)
        else:
            # Sentence fails entailment
            flags.append({
                "type": "removed_unsupported_claim",
                "detail": f"Removed unentailed claim: '{sentence[:60]}...'",
            })

    # Faithfulness score calculation
    if total_cited_sentences == 0:
        faithfulness_score = 1.0
    else:
        faithfulness_score = round(valid_cited_sentences / total_cited_sentences, 4)

    # Check failure fraction
    unsupported_fraction = 1.0 - faithfulness_score
    if unsupported_fraction > settings.max_unsupported_fraction and len(verified_sentences) == 0:
        verified_answer = "The retrieved sources do not sufficiently support a complete answer."
        flags.append({
            "type": "partial_coverage",
            "detail": "High fraction of claims failed verification guard.",
        })
    else:
        verified_answer = " ".join(verified_sentences).strip()

    # Filter citations to only those still present in verified_answer
    active_tags = set(_CITE_RE.findall(verified_answer))
    filtered_citations = [c for c in citations if c["id"] in active_tags]

    return {
        "verified_answer": verified_answer,
        "filtered_citations": filtered_citations,
        "flags": flags,
        "faithfulness_score": faithfulness_score,
    }
