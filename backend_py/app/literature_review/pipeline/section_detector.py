# app/pipeline/section_detector.py
# Stage 3 — Section detection (REWRITTEN — regex-based heading detection)
# Fixes: preserves case, uses regex for numbered/capitalized headings,
#         separates Discussion from Conclusion, robust fallback.
#         Handles "1 Introduction" (no period) and stops at References.

import re
from typing import Dict

from literature_review.utils.logger import get_logger

logger = get_logger(__name__)

# Heading patterns ranked by specificity
HEADING_PATTERNS = [
    # "1. Introduction" or "1 Introduction" or "1) Introduction"
    re.compile(r"^\s*(\d{1,2})\s*[.\):]?\s+([A-Z][A-Za-z\s\-:]{2,50})$", re.MULTILINE),
    # "I. INTRODUCTION" or "II. RELATED WORK" or "II RELATED WORK"
    re.compile(r"^\s*(I{1,3}V?|VI{0,3}|IX|X)\s*[.\):]?\s+([A-Z][A-Za-z\s\-:]{2,50})$", re.MULTILINE),
    # "ABSTRACT" / "INTRODUCTION" (all caps on its own line, 4+ chars)
    re.compile(r"^\s*([A-Z][A-Z\s]{3,40})\s*$", re.MULTILINE),
    # "Abstract" / "Introduction" etc. (title case, standalone line)
    re.compile(
        r"^\s*(Abstract|Introduction|Background|Related Work|Literature Review|"
        r"Methodology|Methods?|Approach|Proposed (?:Method|Approach|System|Model|Framework)|"
        r"(?:Our |The )?(?:Model|Framework|System|Architecture)|"
        r"Experiments?|Experimental (?:Setup|Results)|Results?|Evaluation|"
        r"Discussion|Conclusion|Conclusions|Summary|Future Work|"
        r"Acknowledgements?|References|Bibliography|Appendix)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
]

# Map detected heading text to canonical section name
SECTION_MAP = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "related work": "introduction",
    "literature review": "introduction",
    "method": "methodology",
    "methods": "methodology",
    "methodology": "methodology",
    "approach": "methodology",
    "proposed method": "methodology",
    "proposed approach": "methodology",
    "proposed system": "methodology",
    "proposed model": "methodology",
    "proposed framework": "methodology",
    "our model": "methodology",
    "our framework": "methodology",
    "our system": "methodology",
    "our approach": "methodology",
    "the model": "methodology",
    "the framework": "methodology",
    "the architecture": "methodology",
    "model": "methodology",
    "framework": "methodology",
    "system": "methodology",
    "architecture": "methodology",
    "experiment": "results",
    "experiments": "results",
    "experimental setup": "results",
    "experimental results": "results",
    "results": "results",
    "evaluation": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "summary": "conclusion",
    "future work": "conclusion",
    # Stop markers — we slice content before these
    "acknowledgements": "_stop",
    "acknowledgment": "_stop",
    "references": "_stop",
    "bibliography": "_stop",
    "appendix": "_stop",
}


class SectionDetector:

    def __init__(self, text: str):
        if not text or not text.strip():
            logger.warning("SectionDetector received empty input.")
        self.text = text  # preserve original case

    def _find_headings(self):
        """Find all headings with their positions in the text."""
        headings = []

        for pattern in HEADING_PATTERNS:
            for match in pattern.finditer(self.text):
                # The heading text is the last group (handles numbered vs standalone)
                heading_text = match.group(match.lastindex).strip()
                pos = match.start()

                # Normalize to canonical section name
                heading_lower = heading_text.lower().strip()

                # Skip very short or very long matches (noise)
                if len(heading_lower) < 3 or len(heading_lower) > 50:
                    continue

                # Skip lines that are clearly not headings (too many words = paragraph)
                if len(heading_lower.split()) > 6:
                    continue

                canonical = SECTION_MAP.get(heading_lower)
                if canonical is None:
                    # Try partial match for complex headings
                    for key, val in SECTION_MAP.items():
                        if key in heading_lower:
                            canonical = val
                            break

                if canonical:
                    headings.append((pos, canonical, heading_text))

        # Deduplicate: keep earliest occurrence of each canonical section
        seen = {}
        for pos, canonical, original in sorted(headings, key=lambda x: x[0]):
            if canonical not in seen:
                seen[canonical] = (pos, original)

        return seen

    def detect_sections(self) -> Dict[str, str]:
        """Detect and extract academic paper sections.

        Returns:
            dict with keys: title, abstract, introduction, methodology,
                          results, discussion, conclusion, others
        """
        sections = {
            "title": "",
            "abstract": "",
            "introduction": "",
            "methodology": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "others": "",
        }

        found = self._find_headings()

        if not found:
            logger.warning("No section headings detected — putting all text in 'others'.")
            sections["others"] = self.text
            return sections

        # Remove _stop markers but use their position to limit content
        stop_pos = len(self.text)
        if "_stop" in found:
            stop_pos = found["_stop"][0]
            del found["_stop"]

        if not found:
            sections["others"] = self.text[:stop_pos]
            return sections

        # Sort by position
        sorted_found = sorted(found.items(), key=lambda x: x[1][0])

        # Title = text before first heading
        first_pos = sorted_found[0][1][0]
        title_text = self.text[:first_pos].strip()
        if title_text:
            sections["title"] = title_text

        # Extract each section's content
        for i, (section_name, (start_pos, _original)) in enumerate(sorted_found):
            if i + 1 < len(sorted_found):
                end_pos = sorted_found[i + 1][1][0]
            else:
                end_pos = stop_pos  # Stop at References/Appendix, not end of file

            content = self.text[start_pos:end_pos].strip()
            sections[section_name] = content

        detected = [name for name, content in sections.items() if content]
        logger.info(f"Detected sections: {detected}")

        return sections
