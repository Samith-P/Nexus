# stage-3/section_detector.py

class SectionDetector:

    def __init__(self, text: str):
        self.text = text.lower()

        self.section_keywords = {
            "abstract": ["abstract"],
            "introduction": ["introduction", "background"],
            "methodology": ["method", "methods", "approach"],
            "results": ["results", "experiments"],
            "conclusion": ["conclusion", "discussion"]
        }

    def find_section_positions(self):
        positions = {}

        for section, keywords in self.section_keywords.items():
            for keyword in keywords:
                idx = self.text.find(keyword)
                if idx != -1:
                    positions[section] = idx
                    break

        return positions

    def detect_sections(self):
        sections = {
            "title": "",
            "abstract": "",
            "introduction": "",
            "methodology": "",
            "results": "",
            "conclusion": "",
            "others": ""
        }

        positions = self.find_section_positions()

        if not positions:
            sections["others"] = self.text
            return sections

        # Sort sections by position
        sorted_sections = sorted(positions.items(), key=lambda x: x[1])

        for i, (section, start_idx) in enumerate(sorted_sections):
            end_idx = (
                sorted_sections[i + 1][1]
                if i + 1 < len(sorted_sections)
                else len(self.text)
            )

            sections[section] = self.text[start_idx:end_idx]

        # Title = text before first section
        first_section_start = sorted_sections[0][1]
        sections["title"] = self.text[:first_section_start]

        return sections