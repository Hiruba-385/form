
"""Utility functions for EduXplain."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class StudentFeatures:
    """Student academic and engagement details."""

    attendance: float
    assignment_score: float
    internal_marks: float
    study_hours: float
    previous_score: float
    participation: float

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "StudentFeatures":
        """Read and validate student data from the frontend."""

        if not isinstance(payload, dict):
            raise ValueError("Expected student data as a JSON object.")

        # Support nested payloads.
        for nested_key in ("data", "student", "features", "student_data"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                payload = {**payload, **nested}

        # Normalize keys: attendance_pct -> attendancepct
        normalized = {
            re.sub(r"[^a-z0-9]", "", str(key).lower()): value
            for key, value in payload.items()
        }

        # Frontend field names are included in these aliases.
        aliases = {
            "attendance": [
                "attendance",
                "attendancepercentage",
                "attendancepct",
                "attendancerate",
            ],
            "assignment_score": [
                "assignmentscore",
                "assignmentmarks",
                "avgassignmentscore",
                "assignment",
            ],
            "internal_marks": [
                "internalmarks",
                "internalscore",
                "avgquizscore",
                "internal",
            ],
            "study_hours": [
                "studyhours",
                "dailystudyhours",
                "hoursstudiedperweek",
                "hours",
            ],
            "previous_score": [
                "previousscore",
                "previousmarks",
                "priortopicscore",
                "previousexamscore",
            ],
            "participation": [
                "participation",
                "classparticipation",
                "classparticipationscore",
                "participationscore",
            ],
        }

        values = {}

        for field, possible_names in aliases.items():
            found = False
            found_value = None

            for alias in possible_names:
                if alias in normalized and normalized[alias] not in (None, ""):
                    found_value = normalized[alias]
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Missing required student field: {field}. "
                    f"Received fields: {list(payload.keys())}"
                )

            try:
                value = float(found_value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Student field '{field}' must contain a valid number."
                )

            if field == "study_hours":
                if value < 0:
                    raise ValueError("Study hours cannot be negative.")
            elif not 0 <= value <= 100:
                raise ValueError(
                    f"'{field}' must be between 0 and 100."
                )

            values[field] = value

        return cls(**values)

    def to_prompt_dict(self) -> Dict[str, float]:
        """Convert features to a dictionary."""

        return {
            "attendance": self.attendance,
            "assignment_score": self.assignment_score,
            "internal_marks": self.internal_marks,
            "study_hours": self.study_hours,
            "previous_score": self.previous_score,
            "participation": self.participation,
        }


PROMPT_TEMPLATE = """
You are EduXplain, a student academic performance analysis assistant.

Analyze these student details:
- Attendance: {attendance}%
- Assignment score: {assignment_score}%
- Average quiz score: {internal_marks}%
- Study hours per week: {study_hours}
- Previous topic score: {previous_score}%
- Class participation score: {participation}%

Return ONLY one valid JSON object with these keys:
{{
  "performance_level": "High, Medium, or Low",
  "predicted_score": 0,
  "probability": 0.0,
  "strengths": ["strength 1", "strength 2"],
  "areas_for_improvement": ["area 1", "area 2"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "summary": "A short explanation of the student's performance."
}}

Rules:
- predicted_score must be between 0 and 100.
- probability must be between 0 and 1.
- probability represents estimated confidence, not a guaranteed outcome.
- strengths, areas_for_improvement and recommendations must be arrays of strings.
- Base the analysis on the supplied data.
- Do not include Markdown or text outside the JSON object.
"""


def build_prompt(features: StudentFeatures) -> str:
    """Build the Gemini prompt."""
    return PROMPT_TEMPLATE.format(**features.to_prompt_dict())


def extract_first_json_block(text: str) -> Dict[str, Any]:
    """Extract a valid JSON object from Gemini's response."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini returned an empty response.")

    cleaned = text.strip()
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", cleaned):
        try:
            result, _ = decoder.raw_decode(cleaned[match.start():])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    logger.error("Invalid Gemini JSON response: %r", cleaned[:2000])
    raise ValueError(
        "Gemini response did not contain a valid JSON object. "
        "Check the terminal for details."
    )


def validate_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Gemini's prediction result."""

    if not isinstance(data, dict):
        raise ValueError("Gemini response must be a JSON object.")

    required = [
        "performance_level",
        "predicted_score",
        "probability",
        "strengths",
        "areas_for_improvement",
        "recommendations",
        "summary",
    ]

    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(
            f"Gemini response is missing required fields: {missing}"
        )

    for key in ("predicted_score", "probability"):
        try:
            data[key] = float(data[key])
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a number.")

    if not 0 <= data["predicted_score"] <= 100:
        raise ValueError("'predicted_score' must be between 0 and 100.")

    if not 0 <= data["probability"] <= 1:
        raise ValueError("'probability' must be between 0 and 1.")

    for key in ("strengths", "areas_for_improvement", "recommendations"):
        if isinstance(data[key], str):
            data[key] = [data[key]]
        elif isinstance(data[key], list):
            data[key] = [str(item) for item in data[key]]
        else:
            raise ValueError(f"'{key}' must be a list or string.")

    data["performance_level"] = str(data["performance_level"])
    data["summary"] = str(data["summary"])

    return data
