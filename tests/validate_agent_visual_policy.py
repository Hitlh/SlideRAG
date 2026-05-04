"""Validate AgentLoop visual follow-up policy.

Usage:
    python3 -m tests.validate_agent_visual_policy
"""

from __future__ import annotations

from rag_agent.agent.loop import AgentLoop


def main() -> None:
    should_force = AgentLoop._should_force_visual_verification
    has_direct_signal = AgentLoop._retrieval_has_direct_answer_signal
    is_strong_visual = AgentLoop._is_strong_visual_question

    assert should_force("How many ad types are shown in the figure?") is True
    assert should_force("What comes under Resources for the Center in the column to the left of Craniofacial?") is True
    assert should_force("Which label is closest to the right side of the diagram?") is True
    assert is_strong_visual("What follows The Implementation Process in the flow chart?") is True
    assert is_strong_visual("In the Carpet Model, what occurs directly before cell death?") is True
    assert is_strong_visual("What comes under Resources for the Cancer and Blood Disorders Center?") is True
    assert is_strong_visual("The percentage of total respondents will drop how many percentage points between 10 years ago and in 10 years?") is True
    assert should_force("The percentage of total respondents will drop how many percentage points between 10 years ago and in 10 years?") is True

    assert should_force("What is the difference in hours spent online per day between used-car buyers and new car buyers?") is False
    assert should_force("Do more used or new car buyers go online to have fun?") is False
    assert should_force("Which channel led to the most conversions?") is False
    assert should_force("What percentage of new car buyers switched loyalty?") is False

    clear_numeric_payload = {
        "evidence": {
            "chunks": [
                {
                    "content": "The spotlight image contains one figure standing in the center.",
                }
            ],
            "image_chunks": [],
        }
    }
    assert has_direct_signal(
        payload=clear_numeric_payload,
        question="How many figures are in the spotlight in the image?",
    ) is True
    assert is_strong_visual("How many figures are in the spotlight in the image?") is True

    factoid_payload = {
        "evidence": {
            "chunks": [
                {
                    "content": "Sam Shah is based in Mountain View, California.",
                }
            ],
            "image_chunks": [],
        }
    }
    assert has_direct_signal(
        payload=factoid_payload,
        question="Where is Sam Shah based?",
    ) is True

    weak_factoid_payload = {
        "evidence": {
            "chunks": [
                {
                    "content": "The Cancer and Blood Disorders Center has a Resources section.",
                }
            ],
            "image_chunks": [],
        }
    }
    assert has_direct_signal(
        payload=weak_factoid_payload,
        question="What comes under Resources for the Cancer and Blood Disorders Center?",
    ) is False

    missing_answer_payload = {
        "evidence": {
            "chunks": [
                {
                    "content": "This slide discusses auditorium lighting and presentation design.",
                }
            ],
            "image_chunks": [],
        }
    }
    assert has_direct_signal(
        payload=missing_answer_payload,
        question="How many figures are in the spotlight in the image?",
    ) is False

    print("AGENT_VISUAL_POLICY_TEST: PASS")


if __name__ == "__main__":
    main()
