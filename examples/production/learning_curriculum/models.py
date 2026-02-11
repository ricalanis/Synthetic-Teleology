"""Domain model for the Adaptive Learning Curriculum agent."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Topic:
    """A single topic in the curriculum."""

    name: str
    prerequisites: list[str] = field(default_factory=list)
    difficulty: float = 0.5  # 0-1


@dataclass(frozen=True)
class QuizResult:
    """Result of a quiz attempt."""

    topic: str
    score: float  # 0-1
    passed: bool
    timestamp: float
    weak_areas: list[str] = field(default_factory=list)


@dataclass
class LearnerProfile:
    """Simulated learner with per-topic mastery levels."""

    strengths: dict[str, float] = field(default_factory=dict)  # topic -> mastery 0-1
    weaknesses: list[str] = field(default_factory=list)
    total_time_spent: float = 0.0  # minutes
    quizzes_taken: int = 0


@dataclass
class CurriculumState:
    """Mutable state tracked across the curriculum run."""

    topics_completed: list[str] = field(default_factory=list)
    current_topic: str | None = None
    quiz_results: list[QuizResult] = field(default_factory=list)
    learner: LearnerProfile = field(default_factory=LearnerProfile)
    available_topics: list[Topic] = field(default_factory=list)
    lessons_delivered: int = 0
    resources_found: int = 0


# ---------------------------------------------------------------------------
# React fundamentals curriculum
# ---------------------------------------------------------------------------

REACT_CURRICULUM = [
    Topic("jsx_basics", [], 0.2),
    Topic("components", ["jsx_basics"], 0.3),
    Topic("props", ["components"], 0.3),
    Topic("state_management", ["props"], 0.6),
    Topic("hooks", ["state_management"], 0.7),
    Topic("effects", ["hooks"], 0.6),
    Topic("context", ["state_management"], 0.7),
    Topic("routing", ["components"], 0.5),
    Topic("forms", ["state_management"], 0.5),
    Topic("testing", ["components", "hooks"], 0.8),
]


def create_default_learner() -> LearnerProfile:
    """Create a learner who is strong on basics but weak on advanced topics.

    Mastery values are calibrated so that:
    - Basic topics pass quizzes after 1 lesson (mastery already >= 0.5)
    - state_management fails its first quiz (0.2 + 2 lessons = 0.4 < 0.5),
      triggering goal revision
    - After remedial content, state_management reaches 0.7 and passes
    - hooks / effects / routing need 2-3 lessons to reach 0.5 for passing
    """
    return LearnerProfile(
        strengths={
            "jsx_basics": 0.8,
            "components": 0.7,
            "props": 0.55,
            "state_management": 0.2,
            "hooks": 0.3,
            "effects": 0.3,
            "context": 0.15,
            "routing": 0.40,
            "forms": 0.3,
            "testing": 0.05,
        },
        weaknesses=["state_management", "hooks", "context", "testing"],
    )
