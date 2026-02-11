"""Simulated LangChain-compatible tools for the curriculum agent.

Each tool has ``.name``, ``.description``, and ``.invoke(params)`` following
the LangChain tool interface expected by ``act_node``.  Tools mutate the
shared ``CurriculumState`` to simulate learner progression.
"""

import time

from .models import CurriculumState, QuizResult, Topic

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _topic_by_name(state: CurriculumState, name: str) -> Topic | None:
    """Find a topic in available_topics by name."""
    for t in state.available_topics:
        if t.name == name:
            return t
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class GenerateLessonTool:
    """Deliver a lesson on a given topic.

    Increments ``lessons_delivered``, updates ``current_topic``, and
    slightly boosts the learner's mastery for the topic (+0.1).
    Each lesson costs ~5 minutes of learner time.
    """

    name = "generate_lesson"
    description = "Generate and deliver a lesson on a curriculum topic"

    def __init__(self, state: CurriculumState) -> None:
        self._state = state

    def invoke(self, params: dict) -> str:
        topic_name = params.get("topic", "unknown")
        topic = _topic_by_name(self._state, topic_name)
        if topic is None:
            return f"Error: topic '{topic_name}' not found in curriculum."

        self._state.current_topic = topic_name
        self._state.lessons_delivered += 1
        self._state.learner.total_time_spent += 5.0

        # Delivering a lesson boosts mastery slightly
        current = self._state.learner.strengths.get(topic_name, 0.0)
        self._state.learner.strengths[topic_name] = min(1.0, current + 0.1)

        return (
            f"Lesson delivered: '{topic_name}' "
            f"(difficulty {topic.difficulty:.1f}). "
            f"Covered key concepts and examples. "
            f"Learner mastery now {self._state.learner.strengths[topic_name]:.2f}."
        )


class CreateQuizTool:
    """Administer a quiz on a given topic.

    Simulates the learner taking the quiz based on their mastery level.
    If mastery < 0.5 the quiz fails with score = mastery * 0.8.
    If mastery >= 0.5 the quiz passes with score = min(1.0, mastery * 1.1).
    Each quiz costs ~3 minutes of learner time.
    """

    name = "create_quiz"
    description = "Create and administer a quiz on a curriculum topic"

    def __init__(self, state: CurriculumState) -> None:
        self._state = state

    def invoke(self, params: dict) -> str:
        topic_name = params.get("topic", "unknown")
        topic = _topic_by_name(self._state, topic_name)
        if topic is None:
            return f"Error: topic '{topic_name}' not found in curriculum."

        self._state.learner.total_time_spent += 3.0
        self._state.learner.quizzes_taken += 1

        mastery = self._state.learner.strengths.get(topic_name, 0.0)

        if mastery < 0.5:
            score = mastery * 0.8
            passed = False
            weak_areas = [
                f"{topic_name}_fundamentals",
                f"{topic_name}_application",
            ]
        else:
            score = min(1.0, mastery * 1.1)
            passed = True
            weak_areas = []

        result = QuizResult(
            topic=topic_name,
            score=round(score, 3),
            passed=passed,
            timestamp=time.time(),
            weak_areas=weak_areas,
        )
        self._state.quiz_results.append(result)

        if passed and topic_name not in self._state.topics_completed:
            self._state.topics_completed.append(topic_name)

        status = "PASSED" if passed else "FAILED"
        weak_str = f" Weak areas: {', '.join(weak_areas)}." if weak_areas else ""
        return (
            f"Quiz result for '{topic_name}': {status} "
            f"(score {score:.2f}, mastery {mastery:.2f}).{weak_str}"
        )


class AssessPerformanceTool:
    """Provide an overall assessment of learner progress.

    Examines quiz results, mastery levels, and completion rate.
    Does not mutate state (read-only analysis).
    """

    name = "assess_performance"
    description = "Assess overall learner performance and progress"

    def __init__(self, state: CurriculumState) -> None:
        self._state = state

    def invoke(self, params: dict) -> str:
        learner = self._state.learner
        total_topics = len(self._state.available_topics)
        completed = len(self._state.topics_completed)

        # Average mastery across all topics
        if learner.strengths:
            avg_mastery = sum(learner.strengths.values()) / len(learner.strengths)
        else:
            avg_mastery = 0.0

        # Quiz pass rate
        total_quizzes = len(self._state.quiz_results)
        passed_quizzes = sum(1 for q in self._state.quiz_results if q.passed)
        pass_rate = passed_quizzes / max(total_quizzes, 1)

        # Failed topics
        failed_topics = list({
            q.topic for q in self._state.quiz_results if not q.passed
        })

        # Mastery distribution
        strong = [t for t, m in learner.strengths.items() if m >= 0.7]
        weak = [t for t, m in learner.strengths.items() if m < 0.4]

        return (
            f"Performance Assessment:\n"
            f"  Completion: {completed}/{total_topics} topics\n"
            f"  Average mastery: {avg_mastery:.2f}\n"
            f"  Quiz pass rate: {pass_rate:.0%} ({passed_quizzes}/{total_quizzes})\n"
            f"  Lessons delivered: {self._state.lessons_delivered}\n"
            f"  Time spent: {learner.total_time_spent:.0f} min\n"
            f"  Strong topics: {', '.join(strong) or 'none'}\n"
            f"  Weak topics: {', '.join(weak) or 'none'}\n"
            f"  Failed quizzes: {', '.join(failed_topics) or 'none'}"
        )


class SearchResourcesTool:
    """Find supplementary learning resources for a topic.

    Increments ``resources_found`` and slightly boosts mastery (+0.05)
    for the target topic.
    """

    name = "search_resources"
    description = "Search for supplementary learning resources on a topic"

    def __init__(self, state: CurriculumState) -> None:
        self._state = state

    def invoke(self, params: dict) -> str:
        topic_name = params.get("topic", "unknown")
        difficulty = params.get("difficulty", "intermediate")

        self._state.resources_found += 1

        # Boost mastery slightly
        current = self._state.learner.strengths.get(topic_name, 0.0)
        self._state.learner.strengths[topic_name] = min(1.0, current + 0.05)

        return (
            f"Found resources for '{topic_name}' ({difficulty} level): "
            f"3 tutorials, 2 practice exercises, 1 video lecture. "
            f"Learner mastery now {self._state.learner.strengths[topic_name]:.2f}."
        )


def create_tools(state: CurriculumState) -> list:
    """Create all curriculum tools bound to the given state."""
    return [
        GenerateLessonTool(state),
        CreateQuizTool(state),
        AssessPerformanceTool(state),
        SearchResourcesTool(state),
    ]
