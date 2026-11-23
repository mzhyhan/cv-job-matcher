from __future__ import annotations

import pytest

from cv_matcher import Skill, SkillHit, extract_skills, load_taxonomy
from cv_matcher.skills import _Matcher
from tests.conftest import SAMPLE_CV_TEXT, SAMPLE_JOB_TEXT


def names(text: str, **kwargs: object) -> set[str]:
    return {hit.name for hit in extract_skills(text, **kwargs)}  # type: ignore[arg-type]


def test_taxonomy_loads_with_expected_shape() -> None:
    skills = load_taxonomy()
    assert 80 <= len(skills) <= 200
    assert all(isinstance(skill, Skill) for skill in skills)
    assert len({skill.name for skill in skills}) == len(skills)
    assert {"language", "ml", "web", "data", "cloud", "devops", "tooling"} <= {
        skill.category for skill in skills
    }
    by_name = {skill.name: skill for skill in skills}
    assert by_name["PostgreSQL"].category == "data"
    assert "Postgres" in by_name["PostgreSQL"].aliases
    assert by_name["R"].ambiguous


def test_taxonomy_is_cached() -> None:
    assert load_taxonomy() is load_taxonomy()


def test_taxonomy_aliases_do_not_collide_across_skills() -> None:
    seen: dict[str, str] = {}
    for skill in load_taxonomy():
        for form in (skill.name, *skill.aliases):
            owner = seen.setdefault(form.lower(), skill.name)
            assert owner == skill.name, f"{form!r} belongs to {owner} and {skill.name}"


def test_malformed_taxonomy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResource:
        def read_text(self, encoding: str) -> str:
            return '{"skills": [{"category": "x"}]}'

    class FakeRoot:
        def joinpath(self, *parts: str) -> FakeRoot | FakeResource:
            return FakeResource() if parts == ("skills.json",) else FakeRoot()

    monkeypatch.setattr("cv_matcher.skills.resources.files", lambda package: FakeRoot())
    load_taxonomy.cache_clear()
    try:
        with pytest.raises(ValueError, match="Malformed"):
            load_taxonomy()
    finally:
        load_taxonomy.cache_clear()


def test_exact_match_is_case_insensitive() -> None:
    hits = extract_skills("Experienced with PYTHON, pytorch and Docker.")
    assert {hit.name for hit in hits} == {"Python", "PyTorch", "Docker"}
    assert all(hit.method == "exact" and hit.confidence == 1.0 for hit in hits)


def test_hit_fields_are_populated() -> None:
    (hit,) = extract_skills("We use Postgres daily.")
    assert isinstance(hit, SkillHit)
    assert hit.name == "PostgreSQL"
    assert hit.category == "data"
    assert hit.matched_text == "Postgres"
    assert hit.alias == "Postgres"
    assert hit.method == "alias"
    assert 0.9 <= hit.confidence < 1.0


def test_alias_extraction() -> None:
    found = names("sklearn, k8s, GenAI, golang and a bit of TS")
    assert {"scikit-learn", "Kubernetes", "Generative AI", "Go", "TypeScript"} <= found


def test_multi_word_skills() -> None:
    found = names("Strong machine learning and continuous integration experience.")
    assert {"Machine Learning", "CI/CD"} <= found


def test_multi_word_skill_tolerates_line_breaks() -> None:
    assert "Machine Learning" in names("machine\n  learning")


def test_symbol_skills_with_trailing_punctuation() -> None:
    found = names("Languages: C++, C#, Node.js. Frameworks: ASP.NET, .NET.")
    assert {"C++", "C#", "Node.js", "ASP.NET", ".NET"} <= found


def test_java_is_not_matched_inside_javascript() -> None:
    assert names("Frontend work in JavaScript") == {"JavaScript"}
    assert names("Backend in Java") == {"Java"}


@pytest.mark.parametrize(
    "text",
    [
        "R&D department and go-to-market strategy",
        "C-level stakeholders, go to the shop, I want to go home",
        "Plan A, plan B, plan C-ish and r2d2",
        "go",
        "the letter r and letter c",
    ],
)
def test_ambiguous_short_names_do_not_false_positive(text: str) -> None:
    assert names(text) == set()


def test_ambiguous_names_match_when_clearly_a_skill() -> None:
    hits = {hit.name: hit for hit in extract_skills("Languages: R, Go, C and Python.")}
    assert {"R", "Go", "C", "Python"} <= set(hits)
    assert hits["R"].confidence < hits["Python"].confidence


def test_ambiguous_names_are_case_sensitive() -> None:
    assert names("we go forward and swift action") == set()


def test_ambiguous_name_at_end_of_sentence() -> None:
    assert "Go" in names("My main language is Go.")
    assert "Go" not in names("see github.com/Go.dev")


def test_fuzzy_punctuation_variants() -> None:
    (hit,) = extract_skills("sci-kit learn models")
    assert hit.name == "scikit-learn"
    assert hit.method == "normalised"
    assert names("Hugging-Face and Tensor Flow") >= {"Hugging Face", "TensorFlow"}


def test_fuzzy_typos() -> None:
    hits = {hit.name: hit for hit in extract_skills("Kubernets and tensorflw clusters")}
    assert set(hits) == {"Kubernetes", "TensorFlow"}
    assert all(hit.method == "fuzzy" and 0.7 < hit.confidence < 1.0 for hit in hits.values())


def test_fuzzy_can_be_disabled() -> None:
    assert names("Kubernets", fuzzy=False) == set()


def test_fuzzy_threshold_is_configurable() -> None:
    assert names("Kubernets", fuzzy_threshold=99.0) == set()


def test_fuzzy_does_not_pull_in_unrelated_words() -> None:
    text = (
        "Responsibilities include communication, management, documentation, "
        "presentations, collaboration, leadership, planning and delivery. "
        "Thinking about terrain, transform, tensors and pandemic planning."
    )
    assert names(text) == set()


def test_similar_skills_are_not_confused() -> None:
    assert names("TypeScript") == {"TypeScript"}
    assert names("MySQL and PostgreSQL") == {"MySQL", "PostgreSQL"}


def test_each_skill_reported_once_in_order_of_appearance() -> None:
    hits = extract_skills("Docker, python, DOCKER again and Python once more")
    assert [hit.name for hit in hits] == ["Docker", "Python"]


def test_best_occurrence_wins() -> None:
    (hit,) = extract_skills("Wrote postgres queries, later tuned PostgreSQL")
    assert hit.matched_text == "PostgreSQL"
    assert hit.confidence == 1.0


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_yields_no_hits(text: str) -> None:
    assert extract_skills(text) == ()


def test_text_without_skills() -> None:
    assert extract_skills("Seeking a rewarding opportunity in a friendly team.") == ()


def test_sample_fixtures() -> None:
    assert names(SAMPLE_CV_TEXT) == {"Python"}
    assert {"Python", "Natural Language Processing"} <= names(SAMPLE_JOB_TEXT)


def test_custom_taxonomy() -> None:
    taxonomy = (
        Skill("Widgetry", "tooling", ("widget making",)),
        Skill("Q", "language", (), ambiguous=True),
    )
    assert names("Widget making and Q", taxonomy=taxonomy) == {"Widgetry", "Q"}
    assert names("Python", taxonomy=taxonomy) == set()


def test_empty_taxonomy() -> None:
    assert extract_skills("Python", taxonomy=()) == ()


def test_matcher_ignores_forms_without_alphanumerics() -> None:
    matcher = _Matcher((Skill("---", "tooling"),))
    assert matcher._squashed == {}
