"""
Entity Extractor - Extract entities from text for memory enrichment.

Extracts:
- People names
- Technologies/tools/libraries
- Projects/repositories
- Skills/abilities
- Topics/subjects
- Keywords

Reference: vendor/tools/memori/ entity extraction patterns
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("titan.learning.entity_extractor")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class ExtractedEntities:
    """Entities extracted from text."""

    # People
    people: list[str] = field(default_factory=list)

    # Technology
    technologies: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)

    # Projects
    projects: list[str] = field(default_factory=list)
    repositories: list[str] = field(default_factory=list)

    # Concepts
    topics: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # Structured entities with metadata
    structured_entities: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "people": self.people,
            "technologies": self.technologies,
            "frameworks": self.frameworks,
            "languages": self.languages,
            "projects": self.projects,
            "repositories": self.repositories,
            "topics": self.topics,
            "skills": self.skills,
            "keywords": self.keywords,
        }

    @property
    def all_entities(self) -> list[str]:
        """Get all entities as a flat list."""
        entities = []
        entities.extend(self.people)
        entities.extend(self.technologies)
        entities.extend(self.frameworks)
        entities.extend(self.languages)
        entities.extend(self.projects)
        entities.extend(self.repositories)
        entities.extend(self.topics)
        entities.extend(self.skills)
        return list(set(entities))

    @property
    def all_keywords(self) -> list[str]:
        """Get all keywords including extracted entities."""
        return list(set(self.keywords + self.all_entities))


# ============================================================================
# Pattern-Based Extraction
# ============================================================================

# Known technologies
TECHNOLOGIES = {
    # Languages
    "python", "javascript", "typescript", "java", "go", "golang", "rust",
    "ruby", "php", "c++", "c#", "swift", "kotlin", "scala", "elixir",
    # Frameworks
    "react", "vue", "angular", "django", "flask", "fastapi", "express",
    "spring", "rails", "laravel", "nextjs", "nuxt", "svelte",
    # Tools
    "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins",
    "github", "gitlab", "aws", "azure", "gcp", "redis", "postgres",
    "postgresql", "mysql", "mongodb", "sqlite", "elasticsearch",
    # AI/ML
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "huggingface", "transformers", "langchain",
    "openai", "anthropic", "claude", "gpt", "llm", "rag",
}

# Common topics/skills
SKILLS = {
    "machine learning", "deep learning", "natural language processing",
    "nlp", "computer vision", "data science", "data engineering",
    "backend", "frontend", "full stack", "devops", "sre", "security",
    "api design", "microservices", "distributed systems", "databases",
    "cloud computing", "agile", "scrum", "testing", "ci/cd",
}


def extract_entities_pattern(text: str) -> ExtractedEntities:
    """
    Extract entities using pattern matching.

    This is a fast, rule-based approach.
    For production, consider using spaCy or LLM-based extraction.
    """
    entities = ExtractedEntities()
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))

    # Extract technologies (case-insensitive)
    for tech in TECHNOLOGIES:
        if tech in text_lower:
            if tech in {"python", "javascript", "typescript", "java", "go",
                        "rust", "ruby", "php", "swift", "kotlin", "scala"}:
                entities.languages.append(tech)
            elif tech in {"react", "vue", "angular", "django", "flask",
                          "fastapi", "express", "spring", "rails", "nextjs"}:
                entities.frameworks.append(tech)
            else:
                entities.technologies.append(tech)

    # Extract skills (multi-word)
    for skill in SKILLS:
        if skill in text_lower:
            entities.skills.append(skill)

    # Extract GitHub-style references
    # Repositories: owner/repo pattern
    repo_pattern = re.compile(r"\b([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)\b")
    for match in repo_pattern.findall(text):
        if "/" in match and not match.startswith("http"):
            entities.repositories.append(match)

    # Extract people (capitalized names, simplified)
    # Look for patterns like "John Smith", "Dr. Jane Doe"
    name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    for match in name_pattern.findall(text):
        # Filter out common non-names
        if match.lower() not in {"new york", "los angeles", "san francisco",
                                  "machine learning", "deep learning"}:
            entities.people.append(match)

    # Extract keywords (important nouns, simplified)
    # Look for capitalized words that might be project names
    project_pattern = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:[A-Z][a-z]+)+)\b")  # CamelCase
    for match in project_pattern.findall(text):
        if match not in entities.people:
            entities.projects.append(match)

    # Extract topics from context
    topic_indicators = {
        "about": re.compile(r"about\s+([a-z]+(?:\s+[a-z]+)?)", re.I),
        "regarding": re.compile(r"regarding\s+([a-z]+(?:\s+[a-z]+)?)", re.I),
        "working on": re.compile(r"working on\s+([a-z]+(?:\s+[a-z]+)?)", re.I),
        "project": re.compile(r"(\w+)\s+project", re.I),
    }

    for indicator, pattern in topic_indicators.items():
        for match in pattern.findall(text):
            if len(match) > 2:
                entities.topics.append(match.strip())

    # Generate keywords from unique words (filtered)
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between",
        "and", "but", "or", "nor", "so", "yet", "both", "either",
        "neither", "not", "only", "own", "same", "than", "too", "very",
        "just", "also", "now", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "any", "some", "no", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we",
        "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "its", "our", "their", "what", "which", "who", "whom",
    }

    for word in words:
        if (len(word) > 3 and
            word not in stopwords and
            word.isalpha() and
            word not in text_lower.split("@")):  # Skip email parts
            entities.keywords.append(word)

    # Deduplicate
    entities.people = list(set(entities.people))
    entities.technologies = list(set(entities.technologies))
    entities.frameworks = list(set(entities.frameworks))
    entities.languages = list(set(entities.languages))
    entities.projects = list(set(entities.projects))
    entities.repositories = list(set(entities.repositories))
    entities.topics = list(set(entities.topics))
    entities.skills = list(set(entities.skills))
    entities.keywords = list(set(entities.keywords))[:20]  # Limit keywords

    return entities


# ============================================================================
# LLM-Based Extraction (Enhanced)
# ============================================================================


async def extract_entities_llm(
    text: str,
    llm_complete: Any,  # Function to call LLM
) -> ExtractedEntities:
    """
    Extract entities using LLM for better accuracy.

    Args:
        text: Text to extract entities from
        llm_complete: Async function to call LLM

    Returns:
        ExtractedEntities with extracted data
    """
    prompt = f"""Extract entities from this text. Return JSON with these fields:
- people: List of people mentioned (names)
- technologies: List of technologies, tools, services
- frameworks: List of frameworks and libraries
- languages: List of programming languages
- projects: List of project names
- repositories: List of GitHub-style repos (owner/repo)
- topics: Main subjects discussed
- skills: Skills and abilities mentioned
- keywords: Important terms (max 10)

Text:
{text[:2000]}

Return ONLY valid JSON, no explanation."""

    from adapters.base import LLMMessage

    response = await llm_complete(
        [LLMMessage(role="user", content=prompt)],
        max_tokens=500,
    )

    try:
        import json
        data = json.loads(response.content)
        return ExtractedEntities(
            people=data.get("people", []),
            technologies=data.get("technologies", []),
            frameworks=data.get("frameworks", []),
            languages=data.get("languages", []),
            projects=data.get("projects", []),
            repositories=data.get("repositories", []),
            topics=data.get("topics", []),
            skills=data.get("skills", []),
            keywords=data.get("keywords", []),
        )
    except Exception as e:
        logger.warning(f"Failed to parse LLM entity response: {e}")
        # Fall back to pattern matching
        return extract_entities_pattern(text)


# ============================================================================
# Main API
# ============================================================================


def extract_entities(
    text: str,
    use_llm: bool = False,
    llm_complete: Any = None,
) -> ExtractedEntities:
    """
    Extract entities from text.

    Args:
        text: Text to extract from
        use_llm: Whether to use LLM (requires llm_complete)
        llm_complete: LLM completion function

    Returns:
        ExtractedEntities
    """
    if use_llm and llm_complete:
        import asyncio
        return asyncio.run(extract_entities_llm(text, llm_complete))

    return extract_entities_pattern(text)


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """
    Extract keywords from text (simplified).

    Args:
        text: Text to extract from
        max_keywords: Maximum keywords to return

    Returns:
        List of keywords
    """
    entities = extract_entities_pattern(text)
    keywords = entities.all_keywords
    return keywords[:max_keywords]
