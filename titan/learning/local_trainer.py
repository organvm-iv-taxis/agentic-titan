"""
Local Trainer - Learn from user's local codebase patterns.

Provides:
- Pattern extraction from codebases
- Style adaptation for code generation
- Local fine-tuning support
- Coding convention learning

Reference: vendor/agents/igor/ local learning patterns
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("titan.learning.local_trainer")


# ============================================================================
# Data Structures
# ============================================================================


class PatternType(str, Enum):
    """Types of coding patterns."""
    NAMING = "naming"  # Variable/function naming
    STRUCTURE = "structure"  # Code organization
    STYLE = "style"  # Formatting preferences
    IDIOM = "idiom"  # Common code patterns
    IMPORT = "import"  # Import organization
    DOCSTRING = "docstring"  # Documentation style
    ERROR_HANDLING = "error_handling"  # Exception patterns
    TYPING = "typing"  # Type annotation style


@dataclass
class CodingPattern:
    """A coding pattern extracted from source code."""

    type: PatternType
    name: str
    description: str
    frequency: int = 1
    examples: list[str] = field(default_factory=list)
    confidence: float = 1.0

    # Context
    language: str = "python"
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "frequency": self.frequency,
            "examples": self.examples[:5],  # Limit examples
            "confidence": self.confidence,
            "language": self.language,
        }


@dataclass
class StyleProfile:
    """A user's coding style profile."""

    # Naming conventions
    variable_case: str = "snake_case"  # snake_case, camelCase, PascalCase
    function_case: str = "snake_case"
    class_case: str = "PascalCase"
    constant_case: str = "UPPER_SNAKE_CASE"
    private_prefix: str = "_"

    # Formatting
    indent_size: int = 4
    max_line_length: int = 88
    quote_style: str = "double"  # single, double
    trailing_comma: bool = True

    # Imports
    import_order: list[str] = field(default_factory=lambda: ["stdlib", "third_party", "local"])
    import_style: str = "grouped"  # grouped, alphabetical, length

    # Documentation
    docstring_style: str = "google"  # google, numpy, sphinx
    inline_comments: bool = True
    type_annotations: bool = True

    # Error handling
    exception_style: str = "specific"  # specific, generic
    error_messages: str = "descriptive"  # descriptive, terse

    # Patterns
    patterns: list[CodingPattern] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "naming": {
                "variable_case": self.variable_case,
                "function_case": self.function_case,
                "class_case": self.class_case,
                "constant_case": self.constant_case,
                "private_prefix": self.private_prefix,
            },
            "formatting": {
                "indent_size": self.indent_size,
                "max_line_length": self.max_line_length,
                "quote_style": self.quote_style,
                "trailing_comma": self.trailing_comma,
            },
            "imports": {
                "order": self.import_order,
                "style": self.import_style,
            },
            "documentation": {
                "docstring_style": self.docstring_style,
                "inline_comments": self.inline_comments,
                "type_annotations": self.type_annotations,
            },
            "error_handling": {
                "exception_style": self.exception_style,
                "error_messages": self.error_messages,
            },
            "patterns": [p.to_dict() for p in self.patterns[:20]],
        }

    def to_prompt_context(self) -> str:
        """Generate prompt context from style profile."""
        lines = ["# Coding Style Guide", ""]

        # Naming
        lines.extend([
            "## Naming Conventions",
            f"- Variables: {self.variable_case}",
            f"- Functions: {self.function_case}",
            f"- Classes: {self.class_case}",
            f"- Constants: {self.constant_case}",
            f"- Private members: prefix with '{self.private_prefix}'",
            "",
        ])

        # Formatting
        lines.extend([
            "## Formatting",
            f"- Indent: {self.indent_size} spaces",
            f"- Max line length: {self.max_line_length}",
            f"- Quotes: {self.quote_style}",
            f"- Trailing commas: {'yes' if self.trailing_comma else 'no'}",
            "",
        ])

        # Documentation
        lines.extend([
            "## Documentation",
            f"- Docstring style: {self.docstring_style}",
            f"- Type annotations: {'yes' if self.type_annotations else 'no'}",
            "",
        ])

        # Common patterns
        if self.patterns:
            lines.extend(["## Common Patterns", ""])
            for pattern in self.patterns[:10]:
                lines.append(f"- {pattern.name}: {pattern.description}")
                if pattern.examples:
                    lines.append(f"  Example: `{pattern.examples[0][:100]}`")

        return "\n".join(lines)


@dataclass
class TrainingConfig:
    """Configuration for local training."""

    # Source directories
    source_dirs: list[str] = field(default_factory=list)
    include_patterns: list[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: list[str] = field(default_factory=lambda: ["*test*", "*__pycache__*", "*.pyc"])

    # Analysis options
    min_file_size: int = 100  # bytes
    max_file_size: int = 1_000_000  # 1MB
    min_examples: int = 3  # Min occurrences to be a pattern

    # Learning options
    learn_naming: bool = True
    learn_structure: bool = True
    learn_idioms: bool = True
    learn_docstrings: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dirs": self.source_dirs,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "min_file_size": self.min_file_size,
            "max_file_size": self.max_file_size,
            "min_examples": self.min_examples,
        }


@dataclass
class TrainingResult:
    """Result of local training."""

    style_profile: StyleProfile
    files_analyzed: int = 0
    patterns_found: int = 0
    training_time_ms: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_analyzed": self.files_analyzed,
            "patterns_found": self.patterns_found,
            "training_time_ms": self.training_time_ms,
            "errors": self.errors,
            "style_profile": self.style_profile.to_dict(),
        }


# ============================================================================
# Pattern Extraction
# ============================================================================


class PatternExtractor:
    """Extracts coding patterns from Python source code."""

    def __init__(self) -> None:
        self._naming_stats: Counter = Counter()
        self._import_stats: Counter = Counter()
        self._docstring_stats: Counter = Counter()
        self._patterns: list[CodingPattern] = []

    def analyze_file(self, filepath: Path) -> list[CodingPattern]:
        """Analyze a single file for patterns."""
        try:
            content = filepath.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except Exception as e:
            logger.debug(f"Failed to parse {filepath}: {e}")
            return []

        patterns = []

        # Extract patterns
        patterns.extend(self._extract_naming_patterns(tree, filepath))
        patterns.extend(self._extract_import_patterns(tree, filepath))
        patterns.extend(self._extract_docstring_patterns(tree, filepath))
        patterns.extend(self._extract_structure_patterns(tree, filepath))
        patterns.extend(self._extract_idiom_patterns(content, filepath))

        return patterns

    def _extract_naming_patterns(self, tree: ast.AST, filepath: Path) -> list[CodingPattern]:
        """Extract naming convention patterns."""
        patterns = []

        for node in ast.walk(tree):
            # Function names
            if isinstance(node, ast.FunctionDef):
                case = self._detect_case(node.name)
                self._naming_stats[f"function:{case}"] += 1

                # Check for private prefix
                if node.name.startswith("_"):
                    self._naming_stats["private_prefix:_"] += 1
                if node.name.startswith("__") and not node.name.endswith("__"):
                    self._naming_stats["private_prefix:__"] += 1

            # Class names
            elif isinstance(node, ast.ClassDef):
                case = self._detect_case(node.name)
                self._naming_stats[f"class:{case}"] += 1

            # Variable names (assignments)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        case = self._detect_case(target.id)
                        # Check if constant (all caps)
                        if target.id.isupper():
                            self._naming_stats[f"constant:{case}"] += 1
                        else:
                            self._naming_stats[f"variable:{case}"] += 1

        return patterns

    def _extract_import_patterns(self, tree: ast.AST, filepath: Path) -> list[CodingPattern]:
        """Extract import organization patterns."""
        patterns = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(("import", alias.name, alias.asname))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(("from", module, alias.name))

        # Analyze import grouping
        if imports:
            # Check if imports are grouped
            self._import_stats["has_imports"] += 1

            # Check for common patterns
            if any("from __future__" in str(imp) for imp in imports):
                self._import_stats["future_imports_first"] += 1

        return patterns

    def _extract_docstring_patterns(self, tree: ast.AST, filepath: Path) -> list[CodingPattern]:
        """Extract docstring style patterns."""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    style = self._detect_docstring_style(docstring)
                    self._docstring_stats[style] += 1

        return patterns

    def _extract_structure_patterns(self, tree: ast.AST, filepath: Path) -> list[CodingPattern]:
        """Extract code structure patterns."""
        patterns = []

        # Count decorators usage
        decorator_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorator_count += len(node.decorator_list)

        if decorator_count > 0:
            patterns.append(CodingPattern(
                type=PatternType.STRUCTURE,
                name="decorator_usage",
                description="Uses decorators for cross-cutting concerns",
                frequency=decorator_count,
                source_files=[str(filepath)],
            ))

        return patterns

    def _extract_idiom_patterns(self, content: str, filepath: Path) -> list[CodingPattern]:
        """Extract common Python idioms."""
        patterns = []

        # List comprehensions
        list_comp_count = len(re.findall(r'\[.+\s+for\s+.+\s+in\s+.+\]', content))
        if list_comp_count > 0:
            patterns.append(CodingPattern(
                type=PatternType.IDIOM,
                name="list_comprehension",
                description="Uses list comprehensions for transformations",
                frequency=list_comp_count,
                source_files=[str(filepath)],
            ))

        # Context managers
        with_count = len(re.findall(r'\bwith\s+', content))
        if with_count > 0:
            patterns.append(CodingPattern(
                type=PatternType.IDIOM,
                name="context_manager",
                description="Uses context managers for resource management",
                frequency=with_count,
                source_files=[str(filepath)],
            ))

        # F-strings vs format
        fstring_count = len(re.findall(r'f["\']', content))
        format_count = len(re.findall(r'\.format\(', content))

        if fstring_count > format_count:
            patterns.append(CodingPattern(
                type=PatternType.STYLE,
                name="fstring_preferred",
                description="Prefers f-strings over .format()",
                frequency=fstring_count,
                source_files=[str(filepath)],
            ))

        # Type annotations
        type_hint_count = len(re.findall(r':\s*\w+(?:\[.+\])?\s*[=\)]', content))
        if type_hint_count > 5:
            patterns.append(CodingPattern(
                type=PatternType.TYPING,
                name="type_annotations",
                description="Uses type annotations extensively",
                frequency=type_hint_count,
                source_files=[str(filepath)],
            ))

        return patterns

    def _detect_case(self, name: str) -> str:
        """Detect naming case style."""
        if name.isupper():
            return "UPPER_SNAKE_CASE"
        elif "_" in name:
            return "snake_case"
        elif name[0].isupper():
            return "PascalCase"
        elif any(c.isupper() for c in name[1:]):
            return "camelCase"
        return "snake_case"

    def _detect_docstring_style(self, docstring: str) -> str:
        """Detect docstring style (Google, NumPy, Sphinx)."""
        if "Args:" in docstring or "Returns:" in docstring:
            return "google"
        elif "Parameters" in docstring and "----------" in docstring:
            return "numpy"
        elif ":param" in docstring or ":returns:" in docstring:
            return "sphinx"
        return "plain"

    def get_style_profile(self, min_frequency: int = 3) -> StyleProfile:
        """Generate style profile from collected statistics."""
        profile = StyleProfile()

        # Naming conventions
        func_cases = {k.split(":")[1]: v for k, v in self._naming_stats.items() if k.startswith("function:")}
        if func_cases:
            profile.function_case = max(func_cases, key=func_cases.get)

        class_cases = {k.split(":")[1]: v for k, v in self._naming_stats.items() if k.startswith("class:")}
        if class_cases:
            profile.class_case = max(class_cases, key=class_cases.get)

        var_cases = {k.split(":")[1]: v for k, v in self._naming_stats.items() if k.startswith("variable:")}
        if var_cases:
            profile.variable_case = max(var_cases, key=var_cases.get)

        # Docstring style
        if self._docstring_stats:
            profile.docstring_style = max(self._docstring_stats, key=self._docstring_stats.get)

        # Aggregate patterns
        pattern_map: dict[str, CodingPattern] = {}
        for pattern in self._patterns:
            key = f"{pattern.type}:{pattern.name}"
            if key in pattern_map:
                pattern_map[key].frequency += pattern.frequency
                pattern_map[key].source_files.extend(pattern.source_files)
            else:
                pattern_map[key] = pattern

        # Filter by frequency
        profile.patterns = [
            p for p in pattern_map.values()
            if p.frequency >= min_frequency
        ]
        profile.patterns.sort(key=lambda p: p.frequency, reverse=True)

        return profile


def extract_patterns(
    source_path: str | Path,
    config: TrainingConfig | None = None,
) -> list[CodingPattern]:
    """
    Extract coding patterns from a source directory or file.

    Args:
        source_path: Path to source directory or file
        config: Training configuration

    Returns:
        List of extracted patterns
    """
    config = config or TrainingConfig()
    extractor = PatternExtractor()

    source_path = Path(source_path)

    if source_path.is_file():
        return extractor.analyze_file(source_path)

    patterns = []
    for pattern in config.include_patterns:
        for filepath in source_path.rglob(pattern):
            # Check exclusions
            if any(excl in str(filepath) for excl in config.exclude_patterns):
                continue

            # Check size
            if not (config.min_file_size <= filepath.stat().st_size <= config.max_file_size):
                continue

            patterns.extend(extractor.analyze_file(filepath))

    return patterns


# ============================================================================
# Style Adapter
# ============================================================================


class StyleAdapter:
    """
    Adapts code generation to match a user's coding style.

    Uses the learned style profile to provide context for code generation.
    """

    def __init__(self, style_profile: StyleProfile | None = None) -> None:
        self.style_profile = style_profile or StyleProfile()

    def get_prompt_context(self) -> str:
        """Get style context for prompts."""
        return self.style_profile.to_prompt_context()

    def adapt_code(self, code: str) -> str:
        """
        Adapt generated code to match style profile.

        Currently provides basic transformations. Full adaptation
        would require more sophisticated AST manipulation.
        """
        # Quote style
        if self.style_profile.quote_style == "single":
            code = self._convert_to_single_quotes(code)
        elif self.style_profile.quote_style == "double":
            code = self._convert_to_double_quotes(code)

        return code

    def _convert_to_single_quotes(self, code: str) -> str:
        """Convert double quotes to single quotes (simple strings only)."""
        # This is a simplified implementation
        # A full implementation would need AST parsing
        return re.sub(r'"([^"\\]*)"', r"'\1'", code)

    def _convert_to_double_quotes(self, code: str) -> str:
        """Convert single quotes to double quotes."""
        return re.sub(r"'([^'\\]*)'", r'"\1"', code)

    def suggest_improvements(self, code: str) -> list[str]:
        """Suggest style improvements for code."""
        suggestions = []

        # Check naming conventions
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    expected_case = self.style_profile.function_case
                    actual_case = self._detect_case(node.name)
                    if actual_case != expected_case:
                        suggestions.append(
                            f"Function '{node.name}' uses {actual_case}, "
                            f"but project style is {expected_case}"
                        )

                elif isinstance(node, ast.ClassDef):
                    expected_case = self.style_profile.class_case
                    actual_case = self._detect_case(node.name)
                    if actual_case != expected_case:
                        suggestions.append(
                            f"Class '{node.name}' uses {actual_case}, "
                            f"but project style is {expected_case}"
                        )

        except SyntaxError:
            pass

        # Check docstrings
        if self.style_profile.type_annotations:
            if not re.search(r':\s*\w+', code):
                suggestions.append("Consider adding type annotations (project uses them)")

        return suggestions

    def _detect_case(self, name: str) -> str:
        """Detect naming case."""
        if name.isupper():
            return "UPPER_SNAKE_CASE"
        elif "_" in name:
            return "snake_case"
        elif name[0].isupper():
            return "PascalCase"
        elif any(c.isupper() for c in name[1:]):
            return "camelCase"
        return "snake_case"


# ============================================================================
# Local Trainer
# ============================================================================


class LocalTrainer:
    """
    Local learning system for adapting to user's codebase.

    Analyzes local code to learn patterns, conventions, and style,
    then provides context for more consistent code generation.

    Example:
        trainer = LocalTrainer()

        # Train on local codebase
        result = trainer.train("/path/to/project")

        # Get style adapter
        adapter = trainer.get_adapter()

        # Use in prompts
        context = adapter.get_prompt_context()
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._style_profile: StyleProfile | None = None
        self._extractor = PatternExtractor()

    def train(
        self,
        source_path: str | Path,
        force: bool = False,
    ) -> TrainingResult:
        """
        Train on a local codebase.

        Args:
            source_path: Path to source directory
            force: Force retraining even if cached

        Returns:
            TrainingResult with learned patterns
        """
        import time

        start_time = time.time()
        source_path = Path(source_path)

        # Check cache
        cache_key = self._get_cache_key(source_path)
        if not force and self.cache_dir:
            cached = self._load_cache(cache_key)
            if cached:
                logger.info("Using cached training result")
                return cached

        # Analyze files
        files_analyzed = 0
        errors = []
        all_patterns = []

        for pattern in self.config.include_patterns:
            for filepath in source_path.rglob(pattern):
                # Check exclusions
                if any(excl in str(filepath) for excl in self.config.exclude_patterns):
                    continue

                try:
                    stat = filepath.stat()
                    if not (self.config.min_file_size <= stat.st_size <= self.config.max_file_size):
                        continue

                    patterns = self._extractor.analyze_file(filepath)
                    all_patterns.extend(patterns)
                    files_analyzed += 1

                except Exception as e:
                    errors.append(f"{filepath}: {e}")

        # Generate style profile
        self._style_profile = self._extractor.get_style_profile(
            min_frequency=self.config.min_examples
        )
        self._style_profile.patterns = all_patterns

        # Create result
        result = TrainingResult(
            style_profile=self._style_profile,
            files_analyzed=files_analyzed,
            patterns_found=len(all_patterns),
            training_time_ms=(time.time() - start_time) * 1000,
            errors=errors[:10],  # Limit errors
        )

        # Cache result
        if self.cache_dir:
            self._save_cache(cache_key, result)

        logger.info(
            f"Training complete: {files_analyzed} files, "
            f"{len(all_patterns)} patterns in {result.training_time_ms:.1f}ms"
        )

        return result

    def get_adapter(self) -> StyleAdapter:
        """Get a style adapter for the learned profile."""
        if not self._style_profile:
            self._style_profile = StyleProfile()
        return StyleAdapter(self._style_profile)

    def get_style_profile(self) -> StyleProfile:
        """Get the current style profile."""
        return self._style_profile or StyleProfile()

    def _get_cache_key(self, source_path: Path) -> str:
        """Generate cache key for source path."""
        # Include config in hash
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        key_str = f"{source_path.resolve()}:{config_str}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _load_cache(self, cache_key: str) -> TrainingResult | None:
        """Load cached training result."""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"training_{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text())

            # Reconstruct result
            profile = StyleProfile(**data.get("style_profile", {}))
            profile.patterns = [
                CodingPattern(**p) for p in data.get("style_profile", {}).get("patterns", [])
            ]

            return TrainingResult(
                style_profile=profile,
                files_analyzed=data.get("files_analyzed", 0),
                patterns_found=data.get("patterns_found", 0),
                training_time_ms=data.get("training_time_ms", 0),
            )

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_cache(self, cache_key: str, result: TrainingResult) -> None:
        """Save training result to cache."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"training_{cache_key}.json"

        try:
            cache_file.write_text(json.dumps(result.to_dict(), indent=2))
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


# ============================================================================
# Convenience Functions
# ============================================================================


_trainer: LocalTrainer | None = None


def get_trainer(cache_dir: str | Path | None = None) -> LocalTrainer:
    """Get or create the global trainer instance."""
    global _trainer
    if _trainer is None:
        _trainer = LocalTrainer(cache_dir=cache_dir)
    return _trainer


def train_on_codebase(
    source_path: str | Path,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """
    Train on a local codebase.

    Args:
        source_path: Path to source directory
        config: Training configuration

    Returns:
        Training result with learned patterns
    """
    trainer = LocalTrainer(config=config)
    return trainer.train(source_path)


def get_style_context(source_path: str | Path) -> str:
    """
    Get style context for prompts based on a codebase.

    Args:
        source_path: Path to source directory

    Returns:
        Style context string for LLM prompts
    """
    trainer = get_trainer()
    trainer.train(source_path)
    adapter = trainer.get_adapter()
    return adapter.get_prompt_context()
