"""
Typing helpers for Titan API modules.

These shims keep strict mypy checks useful when CI runs with
``--follow-imports=skip`` and external framework decorators/types
would otherwise degrade to ``Any``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


if TYPE_CHECKING:

    class BaseModel:
        """Minimal pydantic-compatible model protocol for type checking."""

        model_config: dict[str, Any]

        def __init__(self, **data: Any) -> None: ...

        def model_dump(self, **kwargs: Any) -> dict[str, Any]: ...

    def Field(  # noqa: N802 - keep pydantic-compatible import shape
        default: Any = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...
else:
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import Field as _PydanticField

    BaseModel = _PydanticBaseModel
    Field = _PydanticField


def _typed_route_decorator(
    decorator_factory: Callable[..., Any],
    path: str,
    **kwargs: Any,
) -> Callable[[F], F]:
    return cast(Callable[[F], F], decorator_factory(path, **kwargs))


def typed_get(router: Any, path: str, **kwargs: Any) -> Callable[[F], F]:
    return _typed_route_decorator(router.get, path, **kwargs)


def typed_post(router: Any, path: str, **kwargs: Any) -> Callable[[F], F]:
    return _typed_route_decorator(router.post, path, **kwargs)


def typed_put(router: Any, path: str, **kwargs: Any) -> Callable[[F], F]:
    return _typed_route_decorator(router.put, path, **kwargs)


def typed_delete(router: Any, path: str, **kwargs: Any) -> Callable[[F], F]:
    return _typed_route_decorator(router.delete, path, **kwargs)


def typed_websocket(router: Any, path: str, **kwargs: Any) -> Callable[[F], F]:
    return _typed_route_decorator(router.websocket, path, **kwargs)
