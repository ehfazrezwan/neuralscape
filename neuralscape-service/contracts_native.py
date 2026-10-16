"""Native-type helpers shared by the processing contracts.

Everything here decides on a value's *real* type and reads model storage through the base
descriptors.  Nothing consults ``value.__class__`` (a caller-controlled callback on a proxy),
hashes or compares a caller object, or goes through an ABC registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel

_TYPE_MRO = vars(type)["__mro__"]
_MAPPING_BASES = (Mapping, dict, MappingProxyType)
_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]


def is_subclass(cls: type, types: type | tuple[type, ...]) -> bool:
    """Whether ``cls`` inherits from one of ``types``, by identity on its native MRO.

    ``issubclass`` against an ABC (``Mapping``) or pydantic's ABC-based metaclass runs
    ``ABCMeta.__subclasscheck__``: registry lookups that hash and compare ``cls`` and honour
    ``__subclasshook__`` and virtual registration.  ``Mapping`` also recognises ``dict`` and
    ``mappingproxy``; a class only *registered* as a mapping is not one.
    """

    wanted: list[type] = []
    for candidate in types if type(types) is tuple else (types,):
        if candidate is Mapping:
            wanted.extend(_MAPPING_BASES)
        else:
            wanted.append(candidate)
    for base in _TYPE_MRO.__get__(cls):
        for candidate in wanted:
            if base is candidate:
                return True
    return False


MAX_COLLECTION_ENTRIES = 65_536
"""The most entries read from a caller-supplied collection (a public view the caller controls)."""


class CollectionBudgetExceeded(ValueError):
    """A caller-supplied collection produced more entries than the budget allows."""


def bounded_entries(collection: object, limit: int = MAX_COLLECTION_ENTRIES) -> list[object]:
    """Read ``collection`` once, charging each entry as it is produced, and keep the entries.

    An iterator that never ends is cut off at ``limit`` entries with ``CollectionBudgetExceeded``
    instead of being drained.
    """

    entries: list[object] = []
    for entry in collection:  # type: ignore[attr-defined]
        if len(entries) >= limit:
            raise CollectionBudgetExceeded("collection exceeds the budget")
        entries.append(entry)
    return entries


def plain_keyed(pairs: object) -> dict[str, object]:
    """A dict whose keys are exact ``str``, built from ``(key, value)`` pairs without running key hooks.

    A ``str`` subclass key is replaced by its plain text (``str.__str__`` runs none of its hooks);
    any other key type, or two keys with the same text, is refused, so a hostile key can neither run
    code during field lookup nor shadow a plain key.
    """

    result: dict[str, object] = {}
    for key, value in pairs:  # type: ignore[attr-defined]
        if not is_subclass(type(key), str):
            raise TypeError("keys must be text")
        text = str.__str__(key)  # type: ignore[arg-type]
        if text in result:
            raise TypeError("duplicate key text")
        result[text] = value
    return result


def plain_str(value: object) -> object:
    """A built-in ``str`` copy of a ``str`` subclass instance, without running its hooks.

    ``str.__str__`` on a subclass instance returns an exact ``str``; it never calls the
    subclass's own ``__str__``, ``__hash__`` or ``__eq__``.  Any other value is returned
    unchanged for the strict validator to refuse.
    """

    kind = type(value)
    if kind is not str and is_subclass(kind, str):
        return str.__str__(value)  # type: ignore[arg-type]
    return value


def project_model_storage(value: object) -> object:
    """Project a model with native dict extra backing for closed revalidation.

    The caller has already established that ``value`` is a genuine model by its native MRO.
    """

    extras = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(value, BaseModel)
    if extras is None:
        captured: tuple[tuple[object, object], ...] = ()
    elif is_subclass(type(extras), dict):
        # A dict subclass may lie through len/iteration/keys/items/truthiness: snapshot its
        # concrete backing exactly once with the built-in operation.
        captured = tuple(dict.items(extras))
    else:
        # Not a dict: none of its views can be trusted and none is read.  Project a marker the
        # closed model rejects instead of handing the original model back to revalidation.
        captured = (("unreadable_extra_storage", True),)

    # Detach every pair before projected-dict hashing can run a hostile stored-name callback
    # that repairs a later value in the live model backing.
    stored = tuple(dict.items(_BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)))
    projected = plain_keyed(stored)
    if captured:
        projected["native_extra_backing"] = dict(captured)
    return projected


def native_model_input(value: object, model_types: tuple[type, ...]) -> object:
    """The input for validating one of ``model_types``: a projected model, or an exact dict.

    Any other object (a proxy that merely reports a model class, a registered-only base, a
    ``dict`` subclass) is refused with ``TypeError`` before Pydantic can read its ``__class__``.
    """

    kind = type(value)
    if is_subclass(kind, model_types):
        return project_model_storage(value)
    if kind is dict:
        return plain_keyed(dict.items(value))
    raise TypeError("input must be a model of the expected type or a plain dict")
