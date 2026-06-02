"""Lightweight registry mapping model class names to classes."""

registry: dict[str, type] = {}


def register_model(cls):
    """Register ``cls`` in the model registry by its class name."""
    registry[cls.__name__] = cls
    return cls
