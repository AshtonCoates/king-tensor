__all__ = ["Screen"]


def __getattr__(name):
    if name == "Screen":
        # Lazy import to avoid pulling GUI backends into lightweight utilities.
        from .screen import Screen

        return Screen
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
