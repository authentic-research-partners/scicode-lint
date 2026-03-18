from dataclasses import dataclass, field


@dataclass
class Config:
    """Dataclass configuration for experiment tracking."""

    name: str
    tags: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)


def apply_transforms(data, transforms: tuple = ()):
    """Apply a sequence of transforms to the input data."""
    for transform in transforms:
        data = transform(data)
    return data


def process_with_kwargs(item, **kwargs):
    """Process an item with optional keyword parameters."""
    result = {"item": item}
    result.update(kwargs)
    return result


class ImmutableDefaults:
    """Container for labeled item counts."""

    def __init__(self, name: str, count: int = 0, label: str | None = None):
        self.name = name
        self.count = count
        self.label = label or "default"


config = Config(name="experiment")
config.tags.append("test")
