from ovqa.common.lavis.registry import registry
from ovqa.processors.base_processor import BaseProcessor


@registry.register_processor("text_none")
class TextNoneProcessor(BaseProcessor):
    """
    This processor does not change the text.

    In many cases no more text processing is needed, e.g.
    - the dataset already provides clean text
    - the model understands uppercase and UTF-8
    - ...
    """

    def __init__(self, _cfg=None):  # noqa
        pass

    def __call__(self, item):
        return item

    @classmethod
    def from_config(cls, cfg=None):
        return cls(cfg)
