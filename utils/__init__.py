from . import pretraining_parser, downstream_parser, jepa_parser
from .pretraining_parser import get_parser as get_pretraining_parser
from .downstream_parser import get_parser as get_downstream_parser
from .jepa_parser import get_parser as get_jepa_parser


def __getattr__(name):
    if name == "pretraining_Trainer":
        from .pretraining_trainer import Trainer
        return Trainer

    if name == "downstream_Trainer":
        from .downstream_trainer import Trainer
        return Trainer

    if name == "jepa_Trainer":
        from .jepa_trainer import JEPATrainer
        return JEPATrainer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "pretraining_Trainer",
    "downstream_Trainer",
    "jepa_Trainer",
    "pretraining_parser",
    "downstream_parser",
    "jepa_parser",
    "get_pretraining_parser",
    "get_downstream_parser",
    "get_jepa_parser",
]
