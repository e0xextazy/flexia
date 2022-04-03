from dataclasses import dataclass, field, asdict
import json

from attr import fields_dict


@dataclass
class Arguments():
    epochs: int = field(default=5)
    validation_steps: int = field(default=500)
    validation_strategy: str = field(default="step")
    # output_directory: str = field(default="/")
    # model_path: str = field(default="/model.pth")
    # checkpoints_directory: str = field(default="/checkpoints/")
    gradient_norm: float = field(default=1.0)
    gradient_scaling: bool = field(default=False)
    gradient_accumulation_steps: int = field(default=1)
    save_scheduler_state: bool = field(default=False)
    seed: int = field(default=42)
    debug: bool = field(default=True)
    verbose: int = field(default=1)
    device: str = field(default="cpu")
    ignore_warnings: bool = field(default=False)


    def __post_init__(self):
        pass

    def to_dict(self):
        dict_ = asdict(self)
        return dict_

    def to_json(self):
        dict_ = self.to_dict()
        return json.dumps(dict_)

    def from_dict(self, data):
        for k, v in data.items():
            setattr(self, k, v)

        return self

    def from_json(self, data):
        data = json.loads(data)
        self.load_from_dict(data)
        return self