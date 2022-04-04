from dataclasses import dataclass, field, asdict
import json


@dataclass
class Arguments():
    epochs: int = field(default=5)
    validation_steps: int = field(default=500)
    validation_strategy: str = field(default="step") # ["step", "epoch"]
    model_path: str = field(default="model.pth")
    save_model: bool = field(default=True)
    save_scheduler_state: bool = field(default=False)
    gradient_norm: float = field(default=0)
    gradient_scaling: bool = field(default=False)
    gradient_accumulation_steps: int = field(default=1)
    amp: bool = field(default=False)
    seed: int = field(default=42)
    debug: bool = field(default=True)
    verbose: int = field(default=1)
    device: str = field(default="cpu")
    ignore_warnings: bool = field(default=False)
        
    def __post_init__(self):
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"'gradient_accumulation_steps' parameter should be in range [1, +inf), but given '{self.gradient_accumulation_steps}'.")
            
        if isinstance(self.seed, str):
            if self.seed not in ("random", "none"):
                raise ValueError(f"'seed' parameter should be an integer or string with certain values 'random' or 'none'.")
                
        if self.validation_strategy not in ("step", "epoch"):
            raise ValueError(f"'validation_strategy' parameter should one of 'step' or 'epoch', but given '{self.validation_strategy}'.")
        
        if isinstance(self.verbose, str):
            if self.verbose != "epoch":
                raise ValueError(f"'verbose' parameter shoud be an interger or 'epoch'.")
                
        if self.gradient_norm < 0:
            raise ValueError(f"'gradient_norm' parameter should be in range [0, +inf], but given '{self.gradient_norm}'.")
            
        

    def to_dict(self):
        dict_ = asdict(self)
        return dict_

    def to_json(self):
        dict_ = self.to_dict()
        return json.dumps(dict_)
    
    def to_json_file(self, path="arguments.json"):
        with open(path, "w") as file:
            data = self.to_json()
            file.write(data)
        
        return path
    
    def from_json_file(self, path="arguments.json"):
        with open(path, "r") as file:
            data = file.read()
            self.from_json(data)
            
        return self

    def from_dict(self, data):
        for k, v in data.items():
            setattr(self, k, v)

        return self

    def from_json(self, data):
        data = json.loads(data)
        self.from_dict(data)
        return self