from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Status(IntEnum):
    ERROR = 0
    PROCESSING = 1
    ABORTED = 2
    DONE = 3


@dataclass
class TaskData:
    task_id: str
    input_model_token: str
    output_model_token: str
    current_model_token: str


@dataclass
class ModelData:
    status: int = 0


@dataclass
class MetaData:
    task_data: Optional[TaskData] = None
    config_data: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    model_data: ModelData = ModelData()


@dataclass
class ObjectData:
    input_data: Optional[Dict] = None
    preprocessed_data: Optional[np.ndarray] = None
    infer_result: Optional[List] = None
    output_data: Optional[List] = None


@dataclass
class DetectionResponse:
    boxes: list
    scores: list
    labels: list


@dataclass
class ClassificationResponse:
    scores: list
    labels: list
