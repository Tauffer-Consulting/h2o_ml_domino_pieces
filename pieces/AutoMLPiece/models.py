from pydantic import BaseModel, Field
from enum import Enum
from pydantic import FilePath
from typing import List

class SaveLeaderModelTypes(Enum):
    bin = "bin"
    mojo = "mojo"
    bin_and_mojo = "bin_and_mojo"


class InputModel(BaseModel):
    dataset_file_path: str = Field(
        ...,
        description="The path to the dataset file. It can be a local path or a url path"
    )
    target_column: str = Field(
        ...,
        description="The target column from dataset to be predicted"
    )
    categorical_columns: List[str] = Field(
        default=None,
        description="The list of categorical columns from dataset"
    )
    remove_columns: List[str] = Field(
        default=None,
        description="The list of columns to be removed from dataset"
    )
    split_percentage: List[float] = Field(
        default=None,
        description="The list of split percentage for train and test dataset"
    )
    seed: int = Field(
        default=1,
        description="The random seed to be used"
    )
    max_models: int = Field(
        default=5,
        description="The number of models to be trained"
    )
    max_runtime_secs: int = Field(
        default=0,
        description="The maximum runtime in seconds to be used"

    )
    save_leader_model_type: SaveLeaderModelTypes = Field(
        default=SaveLeaderModelTypes.mojo,
        description="The file type of model to be saved"
    )


class OutputModel(BaseModel):
    bin_path_file: FilePath = Field(
        default=None,
        description="The path to the bin file"
    )
    mojo_path_file: FilePath = Field(
        default=None,
        description="The path to the mojo file"
    )