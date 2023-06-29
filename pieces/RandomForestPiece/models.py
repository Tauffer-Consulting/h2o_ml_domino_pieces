from pydantic import BaseModel, Field
from enum import Enum
from pydantic import FilePath
from typing import List, Union

class SaveFormat(Enum):
    bin = "bin"
    mojo = "mojo"


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
    ntrees: int = Field(
        default=50,
        description="The number of trees to be used"
    )
    max_depth: int = Field(
        default=20,
        description="The maximum depth of the tree"
    )
    min_rows: int = Field(
        default=1,
        description="The minimum number of observations (rows) in a leaf"
    )
    sample_rate: float = Field(
        default=0.632,
        description="The sample rate to be used. It specifies the proportion of rows randomly selected for each tree.",
        ge=0,
        le=1,
    )
    col_sample_rate_per_tree: float = Field(
        default=1,
        description="Specify the column sample rate per tree. This method samples without replacement.",
        ge=0,
        le=1,
    )
    stopping_metric: str = Field(
        default="AUTO",
        description="Specify the metric to use for early stopping"
    )
    model_id: str = Field(
        default="random_forest_model",
        description="The model id to be used"
    )
    feature_importance: bool = Field(
        default=False,
        description="Whether to calculate feature importance or not"
    )
    model_save_format: Union[SaveFormat, None] = Field(
        default=None,
        description="The file type of model to be saved"
    )


class OutputModel(BaseModel):
    saved_file_path: str  = Field(
        default=None,
        description="The path to the saved model file"
    )
    feature_importance: List[dict] = Field(
        default=None,
        description="The list of feature importance"
    )