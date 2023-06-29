from pydantic import BaseModel, Field
from enum import Enum
from pydantic import FilePath
from typing import List, Union
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

class MLModels(Enum):
    random_forest = H2ORandomForestEstimator
    gradient_boosting = H2OGradientBoostingEstimator
    linear_regression = H2OGeneralizedLinearEstimator

class SaveFormat(Enum):
    bin = "bin"
    mojo = "mojo"


class InputModel(BaseModel):
    ml_model: MLModels = Field(
        ...,
        description="The machine learning model to be used"
    )
    model_parameters: dict = Field(
        ...,
        description="The dictionary of model parameters"
    )
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
    split_percentage: float = Field(
        default=None,
        description="The split percentage for train and test dataset"
    )
    seed: int = Field(
        default=1,
        description="The random seed to be used"
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
    saved_filed_path: FilePath = Field(
        default=None,
        description="The path to the saved model file"
    )
    feature_importance: List[dict] = Field(
        default=None,
        description="The list of feature importance"
    )