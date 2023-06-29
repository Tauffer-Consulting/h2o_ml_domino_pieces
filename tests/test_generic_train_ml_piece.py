from domino.scripts.piece_dry_run import piece_dry_run
from typing import List

def run_piece(
        ml_model,
        dataset_file_path: str,
        target_column: str,
        categorical_columns: List[str] = None,
        remove_columns: List[str] = None,
        split_percentage: List[float] = None,
        seed: int = 1,
        model_parameters: dict = None,
        feature_importance: bool = False,
        model_save_format: str = None,
):
    
    kwargs = {key: value for key, value in locals().items() if value is not None}

    return piece_dry_run(
  
    #local piece repository path
    repository_folder_path="../",

    #name of the piece
    piece_name="GenericTrainMLPiece",

    #values to the InputModel arguments
    piece_input={
        **kwargs,
    },
)

def test_piece():
    from h2o.estimators import H2ORandomForestEstimator
    piece_kwargs = {
        "ml_model": H2ORandomForestEstimator,
        "model_parameters": {"ntrees": 50, "max_depth": 20, "nfolds": 10},
        "dataset_file_path": "https://github.com/h2oai/h2o-tutorials/raw/master/h2o-world-2017/automl/data/product_backorders.csv",
        "target_column": "went_on_backorder",
        "remove_columns": ["sku"],
        "feature_importance": True,
    }

    output = run_piece(
        **piece_kwargs
    )

    print(output)

if __name__ == "__main__":
    test_piece()
