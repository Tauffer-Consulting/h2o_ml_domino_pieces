from domino.scripts.piece_dry_run import piece_dry_run
from pathlib import PosixPath
from typing import List

def run_piece(
        dataset_file_path: str,
        target_column: str,
        save_leader_model_type: str,
        categorical_columns: List[str] = None,
        remove_columns: List[str] = None,
        split_percentage: List[float] = None,
        seed: int = 1,
        max_models: int = 5,
        max_runtime_secs: int = 0,
):
    return piece_dry_run(
  
    #local piece repository path
    repository_folder_path="../",

    #name of the piece
    piece_name="AutoMLPiece",

    #values to the InputModel arguments
    piece_input={
        "dataset_file_path": dataset_file_path,
        "target_column": target_column,
        "categorical_columns": categorical_columns,
        "remove_columns": remove_columns,
        "split_percentage": split_percentage,
        "seed": seed,
        "max_models": max_models,
        "max_runtime_secs": max_runtime_secs,
        "save_leader_model_type": save_leader_model_type,
    },
)

def test_piece():
    piece_kwargs = {
        "dataset_file_path": "https://github.com/h2oai/h2o-tutorials/raw/master/h2o-world-2017/automl/data/product_backorders.csv",
        "target_column": "went_on_backorder",
        "remove_columns": ["sku"],
        "max_models": 3,
        "save_leader_model_type": "bin",
    }

    output = run_piece(
        **piece_kwargs
    )

if __name__ == "__main__":
    test_piece()
