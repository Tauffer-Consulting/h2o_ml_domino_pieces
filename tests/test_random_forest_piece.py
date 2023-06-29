from domino.scripts.piece_dry_run import piece_dry_run
from pathlib import PosixPath
from typing import List

def run_piece(
        dataset_file_path: str,
        target_column: str,
        categorical_columns: List[str] = None,
        remove_columns: List[str] = None,
        split_percentage: List[float] = None,
        seed: int = None,
        ntrees: int = None,
        max_depth: int = None,
        min_rows: int = None,
        sample_rate: float = None,
        col_sample_rate_per_tree: float = None,
        stopping_metric: str = None,
        feature_importance: bool = None,
        model_save_format: str = None,
):
    
    kwargs = {key: value for key, value in locals().items() if value is not None}

    return piece_dry_run(
  
    #local piece repository path
    repository_folder_path="../",

    #name of the piece
    piece_name="RandomForestPiece",

    #values to the InputModel arguments
    piece_input={
        **kwargs,
    },
)

def test_piece():
    piece_kwargs = {
        "dataset_file_path": "https://github.com/h2oai/h2o-tutorials/raw/master/h2o-world-2017/automl/data/product_backorders.csv",
        "target_column": "went_on_backorder",
        "remove_columns": ["sku"],
        "feature_importance": True,
    }

    output = run_piece(
        **piece_kwargs
    )

if __name__ == "__main__":
    test_piece()
