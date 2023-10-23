from typing import List
from domino.testing import piece_dry_run

def run_piece(
    dataset_file_path: str,
    target_column: str,
    leader_model_save_format: str,
    categorical_columns: List[str] = None,
    remove_columns: List[str] = None,
    split_percentage: List[float] = None,
    seed: int = 1,
    max_models: int = 5,
    max_runtime_secs: int = 0,
):
    return piece_dry_run(
        #name of the piece
        piece_name="AutoMLPiece",
        #values to the InputModel arguments
        input_data={
            "dataset_file_path": dataset_file_path,
            "target_column": target_column,
            "categorical_columns": categorical_columns,
            "remove_columns": remove_columns,
            "split_percentage": split_percentage,
            "seed": seed,
            "max_models": max_models,
            "max_runtime_secs": max_runtime_secs,
            "leader_model_save_format": leader_model_save_format,
        },
)

def test_aut_ml_piece():
    piece_kwargs = {
        "dataset_file_path": "https://github.com/h2oai/h2o-tutorials/raw/master/h2o-world-2017/automl/data/product_backorders.csv",
        "target_column": "went_on_backorder",
        "remove_columns": ["sku"],
        "max_models": 3,
        "leader_model_save_format": "bin",
    }

    output = run_piece(
        **piece_kwargs
    )

    assert type(output.get('bin_path_file')) == str
    assert output.get('mojo_path_file') is None
