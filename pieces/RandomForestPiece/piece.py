from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import pandas as pd
import h2o
from h2o.estimators import H2ORandomForestEstimator


class RandomForestPiece(BasePiece):  
    def piece_function(self, input_model: InputModel):
        h2o.init()

        # Get the input_model parameters
        dataset_file_path = input_model.dataset_file_path
        target_column = input_model.target_column
        categorical_columns = input_model.categorical_columns
        remove_columns = input_model.remove_columns
        split_percentage = input_model.split_percentage
        seed = input_model.seed
        feature_importance = input_model.feature_importance
        model_save_format = input_model.model_save_format

        # Get the model parameters from input_model
        model_parameters = {
            "ntrees": input_model.ntrees,
            "max_depth": input_model.max_depth,
            "min_rows": input_model.min_rows,
            "sample_rate": input_model.sample_rate,
            "col_sample_rate_per_tree": input_model.col_sample_rate_per_tree,
            "stopping_metric": input_model.stopping_metric,
        }
        model_id = input_model.model_id

        # Import dataset
        try:
            df = h2o.import_file(dataset_file_path)
        except:
            raise Exception("The dataset file could not be founded or downloaded")

        # Set categorical columns
        if categorical_columns:
            for column in categorical_columns:
                df[column] = df[column].asfactor()

        # Set target column and traing data
        y = target_column
        x = df.columns
        x.remove(y)
        if remove_columns:
            for column in remove_columns:
                x.remove(column)
        
        model = H2ORandomForestEstimator(model_id=model_id, **model_parameters)
    
        # Split data
        if split_percentage:
            train, test = df.split_frame(ratios=split_percentage, seed=seed)
            model.train(x=x, y=y, training_frame=train, leaderboard_frame=test)
        
        # Train models
        model.train(x=x, y=y, training_frame=df) 

        # Display feature importance table and plot figure
        if feature_importance:
            model_plots = model.explain(test, render=False) if split_percentage else model.explain(df, render=False)
            varimp = model.varimp(use_pandas=True)
            varimp_dict = varimp.to_dict("records")
            self.format_display_result_feature_importance(model_id, model_plots, varimp)
        
        if not model_save_format:
            return OutputModel(feature_importance=varimp_dict if feature_importance else None)
        
        # Save model
        if model_save_format == "bin":
            saved_file_path = f"{self.results_path}/{model_id}_bin"
            h2o.save_model(model, path=saved_file_path)
        if model_save_format == "mojo":
            saved_file_path = f"{self.results_path}/{model_id}_mojo"
            model.download_mojo(path=saved_file_path)

        # Finally, results should return as an Output model
        return OutputModel(
            saved_file_path=saved_file_path if model_save_format else None,
            feature_importance=varimp_dict if feature_importance else None,
        )
    
    def format_display_result_feature_importance(self, model_id, model_plots = None, varimp: pd.DataFrame = None,):
        varimp_markdown_table = varimp.to_markdown(index=False)
        varimp_plot_figure_name = f"varimp_plot_{model_id}.png"
        varimp_plot_figure_file_path = f"{self.results_path}/{varimp_plot_figure_name}"
        model_plots["varimp"]["plots"][model_id].figure().savefig(varimp_plot_figure_file_path, format="png")
        md_text = f"""
## Feature importance table
{varimp_markdown_table}

## Feature importance plot
<img src="./{varimp_plot_figure_name}" alt="varimp plot" width="600" height="400">

"""
        file_path = f"{self.results_path}/display_result_feature_importance.md"
        with open(file_path, "w") as f:
            f.write(md_text)
        self.display_result = {
            "file_type": "md",
            "file_path": file_path
        }