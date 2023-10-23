from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import os
import h2o
from h2o.automl import H2OAutoML


class AutoMLPiece(BasePiece):  
    def piece_function(self, input_data: InputModel):
        h2o.init()

        # Get the input_data parameters
        dataset_file_path = input_data.dataset_file_path
        categorical_columns = input_data.categorical_columns
        target_column = input_data.target_column
        remove_columns = input_data.remove_columns
        split_percentage = input_data.split_percentage
        seed = input_data.seed
        max_models = input_data.max_models
        max_runtime_secs = input_data.max_runtime_secs
        leader_model_save_format = input_data.leader_model_save_format.value

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
        
        # Create aml object
        aml = H2OAutoML(max_models=max_models, max_runtime_secs=max_runtime_secs, seed=seed)
    
        # Split data
        if split_percentage:
            train, test = df.split_frame(ratios=split_percentage, seed=seed)
            aml.train(x=x, y=y, training_frame=train, leaderboard_frame=test)
        
        # Train models
        aml.train(x=x, y=y, training_frame=df)

        # Save leader model
        leader_model = 'model'
        if leader_model_save_format == "bin":
            bin_path_file = f"{self.results_path}/{leader_model}_bin"
            mojo_path_file = None
            h2o.save_model(aml.leader, path=bin_path_file)
        if leader_model_save_format == "mojo":
            mojo_path_file = f"{self.results_path}/{leader_model}_mojo"
            bin_path_file = None
            aml.leader.download_mojo(path=mojo_path_file)
        if leader_model_save_format == "bin_and_mojo":
            bin_path_file = f"{self.results_path}/{leader_model}_bin"
            mojo_path_file = f"{self.results_path}/{leader_model}_mojo"
            h2o.save_model(aml.leader, path=bin_path_file)
            aml.leader.download_mojo(path=mojo_path_file)

        # # Finally, results should return as an Output model
        return OutputModel(
            bin_path_file=bin_path_file,
            mojo_path_file=mojo_path_file
        )
