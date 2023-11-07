from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)
api = Api(
    app, 
    version='1.0', 
    title='My clASSic regression and clASSification app', 
    description='A cool tool for model training'
)
# Dict for trained models
MODELS = {}

# Dict with model types and classes
MODEL_CLASSES = {
    "regression": {
        "linear": LinearRegression,
        "lasso": Lasso,
        "ridge": Ridge
    },
    "classification": {
        "logistic regression": LogisticRegression,
        "decision tree": DecisionTreeClassifier,
        "random forest": RandomForestClassifier
    }
}

train_model_input = api.model('TrainModelInput', {
    'model_type': fields.String(required=True, description='Type of the model - "regression" or "classification"'),
    'model_class': fields.String(required=True, description='Class of the model (e.g., "linear")'),
    'optimization_type': fields.String(required=True, enum=["default", "grid search"], description='Type of optimization - "default" or "grid search"'),
    'X_train': fields.List(fields.List(fields.Float), required=True, description='Training data features'),
    'y_train': fields.List(fields.Float, required=True, description='Training data labels'),
    'param_grid': fields.Raw(required=False, description='Parameter grid for optimization')
})

@api.route('/train')
class TrainModel(Resource):
    @api.expect(train_model_input)
    def post(self):
        '''Train the model via input json with params, X_train and y_train arrays'''
        data = request.json

        # Check whether all the necessary hyperparams are given
        if not all(key in data for key in ["model_type", "model_class", "optimization_type", "X_train", "y_train"]):
            return {"error": "model_type, model_class, optimization_type, X_train, and y_train are required fields"}

        # Check the model type existence
        if data["model_type"] not in MODEL_CLASSES:
            return {"error": f"Unsupported model type {data['model_type']}"}

        # Check the model class existence
        if data["model_class"] not in MODEL_CLASSES[data["model_type"]]:
            return {"error": f"Unsupported model class {data['model_class']}"}

        # Check the optimiztion type existence
        if data["optimization_type"] not in ["default", "grid search"]:
            return {"error": f"Unsupported model optimization type {data['optimization_type']} - try default or grid search"}

        # Convert json into train dataset
        X_train = np.array(data["X_train"])
        y_train = np.array(data["y_train"])
        
        if X_train.shape[0] != y_train.shape[0]:
            return {"error": f"X_train and y_train has different lengths ({X_train.shape[0]} vs {y_train.shape[0]})"}

        # Initialize the model
        model_class = MODEL_CLASSES[data["model_type"]][data["model_class"]]
        model = model_class()
        
        # Train the model through basic params or grid search
        if data["optimization_type"] == "grid search":
            model = model_class()
            param_grid = {
                "regression": {
                    "linear": {
                        "fit_intercept": [True, False],
                    },
                    "lasso": {
                        "alpha": [0.001, 0.01, 0.1, 1, 10]
                    },
                    "ridge": {
                        "alpha": [0.001, 0.01, 0.1, 1, 10]
                    }
                },

                "classification": {
                    "logistic regression": {
                        "penalty": ["l1", "l2"],
                        "C": [0.001, 0.01, 0.1, 1, 10]
                    },
                    "decision tree": {
                        "max_depth": [3, 5, 10],
                        "min_samples_split": [2, 5, 10]
                    },
                    "random forest": {
                        "n_estimators": [10, 50, 100],
                        "max_depth": [3, 5, 10],
                        "min_samples_split": [2, 5, 10]
                    }
                
                }
            }

            # Update the params grid, if it exists
            if "param_grid" in data:
                user_param_grid = data["param_grid"]
                for param in user_param_grid:
                    if param in param_grid[data["model_type"]][data["model_class"]]:
                        param_grid[data["model_type"]][data["model_class"]][param] = user_param_grid[param]

            grid_search = GridSearchCV(model, param_grid[data["model_type"]][data["model_class"]], cv=5)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

        model.fit(X_train, y_train)

        # Save the model
        model_name = f"{data['model_type']}_{data['model_class']}_{len(MODELS) + 1}"
        MODELS[model_name] = model

        return {"message": "Model trained successfully", "model_name": model_name}
        

input_model = api.model('InputModel', {
    'X': fields.List(fields.Float, required=True, description='Input data for prediction')
})

@api.route('/predict/<model_id>')
class Predict(Resource):
    @api.expect(input_model)
    def post(self, model_id):
        '''Predict from json input data through chosen model id'''
        
        # Check whether the model id exists
        if model_id not in MODELS:
            return {"error": f"Model {model_id} does not exist"}, 400

        # Fetch json and check the input file
        data = request.json
        if "X" not in data:
            return {"error": "Missing input data (X)"}, 400

        # Convert json into numpy
        try:
            X = np.array(data["X"])
            if isinstance(X, np.ndarray) and X.ndim == 2:
                if all(len(row) == len(X[0]) for row in X):
                    pass
                else:
                    return {"error": f"X columns are not the same length"}, 400
            else:
                return {"error": f"X is not 2-dimensional array"}, 400
        except Exception as e:
            return {"error": f"Cannot convert X into numpy array: {str(e)}"}, 400
        
        predictions = MODELS[model_id].predict(X)

        return {"predictions": predictions.tolist()}


@api.route('/models')
class TrainedModelList(Resource):
    def get(self):
        '''Show a list of trained models and its params'''
        return {"models": list(MODELS.keys())}


@api.route('/available')
class ModelList(Resource):
    def get(self):
        '''Show available model classes'''
        dct = {}
        for type in MODEL_CLASSES:
            dct[type] = [str(class_) for class_ in MODEL_CLASSES[type]]
        return {"models": dct}


@api.route('/delete/<model_id>')
class ModelDelete(Resource):
    def delete(self, model_id):
        '''Delete a model by its id'''
        if model_id in MODELS:
            del MODELS[model_id]
            return {"message": f"Model {model_id} deleted successfully"}
        else:
            return {"error": f"Model {model_id} does not exist"}, 400


if __name__ == "__main__":
    app.run(debug=True)
