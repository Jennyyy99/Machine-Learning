# import libraries needed for the code to run
import re
import pyspark as ps
from pyspark.ml import PipelineModel
from pyspark.sql import functions as f
from pyspark.sql import types as t
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource



# create a Flask instance
# and create a Flask-RESTful API instance with the created Flask instance
app = Flask(__name__)
api = Api(app)

# create a SparkContext
# load saved pipeline model from the folder 'model'
sc = ps.SparkContext()
sqlContext = ps.sql.SQLContext(sc)
loadedModel = PipelineModel.load('NBmodel')

# create a parser
# fill a parser with information about arguments 
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictClass(Resource):
    def get(self):
        # retrieve query text from API call
        args = parser.parse_args()
        user_text = args['query']
        # create a dictionary with the retrieved query text
        # and make a PySpark dataframe 
        user_input = {'text':user_text}
        schema = t.StructType([t.StructField('text', t.StringType())])
        df=sqlContext.createDataFrame([user_input],schema)
        
        prediction = loadedModel.transform(df)
        # retrieve predicted label ('prediction' column):
        #       trained model will output 0 for negative sentiment and 1 for positive sentiment
        prediction_label = prediction.select(prediction['prediction']).collect()
        # retrieve prediction probability:
        #       it will be an array with two probabilities,
        #       the first number is the probability of the text to be negative sentiment
        #       the second number is the probability of the text to be positive sentiment
        probability = prediction.select(prediction['probability']).collect()
        # store predicted label as integer to a variable 'sentiment'
        classification = int([r['prediction'] for r in prediction_label][0])
        # store the higher probability of the two labels (rounded to 3 decimals) to a variable 'confidence'
        confidence = round(max([r['probability'] for r in probability][0]),3)
        # finally make a dictionary with 'sentiment' and 'confidence'
        output = {'class':classification, 'confidence':confidence}
        # return the dictionary
        return output


# Setup the Api resource routing
# Route the URL to the resource
api.add_resource(PredictClass, '/')


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
