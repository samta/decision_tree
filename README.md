# Python Implementation for building decision trees
python implementation of [id3 classification trees](https://en.wikipedia.org/wiki/ID3_algorithm) and [CART Classification And Regression Tree] (https://en.wikipedia.org/wiki/Decision_tree_learning) is a machine learning algorithm for building decision trees.


## Run the code
Run the code with the python interpreter: 

```>> python3 id3.py ./resources/<config.cfg>```

```>> python3 cart.py ./resources/<config.cfg>```

Where config.cfg is a plain text configuration file. The format of the config file is a python abstract syntax tree representing a dict with the following fields:

``
{
   'data_file' : './resources/tennis.csv',
   'data_project_columns' : ['Outlook', 'Temperature', 'Humidity', 'Windy', 'PlayTennis'],
   'target_attribute' : 'PlayTennis'
}
``

You have to specify:
 + relative path to the data file
 + list of features set
 + the target attribute, that you want to predict.
 
 # Decision Tree

 ![alt tag](https://github.com/samta/decision_tree/blob/master/tennis.png "Description goes here")
