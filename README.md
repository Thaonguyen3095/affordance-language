# affordance-language
- query_wiki.py (in /util) collects all sentences in Wikipedia containing the given objects
- parse_wiki.py (in /util) parses the collected sentences to obtain verb-object pairs
- resnet.py takes in images in the /test-images directory and generate object embeddings
- gen_data.py (in /util) takes in verb-object pairs and object embeddings, and generate language commands-object embedding pairs for training and testing
- To train and test the model, run run.py
- All data should go in the /data directory
- helper.py (in /util) contains helper functions for data reading, cross validation lists generation, and data plotting