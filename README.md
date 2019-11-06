# affordance-language
- query_wiki.py (in /utils) collects all sentences in Wikipedia containing the given objects
- parse_wiki.py (in /utils) parses the collected sentences to obtain verb-object pairs
- imagenet.py (in /utils) generates a dictionary mapping ImageNet images to their labels (object name)
- resnet.py takes in ImageNet images and generate object-embedding pairs corresponding to the image
- gen_data.py (in /utils) takes in verb-object and object-embedding pairs, and generate command-embedding pairs for training and testing
- To train and test the model, run run.py
- All data should go in the /data directory
- helpers.py (in /utils) contains helper functions for data reading, positive & negative example generation, cross validation lists generation, and data plotting