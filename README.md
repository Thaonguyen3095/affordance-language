# Robot Object Retrieval with Contextual Natural Language Queries
- Read the paper: http://www.roboticsproceedings.org/rss16/p080.pdf
- To evaluate the trained model, run eval.py with the desired mode (YCB, robot, eval).
- The annotated dataset of 655 verb-object pairs can be found at data/annotated-vo.csv (a valid verb-object pair is annotated with label '1').
- We also release a smaller dataset of 88 verb-object pairs over a subset of verbs from our original dataset and objects from the YCB object set, this dataset can be found at data/ycb-verb-object.csv.
- To train and evaluate your own model on our dataset, run run.py (you will first need to run resnet.py to generate embeddings for your objects, and gen_data.py to generate train and test data for the model).
