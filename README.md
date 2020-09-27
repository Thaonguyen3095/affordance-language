# Robot Object Retrieval with Contextual Natural Language Queries
- Read the paper: http://www.roboticsproceedings.org/rss16/p080.pdf
- Watch the virtual conference presentation: https://youtu.be/HOYvL5AwX38
- To evaluate the trained model, run eval.py with the desired mode (YCB, robot, eval)
- The annotated dataset of 655 verb-object pairs can be found at data/annotated-vo.csv (a valid verb-object pair is annotated with label '1')
- We also release a smaller dataset of 88 verb-object pairs over a subset of verbs from our original dataset and objects from the YCB object set, this dataset can be found at data/ycb-verb-object.csv
- To train and evaluate your own model on our dataset, run run.py (you will first need to run resnet.py to generate embeddings for your objects, and gen_data.py to generate train and test data for the model)
- corpus-train.csv and object-embedding.csv can be found [here](https://drive.google.com/drive/folders/14h_AxGwJgGFOqdqouaZYM9cqEItXL3TY?usp=sharing)
- If you find the dataset or code useful, please cite:
```
@INPROCEEDINGS{Nguyen-RSS-20,
    AUTHOR    = {Thao Nguyen AND Nakul Gopalan AND Roma Patel AND Matthew Corsaro AND Ellie Pavlick AND Stefanie Tellex},
    TITLE     = {{Robot Object Retrieval with Contextual Natural Language Queries}},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2020},
    ADDRESS   = {Corvalis, Oregon, USA},
    MONTH     = {July},
    DOI       = {10.15607/RSS.2020.XVI.080}
}
```
