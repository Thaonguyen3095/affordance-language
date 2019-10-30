import spacy
import csv
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

data_file = "wiki.txt"
out_file = "verb-object.csv"
o_stat = "obj_counts.csv"
v_stat = "verb_counts.csv"

ov_pair = {}
vo_pair = {}

with open(data_file, encoding='latin-1') as data:
    with open(out_file, "w", newline="", encoding='latin-1') as out:
        with open(o_stat, "w", newline="", encoding='latin-1') as obj_stat:
            with open(v_stat, "w", newline="", encoding='latin-1') as verb_stat:
                csvwriter_out = csv.writer(out, delimiter=',')
                csvwriter_obj = csv.writer(obj_stat, delimiter=',')
                csvwriter_verb = csv.writer(verb_stat, delimiter=',')

                for line in data:
                    obj, sentence = line.split("\t")
                    pos = nltk.pos_tag(sentence.split())
                    for word, tag in pos:
                        if "V" in tag:
                            verb = WordNetLemmatizer().lemmatize(word,'v')
                            if obj not in ov_pair:
                                ov_pair[obj] = []
                            if verb not in ov_pair[obj]:
                                ov_pair[obj].append(verb)

                            if verb not in vo_pair:
                                vo_pair[verb] = []
                            if obj not in vo_pair[verb]:
                                vo_pair[verb].append(obj)

                for obj, verbs in ov_pair.items():
                    csvwriter_obj.writerow((obj,len(verbs),verbs))
                    for v in verbs:
                        csvwriter_out.writerow((obj, v))

                for verb, objects in vo_pair.items():
                    csvwriter_verb.writerow((verb,len(objects),objects))
