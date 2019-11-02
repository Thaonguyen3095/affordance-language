import spacy
import csv
import nltk
import json
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

def create_object_verb_dict(data_file):
    """Reads in the data file and dumps two dictionaries to files in json form.
    object_verb.json contains counts of verbs for each object.
    verb_object.json contains counts of objects for each verb.
    """
    object_verb = {}
    verb_object = {}
    with open(data_file, encoding='latin-1') as data:
        for line in data:
            obj, sentence = line.split("\t")
            pos = nltk.pos_tag(sentence.split())
            for word, tag in pos:
                if tag == "VB":
                    verb = WordNetLemmatizer().lemmatize(word,'v')
                    if obj not in object_verb:
                        object_verb[obj] = {}
                    if verb not in object_verb[obj]:
                        object_verb[obj][verb] = 0
                    object_verb[obj][verb] += 1

                    if verb not in verb_object:
                        verb_object[verb] = {}
                    if obj not in verb_object[verb]:
                        verb_object[verb][obj] = 0

                    verb_object[verb][obj] += 1

    with open("verb-object.json", "w+") as f:
        f.write(json.dumps(verb_object))
    with open("object-verb.json", "w+") as f:
        f.write(json.dumps(object_verb))

def filter_object_verbs(object_verb_file, verb_object_file):
    with open(object_verb_file, encoding='latin-1') as data:
        for line in f:
            object_verb = json.loads(line)

    with open(verb_object_file, encoding='latin-1') as data:
        for line in f:
            verb_object = json.loads(line)

    threshold = 5 # change to higher after looking through

    # filter based on frequency
    filtered_object_verb = {}
    for obj in object_verb:
        for verb in object_verb[obj]:
            if object_verb[obj][verb] > threshold:
                if obj not in filtered_object_verb:
                    filtered_object_verb[obj] = {}
                filtered_object_verb[obj][verb] = object_verb[obj][verb]

    # filter based on tf-idf


def parse_wiki_old():
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

                    # write frequency counts as well
                    for obj, verbs in ov_pair.items():
                        csvwriter_obj.writerow((obj,len(verbs),verbs))
                        for v in verbs:
                            csvwriter_out.writerow((obj, v))

                    for verb, objects in vo_pair.items():
                        csvwriter_verb.writerow((verb,len(objects),objects))


if __name__ == '__main__':
    # parse_wiki_old()
    create_object_verb_data()
