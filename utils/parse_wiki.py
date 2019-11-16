import json
import math
import csv
import spacy


def create_object_verb_dict(data_file):
    """Reads in the data file and dumps two dictionaries to files in json form.
    object_verb.json contains counts of verbs for each object.
    verb_object.json contains counts of objects for each verb.
    """
    object_verb = {}
    verb_object = {}
    nlp = spacy.load('en_core_web_sm')
    with open(data_file, encoding='latin-1') as data:
        for line in data:
            obj, sentence = line.split('\t')
            obj = obj.lower()
            token = nlp(sentence)
            for t in token:
                if t.pos_ == 'VERB':
                    for c in t.children:
                        if c.dep_ == 'dobj' and c.lemma_.lower() == obj:
                            v = t.lemma_.lower()
                            if obj not in object_verb:
                                object_verb[obj] = {}
                            if v not in object_verb[obj]:
                                object_verb[obj][v] = 0
                            object_verb[obj][v] += 1
                            if v not in verb_object:
                                verb_object[v] = {}
                            if obj not in verb_object[v]:
                                verb_object[v][obj] = 0
                            verb_object[v][obj] += 1

    with open("../data/verb-object.json", "w+") as f:
        f.write(json.dumps(verb_object))
    with open("../data/object-verb.json", "w+") as f:
        f.write(json.dumps(object_verb))


def filter_object_verbs(object_verb_file, verb_object_file):
    with open(object_verb_file, encoding='latin-1') as data:
        for line in data:
            object_verb = json.loads(line)

    with open(verb_object_file, encoding='latin-1') as data:
        for line in data:
            verb_object = json.loads(line)

    threshold = 1 # change to higher after looking through

    # filter based on frequency
    filtered_object_verb, verbs = {}, []
    for obj in object_verb:
        for verb in object_verb[obj]:
            if object_verb[obj][verb] >= threshold:
                if obj not in filtered_object_verb:
                    filtered_object_verb[obj] = {}
                filtered_object_verb[obj][verb] = object_verb[obj][verb]
                verbs.append(verb)

    # filter based on tf-idf i.e., calculate scores for verbs for each object
    tf_dict = {verb: {} for verb in verbs}
    idf_dict = {verb: 0 for verb in verbs}
    tf_idf_score = {}
    verbs = list(set(verbs))

    for verb in verbs:
        for obj in filtered_object_verb:
            if verb not in filtered_object_verb[obj]:
                tf_dict[verb][obj] = 0
            else:
                tf_dict[verb][obj] = filtered_object_verb[obj][verb]
                idf_dict[verb] += 1

    for obj in filtered_object_verb:
        tf_idf_score[obj] = {}
        for verb in filtered_object_verb[obj]:
            tf_idf_score[obj][verb] = math.log10(
                tf_dict[verb][obj] / float(idf_dict[verb]))

    with open("../data/filtered-verb-object.json", "w+") as f:
        f.write(json.dumps(verb_object))
    with open("../data/filtered-object-verb.json", "w+") as f:
        f.write(json.dumps(object_verb))

    with open("../data/tf-idf.json", "w+") as f:
        f.write(json.dumps(tf_idf_score))

    with open("../data/filtered-object-verb.tsv", "w+") as f:
        f.write("../data/object\tverb\tfreq\ttf-idf\n")
        for obj in filtered_object_verb:
            for verb in filtered_object_verb[obj]:
                f.write(obj + '\t' + verb + '\t'
                        + str(filtered_object_verb[obj][verb]) + '\t'
                        + str(tf_idf_score[obj][verb]) + '\n')


if __name__ == '__main__':
    create_object_verb_dict("../data/wiki.txt")
    filter_object_verbs("../data/object-verb.json","../data/verb-object.json")
