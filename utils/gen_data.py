import csv
import random
import argparse
import numpy as np
#from nltk import CFG

parser = argparse.ArgumentParser()
parser.add_argument('--inputVO', type=str, default='../data/filtered-object-verb.tsv', help='input verb-object file')
parser.add_argument('--inputOA', type=str, default='../data/object-embedding.csv', help='input object-embedding file')
parser.add_argument('--concreteness', type=str, default='../data/concreteness.csv', help='concreteness scores')
parser.add_argument('--train', type=str, default='../data/corpus-train.csv', help='output file')
parser.add_argument('--test', type=str, default='../data/corpus-test.csv', help='output file')
opt = parser.parse_args()

data_aug = 10 # rate of augmenting each data point
language = ["give", "hand", "get", "me", "the", "pass", "over", "up", "down", "pick", "fetch", "bring", "a", "find", "please", "you", "help", "need", "i", "want", "to", "will", "might", "maybe", "may", "have", "hey", "take", "and", "us", "we", "use", "for", "an", "few", "some"]

THRESHOLD = 5 #frequency threshold for verb-object pair filtering
invalid = ['anus', 'abdomen', 'appendix', 'nipple', 'bladder', 'breast', 'buttocks', 'chest', 'eyelid', 'finger', 'groin', 'gland', 'intestine', 'lip', 'mouth', 'pelvis', 'scrotum', 'stomach']
verb_only = False #only use verb in generating natural language command, if set to False then use both the verb and object (noun)
gen = 'template' #'template' #command generation scheme

def gen_from_grammar(verb, obj):
    """Generates a string from a grammar."""
    #print("Generating strings from context free grammar")
    grammar = CFG.fromstring("""S -> NP VP
    PP -> P NP
    NP -> Det N | NP PP
    VP -> V NP | VP PP
    Det -> 'a' | 'the'
    N -> 'wall' | 'kitchen'
    V -> 'walk' | 'turn' | 'crawl' | 'go'
    P -> 'on' | 'in'
    """)

    print("Grammar:\t", grammar)
    print("Grammar productions:\t")

    sentences = []
    for production in grammar.productions():
        sentences.append(production)
    return sentences


def gen_from_template(verb, obj):
    """Generates a string from templates."""
    #print("Generating strings from templates.")

    pre_obj = ["Give me the ", "Hand me the ", "Pass me the ", "Fetch the ",
           "Get the ", "Bring the ", "Bring me the "
           "I need the ", "I want the ",
           "I need a ", "I want a ",]

    pre_verb = ["An item that can ", "An object that can ",
           "Give me something that can ", "Give me an item that can ",
           "Hand me something with which I can ",
           "Give me something with which I can ",
           "Hand me something to ", "Give me something to ",
           "I want something to ", "I need something to ",]

    if verb_only:
        template = random.choice(pre_verb)
        sentence = template + verb
    else:
        template = random.choice(pre_obj)
        sentence = template + obj + " to " + verb
    return sentence


def gen_random_language(verb, obj):
    """Generates a string of random words around the <verb, object> pair."""
    len = random.randrange(5, 21)
    pos_v = random.randrange(0, len)
    pos_o = pos_v
    # make sure the verb and object have different positions in the sentence
    while pos_o == pos_v:
        pos_o = random.randrange(0, len)
    sentence = ""
    for i in range(len):
        if i == pos_v:
            sentence += verb + " "
        elif i == pos_o:
            if verb_only:
                sentence += random.choice(language) + " "
            else:
                sentence += obj + " "
        else:
            sentence += random.choice(language) + " "
    return(sentence)

concrete = {}
with open(opt.concreteness, 'r') as cFile:
    c = list(csv.reader(cFile))
    for row in c:
        obj = row[0]
        score = float(row[2])
        if obj not in concrete:
            concrete[obj] = score

verbs, objects = [], []
v_count, o_count = 0, 0
with open('../data/v-o.csv', 'r') as f:
    data = list(csv.reader(f))
    for row in data:
        word = str(row[0])
        if word not in concrete:
            score = 0
            v_count += 1
        else:
            score = concrete[word]
        if (int(row[1]) >= 5) and (score >= 3.5):
            verbs.append(word)

with open('../data/o-v.csv', 'r') as f:
    data = list(csv.reader(f))
    for row in data:
        word = str(row[0])
        if word not in concrete:
            score = 0
            o_count += 1
        else:
            score = concrete[word]
        if (int(row[1]) >= 5) and (score >= 3.5) and (word not in invalid):
            objects.append(word)
print('COUNTS', v_count, o_count)

with open(opt.inputVO, 'r') as inputVOFile:
    with open(opt.inputOA, 'r') as inputOAFile:
        with open(opt.train, 'w', newline='') as outputTrain:
            with open(opt.test, 'w', newline='') as outputTest:
                writer_train = csv.writer(outputTrain, delimiter=',')
                writer_test = csv.writer(outputTest, delimiter=',')
                voData = csv.DictReader(inputVOFile, dialect='excel-tab')
                oaData = list(csv.reader(inputOAFile))

                aff_dict = {}
                for row in oaData:
                    obj = str(row[0]).lower()
                    aff = str(row[1])
                    img = str(row[2])
                    if obj not in aff_dict:
                        aff_dict[obj] = []
                        aff_dict[obj].append([aff, img])

                vo_dict, ov_dict = {}, {}
                for row in voData:
                    obj = str(row['object'])
                    verb = str(row['verb'])
                    freq = int(row['freq'])
                    if (verb in verbs) and (obj in objects) and (freq >= THRESHOLD):
                        if verb not in vo_dict:
                            vo_dict[verb] = []
                        if obj not in vo_dict[verb]:
                            vo_dict[verb].append(obj)
                        if obj not in ov_dict:
                            ov_dict[obj] = []
                        if verb not in ov_dict[obj]:
                            ov_dict[obj].append(verb)

                verbs, objects = [], []
                for v, obj in vo_dict.items():
                    if (len(obj) >= 5):
                        verbs.append(v)
                for o, v in ov_dict.items():
                    if (len(v) >= 5):
                        objects.append(o)

                test = random.sample(objects, k=int(len(objects)/5))
                train = []
                with open('../data/train.csv', 'w', newline='') as tr:
                    with open('../data/test.csv', 'w', newline='') as te:
                        w_tr = csv.writer(tr, delimiter=',')
                        w_te = csv.writer(te, delimiter=',')
                        for o in objects:
                            if o in test:
                                w_te.writerow((o, len(ov_dict[o])))
                            else:
                                train.append(o)
                                w_tr.writerow((o, len(ov_dict[o])))

                with open('../data/all.csv', 'w', newline='') as f:
                    writer_all = csv.writer(f, delimiter=',')
                    for v, obj in vo_dict.items():
                        for o in obj:
                            if (v in verbs) and (o in objects):
                                writer_all.writerow((v, o))
                                if o in train:
                                    for i in range(data_aug):
                                        if gen == 'grammar':
                                            sentence = gen_from_grammar(v, o)
                                        elif gen == 'template':
                                            sentence = gen_from_template(v, o)
                                        else:
                                            sentence = gen_random_language(v, o)
                                        aff, img = random.choice(aff_dict[o])
                                        row = (v, o, sentence, aff, img)
                                        writer_train.writerow(row)
                                else:
                                    if gen == 'grammar':
                                        sentence = gen_from_grammar(v, o)
                                    elif gen == 'template':
                                        sentence = gen_from_template(v, o)
                                    else:
                                        sentence = gen_random_language(v, o)
                                    aff, img = random.choice(aff_dict[o])
                                    row = (v, o, sentence, aff, img)
                                    writer_test.writerow(row)
