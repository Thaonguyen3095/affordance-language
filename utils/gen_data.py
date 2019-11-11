import csv
import random
import argparse
import numpy as np
from nltk import CFG

parser = argparse.ArgumentParser()
parser.add_argument('--inputVO', type=str, default='../data/object-verb.csv',
                    help='input verb-object file')
parser.add_argument('--inputOA', type=str,
                    default='../data/object-embedding.csv',
                    help='input object-embedding file')
parser.add_argument('--output', type=str, default='../data/corpus.csv',
                    help='output file')
opt = parser.parse_args()

data_size = 10000
language = ["give", "hand", "get", "me", "the", "pass", "over", "up", "down", "pick", "fetch", "bring", "a", "find", "please", "you", "help", "need", "i", "want", "to", "will", "might", "maybe", "may", "have", "hey", "take", "and", "us", "we", "use", "for", "an", "few", "some"]

# TODO(roma): should just use the template / random function for now
def gen_from_grammar(verb, obj):
    """Generates a string from a grammar."""
    print("Generating strings from context free grammar")
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

def gen_from_template(verb, obj):
    """Generates a string from templates."""
    print("Generating strings from templates.")

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

    sentences = [template + verb for template in pre]

    # returns a list of sentences, can change as needed
    return sentences

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
            sentence += obj + " "
        else:
            sentence += random.choice(language) + " "
    return(sentence)


with open(opt.inputVO, 'r') as inputVOFile:
    with open(opt.inputOA, 'r') as inputOAFile:
        with open(opt.output, 'w', newline='') as outputFile:
            csvwriter = csv.writer(outputFile, delimiter=',')
            voData = list(csv.reader(inputVOFile))
            oaData = list(csv.reader(inputOAFile))

            aff_dict = {}
            for row in oaData:
                obj = str(row[0])
                aff = str(row[1])
                if obj not in aff_dict:
                    aff_dict[obj] = []
                aff_dict[obj].append(aff)

            vo_dict = {}
            verbs = []
            for row in voData:
                obj = str(row[0])
                verb = str(row[1])
                if verb not in vo_dict:
                    verbs.append(verb)
                    vo_dict[verb] = []
                if obj not in vo_dict[verb]:
                    vo_dict[verb].append(obj)

            for i in range(data_size):
                v = random.choice(verbs)
                o = random.choice(vo_dict[v])
                sentence = gen_random_language(v, o)
                aff = random.choice(aff_dict[o])
                row = (v, o, sentence, aff)
                csvwriter.writerow(row)
