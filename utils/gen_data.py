import csv
import random
import argparse
import numpy as np
#from nltk import CFG

parser = argparse.ArgumentParser()
parser.add_argument('--inputVO', type=str, default='../data/filtered-object-verb.tsv', help='input verb-object file')
parser.add_argument('--inputOA', type=str, default='../data/object-embedding.csv', help='input object-embedding file')
parser.add_argument('--output', type=str, default='../data/corpus_template_verb_only.csv', help='output file')
opt = parser.parse_args()

data_size = 10000
language = ["give", "hand", "get", "me", "the", "pass", "over", "up", "down", "pick", "fetch", "bring", "a", "find", "please", "you", "help", "need", "i", "want", "to", "will", "might", "maybe", "may", "have", "hey", "take", "and", "us", "we", "use", "for", "an", "few", "some"]

THRESHOLD = 22 #frequency threshold for verb-object pair filtering
invalid = ['be', '[', ']', '<', '>', '/'] #invalid verbs
verb_only = True #only use verb in generating natural language command, if set to False then use both the verb and object (noun)
gen = 'random' #'template' #command generation scheme

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


with open(opt.inputVO, 'r') as inputVOFile:
    with open(opt.inputOA, 'r') as inputOAFile:
        with open(opt.output, 'w', newline='') as outputFile:
            csvwriter = csv.writer(outputFile, delimiter=',')
            voData = csv.DictReader(inputVOFile, dialect='excel-tab')
            oaData = list(csv.reader(inputOAFile))

            aff_dict = {}
            for row in oaData:
                obj = str(row[0])
                aff = str(row[1])
                if obj not in aff_dict:
                    aff_dict[obj] = []
                aff_dict[obj].append(aff)

            vo_dict = {}
            for row in voData:
                obj = str(row['object'])
                verb = str(row['verb'])
                freq = int(row['freq'])
                if (verb not in invalid) and (freq > THRESHOLD):
                    if verb not in vo_dict:
                        vo_dict[verb] = []
                    if obj not in vo_dict[verb]:
                        vo_dict[verb].append(obj)

            for i in range(data_size):
                v = random.choice(list(vo_dict.keys()))
                o = random.choice(vo_dict[v])
                if gen == 'grammar':
                    sentence = gen_from_grammar(v, o)
                elif gen == 'template':
                    sentence = gen_from_template(v, o)
                else:
                    sentence = gen_random_language(v, o)
                aff = random.choice(aff_dict[o])
                row = (v, o, sentence, aff)
                csvwriter.writerow(row)
