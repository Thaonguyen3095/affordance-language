import csv
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--inputVO', type=str, default='verb-obj.csv', help='input verb-object file')
parser.add_argument('--inputOA', type=str, default='obj-affordance.csv', help='input object-affordance file')
parser.add_argument('--output', type=str, default='corpus.csv', help='output file')
opt = parser.parse_args()

# affordances = [cut, contain, roll, stir, support]
NUM_AF = 5
MEAN = 0
STD = 0.25
num_objs = 6
language = ["give me the ", "hand me the ", "get me the "]

with open(opt.inputVO, 'r') as inputVOFile:
    with open(opt.inputOA, 'r') as inputOAFile:
        with open(opt.output, 'w') as outputFile:
            csvwriter = csv.writer(outputFile, delimiter=',')
            voData = list(csv.reader(inputVOFile))
            oaData = list(csv.reader(inputOAFile))

            affordance_dict = {}
            for row in oaData:
                obj = row[0]
                aff = row[1]
                affordances = []
                for i in range(num_objs):
                    a = ""
                    noise = np.random.normal(MEAN, STD, NUM_AF)
                    for i in range(NUM_AF):
                        if str(i) in aff:
                            if noise[i] > 0:
                                noise[i] = -noise[i]
                            a += str(1 + noise[i]) + " "
                        else:
                            a += str(noise[i]) + " "
                    affordances.append(a)
                affordance_dict[obj] = affordances

            for row in voData:
                verb = row[0]
                obj = row[1]
                affordance = affordance_dict[obj]
                for a in affordance:
                    start = random.choice(language)
                    command = start + str(obj) + " to " + str(verb)
                    csvwriter.writerow((verb, obj, command, a))
