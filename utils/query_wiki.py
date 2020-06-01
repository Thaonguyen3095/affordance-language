file_name = '../data/map_clsloc.txt' #contains mapping between ImageNet classes and labels
#substitute with your own path for the Wikipedia text corpus
data_file = '/home/thao/Downloads/wikipedia-text-dump/text/final.txt'
out_file = '../data/wiki.txt'

'''
Collect all sentences in the Wikipedia text corpus
containing the ImageNet object classes
'''

objects = []
with open(file_name, encoding='latin-1') as f:
    for line in f:
        obj = str(line.split()[2]).split('_')
        object = ' '
        for o in obj:
            object += o + ' '
        objects.append(object)

with open(data_file, encoding='latin-1') as data:
    with open(out_file, 'w', encoding='latin-1') as out:
        for line in data:
            for obj in objects:
                if obj in line:
                    out.write(obj[1:-1]+'\t'+line) #write to file
