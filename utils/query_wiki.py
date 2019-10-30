file_name = "map_clsloc.txt"
data_file = "/data/nlp/wikipedia-text-dump/text/final.txt"
out_file = "wiki.txt"

objects = []
with open(file_name, encoding='latin-1') as f:
    for line in f:
        obj = str(line.split()[2]).split("_")
        object = " "
        for o in obj:
            object += o + " "
        objects.append(object)

with open(data_file, encoding='latin-1') as data:
    with open(out_file, "w", encoding='latin-1') as out:
        for line in data:
            for obj in objects:
                if obj in line:
                    out.write(obj[1:len(obj)-1]+"\t"+line) #write to file
