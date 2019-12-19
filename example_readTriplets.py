import os

# read triplets into a list of list
def read_file(path, file_name):
    f = open(os.path.join(path, file_name),"r")
    lines = f.readlines()
    tri_list = []
    for line in lines:
        data = line.split("\t")
        temp = [ d.split("|") for d in data]
        tri_list.append(temp)
    return tri_list

tri_path = "./data/multi_news/triplets/"
tri_file = "test.src.triplets.txt"
tri_list = read_file(tri_path, tri_file)