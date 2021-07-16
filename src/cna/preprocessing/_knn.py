import scanpy as sc


def knn(data):
    print("computing default knn graph")
    sc.pp.neighbors(data)
