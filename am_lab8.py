#!/usr/bin/env python3

import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm

import numpy as np
from pathlib import Path


def load_table(data: Path):
    f = data.open()
    results = {}
    f.readline()
    for line in f:
       line_parts = line.split()
       results[line_parts[0]] = np.array(

           list(map( lambda n : float(n), line_parts[1:]))

       )
    f.close()
    return results


def lp(x, y, p=3):
    """The norma of p"""
    return sum (
               np.power( np.power( np.abs(x-y), p) , 1/p
                )
    )


def cosine(x, y):
    return 1 - dot(x,y)/norm(x)*norm(y)


def weighted_jaccard_similarity(x, y):
   return  ( sum( np.min( [x,y], axis=0 ) ) / sum( np.max([x,y], axis=0)) )


def weighted_jaccard_distance(x, y):
    return 1 - weighted_jaccard_similarity(x, y)




def samples2distances_table(samples):
    functions = (

                 ('lp',lp),
                 ('cosine',cosine),
                 ('weighted_jaccard_distance', weighted_jaccard_distance)

                 )
    ### Complete this function
    # you may create a tuple of pairs, where each pair is a name of a function (as a string)
    # and the function pointer, then loop over the tuple and fill a dictionary.
    l1 = []
    l2 = []
    l3 = []

    for func in functions:
        # print(func)
        for xk, xv in samples.items():
            for yk, yv in samples.items():
                if func[0] == 'lp':
                    l1.append( func[1](xv, yv,5))
                elif func[0] == 'cosine':
                    l2.append( func[1](xv, yv))
                else:
                    l3.append( func[1](xv, yv) )

    l1 = np.array(l1)
    l2 = np.array(l2)
    l3 = np.array(l3)
    results = {functions[0][0]:l1,
               functions[1][0]:l2,
               functions[2][0]:l3,
               }
    return results


def plot_distances(results, labels, figure):
   fig, ax = plt.subplots(1, len(results))
   i = 0
   for k,v in results.items():
      ax[i].set_xticks(range(len(labels)), labels)
      ax[i].set_yticks(range(len(labels)), labels)
      ax[i].set_title(k)
      ax[i].imshow(v.reshape(4,4), cmap = 'gray')
      i+=1


   if figure is None:
      fig.show()
   else:
      fig.savefig(figure)




def run(data: Path, figure):
    data_table = load_table(data)
    print(data_table)
    distances = samples2distances_table(data_table)
    print(distances)
    plot_distances(distances, list(data_table.keys()), figure)


if __name__ == '__main__':
    PATH=f'{os.getcwd()}\data.tsv'
    run(Path(PATH),None)