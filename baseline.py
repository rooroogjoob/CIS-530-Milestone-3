import pymagnitude as py
import numpy as np
import sys

def create_res(eng_mag, fr_mag, b_dict, t_dict, output_f):
    print("first")
    eng_vectors = py.Magnitude(eng_mag)
    fr_vectors = py.Magnitude(fr_mag)

    # we create two dictionaries from the dict data
    # one goes en->fr and the other fr->en
    # this is the training set
    data_dict_en_to_fr = {}
    data_dict_fr_to_en = {}
    with open(b_dict) as f:
        for line in f:
            pair = line.split(" ")
            pair[1] = pair[1][:-1]
            data_dict_en_to_fr[pair[0]] = pair[1]
            data_dict_fr_to_en[pair[1]] = pair[0]

    en_mat = []
    fr_mat = []
    for key in data_dict_fr_to_en.keys():
        en = eng_vectors.query(data_dict_fr_to_en[key])
        fr = fr_vectors.query(key)
        en_mat.append(en)
        fr_mat.append(fr)
    en_mat = np.array(en_mat)
    fr_mat = np.array(fr_mat)

    u, sig, vt = np.linalg.svd(np.matmul(fr_mat.transpose(), en_mat))

    W = np.matmul(np.transpose(vt), np.transpose(u))

    final = []
    with open(t_dict) as f:
        for i, line in enumerate(f):
            line = line[:-1]
            pair = line.split(" ")
            word = fr_vectors.most_similar(np.matmul(eng_vectors.query(pair[0]), W), topn=1)[0][0]
            line = line + " " + word
            final.append(line)
            print("here")

    np.savetxt(output_f, final, fmt="%s")


def main(argv):
    args = argv[1:]

    print(args)
    create_res(args[0], args[1], args[2], args[3], args[4])

if __name__ == '__main__':
    sys.exit(main(sys.argv))