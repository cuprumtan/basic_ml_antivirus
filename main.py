import pefile
import os
import array
import math
import pickle
import joblib
import sys
import argparse
import signature_finder


def extract_infos(fpath):
    result_dict = {'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'f5': 0, 'f6': 0, 'f7': 0, 'f8': 0, 'f9': 0, 'f10': 0, 'f11': 0, 'f12': 0, 'f13': 0, 'f14': 0}
    binary_file = open(fpath, 'rb').read()
    for signature_dict in signature_finder.signatures:
        for key in signature_dict:
            for signature in signature_dict.get(key):
                if signature in binary_file:
                    result_dict[key] = 1
    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect malicious files')
    parser.add_argument('FILE', help='File to be tested')
    args = parser.parse_args()
    # Load classifier
    clf = joblib.load(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'classifier/classifier.pkl'
    ))
    features = pickle.loads(open(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'classifier/features.pkl'),
        'rb').read()
    )

    vector = extract_infos(args.FILE)
    print('File vector:')
    print(vector)

    pe_features = list(map(lambda x: vector[x], features))
    print(pe_features)

    res = clf.predict([pe_features])[0]
    print('The file %s is %s' % (
        os.path.basename(sys.argv[1]),
        ['malicious', 'legitimate'][res])
    )
