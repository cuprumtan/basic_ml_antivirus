import joblib
import os
import pickle
import re
import signature_finder


def extract_infos(fpath):
    result_dict = {'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'f5': 0, 'f6': 0, 'f7': 0, 'f8': 0, 'f9': 0, 'f10': 0, 'f11': 0, 'f12': 0, 'f13': 0, 'f14': 0}
    binary_file = open(fpath, 'rb').read()
    for signature_dict in signature_finder.signatures:
        for key in signature_dict:
            for signature in signature_dict.get(key):
                if signature in binary_file:
                    result_dict[key] = 1
    for signature_regex in signature_finder.signatures_regex:
        for key in signature_regex:
            for signature_re in signature_regex.get(key):
                if re.search(signature_re, binary_file):
                    result_dict[key] = 1
    return result_dict


if __name__ == '__main__':
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

    binary_files = []
    for dirpath, dirs, files in os.walk('C:\\Users\\cuprumtan\\PycharmProjects\\basic_ml_antivirus\\programs'):
        for f in files:
            if not f.endswith('.txt'):
                binary_files.append(os.path.dirname(dirpath) + '\\' + os.path.basename(dirpath) + '\\' + f)

    for file in binary_files:
        vector = extract_infos(file)
        #print('File vector:')
        print(vector)

        # f1, f4, f7, f6, f14, f5, f3, f10, f2, f9, f11, f8, f13, f12
        pe_features = list(map(lambda x: vector[x], features))
        #print(pe_features)

        res = clf.predict([pe_features])[0]
        if not any(pe_features):
            result = 'OK'
        else:
            result = ['OK', 'malicious'][res]
        print('The file %s is %s' % (os.path.basename(file), result))
