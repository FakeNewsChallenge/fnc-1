# encoding: utf-8

import os
import pandas as pd

try:
    import cPickle as pickle
except:
    import pickle

from pycorenlp import StanfordCoreNLP
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse(text):
    nlp = StanfordCoreNLP('http://localhost:9000')
    text = text.replace('\n', ' ')
    text = text.decode('utf-8', errors='replace')
    text = text.encode('ASCII', errors='replace')
    output = nlp.annotate(text,
                          properties={'annotators': 'tokenize, ssplit, pos, lemma, ner, parse', 'outputFormat': 'json'})
    print output.keys()
    return {'sentences': output['sentences']}


def parse_row(r):
    return r.claimID, parse(r.claimHeadLine), r.articleID, parse(r.articleHeadline)


if __name__ == "__main__":
    input_file = 'dank.csv'
    output_file = 'dank.out.csv'

    raw_data = pd.read_csv(input_file)
    num_ids = len(raw_data['Body ID'])

    raw_data['claimID'] = pd.Series(range(num_ids))

    # make sure these are different
    raw_data['articleID'] = pd.Series(range(num_ids, 2 * num_ids))
    raw_data.to_csv(output_file)

    data = {}
    counter = 0
    TIMEOUT = 60

    with ThreadPoolExecutor(max_workers=10) as thread_pool:
        future_dict = {thread_pool.submit(parse_row, row, TIMEOUT): row for _, row in raw_data.iterrows()}

    for f in as_completed(future_dict):
        row = future_dict[f]
        try:
            (claimId, claimParse, articleId, articleParse) = f.result()
            data[row.claimID] = claimParse
            data[row.articleID] = articleParse
        except Exception as e:
            print 'exception parsing row with claimId = {}, articleID = {}: {}'.format(row.claimID, row.articleID, e)

    with open(os.path.join('..', 'data', 'pickled', 'stanparse-data.pickle'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
