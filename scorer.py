#!/usr/local/env python
"""
Scorer for the Fake News Challenge

Submission is a CSV with the following fields: headline, body_id, label
where label is in {agrees, disagrees, discusses, unrelated}

Scoring is as follows:
  +0.25 for each correct unrelated
  +0.25 for each correct related (label is any of agrees, disagrees, discusses)
  +0.75 for each correct agrees, disagrees, discusses
"""
import csv
import sys


FIELDNAMES = ['Headline', 'Body ID', 'Stance']
RELATED = ['agrees', 'disagrees', 'discusses']

USAGE = """
FakeNewsChallenge FNC-1 scorer - version 1.0
Usage: python scorer.py gold_labels test_labels

  gold_labels - CSV file with reference GOLD stance labels
  test_labels - CSV file with predicted stance labels

The scorer will provide two scores: NULL and Test
  NULL - score as if all predicted stances were unrelated
  TEST - score based on the provided predictions
"""

ERROR_MISMATCH = """
ERROR: Entry mismatch at line {}
 [expected] Headline: {} // Body ID: {}
 [got] Headline: {} // Body ID: {}
"""

SCORE_REPORT = """
NULL - score as if all predicted stances were unrelated
TEST - score based on the provided predictions

||    NULL   ||    TEST   ||\n||{:^11}||{:^11}||
"""


class FNCException(Exception):
    pass


def score_submission(gold_labels, test_labels):
    score = 0.0
    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        if g['Headline'] != t['Headline'] or g['Body ID'] != t['Body ID']:
            error = ERROR_MISMATCH.format(i+2,
                                          g['Headline'], g['Body ID'],
                                          t['Headline'], t['Body ID'])
            raise FNCException(error)
        else:
            g_stance, t_stance = g['Stance'], t['Stance']
            if g_stance == t_stance:
                score += 0.25
                if g_stance != 'unrelated':
                    score += 0.50
            if g_stance in RELATED and t_stance in RELATED:
                score += 0.25
    return score


def score_all_unrelated(gold_labels):
    return 0.25 * len([g for g in gold_labels if g['Stance'] == 'unrelated'])


def load_dataset(filename):
    data = None
    try:
        with open(filename) as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != FIELDNAMES:
                error = 'ERROR: Incorrect headers in: {}'.format(filename)
                raise FNCException(error)
            else:
                data = list(reader)

            if data is None:
                error = 'ERROR: No data found in: {}'.format(filename)
                raise FNCException(error)
    except FileNotFoundError:
        error = "ERROR: Could not find file: {}".format(filename)
        raise FNCException(error)

    return data

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(USAGE)
        sys.exit(0)

    _, gold_filename, test_filename = sys.argv

    try:
        gold_labels = load_dataset(gold_filename)
        test_labels = load_dataset(test_filename)

        test_score = score_submission(gold_labels, test_labels)
        null_score = score_all_unrelated(gold_labels)
        print(SCORE_REPORT.format(null_score, test_score))
    except FNCException as e:
        print(e)
