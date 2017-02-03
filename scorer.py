#!/usr/local/env python
"""
Scorer for the Fake News Challenge
 - @bgalbraith

Submission is a CSV with the following fields: Headline, Body ID, Stance
where Stance is in {agree, disagree, discuss, unrelated}

Scoring is as follows:
  +0.25 for each correct unrelated
  +0.25 for each correct related (label is any of agree, disagree, discuss)
  +0.75 for each correct agree, disagree, discuss
"""
from __future__ import division
import csv
import sys


FIELDNAMES = ['Headline', 'Body ID', 'Stance']
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
RELATED = LABELS[0:3]

USAGE = """
FakeNewsChallenge FNC-1 scorer - version 1.0
Usage: python scorer.py gold_labels test_labels

  gold_labels - CSV file with reference GOLD stance labels
  test_labels - CSV file with predicted stance labels

The scorer will provide three scores: MAX, NULL, and TEST
  MAX  - the best possible score (100% accuracy)
  NULL - score as if all predicted stances were unrelated
  TEST - score based on the provided predictions
"""

ERROR_MISMATCH = """
ERROR: Entry mismatch at line {}
 [expected] Headline: {} // Body ID: {}
 [got] Headline: {} // Body ID: {}
"""

SCORE_REPORT = """
MAX  - the best possible score (100% accuracy)
NULL - score as if all predicted stances were unrelated
TEST - score based on the provided predictions

||    MAX    ||    NULL   ||    TEST   ||\n||{:^11}||{:^11}||{:^11}||
"""


class FNCException(Exception):
    pass


def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

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

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm


def score_defaults(gold_labels):
    """
    Compute the "all false" baseline (all labels as unrelated) and the max
    possible score
    :param gold_labels: list containing the true labels
    :return: (null_score, best_score)
    """
    unrelated = [g for g in gold_labels if g['Stance'] == 'unrelated']
    null_score = 0.25 * len(unrelated)
    max_score = null_score + (len(gold_labels) - len(unrelated))
    return null_score, max_score


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


def print_confusion_matrix(cm):
    lines = ['CONFUSION MATRIX:']
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    lines.append("ACCURACY: {:.3f}".format(hit / total))
    print('\n'.join(lines))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(USAGE)
        sys.exit(0)

    _, gold_filename, test_filename = sys.argv

    try:
        gold_labels = load_dataset(gold_filename)
        test_labels = load_dataset(test_filename)

        test_score, cm = score_submission(gold_labels, test_labels)
        null_score, max_score = score_defaults(gold_labels)
        print_confusion_matrix(cm)
        print(SCORE_REPORT.format(max_score, null_score, test_score))

    except FNCException as e:
        print(e)
