# Stance Detection dataset for FNC-1

For details of the task, see [FakeNewsChallenge.org](http://fakenewschallenge.org)


The data provided is `(headline, body, stance)` instances, where `stance` is one of `{unrelated, discuss, agree, disagree}`. The dataset is provided as two CSVs:


### `train_bodies.csv`

This file contains the body text of articles (the `articleBody` column) with corresponding IDs (`Body ID`)

### `train_stances.csv`

This file contains the labeled stances (the `Stance` column) for pairs of article headlines (`Headline`) and article bodies (`Body ID`, referring to entries in `train_bodies.csv`).


### JSON files

`train_combined.json`: JSON file with following format:
```
{
    "bodies": [...list of body items...], 
    "stances": [...list of stance items...],
}
```

The `bodies` key has data in the following format:
```
{
    "Body ID": <body id>,
    "articleBody": <body text>,
}
```

The `stances` key has data in the following format:
```
{
    "Headline": <headline text>,
    "Body ID": <body id>,
    "Stance": <body text>,
}
```


### Distribution of the data

The distribution of `Stance` classes in `train_stances.csv` is as follows:

|   rows |   unrelated |   discuss |     agree |   disagree |
|-------:|------------:|----------:|----------:|-----------:|
|  49972 |    0.73131  |  0.17828  | 0.0736012 |  0.0168094 |

Credits:

- Edward Misback
- Craig Pfeifer
