"""
Auther: LetianY
Date: 2025-4-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from const import constants
from zip_codes import df_zip_crosswalk, df_reverse_zcta_crosswalk
from surgeo.models.base_model import BaseModel
from joblib import Parallel, delayed


# configure constants: WRU
RACE_COLS = constants.RACE_COLS_WRU
RACE_MAPPING = constants.RACE_MAPPING_WRU

def load_voter_data_txt(file_path: str, delimiter: str = '\t', **kwargs) -> pd.DataFrame:
    """Load a .txt file and return a pandas DataFrame."""
    return pd.read_csv(file_path, delimiter=delimiter, **kwargs)

def clean_voter_data(df: pd.DataFrame, county_name: str='') -> pd.DataFrame:
    """Clean the NC 2022 voter data."""
    # filter by county_name
    print("filtering by county_name...")
    if county_name in df['county_desc'].unique():
        county_df = df[df['county_desc'] == county_name]
    else:
        county_df = df.copy()

    # all variables here: https://s3.amazonaws.com/dl.ncsbe.gov/data/layout_ncvoter.txt
    # we will only be using the following variables
    print("selecting columns...")
    usecols = ['county_id', 'county_desc', 'zip_code', 'ncid', 'last_name', 'first_name', 
               'middle_name', 'race_code', 'ethnic_code', 'party_cd']
    county_df = county_df[usecols]
    county_df = county_df.rename(columns={
        'last_name': 'surname', 
        'first_name': 'first', 
        'middle_name': 'middle',
        })
    county_df['state'] = 'NC'
    county_df['ethnic_code'] = county_df['ethnic_code'].astype(str)

    # clean surname
    print("cleaning surname...")
    # basemodel = BaseModel()
    county_df['surname'] = county_df['surname'].str.upper()
    # county_df['surname'] = basemodel._normalize_names(county_df['surname'])

    # clean middle name
    print("cleaning middle name...")
    county_df['middle'] = county_df['middle'].str.upper()
    # county_df['middle'] = basemodel._normalize_names(county_df['middle'])

    # clean first name
    print("cleaning first name...")
    county_df['first'] = county_df['first'].str.upper()
    # county_df['first'] = basemodel._normalize_names(county_df['first'])

    # clean ztac
    print("cleaning ztac...")
    county_df = df_zip_crosswalk(
        dataframe=county_df, 
        zip_field_name='zip_code', 
        year=2020,
        zcta_field_name='zcta',
        use_postalcode_if_error=False,
        suppress_prints=False
        )
    county_df['zcta'] = county_df['zcta'].str.zfill(5)

    # clean true race: map to Surgeo/WRU categories
    """
    Set the race value to hispanic when ethnicity is hispanic 

    HL                 HISPANIC or LATINO
    NL                 NOT HISPANIC or NOT LATINO
    UN                 UNDESIGNATED
    """
    print("cleaning race...")
    county_df['true_race'] = county_df['race_code'].map(RACE_MAPPING)
    county_df.loc[county_df['ethnic_code'] == "HL", 'true_race'] = 'hispanic'

    # remove invalid records
    print("removing invalid records...")
    return county_df.dropna(subset=['surname', 'zcta', 'party_cd', 'true_race'])

def map_zcta_to_zip(df: pd.DataFrame, zcta_col: str='perturbed_zcta', zip_col: str='perturbed_zip_code') -> pd.DataFrame:
    print("mapping perturbed ztac back to zipcodes...")
    df = df_reverse_zcta_crosswalk(
        dataframe=df, 
        zcta_field_name=zcta_col, 
        year_zip=2020,
        zip_field_name=zip_col,
        use_zcta_if_error=True,
        suppress_prints=False
        )
    df[zip_col] = df[zip_col].apply(lambda x: str(x[0]) if x else '')
    df[zip_col] = df[zip_col].str.zfill(5)
    return df

def plot_confusion_matrix(df, true_col='true_race', pred_col='pred_race', labels=None, party='', method=''):
    y_true = df[true_col]
    y_pred = df[pred_col]

    if labels is None:
        labels = list(set(list(y_true.unique())+list(y_pred.unique())))  # ensure all classes are considered
        print(labels)

    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    
    # labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for Party {party} using {method}")
    plt.show()

def plot_voter_data(df: pd.DataFrame) -> None:
    """Result evaluations."""
    PARTIES = df['party_cd'].unique()
    for party in PARTIES:
        party_df = df[df['party_cd']==party]
        print(f"Prediction accuracy for party {party}: ", accuracy_score(party_df['true_race'], party_df['pred_race']))
        plot_confusion_matrix(df=party_df, true_col='true_race', pred_col='pred_race', party=party)

def plot_estimations(estimated: dict, true: dict, party: str) -> None:
    assert estimated.keys() == true.keys(), "Both dictionaries must have the same keys."

    est_values = [estimated[race] for race in RACE_COLS]
    true_values = [true[race] for race in RACE_COLS]

    x = np.arange(len(RACE_COLS))  # label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, est_values, width, label='Estimated')
    bars2 = ax.bar(x + width/2, true_values, width, label='True')

    ax.set_xlabel('Race')
    ax.set_ylabel('Pr(Y=1|R=r)')
    ax.set_title(f'Estimated vs. True Party ({party}) Distribution Given Race')
    ax.set_xticks(x)
    ax.set_xticklabels(RACE_COLS)
    ax.legend()
    plt.show()

def weighted_estimator(df: pd.DataFrame, party):
    df = df.copy()
    df = df[(df['true_race'].isin(RACE_COLS))&(df['pred_race'].isin(RACE_COLS))]
    
    estimator_results = {}
    true_results = {}
    df[f'{party}_binary'] = np.where(df['party_cd']==party, 1, 0)
    for race in RACE_COLS:
        estimator = (df[race] * df[f'{party}_binary']).sum() / df[race].sum()
        estimator_results[race] = estimator
        true_results[race] =  df[df['true_race']==race][f'{party}_binary'].mean() # mu = Pr(Y=1|R=r)

    print(f"Weighted estimation ({party}):", estimator_results)
    print(f"True distribution ({party}):", true_results)

    # plot_estimations(estimated=estimator_results, true=true_results, party=party)
    return estimator_results, true_results

def add_noise_to_surnames(df, alpha=0.05, noise_str='xyz', seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(df)
    n_perturb = int(alpha * n_samples)
    
    indices_to_perturb = np.random.choice(df.index, size=n_perturb, replace=False)
    df.loc[indices_to_perturb, 'surname'] = df.loc[indices_to_perturb, 'surname'] + noise_str
    return df

def get_zip_in_range(zcta, zctas, distances, dist_min, dist_max):
    """
    Refer to BISG playground notebook (reference/zip-codes-perturbation) for the details of this function.
    This function is used to get a random zip code in the range of dist_min and dist_max from the given zcta.
    """
    index = zctas.index(zcta)
    (distances[index, :] >= dist_min) & ((distances[index, :] <= dist_max))
    return zctas[np.random.choice(np.nonzero((distances[index, :] >= dist_min) & (distances[index, :] <= dist_max))[0], 1)[0]]

def get_zip_with_error(zcta, zctas, distances, err_distances, err_probs):
    """
    Refer to BISG playground notebook (reference/zip-codes-perturbation) for the details of this function. 
    """
    dist_max_idx = np.random.choice(np.arange(len(err_distances)), p=err_probs)
    dist_max = err_distances[dist_max_idx]
    if dist_max_idx == 0:
        dist_min = 0
    else:
        dist_min = err_distances[dist_max_idx-1]
    return get_zip_in_range(zcta, zctas, distances, dist_min, dist_max)

def perturb_zcta(df, gamma=0.05, seed=None, **perturb_kwargs):
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(df)
    n_perturb = int(gamma * n_samples)

    indices_to_perturb = np.random.choice(df.index, size=n_perturb, replace=False)

    zctas_to_perturb = df.loc[indices_to_perturb, 'zcta'].unique().tolist()

    # parallel create a map: zcta -> perturbed_zcta
    perturbed_zctas = Parallel(n_jobs=-1)(
        delayed(get_zip_with_error)(zcta, **perturb_kwargs) for zcta in zctas_to_perturb
    )
    zcta_perturb_map = dict(zip(zctas_to_perturb, perturbed_zctas))

    df['perturbed_zcta'] = df['zcta']
    df.loc[indices_to_perturb, 'perturbed_zcta'] = df.loc[indices_to_perturb, 'zcta'].map(zcta_perturb_map)

    return df


def compute_ece(y_true, y_probs, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE) for a multi-class classification problem.
    """ 
    class_names = ['white', 'black', 'hispanic', 'api', 'other']
    true_labels = np.array([class_names.index(label) for label in y_true])

    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    correct = (predictions == true_labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = (bin_indices == i)
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = correct[mask].mean()
            ece += np.abs(bin_confidence - bin_accuracy) * (bin_size / len(y_true))

    return ece