import numpy as np
import surgeo
from time import sleep
from tqdm import tqdm
import sklearn
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


class BaseProxy:
    """
    Base proxy class which implements various proxy methods
    """
    def __init__(self):
        pass

    def train(self, data: pd.DataFrame):
        pass

    def inference(self, data: pd.DataFrame, races:[str] = None, drop_geo:bool = False, probs_only:bool =False):

        pass

    def validate_data(self, data) -> pd.DataFrame:
        pass

    def consolidate_race_probs(self, probs: pd.DataFrame, races: [str]) -> pd.DataFrame:
        # assert set(races).issubset(set(probs.columns)), "all races must be contained in df"

        incl_indexes = [list(probs.columns).index(r) for r in races if r in list(probs.columns)]
        all_index = np.arange(probs.shape[1])

        excl_indexes = [ix for ix in all_index if ix not in incl_indexes]

        probs_new = probs.iloc[:,incl_indexes]
        if excl_indexes: #if list is not empty
            probs_new["other"] = probs.iloc[:,excl_indexes].sum(axis=1, min_count=1)

        return probs_new


class gBisg(BaseProxy):
    #TODO replace with ACS

    def __init__(self, races: [str] = None):
        self.races = races
        self.geo = surgeo.GeocodeModel(geo_level="tract")
        self.geo_headers = ["tract", "state", "county"]

    def validate_data(self, data) -> bool:
        assert "state" in data.columns, "Must contain a column labeled \"state\" " \
                                          "This is case sensitive"

        assert "tract" in data.columns, "Must contain a column labeled \"tract\" " \
                                        "This is case sensitive"

        assert "county" in data.columns, "Must contain a column labeled \"county\" "\
                                        "This is case sensitive"

        return data[["state","county", "tract"]]





    def inference(self, data: pd.DataFrame, races:[str] = None, drop_geo=False, probs_only=False):
        geo_columns = self.validate_data(data)
        probs = self.geo.get_probabilities_tract(geo_columns).iloc[:,-6:]
        probs = self.consolidate_race_probs(probs, races)
        probs = probs.fillna(probs.mean())



        if probs_only:
            return probs
        else:
            if drop_geo:
                data = data.drop(self.geo_headers, axis=1)

            data = pd.concat([data,probs], axis=1)
            return data


class Bisg(BaseProxy):

    def __init__(self, geo_level="tract", surname_only=False):
        self.wru = importr('wru')
        self.surname_only = surname_only
        self.call_wru = lambda x: self.wru.predict_race(
            voter_file = x,
            census_geo = geo_level,
            surname_only = surname_only,
        )

    def validate_data(self, data) -> (pd.DataFrame):
        assert type(data) is pd.DataFrame, "Data must be pandas dataframe"

        #TODO: improve functionality to include other census.geo fields

        assert "surname" in data.columns, "Must contain a column labeled \"surname\" "\
                                          "This is case sensitive"
        if not self.surname_only:
            assert "state" in data.columns, "Must contain a column labeled \"state\" " \
                                            "This is case sensitive"

            assert "tract" in data.columns, "Must contain a column labeled \"tract\" " \
                                            "This is case sensitive"

            assert "county" in data.columns, "Must contain a column labeled \"county\" " \
                                             "This is case sensitive"

        return data


    def inference(self, data: pd.DataFrame):


        # Validate data
        data = self.validate_data(data)

        #Convert data to R
        with (ro.default_converter + pandas2ri.converter).context():
            rdf_data = ro.conversion.get_conversion().py2rpy(data)

        wru_prediction = self.call_wru(rdf_data)

        #Convert data back to python
        with (ro.default_converter + pandas2ri.converter).context():
            df_results = ro.conversion.get_conversion().rpy2py(wru_prediction)

        return df_results


class fBisg(Bisg):
    def __init__(self, geo_level="tract", surname_only=False):
        self.wru = importr('wru')
        self.surname_only = surname_only
        self.call_wru = lambda x: self.wru.predict_race(
            voter_file=x,
            census_geo=geo_level,
            surname_only=surname_only,
            skip_bad_geos=True,
            impute_missing=True,
            model="fBISG"
        )

    def validate_data(self, data) -> (pd.DataFrame):
        assert type(data) is pd.DataFrame, "Data must be pandas dataframe"

        #TODO: improve functionality to include other census.geo fields

        assert "last" in data.columns, "Must contain a column labeled \"last\" "\
                                          "This is case sensitive"

        assert "first" in data.columns, "Must contain a column labeled \"last\" " \
                                       "This is case sensitive"

        if not self.surname_only:
            assert "state" in data.columns, "Must contain a column labeled \"state\" " \
                                            "This is case sensitive"

            assert "tract" in data.columns, "Must contain a column labeled \"tract\" " \
                                            "This is case sensitive"

            assert "county" in data.columns, "Must contain a column labeled \"county\" " \
                                             "This is case sensitive"

        return data

class Bifsg(Bisg):
    def __init__(self, geo_level="county", surname_only=False):
        self.wru = importr('wru')
        self.surname_only = surname_only
        self.call_wru = lambda x: self.wru.predict_race(
            voter_file=x,
            census_geo=geo_level,
            surname_only=surname_only,
            names_to_use='surname, first',
            skip_bad_geos=True,
        )

    def validate_data(self, data) -> (pd.DataFrame):
        assert type(data) is pd.DataFrame, "Data must be pandas dataframe"

        #TODO: improve functionality to include other census.geo fields

        assert "surname" in data.columns, "Must contain a column labeled \"last\" "\
                                          "This is case sensitive"

        assert "first" in data.columns, "Must contain a column labeled \"last\" " \
                                       "This is case sensitive"

        if not self.surname_only:
            assert "state" in data.columns, "Must contain a column labeled \"state\" " \
                                            "This is case sensitive"

            assert "tract" in data.columns, "Must contain a column labeled \"tract\" " \
                                            "This is case sensitive"

            assert "county" in data.columns, "Must contain a column labeled \"county\" " \
                                             "This is case sensitive"

        return data

class mlBisg(BaseProxy):
    def __init__(self, model: str, proxy: BaseProxy = None):
        self.proxy = proxy
        if self.proxy is not None:
            assert isinstance(proxy, BaseProxy), "must be implemented proxy type"
            self.proxy = proxy

        if model.lower() == "mr":
            self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                            penalty="l2", tol=1e-7, max_iter=int(1e6))

        elif model.lower() == "gb":
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                                    max_depth=2, random_state=0)

        elif model.lower() == "rf":
            self.model = RandomForestClassifier(max_depth=2, random_state=0)

        else:
            raise Exception("Model not implemented")


    def train(self, X, Y):
        if self.proxy is not None:
            proxy_input = self.proxy.validate_data(X)

            X = self.proxy.inference(proxy_input, drop_geo=True)
        X = X.to_numpy().astype(float)

        self.model.fit(X,Y)

    def inference(self, X: pd.DataFrame):
        if self.proxy is not None:
            X = self.proxy.validate_data(X)
            X = self.proxy.inference(X)

        return self.model.predict_proba(X)

    def validate_data(self, data) -> bool:
        pass


class ftBisg(BaseProxy):
    def __init__(self, census, races=None):
        self.races = races
        self.census = census
        self.surname_race = self.census.load_surname_table(counts=False)


    def train(self, data: pd.DataFrame, target, outcome, eta=None):


        #Get census geo_counts
        census_geo = self.census.load_geo_table(races=self.races, index="CENSUS_TRACT")


        #Get the prior counts (these are actually the counts from the voter file)
        mask = (data[target] == outcome)
        masked = data[mask]
        outcome_geo = masked.groupby("geoid")["race"].value_counts().unstack()
        outcome_geo = self.consolidate_race_probs(outcome_geo.fillna(0), self.races)

        # eta = 1 #outcome_geo.to_numpy().sum() / census_geo.to_numpy().sum()

        etas = np.array([0, 0.01, 0.05, .1, .25, .5, .75, 1])
        cond_race_probs = []
        keep_etas = []

        print("Beginning Training...")
        for g in tqdm(census_geo.index):

            best_score = np.inf
            best_sample = None
            best_eta = None

            #Skip is no one is observed to live there, use census_prob
            if g in outcome_geo.index:

                for eta in etas:
                    counts = census_geo.loc[g].to_numpy() * eta

                    obs_counts = outcome_geo.loc[g].to_numpy()
                    counts += obs_counts

                    samples = np.random.dirichlet(counts + 1e-4, size=5000).mean(axis=0)

                    filter_by_g = masked[masked.geoid == g]
                    hpo_probs = self._hpo_helper(data=filter_by_g, geo_prob=samples)

                    hpo_guess = hpo_probs.mean().sub(filter_by_g.race.value_counts(normalize=True), fill_value=0)
                    heuristic = np.abs(hpo_guess.to_numpy()).sum()

                    if heuristic < best_score:
                        best_score = heuristic
                        best_sample = samples
                        best_eta = eta
            else:
                counts = census_geo.loc[g].to_numpy()
                best_sample = np.random.dirichlet(counts + 1e-4, size=5000).mean(axis=0)
                best_eta = -1
                keep_etas.append(best_eta)

                # p = predicted[mask].mean(axis=0)
                # top = (p[i] * X_test["Loan Originated"].mean())
                # bottom = predicted.mean(axis=0)[i]
                # our_rate = top / bottom



            cond_race_probs.append(best_sample)



        self.best_etas = np.array(keep_etas)


        self.race_geo = pd.DataFrame(data=cond_race_probs,
                                  index=census_geo.index,
                                  columns=census_geo.columns)


    def prob_fetcher(self, id, df):
        if id in df.index:
            return df.loc[id].to_numpy()
        else:
            return np.ones(df.shape[1])

    def _hpo_helper(self, data: pd.DataFrame, geo_prob):

        sur_probs = pd.DataFrame(
            data= map(lambda x: self.prob_fetcher(x, self.surname_race), data.surname),
            index = data.index,
            columns = self.surname_race.columns
        )

        sur_probs = self.consolidate_race_probs(sur_probs, races=self.races)

        race_probs = pd.DataFrame(
            data = np.repeat(geo_prob.reshape(1,-1), sur_probs.shape[0], axis=0),
            index= data.index,
            columns = sur_probs.columns
        )

        assert race_probs.shape == sur_probs.shape

        bisyg = race_probs.mul(sur_probs, fill_value=0)
        bisyg = bisyg.divide(bisyg.sum(axis=1), axis=0)

        return bisyg

    def inference(self, data: pd.DataFrame, races:[str] = None, drop_geo:bool = False, probs_only:bool =False):




        # race_probs = data.geoid.apply()
        race_probs = pd.DataFrame(
            data= map(lambda x: self.prob_fetcher(x, self.race_geo), data.geoid),
            index = data.index,
            columns = self.race_geo.columns
        )

        sur_probs = pd.DataFrame(
            data= map(lambda x: self.prob_fetcher(x, self.surname_race), data.surname),
            index = data.index,
            columns = self.surname_race.columns
        )

        sur_probs = self.consolidate_race_probs(sur_probs, races=self.races)

        assert race_probs.shape == sur_probs.shape

        bisyg = race_probs.mul(sur_probs, fill_value=0)
        bisyg = bisyg.divide(bisyg.sum(axis=1), axis=0)

        return pd.concat([data, bisyg], axis=1)

    def _make_name_table(self):
        names = self.voters["surname"].unique()
        voters = self.voters
        for n in names:
            name_df = voters[voters["surname"] == n]
            pass

    def make_geo_table(self, voters, party=None, races=None):
        if party is not None:
            voters = voters[voters["party"] == party]


        table = voters.groupby("geoid").race.value_counts().unstack(fill_value=0)
        if races is not None:
            conslidated_table = self.consolidate_race_probs(table, races)

        return conslidated_table








