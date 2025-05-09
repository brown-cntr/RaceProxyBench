import os
import numpy as np
import pandas as pd
from collections.abc import Iterable
from src.constants import constants as const



class BaseIndex:
    """
    This is the class responsible for searching, and indexing
    the CSV files which contain the probabilities
    from which BISG is constructed
    """
    def __init__(self, path, index_col):
        """

        :param path: The path to the CSV which will be indexed
        :param index_col: The column name, that will serve as the index
        """

        #Check that the path points to a real file
        assert os.path.exists(path), "Make sure clean data exists"



        #Read the CSV and store as df
        if index_col != const.zipcode:
            self.base_data = pd.read_csv(path)
        else:
            self.base_data = pd.read_csv(path, dtype={const.zipcode: str})




        self._n = len(self.base_data)


        """
        The index is a dictionary mapping 
            value --> row # in table 
            
        So for eg. if the index is "name" then I could do 
            "Grayson" --> 14053 
            
        The point of doing this is for fast look up 
        """
        #TODO: I'm sure there is a smarter way to do this
        self.index = dict(
            zip(self.base_data[index_col].astype(str).to_list(), np.arange(self._n))
        )

        self._columns = self.base_data.columns

    def __getitem__(self, i):
        return self.base_data[i]

    def __len__(self):
        return len(self.base_data)




    def query(self, entries):
        """

        :param entries: The entries that we'd like to index
        :return:
        """

        if type(entries) is str:
            entries = [entries]

        assert isinstance(entries, list) or isinstance(entries, pd.Series)


        probs = []
        for e in entries:
            try:
                prob = self.base_data.iloc[self.index[e]]
                prob = prob.to_numpy()
            except:
                prob = np.empty(len(const.races))
                prob[:] = np.nan
            probs.append(prob)

        query_df = pd.DataFrame(probs, columns=self._columns)

        return query_df

class Bisg:
    def __init__(self, surname_path=None, geo_path=None):
        if surname_path is None:
            surname_path = "../data/clean/census/surnames.csv"

        if geo_path is None:
            geo_path = "../data/clean/census/zipcodes.csv"

        self._race_surname_probs = BaseIndex(surname_path, index_col=const.name)
        self._geo_race_probs = BaseIndex(geo_path, index_col=const.zipcode)


    def inference(self, surnames, zips):
        assert isinstance(zips, Iterable)
        assert isinstance(surnames, Iterable)

        geo_race = self._geo_race_probs.query(zips)
        race_surname = self._race_surname_probs.query(surnames)


        probs = self._combined_probs(sur_probs=race_surname, geo_probs=geo_race)

        surgeo_data = pd.concat([
            geo_race[const.zipcode].to_frame(),
            race_surname[const.name].to_frame(),
            probs
        ], axis=1)

        return surgeo_data

    def _combined_probs(self,
                        sur_probs: pd.DataFrame,
                        geo_probs: pd.DataFrame) -> pd.DataFrame:
        """Performs the BISG calculation"""
        # Calculate each of the numerators
        # sur_probs = sur_probs.reset_index(inplace=False)
        # geo_probs = geo_probs.reset_index(inplace=False)

        surgeo_numer = geo_probs.iloc[:, 2:] * sur_probs.iloc[:, 2:]

        # Calculate the denominator
        surgeo_denom = surgeo_numer.sum(axis=1)
        # Caluclate the surgeo probabilities (each num / denom)
        surgeo_probs = surgeo_numer.div(surgeo_denom, axis=0)
        return surgeo_probs

        print()



class fBisg():
    def __init__(self, path, surname_path=None, geo_path=None, max_iter=10):

        self.max_iter = max_iter
        assert os.path.exists(path), "Make sure cleaned data exists"

        self.records = pd.read_csv(path)
        self.n = len(self.records)


        if surname_path is None:
            surname_path = "../data/clean/census/surnames.csv"

        if geo_path is None:
            geo_path = "../data/clean/census/zipcodes_counts.csv"

        self._race_surname_probs = BaseIndex(surname_path, index_col=const.name)
        self._geo_race_probs = BaseIndex(geo_path, index_col=const.zipcode)



        print()

    def train(self):


        num_geo = len(self._geo_race_probs.base_data[const.zipcode].unique())

        num_race_geo = np.zeros(shape=(num_geo, const.num_races))

        geo_index = self._geo_race_probs.index
        race_index = const.race_index
        name_index = self._race_surname_probs.index

        #Initialize counts of people of each race living in each location
        #This serves as our prior
        for i in range(self.n):

            race = self.records.iloc[i][const.race]
            geo = str(self.records.iloc[i][const.zipcode])

            geo_ix = geo_index[geo]
            race_ix = race_index[race]

            #Check if geography is in index, if not add it
            if geo in geo_index.keys():
                geo_ix = geo_index[geo]
            else:
                geo_ix = len(geo_index)
                geo_index[geo] = geo_ix

            num_race_geo[geo_ix, race_ix] += 1


        for iter in range(self.max_iter):

            for i in range(self.n):
                race = self.records.iloc[i][const.race]
                geo = str(self.records.iloc[i][const.zipcode])
                name = self.records.iloc[i][const.name]

                geo_ix = geo_index[geo]
                race_ix = race_index[race]

                num_race_geo[geo_ix, race_ix] -= 1

                race_probs = np.zeros(const.num_races)
                for r in const.races:
                    race_jx = race_index[r]
                    census_n = self._geo_race_probs.query(geo)[r].values[0]

                    #Add geo|race probability
                    race_probs[race_jx] += np.log(num_race_geo[race_jx, geo_ix] + census_n + 1)

                    #Add surname probability




# nc_path = "../data/clean/nc/ncvoter1.csv"
# fbisg = fBisg(path=nc_path)
# fbisg.train()
# print()


# surname_path = "../data/clean/census/surnames.csv"
# race_surname_probs = BaseIndex(surname_path, index_col=const.name)
# print()












# bisg = Bisg()
# surnames = pd.Series(['DIAZ', 'JOHNSON', 'WASHINGTON'])
# zctas = pd.Series(['65201', '63144', '63110'])
#
# bisg_results = bisg.inference(surnames=surnames, zips=zctas)
#
#
# nc_voter_recrod_path = os.path.join("../data/clean/nc", "ncvoter1.csv")
#
# fBisg = fBisg(path=nc_voter_recrod_path)