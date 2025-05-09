import rpy2
import rpy2.robjects as ro
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, data



import pandas as pd


from src.proxies import Bisg
bisg = Bisg()



wru = importr('wru')
voters = data(wru).fetch("voters")["voters"]

#R to Pandas
with (ro.default_converter + pandas2ri.converter).context():
  voters = ro.conversion.get_conversion().rpy2py(voters)

voters = voters.loc[:,["state", "county", "tract", "surname"]]


surnames = pd.Series(['Diaz', 'Johnson', 'Washington'])
tracts = pd.DataFrame({"tract":['37119004400', '37119004400', '37119004400']})
def parse_tract(df):
    df["state"] = "NC"
    df["county"] = df["tract"].apply(lambda x: str(x)[2:5])
    df["tract"] = df["tract"].apply(lambda x: str(x)[5:])

    return df

parse_tract(tracts)
tracts["surname"] = surnames

hold = bisg.inference(tracts)
print()





# #Pandas to R
# with (ro.default_converter + pandas2ri.converter).context():
#   voters = ro.conversion.get_conversion().py2rpy(voters)
#
# hmm = wru.predict_race(voter_file=voters, census_geo="tract")
#
#
# with (ro.default_converter + pandas2ri.converter).context():
#   pd_from_r_df = ro.conversion.get_conversion().rpy2py(voters)
#
# pd_from_r_df
#
# print()
