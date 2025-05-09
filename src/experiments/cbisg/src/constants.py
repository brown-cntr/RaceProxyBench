"""
Defining constants
"""
import numpy as np

class constants:
    white = "white"
    black = "black"
    hisp = "hispanic"

    #Asian, Native Hawaiian, Pacific Islander
    api = "api"

    #Asian
    asian = "asian"

    #American-Indian // Alaskan Native
    aian = "native"

    #mixed
    multiple = "multiple"

    #Native Hawaiian, pacific islander
    nhpi = "nhpi"

    zipcode = "zipcode"
    count = "count"
    name = "name"

    unknown = "unknown"

    race = "race"

    races = [white, black, hisp, api, aian, multiple]

    num_races = 6
    race_index = dict(zip(races, np.arange(num_races)))





