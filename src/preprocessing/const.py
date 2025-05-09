"""
Defining constants

Author: LetianY
"""
import numpy as np

class constants:
    RACE_COLS_SURGEO = ['white', 'black', 'api', 'native', 'multiple', 'hispanic']

    RACE_MAPPING_SURGEO = {
    'A': 'api',        # Asian
    'B': 'black',      # Black or African American
    'I': 'native',     # American Indian or Alaska Native
    'M': 'multiple',   # Two or More Races
    'O': 'other',      # Other - not mappable
    'P': 'api',        # Native Hawaiian or Pacific Islander
    'U': 'unknown',    # Undesignated - not mappable
    'W': 'white'       # White
    }

    RACE_COLS_WRU = ['white', 'black', 'api', 'hispanic', 'other']
    
    RACE_MAPPING_WRU = {
    'A': 'api',        # Asian
    'B': 'black',      # Black or African American
    'I': 'other',      # American Indian or Alaska Native??
    'M': 'other',      # Two or More Races
    'O': 'other',      # Other
    'P': 'api',        # Native Hawaiian or Pacific Islander
    'U': np.nan,    # Undesignated - not mappable
    'W': 'white'       # White
    }

