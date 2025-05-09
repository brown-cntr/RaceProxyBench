"""
ZIP-Code Crosswalks.
Takes USA ZIP-Codes and returns 5 digit US Census Zip Code Tabulation Areas (ZCTAs)
and/or their latitude & longitude centroid coordinates.

By J. A. Cooper https://github.com/cooperjaXC
"""

import os, json, inspect, numpy as np, pandas as pd
from collections import defaultdict


def reverse_dict(dictionary: dict):
    """Takes an input dictionary and reverses the keys and values.
    Input dictionary values do not need to be unique, and they can be both individual and list types."""
    # Create a defaultdict to store reversed values
    reverse_dictionary = defaultdict(list)

    for key, value in dictionary.items():
        # https://stackoverflow.com/questions/13675296/how-to-overcome-typeerror-unhashable-type-list
        if isinstance(value, list):
            for v in value:
                reverse_dictionary[v].append(key)
        else:
            reverse_dictionary[value].append(key)

    # If all the values in the dict have a len <=1, set them as individual, non_listed values.
    if all(len(v) <= 1 for v in reverse_dictionary.values()):
        reverse_dictionary = {
            k: v[0] if v else None for k, v in reverse_dictionary.items()
        }

    # Convert defaultdict item back to a regular dictionary
    dict_return = dict(reverse_dictionary)
    return dict_return


class ZipCodes:
    """Contains variables relating to ZIP-Codes and ZCTAs.
    2010 Crosswalk data here comes from https://udsmapper.org/zip-code-to-zcta-crosswalk/ .
    This dictionary is based on the 2010 Census' ZCTA data and will need to be updated with the new 2020 geographies."""

    def __init__(self, year: int = 2020):
        # Establish the Census year in question
        try:
            year = int(year)
        except:
            print(
                "Year value of",
                year,
                "is an invalid input for the Zip Code Class. Default of '2020' will be set.",
            )
            year = 2020
        if year not in [2010, 2020]:  # Census years
            if year < 2020:
                # years in the 2010s (before 2020) all operate on 2010 census definitions.
                year = 2010
            else:
                year = 2020
        self.year = str(year)

        # Find the path to the directory of JSONs containing ZIP-Code data.
        filepath = os.path.join(
            os.path.realpath(
                os.path.abspath(
                    os.path.split(inspect.getfile(inspect.currentframe()))[0]
                )
            ),
            "json",
        )
        # If the `json` directory is in a sibling directory to the script
        if os.path.exists(filepath) is False:
            filepath = os.path.join(
                os.path.dirname(os.path.dirname(filepath)), os.path.basename(filepath)
            )

        # Establish year-specific variables
        # # ZIP-Code -> ZCTA Crosswalks
        crosswalk_path = os.path.join(
            filepath, "zipzcta_crosswalk_" + self.year + ".json"
        )
        with open(crosswalk_path) as open_cross:
            # A dictionary containing the ZIP-Code (key; string)
            # and corresponding ZCTA (value; string) for the given [year].
            self.crosswalk = json.load(open_cross)

        # # ZCTA -> ZIP-Code Reverse Crosswalks
        self.reverse_crosswalk = reverse_dict(self.crosswalk)

        # # ZCTA -> Latitude & Longitude Centroids
        latlon_path = os.path.join(
            filepath, "zcta_latloncentroid_" + self.year + ".json"
        )
        with open(latlon_path) as open_ll:
            # A dictionary containing the ZCTA (key; string)
            # and corresponding [Latitude, Longitude] coordinates (value; list) for the given [year].
            self.latlon_centroids = json.load(open_ll)


def zip_code_formatter(postal_code):
    """Formats a USA ZIP-Code into the correct 5-digit format."""
    # Put in some safeguards here in case you get entries with zip 9s or zips w/o the leading 0s.
    postal_code = str(postal_code)
    if len(postal_code) > 5:
        postal_code = postal_code[:5]
    if "-" in postal_code:
        postal_code = postal_code.replace("-", "").replace(" ", "")
    # Use zfill()? https://stackoverflow.com/questions/733454/best-way-to-format-integer-as-string-with-leading-zeros
    if len(postal_code) == 3:
        # No longer only uses postal codes "501" and "544". Expanded to include US overseas territories.
        postal_code = "00" + postal_code
    if len(postal_code) == 4:
        postal_code = "0" + postal_code
    # Catch nulls and return None
    null_list = ["0", "nan", "null", "none", "0none", "00nan"]
    if (not postal_code) or (postal_code.lower() in null_list):
        postal_code = None
    return postal_code


def zip_code_crosswalk(
    postal_code,
    year=2020,
    zipcrosswalk=None,
    use_postalcode_if_error: bool = False,
    suppress_prints: bool = False,
):
    """This function takes a (1) postal ZIP Code and transforms it into a Zip Code Tabulation Area,
    the US Census-defined polygonal region for a ZIP Code.
    Postal ZIP Codes are not indicative of a continuous region; rather, they are functional attributes used by the
    US Postal Service to deliver mail & goods. They can refer to a single post office (point), a discontinuous
    region, or an area that transcends state borders. The ZCTA is the Census' way of polygonizing and ordering this
    messy but commonly-used geographic identifying attribute.
    Crosswalk data here comes from https://udsmapper.org/zip-code-to-zcta-crosswalk/ .
    This function has been updated with the 2020 census' new ZCTA definitions."""
    if not zipcrosswalk:
        zipcrosswalk = ZipCodes(year).crosswalk

    # Put in some safeguards here in case you get entries with zip 9s or zips w/o the leading 0s.
    postal_code = zip_code_formatter(postal_code)

    # Get ZCTA
    if postal_code in zipcrosswalk:
        zcta = zipcrosswalk[postal_code]
    else:
        nozctawarning = str(postal_code) + " is not in this repository's records."
        if use_postalcode_if_error is True:
            if suppress_prints is False:
                print(nozctawarning, "The input postal code will be returned instead.")
            zcta = postal_code
        else:
            if suppress_prints is False and str(postal_code).lower() != "none":
                print(
                    nozctawarning,
                    "No ZCTA will be returned. Please double check your entry and try again.",
                )
            zcta = None

    return zcta


def df_zip_crosswalk(
    dataframe: pd.DataFrame,
    zip_field_name: str,
    year: int = 2020,
    zcta_field_name: str = "zcta",
    use_postalcode_if_error: bool = False,
    suppress_prints: bool = False,
):
    """Takes a Pandas Dataframe with a ZIP-Code field and returns a ZCTA field using the crosswalk function.
    Returns a Pandas dataframe."""
    zipcrosswalk = ZipCodes(year).crosswalk
    if zip_field_name not in dataframe.columns.to_list():
        print(
            zip_field_name,
            "not in the submitted dataframe. No ZCTA field will be added.",
        )
        return dataframe
    else:
        outdf = dataframe.copy()
        outdf[zcta_field_name] = (
            outdf[zip_field_name]
            .fillna("0")
            .astype(int)
            .astype(str)
            .apply(
                lambda x: zip_code_crosswalk(
                    x,
                    year=year,
                    zipcrosswalk=zipcrosswalk,
                    use_postalcode_if_error=use_postalcode_if_error,
                    suppress_prints=suppress_prints,
                )
            )
        )
        return outdf


def reverse_zcta_crosswalk(
    zcta, year=2020, zcta_reverse_xwalk = None, suppress_prints: bool = False, use_zcta_if_error: bool = True
):
    """Function takes a ZCTA and returns a list of all ZIP Codes that correspond to that ZCTA."""

    # Dictionary of all ZCTAs and their
    if not zcta_reverse_xwalk:
        zcta_reverse_xwalk = ZipCodes(year).reverse_crosswalk

    # Put in some safeguards here in case you get entries with zip 9s or zips w/o the leading 0s.
    zcta_formatted = zip_code_formatter(zcta)

    # Get ZIP Codes
    if zcta_formatted in zcta_reverse_xwalk:
        zips = zcta_reverse_xwalk[zcta_formatted]
    else:
        nozctawarning = (
            str(zcta_formatted) + " is not in this repository's records of ZCTAs."
        )
        # Check if the passed ZCTA is actually a non-ZCTA ZIP Code.
        if zcta_formatted in ZipCodes(year).crosswalk:
            if suppress_prints is False:
                print(
                    nozctawarning,
                    "However, it is already a ZIP Code.",
                    "It will be returned inside a list as the only item.",
                )
            zips = [zcta_formatted]
        elif use_zcta_if_error is True:
            if suppress_prints is False:
                print(
                    nozctawarning,
                    "It will be returned inside a list as the only item instead.",
                )
            zips = [zcta_formatted]
        else:
            if suppress_prints is False and str(zcta_formatted).lower() != "none":
                print(
                    nozctawarning,
                    "No zips will be returned. Please double check your entry and try again.",
                )
            zips = []
    # print("zips", zips)
    return zips


def df_reverse_zcta_crosswalk(
    dataframe: pd.DataFrame,
    zcta_field_name: str = "zcta",
    year_zip: int = 2020,
    zip_field_name: str = "zip_codes",
    use_zcta_if_error: bool = True,
    suppress_prints: bool = False,
):
    """Takes a Pandas Dataframe with a ZCTA field and returns a ZIP-Code field using the reverse crosswalk function.
    Returns a Pandas dataframe."""
    zcta_reverse_xwalk = ZipCodes(year_zip).reverse_crosswalk
    if zcta_field_name not in dataframe.columns.to_list():
        print(
            zcta_field_name,
            "not in the submitted dataframe. No ZIP Code field will be added.",
        )
        return dataframe
    else:
        outdf = dataframe.copy()
        # Create and fill the ZCTA field.
        outdf[zip_field_name] = (
            outdf[zcta_field_name]
            .fillna("0")
            .astype(int)
            .astype(str)
            .apply(
                lambda x: reverse_zcta_crosswalk(
                    x,
                    year=year_zip,
                    zcta_reverse_xwalk=zcta_reverse_xwalk,
                    use_zcta_if_error=use_zcta_if_error,
                    suppress_prints=suppress_prints,
                )
            )
        )
        # Format the input ZCTA field for output.
        # outdf[zcta_field_name] = (
        #     outdf[zcta_field_name]
        #     .fillna("0")
        #     .astype(int)
        #     .astype(str)
        #     .apply(lambda x: zip_code_formatter(x))
        # )
        # Fix: outdf[zip_field_name] will be a list of zip codes.
        return outdf


def lat_lon_centroid(
    postal_code,
    year: int = 2020,
    use_postalcode_if_error: bool = False,
    suppress_prints: bool = False,
):
    """Returns the latitude and longitude coordinates in the centroid of the postal ZIP code's ZCTA
    as defined by the US Census Bureau's TIGER shapefiles. The function will return a list: [lat, lon].
    These centroids are not guaranteed to be on land.
    If there is a body of water near the geometric center of the ZCTA, the centroid may be placed offshore."""
    zcta = zip_code_formatter(
        zip_code_crosswalk(postal_code, year, use_postalcode_if_error, suppress_prints)
    )
    latlon_crosswalk = ZipCodes(year).latlon_centroids

    if zcta in latlon_crosswalk:
        centroid = latlon_crosswalk[zcta]
    else:
        if suppress_prints is False:
            zip_is_zcta = zcta == postal_code
            if zip_is_zcta:
                no_centroid_warning = (
                    str(postal_code)
                    + " does not have a centroid in this repository's records"
                )
            else:
                no_centroid_warning = (
                    str(postal_code)
                    + "'s tabulation area "
                    + str(zcta)
                    + " does not have a centroid in this repository's records"
                )
            no_centroid_warning = (
                no_centroid_warning
                + " for the census year "
                + str(year)
                + ".\n No centroid will be returned. Please double check your entry and try again."
            )
            print(no_centroid_warning)
        # There is hot debate in coding communities on whether to reutrn None or an empty list for situations like this.
        # https://softwareengineering.stackexchange.com/questions/120355/is-it-better-to-return-null-or-empty-values-from-functions-methods-where-the-ret
        # https://www.reddit.com/r/Python/comments/30yb5t/return_none_or_not_to_return_none/
        # I have chosen to return the None as coordinates within a list
        # # so as to not destroy downstream list parsing efforts.
        centroid = [None, None]
    return centroid


def df_latlon_centroids(
    dataframe: pd.DataFrame,
    zip_in_field_name: str,
    year: int = 2020,
    keep_coordinates_field: bool = False,
    use_postalcode_if_error: bool = False,
    suppress_prints: bool = False,
):
    """
    Takes a Pandas Dataframe with a ZIP-Code field and returns a [latitude, longitude] coordinates field
    using the `lat_lon_centroid` function. Returns a Pandas dataframe.
    """
    if zip_in_field_name not in dataframe.columns.to_list():
        print(
            zip_in_field_name,
            "not in the submitted dataframe. No ZCTA centroid fields will be added.",
        )
        return dataframe
    else:
        outdf = dataframe.copy()
        coordfieldname = "coordinates"
        # Generate the coordinates field with a [lat, lon] list
        outdf[coordfieldname] = outdf[zip_in_field_name].apply(
            lambda x: lat_lon_centroid(
                x,
                year=year,
                use_postalcode_if_error=use_postalcode_if_error,
                suppress_prints=suppress_prints,
            )
        )
        # Split the coordinates field into a lat and lon field separately.
        outdf["lat"] = outdf[coordfieldname].apply(lambda x: x[0])
        outdf["lon"] = outdf[coordfieldname].apply(lambda x: x[1])
        # Remove the coordinates field unless the input argument says otherwise
        if keep_coordinates_field is False:
            outdf = outdf.drop(coordfieldname, axis=1)
        return outdf
