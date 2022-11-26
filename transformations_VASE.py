
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np
import math

class wind_transformer(TransformerMixin, BaseEstimator):
    """
    Args:
        TransformerMixin (_type_): _description_
        BaseEstimator (_type_): _description_
    """
    def __init__( self,component=False ):
        self.component=component

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transforms wind speed to its horizontal and vertical components

        Args:
            speed (np_array): Array of wind speed
            direction (np_array): Arrays of wind directions in Degrees.

        Returns:
            np.array, np.array: Horizontal speed component, Vertical speed component
        """
        X = convertDirection(X)
        if( self.component==False):
            cos = np.array(np.cos(np.array(X.wwd*math.pi/180)))
            sin = np.array(np.sin(np.array(X.wwd*math.pi/180)))
            return np.column_stack((cos, sin))

        else: 
            u = np.array((X.Speed)*(np.cos(np.array(X.wwd*math.pi/180))))
            v = np.array(X.Speed)*(np.sin(np.array(X.wwd*math.pi/180)))
            return np.column_stack((u, v))

def upscale(df):
     # Remove all records before first wind estimation and all record afte
    df= df[df.index.to_series().between(
    df[~pd.isna(df['Direction'])].index.min(), df[~pd.isna(df['Direction'])].index.max())]
    df['wwd'] = df['wwd'].interpolate(method='quadratic', order=2)
    df['Speed'] = df['Speed'].interpolate(method='quadratic', order=2)
    return convertDirection(df)

def downscale(df):
    df['Direction']=df['Direction'].fillna('-1')
    df = df.resample('3h').agg({"ANM": np.mean , "Non-ANM": np.mean, "Total":np.mean, "Direction": np.max, "Source_time": np.max, "Speed": np.max})
    df = df[(~df['Direction'].isin(['-1'])) & ~df['Total'].isna()]
    return (df)

def convertDirection(df):
    windDirectionMap = {"N": 12,
                        "NNE": 11, "NE": 10, "ENE": 9,
                        "E": 8, "ESE": 7, "SE": 6, "SSE": 5,
                        "S": 4, "SSW": 3, "SW": 2, "WSW": 1,
                        "W": 0, "WNW": 13, "NW": 14,
                        "NNW": 15}
    
    df['wwd'] = pd.Series(df.Direction).map(windDirectionMap)*22.5
    return df