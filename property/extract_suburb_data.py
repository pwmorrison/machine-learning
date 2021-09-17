import pandas as pd

"""
Extract Valuer General data for a given set of suburbs.
"""

def get_unique_properties(df):
    cols = ['unit_number', 'house_number', 'street_name', 'locality']
    df_cols = df[cols]
    df_cols.drop_duplicates
    return df_cols


def main():
    filenames = [
        r'D:\data\property\2015.csv',
        r'D:\data\property\2016.csv',
        r'D:\data\property\2017.csv',
        r'D:\data\property\2018.csv',
        r'D:\data\property\2019.csv',
        r'D:\data\property\2020.csv',
    ]
    locality = 'ERSKINE PARK'

    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)

        df['locality'] = df['locality'].astype('string')
        df['primary_purpose'] = df['primary_purpose'].astype('string')

        df = df[df['locality'] == locality]
        df = df[df['primary_purpose'] == 'RESIDENCE']
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(f'D:\data\property\{locality}.csv', sep=',')

    df_unique = get_unique_properties(df)
    df.to_csv(f'D:\data\property\{locality}_properties.csv', sep=',')


if __name__ == '__main__':
    main()
