import requests
import pandas as pd
import matplotlib.pyplot as plt

"""
https://maps.six.nsw.gov.au/sws/AddressLocation.html
"""

vg_road_types = [
    'ALLY', 'AV', 'AVE', 'AVE E', 'AVE N', 'AVE S', 'AVE W', 'AVENUE', 'BVD', 'BVDE',
    'CCT', 'CH', 'CHASE', 'CIR', 'CL', 'CLT', 'CNR', 'CR', 'CRES', 'CT',
    'DR', 'DR E', 'DR N', 'DR S', 'DR W', 'DRIVE', 'ESP', 'GDNS', 'GLD', 'GLEN', 'GR', 'GRA',
    'HWY', 'LANE', 'LINK', 'LOOP',
    'PATH', 'PDE', 'PDE E', 'PDE N', 'PDE S', 'PDE W', 'PKWY', 'PL', 'PWY',
    'RD', 'RD E', 'RD N', 'RD S', 'RD W', 'RDWY', 'RDGE', 'RISE', 'ROAD', 'ROW', 'RTT', 'RUN', 'RVR',
    'ST', 'ST E', 'ST N', 'ST S', 'ST W', 'STRAIT', 'SQ', 'TCE', 'TRK', 'TRL', 'WALK', 'WAY'
]

def get_geocode(house_number, road_name, road_type, suburb, post_code, projection):
    #r = requests.get(f'https://maps.six.nsw.gov.au/services/public/Address_Location?houseNumber=2&roadName=Hocking&roadType=Pl&suburb=ErskinePark&postCode=2759&projection=EPSG%3A4326')
    r = requests.get(
        f'https://maps.six.nsw.gov.au/services/public/Address_Location?houseNumber={house_number}&roadName={road_name}&roadType={road_type}&suburb={suburb}&postCode={post_code}&projection={projection}')
    response = r.json()
    lat = response['addressResult']['addresses'][0]['addressPoint']['centreX']
    long = response['addressResult']['addresses'][0]['addressPoint']['centreY']
    return lat, long


def plot_geocodes(df):
    fig, axes = plt.subplots(1, 1, squeeze=False)
    ax = axes[0, 0]
    ax.scatter(df['lat'], df['long'])
    ax.axis('equal')
    plt.show()



def extract_property_addresses(vg_sales_price_csv_filename):
    """
    Extracts the property addresses in a format that can be used to get the geocode.
    The input is the csv constructed from Valuer General data.
    """
    df = pd.read_csv(vg_sales_price_csv_filename)

    #df = df.iloc[:10, :]

    df['street_name'] = df['street_name'].astype('string')
    df['post_code'] = df['post_code'].astype(int)
    df.dropna(axis=0, subset=['street_name'], inplace=True)

    def extract_road_name(row):
        street_name = row['street_name'].upper()
        street_types = [t for t in vg_road_types if street_name.endswith(t)]
        street_type = street_types[0]
        # Get the index of the last occurrence of the street type
        index = street_name.rfind(street_type)
        street = street_name[:index]
        return street.strip()
    df['road_name'] = df.apply(extract_road_name, axis='columns')

    def extract_road_type(row):
        street_name = row['street_name'].upper()
        street_types = [t for t in vg_road_types if street_name.endswith(t)]
        street_type = street_types[0]
        return street_type
    df['road_type'] = df.apply(extract_road_type, axis='columns')

    # Get the geocodes.
    def calculate_geocode(row):
        house_number = row['house_number']
        road_name = row['road_name']
        road_type = row['road_type']
        suburb = row['locality']
        post_code = row['post_code']
        projection = 'EPSG:4326'
        print(f'Getting geocode for address: ', house_number, road_name, road_type, suburb, post_code, projection)
        lat, long = get_geocode(house_number, road_name, road_type, suburb, post_code, projection)
        return f'{lat},{long}'
    df['geocode'] = df.apply(calculate_geocode, axis='columns')
    df[['lat', 'long']] = df['geocode'].str.split(',', 1, expand=True)
    df['lat'] = df['lat'].astype(float)
    df['long'] = df['long'].astype(float)

    return df


def main():
    house_number = 2
    road_name = 'Hocking'
    road_type = 'Pl'
    suburb = 'ErskinePark'
    post_code = 2759
    projection = 'EPSG:4326'  # The address location's coordinate system in the output. Valid values are 'EPSG:3857' (Web Mercator) or 'EPSG:4326' (Geographic)

    lat, long = get_geocode(house_number, road_name, road_type, suburb, post_code, projection)
    print(lat, long)


if __name__ == '__main__':

    df = extract_property_addresses(r'D:\data\property\ERSKINE PARK_properties.csv')
    df.to_csv(f'D:\data\property\ERSKINE PARK_properties_geocodes.csv', sep=',')

    plot_geocodes(df)

    #main()

