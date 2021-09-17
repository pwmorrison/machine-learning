from pathlib import Path
from dataclasses import dataclass
import pandas as pd


"""
Parses the sales price files from the Valuer General.
https://www.valuergeneral.nsw.gov.au/land_values/where_can_you_learn_more_about_your_land_value/property_sales
https://valuation.property.nsw.gov.au/embed/propertySalesInformation

Record type ‘A’: Is a header record and will be the first record in the file. It is to include the file type, district code, date and time of lodgement.
Record type ‘B’: Will contain property address and sales information.
Record type ‘C’: Will contain Property description details.
Record type ‘D’: Owner details suppressed.
Record type ‘Z’: Will be a trailer record and is to be the last record in the file. It is to include a property count and a total record count.
"""


def parse_dat_file(filename):
    lines = open(filename, 'r').readlines()
    property_sales = []
    for line in lines:
        cols = line.split(';')

        if cols[0] == 'B':
            property_sale = {
                'district_code': cols[1],
                'id': cols[2],
                'name': cols[5],
                'unit_number': cols[6],
                'house_number': cols[7],  # house number
                'street_name': cols[8],  # street name
                'locality': cols[9],  # locality
                'post_code': cols[10],  # post code
                'area': cols[11],  # area
                'area_type': cols[12],  # area type. The metric used to measure area (M=square metres, H=hectares) as recorded in the register of Land Values
                'contract_date': cols[13],  # The calander date on which contracts were exchanged as recorded in the Register of Land Values and sourced from the Notice of Sale. Format is CCYYMMDD
                'settlement_date': cols[14],  # The calander date on which a contract was settled as recorded in the Register of Land values. Format is CCYYMMDD
                'purchase_price': cols[15],  # The purchase price of a property as recorded in the register of Land Values.
                'zoning': cols[16],  # The zone classification applied to a property as recorded in the Register of Land Values.
                'nature_of_property': cols[17],  # The nature of property classification applied to a property (V=Vacant, R=Residence, 3=Other) as recorded in the Register of Land Values.
                'primary_purpose': cols[18],  # The main use of a property as recorded in the Register of Land Values. Description supplied when Nature of Property = 3.
                'strata_lot_number': cols[19],  # The strata lot identifier as recorded in the Register of Land Values
                'component_code': cols[20],
                'sale_code': cols[21],
                'percent_interest_of_sale': cols[22],  # The percentage of ownership applied to each party in a sale as recorded in the Register of Land Values. A 0% is displayed in this field if the percentage of share provided in the Notice of Sale is 0%.
                'dealing_number': cols[23],  # A unique identifier applied to a dealing created within the State of New South Wales.
            }
            property_sale = pd.Series(property_sale)
            property_sales.append(property_sale)

    df = pd.DataFrame(columns=property_sales[0].keys())
    for s in property_sales:
        df = df.append(s, ignore_index=True)

    return df



def main():
    year = 2006
    directories = f'D:\data\property\{year}'
    filename_template = '*.DAT'

    filenames = sorted(Path(directories).rglob(filename_template))

    # filenames = filenames[:100]

    print(len(filenames))
    print(filenames)

    dfs = []
    for i, filename in enumerate(filenames):
        print(f'File {i} of {len(filenames)}: {filename}')
        df_file = parse_dat_file(filename)
        dfs.append(df_file)

    df = pd.concat(dfs, axis=0)
    df.to_csv(f'D:\data\property\{year}.csv', sep=',')



if __name__ == '__main__':
    main()
