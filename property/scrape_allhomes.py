import requests
from bs4 import BeautifulSoup

def main():
    url = "https://www.allhomes.com.au/ah/research/erskine-park/12733912/sale-history?year=2008"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    #print(page.text)

    table_rows = soup.find_all("tbody", class_="research-table-row")

    for table_row in table_rows:
        # Get the address.
        address_div = table_row.find("div", class_="research-address-link")
        address = address_div.find("a").text

        columns = table_row.find_all("td", class_="research-table-column")
        contract_date = columns[0].text
        transfer_date = columns[1].text
        list_date = columns[2].text
        price = columns[3].text
        block_size = columns[4].text
        transfer_type = columns[5].text

        print(f'Address: {address}')
        print(f'Contract date: {contract_date}, transfer date: {transfer_date}, list date: {list_date}, price: {price}, block_size: {block_size}, transfer type: {transfer_type}.')


if __name__ == '__main__':
    main()
