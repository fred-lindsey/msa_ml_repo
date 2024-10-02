
import pandas as pd
import requests
from bs4 import BeautifulSoup

companies = ["AAPL","NVDA","MSFT","PFE","GSK","MRNA","F","GM","TSLA","PYPL","JPM","SQ"]
years = [2021,2022,2023,2024]

# Create a base URL of the page you want to scrape
base_url = "https://www.roic.ai/quote/{company}/transcripts/{year}-year/{quarter}-quarter"

# Make a list of years and quarters to scrape from
companies = ["AAPL","NVDA","MSFT","PFE","GSK","MRNA","F","GM","TSLA","PYPL","JPM","SQ"]
years = [2024, 2023, 2022, 2021]
quarters = [3, 2]

# Add headers to bypass the browser request (acknowledges it is not a bot)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

# Create an empty dictionary that includes the transcript for each quarter and year
transcripts_by_company = {}

# can choose between tech, pharma, cars, or financial 
for company in companies:
    transcripts_by_quarter = {}
    # loop through the years and pull Q3, unless its 2024, then pull Q2
    for year in [2021,2022,2023]:
        quarter = 3

        # Create URL for the iterated year and quarter
        url = base_url.format(company=company, year=year, quarter=quarter)

        # Send a GET request with headers to input the data
        response = requests.get(url, headers=headers)

        # Check if the request was successful just to make sure it is working
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Create an empty list to put all the text into
            all_transcripts = []

            # List all the classes to retrieve the text
            class_list = [
                'relative max-w-xl rounded-xl rounded-tl-none bg-muted px-4 py-2 leading-normal shadow',
                'relative max-w-xl rounded-xl rounded-tr-none border px-4 py-2 leading-normal shadow'
            ]

            # Create a for loop to find every iteration of the 'div' elements -- keeps convo in order
            for div in soup.find_all('div'):
                # Find the class element and make it a string
                div_class = ' '.join(div.get('class', []))

                # Find all the 'div' elements that are in the listed classes in class_list
                if div_class in class_list:
                    # Finds the text in the class and adds it to the all_transcripts list
                    all_transcripts.append(div.get_text(separator='\n\n'))

                # Check if the class equals 'flex justify-start' and if the data-cy attribute equals 'transcripts_call_message'
                # This indicates the beginning of a new speaker
                elif div_class == 'flex justify-start' and div.get('data-cy') == 'transcripts_call_message':
                    # Adds two new lines to separate speakers
                    all_transcripts.append('\n\n')

            # Join the list into a single string
            conversation_transcript = "\n".join(all_transcripts)
            transcripts_by_quarter[f'{year}_Q{quarter}'] = conversation_transcript  
            
    for year in [2024]:
        quarter = 2

        # Create URL for the iterated year and quarter
        url = base_url.format(company=company, year=year, quarter=quarter)

        # Send a GET request with headers to input the data
        response = requests.get(url, headers=headers)

        # Check if the request was successful just to make sure it is working
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Create an empty list to put all the text into
            all_transcripts = []

            # List all the classes to retrieve the text
            class_list = [
                'relative max-w-xl rounded-xl rounded-tl-none bg-muted px-4 py-2 leading-normal shadow',
                'relative max-w-xl rounded-xl rounded-tr-none border px-4 py-2 leading-normal shadow'
            ]

            # Create a for loop to find every iteration of the 'div' elements -- keeps convo in order
            for div in soup.find_all('div'):
                # Find the class element and make it a string
                div_class = ' '.join(div.get('class', []))

                # Find all the 'div' elements that are in the listed classes in class_list
                if div_class in class_list:
                    # Finds the text in the class and adds it to the all_transcripts list
                    all_transcripts.append(div.get_text(separator='\n\n'))

                # Check if the class equals 'flex justify-start' and if the data-cy attribute equals 'transcripts_call_message'
                # This indicates the beginning of a new speaker
                elif div_class == 'flex justify-start' and div.get('data-cy') == 'transcripts_call_message':
                    # Adds two new lines to separate speakers
                    all_transcripts.append('\n\n')

            # Join the list into a single string
            conversation_transcript = "\n".join(all_transcripts)
            transcripts_by_quarter[f'{year}_Q{quarter}'] = conversation_transcript
    

        # If the data is not scraped right, then it will spit out this error
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

    transcripts_by_company[f'{company}'] = transcripts_by_quarter

# Print sample transcript
print(transcripts_by_company)

# Create DataFrame
earnings_df2 = pd.DataFrame(transcripts_by_company)
earnings_df2.reset_index(inplace=True)
earnings_df2.rename(columns={'index':'earnings_date'}, inplace=True)
print(earnings_df2.head())

#Create CSV
earnings_df2.to_csv('earnings_df.csv')

# Check that it worked
final_earnings_df = pd.read_csv('earnings_df.csv')
print(final_earnings_df.head())