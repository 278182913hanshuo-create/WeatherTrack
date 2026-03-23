import requests
import pandas as pd

# Constants
API_KEY = 'YOUR_API_KEY'
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

# Download weather data function
def download_weather_data(city):
    url = f'{BASE_URL}?q={city}&appid={API_KEY}'
    response = requests.get(url)
    data = response.json()
    return data

# Process weather data function

def process_weather_data(data):
    # Extract relevant information
    weather_info = {
        'city': data['name'],
        'temperature': data['main']['temp'],
        'humidity': data['main']['humidity'],
        'description': data['weather'][0]['description'],
        'date': pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
    }
    return weather_info

# Main function
def main():
    cities = ['London', 'New York', 'Tokyo']  # Add more cities as needed
    weather_data_list = []

    for city in cities:
        data = download_weather_data(city)
        processed_data = process_weather_data(data)
        weather_data_list.append(processed_data)

    # Convert to DataFrame
    df = pd.DataFrame(weather_data_list)
    df.to_csv('weather_data.csv', index=False)
    print('Weather data has been downloaded and processed.');

if __name__ == '__main__':
    main()