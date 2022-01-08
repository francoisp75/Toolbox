# pylint: disable=missing-module-docstring

import sys
import urllib.parse
import requests
from requests.models import Response

BASE_URI = "https://www.metaweather.com/"


def search_city(query):
    '''Look for a given city and disambiguate between several candidates. Return one city (or None)'''
    url = urllib.parse.urljoin(BASE_URI, "/api/location/search")
    params = {'query': query}
    response = requests.get(url, params=params).json()

    if not response:
        print("please enter a name")
        return None
    else:
        return response[0]


def weather_forecast(woeid):
    '''Return a 5-element list of weather forecast for a given woeid'''

    url = f"{BASE_URI}api/location/{woeid}"

    #METHODE GET SANS PARAm
    #response = requests.get(url).json()
    return requests.get(url).json()['consolidated_weather']
    #print(response['woeid'])
    #return response['woeid']
    #Faire 5 lignes/
    #return (response[0])


def main():
    '''Ask user for a city and display weather forecast'''
    query = input("City?\n> ")
    city = search_city(query)
    # TODO: Display weather forecast for a given city
    #forecast = weather_forecast()
    if city:
        daily_forecasts = weather_forecast(city['woeid'])
        for forecast in daily_forecasts:
            max_temp = round(forecast['max_temp'])
            print(
                f"{forecast['applicable_date']}: {forecast['weather_state_name']} ({max_temp}°C)"
            )


if __name__ == '__main__':
    try:
        while True:
            main()
    except KeyboardInterrupt:
        print('\nGoodbye!')
        sys.exit(0)
