# Testing

import requests
import json

payload1 = {'load': 12000, 'fuels': {'gas(euro/MWh)': 13.4, 'kerosine(euro/MWh)': 50.8, 'co2(euro/ton)': 20, 'wind(%)': 60}, 'powerplants': [{'name': 'gasfiredbig1', 'type': 'gasfired', 'efficiency': 0.53, 'pmin': 100, 'pmax': 460}, {'name': 'gasfiredbig2', 'type': 'gasfired', 'efficiency': 0.53, 'pmin': 100, 'pmax': 460}, {'name': 'gasfiredsomewhatsmaller', 'type': 'gasfired', 'efficiency': 0.37, 'pmin': 40, 'pmax': 210}, {'name': 'tj1', 'type': 'turbojet', 'efficiency': 0.3, 'pmin': 0, 'pmax': 16}, {'name': 'windpark1', 'type': 'windturbine', 'efficiency': 1, 'pmin': 0, 'pmax': 150}, {'name': 'windpark2', 'type': 'windturbine', 'efficiency': 1, 'pmin': 0, 'pmax': 36}]}
payload2 = {'load': 480, 'fuels': {'gas(euro/MWh)': 13.4, 'kerosine(euro/MWh)': 50.8, 'co2(euro/ton)': 20, 'wind(%)': 0}, 'powerplants': [{'name': 'gasfiredbig1', 'type': 'gasfired', 'efficiency': 0.53, 'pmin': 100, 'pmax': 460}, {'name': 'gasfiredbig2', 'type': 'gasfired', 'efficiency': 0.53, 'pmin': 100, 'pmax': 460}, {'name': 'gasfiredsomewhatsmaller', 'type': 'gasfired', 'efficiency': 0.37, 'pmin': 40, 'pmax': 210}, {'name': 'tj1', 'type': 'turbojet', 'efficiency': 0.3, 'pmin': 0, 'pmax': 16}, {'name': 'windpark1', 'type': 'windturbine', 'efficiency': 1, 'pmin': 0, 'pmax': 150}, {'name': 'windpark2', 'type': 'windturbine', 'efficiency': 1, 'pmin': 0, 'pmax': 36}]}
payload3 = {'load': 910, 'fuels': {'gas(euro/MWh)': 13.4, 'kerosine(euro/MWh)': 50.8, 'co2(euro/ton)': 20, 'wind(%)': 60}, 'powerplants': [{'name': 'gasfiredbig1', 'type': 'gasfired', 'efficiency': 0.53, 'pmin': 100, 'pmax': 460}, {'name': 'gasfiredbig2', 'type': 'gasfired', 'efficiency': 0.53, 'pmin': 100, 'pmax': 460}, {'name': 'gasfiredsomewhatsmaller', 'type': 'gasfired', 'efficiency': 0.37, 'pmin': 40, 'pmax': 210}, {'name': 'tj1', 'type': 'turbojet', 'efficiency': 0.3, 'pmin': 0, 'pmax': 16}, {'name': 'windpark1', 'type': 'windturbine', 'efficiency': 1, 'pmin': 0, 'pmax': 150}, {'name': 'windpark2', 'type': 'windturbine', 'efficiency': 1, 'pmin': 0, 'pmax': 36}]}


# post request - change the payload number to use one of the payload above (payload1, payload2, payload3)
res = requests.post('http://localhost:5000/productionplan', json=payload1)
if res.ok:
    print(res.json())
