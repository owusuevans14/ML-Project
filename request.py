import requests

url = 'http://127.0.0.1:5000/predict_api'
r = requests.post(url,json={'date':450, 'HOSPITALIZED_COUNT':30000, 'DEATH_COUNT':1000, 'DEATH_COUNT_7DAY_AVG': 2000})

print('TOTAL CASE COUNT WOULD BE {}'.format(r.text))