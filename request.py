import requests

url = 'http://localhost:5000/cancerpredict_api'
r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})

print(r.json())