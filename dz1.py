import requests

def check_id():
    r = requests.get('https://vk.com/80814747')
    print(r.status_code)

check_id()
