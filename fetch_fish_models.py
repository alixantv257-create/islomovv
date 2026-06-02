import urllib.request
import urllib.error
import json

API_KEY = "4d39eba700c74932b51e07a0e7512cd4"
BASE_URL = "https://api.fish.audio"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Pagination bilan ko'proq model olish
all_models = []
limit = 100
offset = 0

print("Modellar yuklanmoqda...")

# Birinchi navbatda barcha modellarni olish
while True:
    url = f"{BASE_URL}/model?limit={limit}&offset={offset}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get("items", [])
            all_models.extend(items)
            print(f"{len(all_models)} ta model yuklandi...")
            
            if not items or len(items) < limit:
                break
            offset += limit
    except urllib.error.HTTPError as e:
        print(f"Xatolik: {e.code}")
        print(e.read().decode('utf-8'))
        break

# Endi female modellarini qidirish
print("\nFemale modellar qidirilmoqda...")
offset = 0
while offset < 500:  # Maksimum 500 ta female model
    url = f"{BASE_URL}/model?limit={limit}&offset={offset}&tags=female"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get("items", [])
            all_models.extend(items)
            print(f"Female: {len(items)} ta model qo'shildi (jami: {len(all_models)})")
            
            if not items or len(items) < limit:
                break
            offset += limit
    except urllib.error.HTTPError as e:
        print(f"Xatolik: {e.code}")
        print(e.read().decode('utf-8'))
        break

# Woman modellarini qidirish (middle-aged female)
print("\nWoman (middle-aged female) modellar qidirilmoqda...")
offset = 0
while offset < 500:
    url = f"{BASE_URL}/model?limit={limit}&offset={offset}&tags=middle-aged,female"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get("items", [])
            all_models.extend(items)
            print(f"Woman: {len(items)} ta model qo'shildi (jami: {len(all_models)})")
            
            if not items or len(items) < limit:
                break
            offset += limit
    except urllib.error.HTTPError as e:
        print(f"Xatolik: {e.code}")
        print(e.read().decode('utf-8'))
        break

# Young male (boy) modellarini qidirish
print("\nYoung male (boy) modellar qidirilmoqda...")
offset = 0
while offset < 500:
    url = f"{BASE_URL}/model?limit={limit}&offset={offset}&tags=young,male"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get("items", [])
            all_models.extend(items)
            print(f"Boy: {len(items)} ta model qo'shildi (jami: {len(all_models)})")
            
            if not items or len(items) < limit:
                break
            offset += limit
    except urllib.error.HTTPError as e:
        print(f"Xatolik: {e.code}")
        print(e.read().decode('utf-8'))
        break

# Young female (girl) modellarini qidirish
print("\nYoung female (girl) modellar qidirilmoqda...")
offset = 0
while offset < 500:
    url = f"{BASE_URL}/model?limit={limit}&offset={offset}&tags=young,female"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get("items", [])
            all_models.extend(items)
            print(f"Girl: {len(items)} ta model qo'shildi (jami: {len(all_models)})")
            
            if not items or len(items) < limit:
                break
            offset += limit
    except urllib.error.HTTPError as e:
        print(f"Xatolik: {e.code}")
        print(e.read().decode('utf-8'))
        break

# Boshqa usul: language parametri bilan qidirish
print("\nEnglish language modellar qidirilmoqda...")
offset = 0
while offset < 200:
    url = f"{BASE_URL}/model?limit={limit}&offset={offset}&language=en"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get("items", [])
            all_models.extend(items)
            print(f"English: {len(items)} ta model qo'shildi (jami: {len(all_models)})")
            
            if not items or len(items) < limit:
                break
            offset += limit
    except urllib.error.HTTPError as e:
        print(f"Xatolik: {e.code}")
        print(e.read().decode('utf-8'))
        break

# Search parametri bilan qidirish
print("\n'young' so'zi bilan qidirish...")
offset = 0
while offset < 200:
    url = f"{BASE_URL}/model?limit={limit}&offset={offset}&search=young"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get("items", [])
            all_models.extend(items)
            print(f"Young search: {len(items)} ta model qo'shildi (jami: {len(all_models)})")
            
            if not items or len(items) < limit:
                break
            offset += limit
    except urllib.error.HTTPError as e:
        print(f"Xatolik: {e.code}")
        print(e.read().decode('utf-8'))
        break

print(f"\nJami topilgan modellar soni: {len(all_models)}")

# JSON faylga saqlash
with open("fish_models.json", "w", encoding="utf-8") as f:
    json.dump(all_models, f, indent=2, ensure_ascii=False)
print("\nModellar fish_models.json fayliga saqlandi")
