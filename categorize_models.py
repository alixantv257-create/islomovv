import json

# Fish modellarni o'qish
with open("fish_models.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Agar data dict bo'lsa items'ni olish, aks holda to'g'ridan-to'g'ri ishlatish
if isinstance(data, dict) and "items" in data:
    models = data["items"]
else:
    models = data

# Kategoriyalar
categories = {
    "boy": [],
    "girl": [],
    "man": [],
    "woman": [],
    "old": []
}

# Dublikatlarni oldini olish uchun ID larni saqlash
seen_ids = set()

for model in models:
    tags = model.get("tags", [])
    model_id = model.get("_id")
    title = model.get("title")
    description = model.get("description", "")
    
    # Dublikatni tekshirish
    if model_id in seen_ids:
        continue
    seen_ids.add(model_id)
    
    # Tags va descriptiondan kalit so'zlarni tekshirish
    tags_lower = [tag.lower() for tag in tags]
    desc_lower = description.lower()
    
    # Saralash mantiqi
    if "young" in tags_lower and "male" in tags_lower:
        categories["boy"].append({
            "id": model_id,
            "title": title,
            "tags": tags,
            "description": description
        })
    elif "young" in tags_lower and "female" in tags_lower:
        categories["girl"].append({
            "id": model_id,
            "title": title,
            "tags": tags,
            "description": description
        })
    elif "middle-aged" in tags_lower and "male" in tags_lower:
        categories["man"].append({
            "id": model_id,
            "title": title,
            "tags": tags,
            "description": description
        })
    elif "middle-aged" in tags_lower and "female" in tags_lower:
        categories["woman"].append({
            "id": model_id,
            "title": title,
            "tags": tags,
            "description": description
        })
    elif "old" in tags_lower:
        categories["old"].append({
            "id": model_id,
            "title": title,
            "tags": tags,
            "description": description
        })
    # Agar female bo'lsa lekin young/middle-aged/old bo'lmasa
    elif "female" in tags_lower:
        # Descriptiondan yoshni taxmin qilish
        if "young" in desc_lower or "girl" in desc_lower or "child" in desc_lower or "teen" in desc_lower:
            categories["girl"].append({
                "id": model_id,
                "title": title,
                "tags": tags,
                "description": description
            })
        elif "woman" in desc_lower or "lady" in desc_lower or "mature" in desc_lower:
            categories["woman"].append({
                "id": model_id,
                "title": title,
                "tags": tags,
                "description": description
            })
        else:
            # Default: woman ga qo'shish
            categories["woman"].append({
                "id": model_id,
                "title": title,
                "tags": tags,
                "description": description
            })
    # Agar male bo'lsa lekin young/middle-aged/old bo'lmasa
    elif "male" in tags_lower:
        # Descriptiondan yoshni taxmin qilish
        if "young" in desc_lower or "boy" in desc_lower or "child" in desc_lower or "teen" in desc_lower:
            categories["boy"].append({
                "id": model_id,
                "title": title,
                "tags": tags,
                "description": description
            })
        elif "man" in desc_lower or "gentleman" in desc_lower or "mature" in desc_lower:
            categories["man"].append({
                "id": model_id,
                "title": title,
                "tags": tags,
                "description": description
            })
        else:
            # Default: man ga qo'shish
            categories["man"].append({
                "id": model_id,
                "title": title,
                "tags": tags,
                "description": description
            })

# Natijalarni chiqarish
print("=== Model Saralash Natijalari ===\n")
for category, models_list in categories.items():
    print(f"{category.upper()}: {len(models_list)} ta model")
    for i, model in enumerate(models_list[:5], 1):  # Birinchi 5 tasini ko'rsatish
        print(f"  {i}. ID: {model['id']}")
        print(f"     Title: {model['title']}")
        print(f"     Tags: {model['tags']}")
        print()
    if len(models_list) > 5:
        print(f"  ... va yana {len(models_list) - 5} ta model")
    print("-" * 50)

# .env formatida saqlash
env_lines = []
for category, models_list in categories.items():
    if models_list:
        ids = [m["id"] for m in models_list[:10]]  # Har kategoriyadan 10 ta
        env_line = f"FISH_TTS_{category.upper()}_MODELS={','.join(ids)}"
        env_lines.append(env_line)

print("\n=== .env uchun ===")
for line in env_lines:
    print(line)

# JSON formatida saqlash
with open("fish_models_categorized.json", "w", encoding="utf-8") as f:
    json.dump(categories, f, indent=2, ensure_ascii=False)

print("\nModellar fish_models_categorized.json fayliga saqlandi")
