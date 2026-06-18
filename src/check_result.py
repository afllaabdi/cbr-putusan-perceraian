import os

folder = "data/cleaned"

files = os.listdir(folder)

for file in files[:3]:

    path = os.path.join(folder, file)

    with open(path, encoding="utf-8") as f:
        text = f.read()

    print("=" * 100)
    print(file)
    print("=" * 100)

    print(text[:2000])

    print("\n")