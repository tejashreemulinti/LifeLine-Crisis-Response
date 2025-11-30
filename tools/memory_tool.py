import json, os

class MemoryBank:
    def __init__(self, path="memory.json"):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump([], f)

    def add(self, item):
        data = json.load(open(self.path))
        data.append(item)
        json.dump(data, open(self.path, "w"), indent=4)

    def get_all(self):
        return json.load(open(self.path))
