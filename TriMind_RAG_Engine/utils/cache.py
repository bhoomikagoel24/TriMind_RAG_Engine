from typing import Dict

class SimpleCache:

    def __init__(self):
        self.storage: Dict[str, dict] = {}

    def get(self, key: str):
        return self.storage.get(key)

    def set(self, key: str, value: dict):
        self.storage[key] = value

    def clear(self):
        self.storage.clear()