import json
import os


FILE = "session_memory.json"


class SessionMemory:
    def __init__(self):
        self.memory = []
        self.load()

    def add(self, text: str):
        if text and text not in self.memory:
            self.memory.append(text)
            self.save()

    def get(self):
        return self.memory[-5:]

    def clear(self):
        self.memory = []
        self.save()

    def save(self):
        with open(FILE, "w") as f:
            json.dump(self.memory, f)

    def load(self):
        if os.path.exists(FILE):
            with open(FILE, "r") as f:
                self.memory = json.load(f)


session_memory = SessionMemory()