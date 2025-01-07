import os

class Logger:
    def __init__(self, file_path):
        print()

        if not os.path.exists("/".join(file_path.split("/")[:-1])):
            os.mkdir("/".join(file_path.split("/")[:-1]))
            
        self.file_path = file_path

    def __call__(self, message):
        with open(self.file_path, "a") as f:
            f.write(f"{message}\n")
            print(message)


if __name__ == "__main__":
    logger = Logger("./logs/log.txt") 
    logger("Hello, World!")