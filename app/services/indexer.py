import os

ALLOWED_EXTENSIONS = (".py", ".js", ".ts", ".cpp", ".java")

def get_code_files(base_path):
    code_files = []

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(ALLOWED_EXTENSIONS):
                full_path = os.path.join(root, file)
                code_files.append(full_path)

    return code_files