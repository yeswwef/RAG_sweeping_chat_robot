import os

def get_project_root() -> str:
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    project_dir=os.path.dirname(current_dir)
    return project_dir

def get_abs_path (path:str) -> str:
    project_path=get_project_root()
    return os.path.join(project_path,path)
if __name__ == '__main__':
    print(get_project_root())
    print(get_abs_path("AIChat.py"))
