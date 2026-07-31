import os
from utils.path_tool import get_abs_path
import yaml

def load_rag_config(config_path:str=get_abs_path("config/rag.yml")):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
def load_chroma_config(config_path:str=get_abs_path("config/chroma.yml")):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
def load_prompts_config(config_path:str=get_abs_path("config/prompts.yml")):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
def load_agent_config(config_path:str=get_abs_path("config/agent.yml")):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

rag_config=load_rag_config()
chroma_config=load_chroma_config()
prompts_config=load_prompts_config()
agent_config=load_agent_config()

if __name__ == '__main__':
    print(agent_config["qwe"])
