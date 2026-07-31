import os
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from utils.logger_handler import logger
from langchain_community.document_loaders import TextLoader



#获取文件的MD5的十六进制字符串
def get_file_md5(filepath: str):
    if not os.path.exists(filepath):
        logger.error("文件不存在")
        return None
    if not os.path.isfile(filepath):
        logger.error("这不是一个文件")
        return None
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] {path}不是文件夹")
        return allowed_types
    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))
    return tuple(files)

def pdf_loader(filepath :str , password=None) -> list[Document]:
    return PyPDFLoader(filepath,password).load()

def txt_loader(filepath :str ) -> list[Document]:
    return TextLoader(filepath,encoding='utf-8').load()
