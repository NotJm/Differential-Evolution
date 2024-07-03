import os
import random
import string

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        

def generate_random_filename():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=10))