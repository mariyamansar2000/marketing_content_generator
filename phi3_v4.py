%pip install onnx
%pip install onnxruntime
%pip install onnxruntime_genai
%pip install transformers
%pip install openpyxl
%pip install onnx2pytorch

%restart_python or dbutils.library.restartPython()

import onnxruntime_genai as og
import pandas as pd
import gc
import datetime
import numpy as np
import os

storage_account_name = "inputmodels"
storage_account_key = "give the account key"
container_name = "input"
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_key)


st=datetime.datetime.now()
container_name = "input"
storage_account_name = "inputmodels"
phi_3_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/phi3/"
# Define local directory for model files
local_model_dir = "/tmp/phi-3_model/"
dbutils.fs.mkdirs(local_model_dir)
# Download all files in the phi-3 folder to the local directory
files = dbutils.fs.ls(phi_3_path)
for file in files:
   dbutils.fs.cp(file.path, f"file:{local_model_dir}/{file.name}")
# Define model path (assuming ONNX file is in the downloaded directory)
__model_path = f"{local_model_dir}"
model = og.Model(__model_path)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
# Set search options
search_options = {'max_length': 3000, 'batch_size': 1}

# Prepare the input text
def format_prompt(platform,audience,topic,word_limit,product_desc,offers,other_info):
    prompt = f"""
    Generate a {word_limit}-word promotional post for {platform}.
    - Target audience: {audience}
    - Topic: {topic}
    - Product description: {product_desc}
    - Offers: {offers}
    - Additional information: {other_info}

    The post should be engaging and relevant to the audience.
    """
    return prompt
# def generate_content(platform,audience,topic,word_limit,product_desc,offers,other_info):
#     prompt = format_prompt(platform,audience,topic,word_limit,product_desc,offers,other_info)

#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=word_limit)
#     response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     return response_text

#manual prompt
platform = "LinkedIn"
audience = "IT professionals"
topic = "AI"
word_limit = 200
product_desc = "AI-powered chatbot"
offers = "Get 30% off on first purchase"
other_info = "No other information"

prompt = format_prompt(platform,audience,topic,word_limit,product_desc,offers,other_info)
input_tokens = tokenizer.encode(prompt)

# Initialize the generator
params = og.GeneratorParams(model)
params.set_search_options(**search_options)
generator = og.Generator(model, params)

# Generate the output
print("Output: ", end='', flush=True)
try:
  generator.append_tokens(input_tokens)
  while not generator.is_done():
    generator.generate_next_token()
    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)
except KeyboardInterrupt:
  print(" --control+c pressed, aborting generation--")

# print()
del generator

