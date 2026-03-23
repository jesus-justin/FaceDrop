#!/usr/bin/env python3
import os
import urllib.request
import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context

output_dir = os.path.expanduser('~/.insightface/models')
output_file = os.path.join(output_dir, 'inswapper_128.onnx')

os.makedirs(output_dir, exist_ok=True)

# Try multiple sources
sources = [
    ('https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx', 'HuggingFace'),
    ('https://civitai.com/api/download/models/89146', 'Civitai'),
    ('https://modelscope.cn/models/deepinsight/inswapper/resolve/main/inswapper_128.onnx', 'ModelScope'),
    ('https://github.com/facefusion/facefusion-assets/raw/refs/heads/main/models/inswapper_128.onnx', 'GitHub'),
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

for url, source_name in sources:
    try:
        print(f'[Download] Trying {source_name}: {url}')
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            file_size = response.headers.get('Content-Length')
            print(f'[Download] Got response, size: {file_size} bytes')
            
            # Download with progress
            with open(output_file, 'wb') as out_f:
                chunk_size = 8192
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if int(file_size or 0) > 0:
                        percent = (downloaded / int(file_size)) * 100
                        print(f'\r[Download] Progress: {percent:.1f}% ({downloaded}/{file_size} bytes)', end='', flush=True)
            
            print(f'\n[Download] Success! Downloaded to {output_file}')
            print(f'[Download] Final size: {os.path.getsize(output_file)} bytes')
            sys.exit(0)
            
    except Exception as e:
        print(f'[Download] Failed: {e}')
        if os.path.exists(output_file):
            os.remove(output_file)

print('[Download] All download attempts failed. The model will need to be downloaded manually.')
print(f'[Download] Place the inswapper_128.onnx file at: {output_file}')
