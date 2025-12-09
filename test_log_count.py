import re

with open('task11_backtest.log', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

print(f'File size: {len(content)} chars')
print(f'DSSMS_FORCE_CLOSE_START count: {len(re.findall(r"DSSMS_FORCE_CLOSE_START", content))}')
print(f'DSSMS_FORCE_CLOSE_SUPPRESS count: {len(re.findall(r"DSSMS_FORCE_CLOSE_SUPPRESS", content))}')
print(f'DSSMS_FORCE_CLOSE_END count: {len(re.findall(r"DSSMS_FORCE_CLOSE_END", content))}')
print(f'FORCE_CLOSE_START count: {len(re.findall(r"\\[FORCE_CLOSE_START\\]", content))}')
