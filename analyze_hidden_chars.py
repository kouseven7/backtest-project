#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hidden Character Detection - Line 63 and Line 120 Investigation
"""

with open('src/dssms/dssms_integrated_main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

def analyze_line_characters(line_num, line_content):
    """Analyze characters in a specific line"""
    print(f"\n=== Line {line_num} Analysis ===")
    print(f"Raw content: {repr(line_content)}")
    print(f"Length: {len(line_content)}")
    
    # Character analysis
    for i, char in enumerate(line_content):
        if ord(char) > 127 or char in '\t\r\n\xa0':  # Non-ASCII or special
            print(f"Pos {i:2d}: {repr(char)} (U+{ord(char):04X}) - {ord(char)}")

# Analyze Line 63 and Line 120
if len(lines) >= 63:
    analyze_line_characters(63, lines[62])  # Line 63 (0-indexed)
    
if len(lines) >= 120:
    analyze_line_characters(120, lines[119])  # Line 120 (0-indexed)

# Also check surrounding lines for context
print(f"\n=== Context Analysis ===")
print(f"Lines around 63:")
for i in range(max(0, 61), min(len(lines), 66)):
    print(f"Line {i+1}: {repr(lines[i])}")

print(f"\nLines around 120:")
for i in range(max(0, 118), min(len(lines), 123)):
    print(f"Line {i+1}: {repr(lines[i])}")