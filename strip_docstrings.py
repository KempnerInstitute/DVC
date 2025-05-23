#!/usr/bin/env python3
import re
import sys

def remove_docstrings(content):
    # Remove triple quoted docstrings
    content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
    content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)
    
    # Remove empty lines
    content = re.sub(r'\n\s*\n+', '\n\n', content)
    
    return content

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python strip_docstrings.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    content = remove_docstrings(content)
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Processed {input_file} -> {output_file}") 