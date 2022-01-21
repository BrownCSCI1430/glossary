import json
import sys
import re

glossary_map = {}

curr_word = ''

def slug(word):
    return re.sub(r'\s', '-', re.sub(r'[\u2000-\u206F\u2E00-\u2E7F\'\\!"#$%&()*+,./:;<=>?@[\]^`{|}~]', '', re.sub(r'<[!\/a-z].*?>', '', word)))

with open(sys.argv[1], 'r') as f:
    for line in f:
        stripped_line = line.strip()
        if stripped_line.startswith('---'):
            curr_word = ''
        if stripped_line.startswith('###'):
            curr_word = slug(stripped_line[4:]).lower()
            glossary_map[curr_word] = ''
        elif len(curr_word) > 0:
            # reset current word if new section is found
            if stripped_line.startswith('##'):
                curr_word = ''
            # otherwise add to definition
            else:
                glossary_map[curr_word] += stripped_line + '\n'

with open('glossary.json', 'w') as outfile:
    outfile.write(json.dumps(glossary_map, indent=4))
