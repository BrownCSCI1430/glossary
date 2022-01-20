import json
import sys

glossary_map = {}

curr_word = ''

with open(sys.argv[1], 'r') as f:
    for line in f:
        stripped_line = line.strip()
        if stripped_line.startswith('###'):
            curr_word = stripped_line[4:]
            glossary_map[curr_word] = ''
        elif len(curr_word) > 0:
            # reset current word if new section is found
            if stripped_line.startswith('##'):
                curr_word = ''
            # otherwise add to definition
            else:
                glossary_map[curr_word] += stripped_line

with open('glossary.json', 'w') as outfile:
    outfile.write(json.dumps(glossary_map, indent=4))
