'''
Patch the geoffrey_poore.rst that is generated from poore.md with pandoc

```
codebraid pandoc -f markdown -t rst --overwrite -o geoffrey_poore.rst poore.txt
```
'''

import pathlib
import re

p = pathlib.Path('geoffrey_poore.rst')
text = p.read_text(encoding='utf8')
# Replace languages not supported by default docutils
text = re.sub('code:: (?:stdout|stderr|markdown|sourceError)', 'code:: text', text)
# Remove unsupported line numbering inserted by Pandoc
text = re.sub('   :number-lines:.*\n', '', text)
p.write_text(text, encoding='utf8')
