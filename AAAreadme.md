<h1> AAAreadme file </h1>

This markdown file is written in Github markdown.

When editing this file in Vim, the vim-markdown-preview plugin will display the markdown
file inside of Safari. It will update the browser's view automatically when you
save this file. 

Do not forget to source the scipy21 conda env before running `./update.sh`.

```bash
conda activate scipy21
```

After making some edits of the manuscript file, compile it with the `./update.sh` script.
Preview will display the resulting PDF.
If Preview fails, your compile job failed.

After making some edits the manuscript that compile well, save your changes to 
the SciPy21 GitHub repo as follows.

```bash
mate ./papers/blaine_mooers/mooersblaine.rst 
git add ./papers/blaine_mooers/mooersblaine.rst 
git commit -m "Added section on using Phenix" 
git push origin 2021
```

You really only need to enter the following:

```bash
gac ./papers/blaine_mooers/mooersblaine.rst "Updated the text."
git push 
```

I should write an alias for this:

```bash
alias gup='gac ./papers/blaine_mooers/mooersblaine.rst "Updated the text." && git push'
```

The comment symbol for reStructuredText is (.. ).

SublimeText3 recognizes it.
It will it when text is selected when using the command-forward slash keybinding.

The make_paper.sh script compiles the rst document, which is in a subsubfolder.
But, you really only need to run ./update.sh which in turn runs make_paper.sh and
then opens the output PDF.



