codebraid pandoc -f markdown -t rst --overwrite -o geoffrey_poore.rst poore.txt
python3 patch_rst_conversion.py
