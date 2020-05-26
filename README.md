# Generate a baubled word cloud

This script fills a specified mask image (a xmas tree by default) with a word cloud, and images of faces as baubles on the tree.

Requirements:

* Python >= 3.6
* https://pypi.org/project/wordcloud/


With [`pipenv`](https://pipenv.kennethreitz.org/en/latest/) installed, you don't need to worry about other requirements, and
can just do:

```sh
pipenv run python baubled_wordcloud.py faces/*.jpg
```
given a directory of images at `faces/`, and the generated output will be in
`out.png`.


Contributors are welcome. See [Issues](https://github.com/Sydney-Informatics-Hub/baubled_wordcloud/issues/).

## Developed by the Sydney Informatics Hub

This tool was developed by the Sydney Informatics Hub, a core research facility of The University of Sydney.

         /  /\        ___          /__/\
        /  /:/_      /  /\         \  \:\
       /  /:/ /\    /  /:/          \__\:\
      /  /:/ /::\  /__/::\      ___ /  /::\
     /__/:/ /:/\:\ \__\/\:\__  /__/\  /:/\:\
    \  \:\/:/~/:/    \  \:\/\ \  \:\/:/__\/
      \  \::/ /:/      \__\::/  \  \::/
       \__\/ /:/       /__/:/    \  \:\
         /__/:/ please \__\/      \  \:\
         \__\/ acknowledge your use\__\/
