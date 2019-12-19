# Generate a baubled word cloud

This script creates an image to a mask (a xmas tree) with a word clound and images of faces as baubles on the tree.

Requirements:

* Python >= 3.6
* https://pypi.org/project/wordcloud/


With `pipenv` installed, you don't need to worry about other requirements, and
can just do:

```sh
pipenv run python baubled_wordcloud.py faces/*.jpg
```
given a directory of images at `faces/`, and the generated output will be in
`out.png`.


Contributors are welcome. See [Issues](https://github.sydney.edu.au/informatics/baubled_wordcloud/issues/).
