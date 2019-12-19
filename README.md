# Generate a baubled word cloud

This script creates an image to a mask (a xmas tree) with a word clound and images of faces as baubles on the tree.

Requirements:

* Python >= 3.6
* https://pypi.org/project/wordcloud/


With `pipenv` installed, you can do:

```sh
pipenv run python baubled_wordcloud.py faces/*.jpg
```
and the generated output will be in `out.png`.
