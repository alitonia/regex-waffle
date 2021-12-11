# Intro

# How to run

* Go to _/server_
* (Optional): `virtualenv -p python3 sample && source sample/bin/active`
* Install `pip install -r requirements.txt`
* `python main.py`
* Visit __localhost:8000__.

* Put log in input
* Submit, retrieve filename if valid (ex: file_(.*)\.txt)
* in __processing_mirror.py__, replace `FILE_NAME` with your file name
* `python processing_mirror.py`, and wait

## Note

* Can increase accuracy by changing population size, generation time, fitness function, but beware of running time

## Credit

Based on implementation of Regex Golf game using Genetic Programming based
on [this paper](https://www.human-competitive.org/sites/default/files/bartoli-paper.pdf).

## Developers

[alitonia](https;//github.com/alitonia) <br/>
[Andjela Ilic 105/2017](https://github.com/ilicandjela) <br/>
[Mina Milosevic 81/2017](https://github.com/sardothien)
