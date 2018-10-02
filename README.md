# Machine Learning Workshop: Obecný machine learning

Jak otestovat že vše běží tak jak má?
```bash
git clone https://github.com/tomasprinda/ml-workshop-1.git
cd ml-workshop-1
docker build -t ml-workshop-1 .
docker run -v $(pwd):/src -it -p 7777:7777 -p 8888:8888  ml-workshop-1
```
Pokud je vše v pořádku měl by se vyklonovat projekt z GitHubu, vybuildit docker image a spustit docker container. 
`$(pwd)` označuje absolutní cestu k adresáři d projektem.

Nádledně spustíme test pro ověření, že knihovny jsou správně nainstalované a data stažena.
```bash
pytest test.py
```

A vyzkoušíme zda funguje jupyter notebook jeho spuštěním a v otevřením adresy [http://localhost:8888/](http://localhost:8888/)
```bash
jupyter notebook --port=8888
```

Nakonec vyzkoušíme *flexp-browser* na adrese [http://localhost:7777/](http://localhost:7777/)
```bash
flexp-browser --port 7777
```





