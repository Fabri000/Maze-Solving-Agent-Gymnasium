# Sviluppo e  Sperimentazione di Strategie di Reinforcement Learning per la risoluzione di labirinti

## Requisiti
Librerie necessarie:

* Numpy 2.2.2
* Pandas 2.2.3
* Gymnasium 1.0.0
* NetworkX 3.4.2
* Pygame 2.6.1
* Torch 2.4.2
* Torch-directml 0.2.5.dev240914 (se eseguito su GPU AMD)
* Sympy 1.13.3

## Project overview
Progetto di tesi di laurea magistrale in ingegneria informatica incentrato sull'addestramento di agenti tramite tecniche di apprendimento afferenti al reinforcement learning sulla risoluzione di labirinti.
L'obiettivo centrale è stato quello di valutare la capacità di diversi algoritmi di RL di realizzare agenti capaci di risolvere insiemi di labirinti di diversi tipi e di generalizzare la risoluzione di nuovi labirinti.

## Labirinti usati per gli esperimenti
Nella letteratura sono definibili diverse tipologie di labirinti che possono essere distinte sulla base del modo in cui sono connesse le celle tra loro, se sono presenti vicoli ciechi o loop  e il tipo di superficie su cui esso si sviluppa.
In aggiunta a ciò è necessario distinguere labirinti consistenti (ovvero quelli validi) su cui ha senso ricercare una soluzione.
In questo lavoro si sono sfruttati principalmente labirinti con topologia euclidea e toroidale, con tessellazione ortogonale e laddove possibile senza loop. \
Aggiungendo un vincolo di aciclicità sui percorsi nel labirinto si ottiene una importante equivalenza tra l'insieme dei labirinti di dimensione $N\times N$ e degli alberi ricoprendi su $N^2$ nodi.

## Costruzione dei labirinti per esperimenti
L'equivalenza tra labirinto e albero ricoprente rende possibile costruire l'insieme di labirinti tramite algoritmi di esplorazione come *DFS* e *r-prim* oppure più complessi come *prim&kill* che combina la selezione prim con un random walk per la costruzione dei passaggi.
Sfuttando alcune metriche sul labirinto per calcolare la difficoltà si ottengono i seguenti valori su campioni di 1000 labirinti di dimensione $40\times40$.

| **Algorithm**  | **MD**  | **Max D** | **MC**  | **ML**  | **MDE** | **MDs** |
|-----------------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| **DFS**         | 106.41   | 152.22    | 12.24    | 0.47     | 0.05     | 0.03     |
| **R-Prim**      | 71.90    | 103.43    | 8.43     | 0.04     | 1.34     | 0.33     |
| **Prim&kill**   | 99.08    | 157.15    | 10.16    | 0.14     | 0.14     | 0.07     |




## Algoritmi di learning Testati
I principali algoritmi di RL usati sono tutti afferenti alla tipologia Value Based quali:
* Q-learning
* Double Q-learning
* Deep Q network
* Double Deep Q network

## Esperimenti e risultati