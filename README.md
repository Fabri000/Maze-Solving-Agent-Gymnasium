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
| **R-Prim**      | 71.90 | 103.43 | 8.43 | 0.04 | 1.34 | 0.33 |
| **Prim&kill**   | 99.08 | **157.15** | 10.16 | 0.14 | 0.14 | 0.07 |
| **DFS**         | **106.41** | 152.22 | **12.24** | **0.47** | **0.05** | **0.03** |


Da questi risultati si possono osservare che, mediamente, i labirinti generati presentano delle strutture caratteristiche che si differenziano cambiando algoritmo di generazione.
Per questo motivo si sono provati ad utilizzare insiemi di labirinti in cui essi sono stati generati usando un solo algoritmo o molteplici.

## Algoritmi di learning Testati
I principali algoritmi di RL usati sono tutti afferenti alla tipologia Value Based quali:
* Q-learning
* Double Q-learning
* Deep Q network
* Double Deep Q network

## Esperimenti e risultati
Gli esperimenti sono stati principalmente effettuati su tre livelli di robustezza relativi all'insieme di labirinti usati:
* Singolo labirinto
* Labirinti di dimensione costante
* Labirinti di dimensione variabile

Per gli ultimi due esperimenti si è inoltre provato a sperimentare su insiemi di labirinti con topologia toroidale o euclidea generati usando o un solo algoritmo o molteplici.
Nella tabella di seguito si riportano i risultati su un singolo labirinto i quali sono abbastanza triviali.

| **Algoritmo**  | **Tempo**  | **Catastrophic Forget** | **MC**  |
|-----------------|:----------:|:----------:|:----------:|
| **Q-learning** | 120s| No | 0% |
| **Double Q-learning** |112s| No | 0% |
| **DQN** | 1020s | No | 4% |
| **DDQN** | 1620s | No | 14% |

Per questo motivo ci si è concentrati su gli altri due tipi di esperimenti, prima su insiemi di labirinti di dimensione costante generati usando l'algoritmo r-prim

| **Algoritmo**  | **Tempo**  | **W/R labirinti esplorati** | **W/R nuovi labirinti**  |
|-----------------|:----------:|:----------:|:----------:|
| **Q-learning** | 300s | 80.49% | 0% |
| **Double Q-learning** |510s| 80% | 0% |
| **DQN** | 600s | 98.67% | 96% |
| **DDQN** | 1200s | 100% | 99.6% |

e successivamente su insiemi di labirinti generati usando diversi algoritmi di costruzione

| **Algoritmo**  | **Tempo**  | **W/R labirinti esplorati** | **W/R nuovi labirinti**  |
|-----------------|:----------:|:----------:|:----------:|
| **Q-learning** | 840s| 70.83% | 0% |
| **Double Q-learning** |400s| 22.5% | 0% |
| **DQN** | 9600s | 18.18% | 13.33% |
| **DDQN** | 12000s | 16.6% | 1.32% |

Da questi primi risultati si osserva come DQN e DDQN, rispetto alle controparti tabellari, garantiscano migliori performanze in termini di generalizzazione a (quasi) parità di capacità di risolvere i labirinti già esplorati.
In aggiunta a ciò si osserva che su insiemi di labirinti con strutture caratteristiche differenti si hanno tempi più lunghi di addestramento e performance più basse.\
Successivamente si è sperimentato su labirinti di dimensione variabile prima generati usando l'algoritmo r-prim

| **Algoritmo**  | **Tempo**  | **W/R labirinti esplorati** | **W/R nuovi labirinti**  |
|-----------------|:----------:|:----------:|:----------:|
| **Q-learning** | 430s | 78.57% | 0% |
| **Double Q-learning** | 360s | 71.43% | 0% |
| **DQN** | 340s | 100% | 86.67% |
| **DDQN** | 309s | 100% | 98.67% |

e si è osservata una riduzione dei tempi di addestramento degli approcci neurali grazie all'introduzione di una regola di early stop sulla dimensione massima del labirinto raggiounta in fase di addestramento.
Conseguentemente si è esperimentato su labirinti di dimensione variabile generati usando diversi approcci di costruzione

| **Algoritmo**  | **Tempo**  | **W/R labirinti esplorati** | **W/R nuovi labirinti**  |
|-----------------|:----------:|:----------:|:----------:|
| **Q-learning** | 600s| 85.71% | 0% |
| **Double Q-learning** |586s| 36.36% | 0% |
| **DQN** | 25000s | 33% | 0% |
| **DDQN** | 40000s | 8.33% | 1.32% |

e per cui so osserva lo stesso fenomeno visto nel caso precedente.
Infine, per brevità si è sperimentato su labirinti toroidali di dimensione variabile o costante, ma generati tutti usando l'algoritmo r-prim.\
Prima su insieme di labirinti toroidali di dimensione costante

| **Algoritmo**  | **Tempo**  | **W/R labirinti esplorati** | **W/R nuovi labirinti**  |
|-----------------|:----------:|:----------:|:----------:|
| **Q-learning** | 115s | 78.57% | 0% |
| **Double Q-learning** | 180s | 58% | 0% |
| **DQN** | 1020s | 93.33% | 1.3% |
| **DDQN** | 780s | 81.22% | 16% |

e successivamente su insieme di labirinti toroidali di dimensione variabile

| **Algoritmo**  | **Tempo**  | **W/R labirinti esplorati** | **W/R nuovi labirinti**  |
|-----------------|:----------:|:----------:|:----------:|
| **Q-learning** | 110s | 71.43% | 0% |
| **Double Q-learning** | 120s | 50% | 0% |
| **DQN** | 1020s | 64.29% | 0% |
| **DDQN** | 1320s | 81.2% | 1.42% |

Da questi esperimenti si osserva come la perdita di aciclicità nei labirinti rappresenta un livello di sfida maggiore sia in termini di capacità di risoluzione di labirinti già esplorati, ma soprattutto in termini di generalizzazione a nuovi labirinti.