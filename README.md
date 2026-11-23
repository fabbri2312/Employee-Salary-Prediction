# Employee-Salary-Prediction

## Descrizione

Questo progetto è stato realizzato come progetto universitario con l'obiettivo di confrontare un modello di regressione supervisionata e una rete neurale feedforward per stimare lo stipendio annuale di un dipendente a partire da caratteristiche anagrafiche, professionali e lavorative.

Il modello utilizza informazioni come gli anni di esperienza, il titolo di studio, il ruolo lavorativo, la fascia della città, la valutazione delle prestazioni, il numero di competenze e il lavoro da remoto. La variabile da prevedere è `annual_salary_usd`.

## Struttura del progetto

```text
.
├── progetto.ipynb
├── requirements.txt
└── data/
    └── employee_salary_regression.csv
```

- `progetto.ipynb` contiene l'analisi esplorativa, il preprocessing, l'addestramento e la valutazione dei modelli.
- `data/employee_salary_regression.csv` contiene 1.000 osservazioni e 10 colonne.
- `requirements.txt` contiene le dipendenze Python necessarie.

## Dataset

Il dataset comprende 1.000 righe senza valori nulli. Le colonne disponibili sono:

- `employee_id`
- `age`
- `years_experience`
- `education_level`
- `job_role`
- `city_tier`
- `performance_score`
- `num_skills`
- `remote_work`
- `annual_salary_usd`

`employee_id` viene escluso perché è un identificativo e non una caratteristica predittiva. Dopo l'analisi esplorativa, anche `age` viene rimossa perché fortemente correlata con `years_experience`, evitando ridondanza tra le variabili.

## Modelli utilizzati

### 1. Regressione lineare

Il modello di regressione lineare utilizza:

- standardizzazione delle variabili numeriche;
- codifica One-Hot delle variabili categoriche;
- divisione dei dati in 67% training set e 33% test set;
- seme casuale `random_state=42`;
- valutazione con MAE, MSE, RMSE e R².

Il preprocessing viene eseguito dopo la divisione dei dati: `StandardScaler` e `OneHotEncoder` vengono addestrati esclusivamente sul training set e successivamente applicati al test set. Il target viene scalato allo stesso modo con uno `StandardScaler` separato, sempre addestrato esclusivamente sui valori di training.

### 2. Rete neurale feedforward

La rete neurale implementata in PyTorch utilizza:

- livelli fully connected con architettura `64 -> 32 -> 1`;
- attivazione ReLU;
- ottimizzatore Adam;
- funzione di perdita MSE;
- 200 epoche di addestramento.

Per rendere l'esperimento riproducibile, vengono impostati lo stesso seed `42` per Python, NumPy, Scikit-learn e PyTorch. Quando disponibile, il seed viene configurato anche per tutte le GPU CUDA.

## Risultati dell'esecuzione

Dal notebook già eseguito sono stati ottenuti i seguenti risultati sul test set:

| Modello | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Regressione lineare | $3.490,76 | $19.243.274,19 | $4.386,72 | 0,9853 |
| Rete neurale | $3.955,69 | $24.436.800,00 | $4.943,36 | 0,9813 |

Durante l'addestramento della rete neurale, il valore della loss MSE normalizzata è passato da circa `0,05` all'epoca 50 a circa `0,01` alle epoche 150 e 200. Questo indica che il modello ha continuato ad apprendere e ha raggiunto una buona convergenza.

Nel confronto finale, la regressione lineare ha ottenuto metriche migliori rispetto alla rete neurale: errore assoluto medio, MSE e RMSE più bassi, oltre a un valore R² più alto. La rete neurale resta comunque efficace, con un R² pari a `0,9813`, ma i dati mostrano una relazione prevalentemente lineare e, in questo caso, il modello più semplice risulta il più preciso.

I risultati possono variare leggermente in base alla versione delle librerie e alla configurazione hardware, in particolare per l'addestramento della rete neurale.

## Installazione

È consigliato Python 3.9 o una versione successiva. Dalla directory principale del progetto, creare un ambiente virtuale e installare le dipendenze:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Su macOS o Linux, attivare l'ambiente con:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Esecuzione

Aprire il notebook con Jupyter:

```bash
jupyter notebook progetto.ipynb
```

In alternativa, aprire `progetto.ipynb` con JupyterLab o Visual Studio Code, selezionare il kernel dell'ambiente virtuale ed eseguire tutte le celle in ordine.
