# Employee Salary Prediction

Progetto di regressione supervisionata per stimare lo stipendio annuale di un dipendente a partire da caratteristiche anagrafiche, professionali e lavorative.

## Contenuto del progetto

```
.
├── progetto.ipynb
└── data/
    └── employee_salary_regression.csv
```

- `progetto.ipynb` contiene l'analisi esplorativa, il preprocessing, l'addestramento e la valutazione dei modelli.
- `data/employee_salary_regression.csv` contiene 1.000 osservazioni e 10 colonne.

## Dataset

La variabile obiettivo è `annual_salary_usd`. Le caratteristiche utilizzate sono:

- `years_experience`
- `education_level`
- `job_role`
- `city_tier`
- `performance_score`
- `num_skills`
- `remote_work`

La colonna `employee_id` viene esclusa dal modello. Anche `age` viene rimossa dopo l'analisi esplorativa perché fortemente correlata con `years_experience`.

## Modelli

Il notebook confronta due approcci:

1. Regressione lineare
   - standardizzazione delle variabili numeriche;
   - codifica One-Hot delle variabili categoriche;
   - valutazione tramite MAE, MSE, RMSE e R².
2. Rete neurale
   - architettura fully connected con livelli `64 -> 32 -> 1`;
   - attivazione ReLU;
   - ottimizzatore Adam;
   - 200 epoche di addestramento.

Il dataset viene suddiviso in training set (67%) e test set (33%), usando `random_state=42`.

## Requisiti

Python 3.9 o versione successiva, con i seguenti pacchetti:

```bash
pip install jupyter pandas numpy matplotlib scikit-learn torch
```

## Esecuzione

Dalla directory principale del progetto:

```bash
jupyter notebook progetto.ipynb
```

In alternativa, il notebook può essere aperto ed eseguito con JupyterLab o Visual Studio Code. Eseguire le celle in ordine per caricare i dati, addestrare i modelli e visualizzare le metriche e i grafici delle predizioni.

## Nota

I risultati dipendono dall'ambiente di esecuzione e dalla versione delle librerie. Il dataset è incluso nel repository e viene caricato tramite il percorso relativo `data/employee_salary_regression.csv`.
