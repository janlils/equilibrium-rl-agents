# sarsa-farmers

Symulacja rynku rolnego oparta o wieloagentowe uczenie SARSA z funkcją wartości aproksymowaną liniowo. Projekt pozwala badać, w jaki sposób agenci (rolnicy) uczą się wybierać rynek zbytu w odpowiedzi na ceny, koszty i ex-post szoki, a także porównywać ich zachowanie z rozkładem optymalnym.

## Spis treści
1. [Architektura](#architektura)
2. [Instalacja](#instalacja)
3. [Uruchamianie symulacji](#uruchamianie-symulacji)
4. [Scenariusze eksperymentalne](#scenariusze-eksperymentalne)
5. [Analiza wyników i wykresy](#analiza-wyników-i-wykresy)
6. [Eksport zbiorczy do Excela](#eksport-zbiorczy-do-excela)
7. [Struktura katalogów](#struktura-katalogów)

## Architektura

- `simulation.py` — główny silnik symulacji, tworzy agentów (`farmer.py`), rynki (`market.py`) oraz zapisuje wyniki do plików `.pickle`.
- `farmer.py` — definicja strategii agentów z liniową aproksymacją Q i opcjonalnym trybem losowym.
- `market.py` + `config.py` — definicja rynków, funkcji cenowych i kosztów, w tym możliwość dodania rynku „rządowego”.
- `scenario.py` — generator szoków makro (np. zmiana ceny, inflacja, zwiększenie liczby uczestników).
- `optimal_allocation.py` — dynamiczne programowanie do obliczenia maksymalnego potencjału (ref. optimum).
- `summary.py` — generacja wykresów i arkuszy diagnostycznych z wygenerowanych wyników.
- `experiments.py` — lista gotowych scenariuszy oraz narzędzie do uruchamiania wielu eksperymentów.
- `export_runs_excel.py` — dodatkowy raport Excel agregujący metryki dla wybranych katalogów z wynikami.

## Instalacja

1. **Środowisko**: Python 3.11+ (zalecany virtualenv/conda).
2. **Wymagania**: `pip install -r requirements.txt`. W projekcie wykorzystywane są m.in. `numpy`, `pandas`, `matplotlib`, `seaborn`, `openpyxl`.
3. **Uwagi MacOS**: W niektórych środowiskach może brakować modułu `numpy._core` podczas ładowania pickli zapisanych przez NumPy 2.x — upewnij się, że masz spójną wersję NumPy (`pip install "numpy>=2.0"`).

## Uruchamianie symulacji

Najprostszy sposób to wywołać bezpośrednio:

```bash
python simulation.py
```

Domyślnie uruchomionych zostanie `N=100` agentów, 1000 iteracji w każdej z 10 rund. Wyniki zapisywane są w katalogu `results/run_<timestamp>` jako `results_market*.pickle`, `results_profits*.pickle`, `results_potential*.pickle` oraz (dla nowych wersji) `results_theta*.pickle`.

Parametry możesz nadpisać poprzez wywołanie `run_experiment` z odpowiednimi argumentami (np. `add_gov=True`, `switch_cost`, `scenario`, `model_type="Q1"/"Q2"/"full"`, `random_policy=True`).

## Scenariusze eksperymentalne

Skrypt `experiments.py` definiuje gotowe scenariusze (np. `4m_random`, `5m_Q1_shocks`) oraz narzędzie do zbiorczego uruchomienia:

```bash
python experiments.py --scenario 5m_Q1_shocks
# lub wszystkie:
python experiments.py --all
```

Podczas działania:
- wywoływana jest symulacja z parametrami scenariusza,
- tworzony jest `config.json` z metadanymi,
- automatycznie generowane są wykresy podsumowujące (patrz sekcja poniżej),
- wyniki pozwalają później spiąć wiele eksperymentów z tabelą `results/experiments_summary.csv`.

## Analiza wyników i wykresy

Narzędzie `summary.py` przyjmuje parę `results_market*.pickle` + `results_profits*.pickle` (opcjonalnie `results_potential*.pickle` i `results_theta*.pickle`) i generuje zestaw wykresów PNG oraz arkusze CSV:

```bash
python summary.py --market results/run_2025-12-15_11-27-10_4m_random/results_market_4m.pickle \
                  --profits results/run_2025-12-15_11-27-10_4m_random/results_profits_4m.pickle \
                  --outdir results/run_2025-12-15_11-27-10_4m_random/plots
```

Najważniejsze wykresy:
- `art_Number_Iter.png` — średnia liczba agentów na rynku w kolejnych iteracjach,
- `art_Profit_Iter.png` — średnie zyski (całkowite i per rynek, zakres domyślnie ±10),
- `art_Efficiency_Stability_Iter.png` — porównanie efektywności (`Phi/Phi_max`) i stabilności (procent agentów, którzy pozostali na rynku),
- `art_Allocation_Optimal_Iter.png` — faktyczna vs. optymalna alokacja z zaznaczonymi szokami (etykiety przy górnej krawędzi),
- `art_Q_Params_Iter.png` — średnie współczynniki funkcji Q, jeśli zapisano `results_theta*.pickle`.

Ponadto generowane są arkusze CSV:
- `diagnostic_episodes.csv` (szczegółowe dane na poziomie iteracji),
- `episode_efficiency_stats.csv` (odsetek efektywnych iteracji, szacowany czas dostosowania, odchylenie, itp.),
- `plots/art_*` — gotowe grafiki do raportów.

## Eksport zbiorczy do Excela

Skrypt `export_runs_excel.py` agreguje metryki z wielu katalogów `run_*`. Wspiera dwie formy podania listy:

1. Argumenty pozycyjne:
   ```bash
   python export_runs_excel.py results/run_... results/run_... --out results/summary.xlsx
   ```
2. Plik tekstowy `runs.txt` (po jednej ścieżce na linię, puste linie i wiersze z `#` są ignorowane):
   ```bash
   python export_runs_excel.py --runs-file results/runs.txt --out results/summary.xlsx
   ```

Każdy wiersz w Excelu zawiera m.in.:
- `efficient_pct` / `adjustment_pct` — średni odsetek efektywnych iteracji i udział czasu dostosowania,
- `mean_efficiency`, `min_efficiency`, `max_efficiency`, `std_efficiency`, `mean_variance_efficiency` — agregaty potencjału,
- `mean_stability` — odsetek agentów, którzy nie zmieniali rynku,
- `avg_profit` — średni zysk w ostatnich 100 iteracjach,
- `recent_allocation_gap` — suma bezwzględnych różnic między alokacją faktyczną a optymalną (ostatnie 100 iteracji),
- `eq_start_iteration` — średnia iteracja (po rundach), od której każdorazowo osiągano równowagę (5 kolejnych iteracji z `Phi_ratio > 0.99`),
- `avg_alloc_*` i `avg_opt_alloc_*` — przeciętne obsady rynków i optimum w końcówce symulacji.

## Struktura katalogów

- `results/`
  - `run_<timestamp>[_expid]` — pojedynczy eksperyment.
    - `config.json` — parametry wejściowe.
    - `results_market*.pickle`, `results_profits*.pickle`, `results_potential*.pickle`, `results_theta*.pickle`.
    - `plots/` — wygenerowane wykresy PNG + CSV.
  - `runs.txt` — przykładowa lista 10 ostatnich katalogów (od najstarszego do najnowszego).
  - `experiments_summary.csv` — zbiorcze zestawienie scenariuszy po uruchomieniu `experiments.py`.
- `requirements.txt` — zależności Pythona.
- `old/` — archiwalne materiały (niewykorzystywane w aktualnej wersji).

## Przydatne wskazówki

- **Reproducible runs**: `simulation.py` przyjmuje ziarno (`seed`), więc możesz odtwarzać eksperymenty.
- **Tryb losowy**: ustaw `random_policy=True`, by agenci losowo zmieniali rynki (przydatne jako baseline).
- **Koszty i szoki**: definiuj w `scenario.py` i przekazuj listę zdarzeń przez `run_experiment(..., scenario=[...])`.
- **Wydajność**: korzystaj z `experiments.py`, aby hurtowo uruchamiać wiele konfiguracji i automatycznie generować raporty.
- **Eksport**: `export_runs_excel.py` wymaga `openpyxl`; w Excelu łatwo porównać metryki między runami (np. filtry, wykresy pivot).

## Licencja

Projekt ma charakter badawczy i edukacyjny. Brak formalnej licencji — jeśli chcesz wykorzystać kod komercyjnie, skontaktuj się z autorem repozytorium.
