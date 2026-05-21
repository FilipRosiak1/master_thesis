# ML zamiast klasycznej ewolucji dla genotypów Framsticks f1

**Cel dokumentu:** zaproponować architekturę i plan eksperymentu, w którym klasyczne operatory ewolucyjne dla genotypów Framsticks `f1` są częściowo zastępowane modelem uczenia maszynowego oraz optymalizacją typu CEM/CMA-ES w przestrzeni latentnej lub w przestrzeni produkcji gramatyki.

**Kontekst:** dane wejściowe to zmiennej długości ciągi znaków w reprezentacji `f1` oraz odpowiadające im wartości fitness. Reprezentacja `f1` opisuje ciała istot Framsticks jako struktury drzewiaste z elementami kontrolnymi/neuronowymi przyczepionymi do segmentów.

**Wniosek główny:** wcześniejsze podejście nadal jest zasadne, ale dla `f1` trzeba je doprecyzować: **nie optymalizować surowych znaków**, tylko pracować na gramatyce/AST/drzewie/heterogenicznym grafie, z dekoderem generującym poprawne genotypy zmiennej długości oraz z ograniczeniem eksploracji poza rozkładem danych.

---

## 1. Co oznacza f1 i dlaczego zmienna długość nie psuje podejścia

Dokumentacja Framsticks opisuje `f1` jako format, w którym podstawowymi symbolami ciała są:

- `X` — stick/segment,
- `()` — rozgałęzienie.

Struktura ciała jest budowana jak drzewo: nowe segmenty są dołączane do końców poprzednich segmentów. Przykłady z dokumentacji to między innymi `X(X,X)`, `X(X,X,X)`, `X(X,X,)` oraz `X(,X,,,X,,X,,)`. W nawiasach pełny kąt rozgałęzienia jest dzielony zależnie od liczby przecinków, a modyfikatory takie jak `R/r`, `Q/q`, `C/c`, `L/l`, `W/w`, `F/f`, `A/a`, `S/s`, `M/m`, `I/i`, `E/e` wpływają na pozycję i właściwości segmentów. Neurony i elementy sterujące mogą być zapisane po segmentach w nawiasach kwadratowych `[]`.

To ma kilka konsekwencji dla ML:

1. Genotypy są **sekwencjami zmiennej długości**.
2. Fenotyp ciała ma naturalną strukturę **drzewa**, nie zwykłego tekstu.
3. Jeśli uwzględniasz neurony, dostajesz raczej **strukturę mieszaną**: drzewo ciała + graf połączeń neuronowych.
4. Nie wszystkie sekwencje znaków są poprawnymi genotypami.
5. Drobne zmiany składniowe mogą mieć różny wpływ: czasem lokalny, czasem globalny przez propagację modyfikatorów.

Dlatego zmienna długość **nie unieważnia** podejścia z przestrzenią latentną. Wręcz przeciwnie: jest jednym z powodów, dla których latent space jest atrakcyjny. CMA-ES/CEM potrzebuje przestrzeni o stałym wymiarze, a model generatywny może mapować:

```text
genotyp f1 zmiennej długości → z ∈ R^d → genotyp f1 zmiennej długości
```

Najważniejsze jest, aby dekoder potrafił kończyć generację, zachować poprawność gramatyczną i kontrolować rozmiar struktury.

---

## 2. Dlaczego nie optymalizować bezpośrednio stringów f1

Bezpośrednie stosowanie CMA-ES do znaków lub tokenów byłoby nienaturalne:

```text
CMA-ES: R^d → R
f1: dyskretny, zmiennej długości, ograniczony gramatyką
```

Nawet gdyby zakodować tokeny jako liczby, operacje typu średnia, kowariancja i perturbacja Gaussowska nie mają sensownej interpretacji składniowej.

Zamiast tego należy rozdzielić problem na:

```text
f1 string
  → parser
  → AST / drzewo ciała / heterogeniczny graf
  → encoder
  → z, stały wymiar
  → CMA-ES/CEM
  → decoder constrained by grammar
  → poprawny f1 string
  → prawdziwa ewaluacja fitness
```

---

## 3. Rekomendowana reprezentacja wewnętrzna

### 3.1 Nie traktuj f1 jako zwykłego tekstu

Możesz trenować Transformer na surowych tokenach jako baseline, ale docelowo lepiej użyć struktury po parsowaniu.

Proponowana reprezentacja:

```text
F1Genotype
├── BodyTree
│   ├── StickNode
│   │   ├── modifiers_before_X
│   │   ├── effective_properties
│   │   ├── children/branches
│   │   └── attached_neurons
│   └── BranchNode
│       ├── comma_slots
│       └── children
└── NeuralGraph opcjonalnie
    ├── neurons
    ├── receptors
    ├── effectors
    └── weighted_connections
```

W zależności od datasetu:

- jeśli masz tylko morfologię: wystarczy `BodyTree`,
- jeśli masz morfologię + sterowanie: użyj `BodyTree + NeuralGraph`,
- jeśli zależy ci na zachowaniu dokładnej składni `f1`: przechowuj też tokeny i mapowanie AST → oryginalny/kano­niczny string.

### 3.2 Kanonizacja

Przed trenowaniem warto zrobić kanonizację:

```text
f1 string → parser Framsticks → f0/phenotype/AST → canonical f1 albo canonical structural representation
```

Powody:

- różne stringi mogą opisywać bardzo podobne albo równoważne struktury,
- model może nauczyć się składniowego szumu zamiast relacji struktura–fitness,
- duplikaty i prawie-duplikaty będą zafałszowywać walidację.

Minimalna kanonizacja:

1. usuń nieistotne spacje,
2. ujednolić zapis liczb,
3. ujednolić kolejność lub reprezentację neuronów, jeśli jest możliwe,
4. rozważ deduplikację po fenotypie, nie tylko po stringu,
5. dodaj cechy fenotypowe z Framsticks/f0 jako dodatkowy input do predyktora.

---

## 4. Architektura modelu

Najbardziej sensowny wariant:

```text
f1 string
   ↓
parser f1
   ↓
AST / tree / graph
   ↓
Encoder strukturalny
   ↓
z ∈ R^d
   ├───────────────→ fitness head / ensemble
   ↓
Grammar-aware decoder
   ↓
poprawny f1 string zmiennej długości
```

### 4.1 Encoder

Możliwe enkodery:

| Przypadek | Encoder |
|---|---|
| Sam string jako baseline | Transformer encoder / BiLSTM |
| Ciało jako drzewo | Tree-LSTM, recursive Transformer, Tree Transformer |
| Ciało jako graf | GNN z typami węzłów i krawędzi |
| Ciało + mózg | heterogeniczny GNN: segmenty, neurony, sensory, mięśnie, połączenia |
| Duży dataset | Transformer na tokenach + dodatkowe cechy AST |
| Mały dataset | cechy ręczne + LightGBM/XGBoost + później model generatywny |

Dla `f1` preferowałbym:

```text
Tree/Graph encoder + cechy globalne + fitness head
```

Cechy globalne mogą zawierać:

- liczba segmentów `X`,
- głębokość drzewa,
- liczba rozgałęzień,
- liczba pustych slotów w rozgałęzieniach,
- histogram modyfikatorów,
- liczba neuronów,
- liczba połączeń neuronowych,
- liczba sensorów i efektorów,
- cechy geometryczne po dekodowaniu do fenotypu/f0,
- metryki symetrii, modularności i długości.

### 4.2 Latent space

Latent powinien być stałowymiarowy:

```text
z ∈ R^d, np. d = 16, 32, 64, 128
```

Zmienna długość genotypu jest obsługiwana przez dekoder, nie przez CMA-ES.

Dobry latent powinien spełniać:

1. **lokalność**: bliskie `z` dają podobne struktury,
2. **ciągłość fitnessu**: mała zmiana `z` zwykle nie powoduje katastrofalnej zmiany fitness,
3. **pokrycie topowych struktur**: dobre genotypy z datasetu nie powinny leżeć w odizolowanych, chaotycznych regionach,
4. **dekodowalność**: większość sensownych `z` powinna dekodować się do poprawnych `f1`,
5. **kontrolę długości**: model nie powinien zawsze generować średniej długości ani eksplodować długością.

### 4.3 Decoder

Dla `f1` dekoder musi obsługiwać:

- tokeny zmiennej długości,
- nawiasy `()` i `[]`,
- ograniczenia kolejności,
- zakończenie generacji,
- opcjonalne modyfikatory przed `X` lub `(`,
- wartości liczbowe w neuronach,
- relatywne referencje neuronowe, jeśli są w danych.

Najlepsze warianty dekodera:

#### Wariant A — decoder produkcji gramatyki

Model generuje nie znaki, tylko reguły:

```text
Body       → Stick
Stick      → Modifiers X NeuronList BranchList?
BranchList → "(" BranchItems ")"
BranchItems→ Empty | Stick | Stick "," BranchItems | "," BranchItems
Modifiers  → ε | Modifier Modifiers
NeuronList → ε | Neuron NeuronList
Neuron     → "[" NeuronSpec "]"
```

W praktyce trzeba doprecyzować gramatykę według realnego subsetu `f1` w datasetcie.

Zaletą jest możliwość maskowania nielegalnych produkcji.

#### Wariant B — decoder AST/top-down

Model rozwija drzewo od korzenia:

```text
sample root stick
while stack not empty:
    pop node
    sample modifiers
    sample attached neurons
    sample number of branch slots
    sample children in slots
```

To jest bardzo naturalne dla f1, bo ciało jest drzewem.

#### Wariant C — decoder grafowy

Jeżeli neurony i połączenia są istotne:

1. wygeneruj drzewo ciała,
2. wygeneruj węzły neuronów przy segmentach,
3. wygeneruj połączenia neuronowe z ograniczeniami,
4. wyrenderuj do f1.

To jest bardziej skomplikowane, ale daje lepszą kontrolę nad mózgiem.

---

## 5. Co zmienia zmienna długość

Zmienna długość wymaga kilku usprawnień.

### 5.1 Padding i maski w modelu sekwencyjnym

Jeżeli używasz Transformera na tokenach:

```text
batch:
  [X, (, X, ,, X, ), EOS, PAD, PAD]
  [X, X, X, EOS, PAD, PAD, PAD, PAD]
```

Potrzebujesz:

- `EOS` jako token końca,
- `PAD` jako token do batchowania,
- attention mask,
- loss mask ignorujący `PAD`,
- ograniczenie maksymalnej długości.

### 5.2 Loss normalizowany po długości

Bez normalizacji długie genotypy mogą dominować stratę rekonstrukcji.

Stosuj:

```text
reconstruction_loss = sum(token_loss * mask) / number_of_non_pad_tokens
```

Dla AST analogicznie:

```text
loss = średnia po decyzjach dekodera, nie suma po całym drzewie
```

### 5.3 Bucketing po długości

Warto trenować w batchach o podobnych długościach:

```text
bucket 1: 1–20 tokenów
bucket 2: 21–50 tokenów
bucket 3: 51–100 tokenów
bucket 4: 101–200 tokenów
```

Zmniejsza to padding i stabilizuje trening.

### 5.4 Warunkowanie długością

CMA-ES może znaleźć latent, który dekoduje się do zbyt krótkich lub zbyt długich struktur. Dlatego warto dodać jawny warunek:

```text
decoder(z, target_size)
```

gdzie `target_size` może oznaczać:

- liczbę segmentów,
- maksymalną głębokość,
- liczbę neuronów,
- budżet tokenów,
- klasę długości.

Możesz wtedy optymalizować:

```text
z ciągły + size_bin dyskretny
```

albo uruchamiać osobne optymalizacje dla różnych długości:

```text
CMA-ES dla 5–10 segmentów
CMA-ES dla 10–20 segmentów
CMA-ES dla 20–40 segmentów
```

### 5.5 Kara za wyjście poza rozkład długości

Jeśli dataset ma długości głównie 20–80 tokenów, a model generuje 500 tokenów, to jest to zwykle model exploitation.

Dodaj do acquisition function:

```text
length_penalty = distance_to_training_length_distribution(x)
```

Na przykład:

```text
score(z) =
    predicted_fitness(z)
    - λ_unc * uncertainty(z)
    - λ_len * length_penalty(x)
    - λ_prior * max(0, ||z|| - r)
    + λ_novelty * novelty(x)
```

---

## 6. Model fitnessu

### 6.1 Nie zaczynaj od generowania

Pierwszy etap powinien odpowiedzieć na pytanie:

> Czy z samych genotypów/struktur można przewidzieć albo przynajmniej poprawnie uporządkować fitness?

Modele baseline:

1. **LightGBM/XGBoost na cechach ręcznych**  
   Bardzo ważny baseline, szczególnie przy małym datasetcie.

2. **Transformer na tokenach f1**  
   Szybki baseline dla surowej sekwencji.

3. **Tree/GNN predictor**  
   Docelowy model strukturalny.

Metryki:

| Cel | Metryka |
|---|---|
| Regresja fitness | MAE, RMSE, R² |
| Ranking | Spearman, Kendall tau |
| Wybór najlepszych | top-k recall, precision@k |
| Stabilność | metryki osobno dla długości, głębokości, liczby neuronów |
| OOD | błąd względem odległości od danych treningowych |

Dla optymalizacji ważniejszy jest ranking niż idealna regresja.

### 6.2 Ranking loss

Oprócz MSE/MAE dodałbym ranking loss:

```text
L = L_regression + α L_pairwise_ranking
```

Przykład pairwise:

```text
jeśli fitness_i > fitness_j:
    L_rank = max(0, margin - (pred_i - pred_j))
```

To pomaga modelowi odróżniać dobre osobniki od bardzo dobrych.

### 6.3 Niepewność

CMA-ES/CEM będzie wykorzystywał błędy predyktora. Dlatego potrzebujesz estymacji niepewności.

Najpraktyczniejsze opcje:

1. ensemble 5–10 predyktorów,
2. MC dropout,
3. quantile regression,
4. heteroscedastic regression,
5. Gaussian Process na niskowymiarowym latencie.

Najprostsza rekomendacja:

```text
ensemble fitness_head_1...fitness_head_K
mean = średnia predykcji
uncertainty = odchylenie standardowe predykcji
```

---

## 7. Model generatywny

### 7.1 Minimalny model

```text
Encoder(AST) → z
Decoder(z) → AST/f1
FitnessHead(z) → fitness
```

Strata:

```text
L =
    L_reconstruction
  + α L_fitness
  + β L_ranking
  + γ L_KL
  + δ L_length
```

Gdzie:

- `L_reconstruction` — poprawne odtworzenie produkcji/tokenów/AST,
- `L_fitness` — regresja fitness,
- `L_ranking` — ranking topowych struktur,
- `L_KL` — regularyzacja VAE,
- `L_length` — predykcja lub kontrola długości.

### 7.2 VAE czy zwykły autoencoder?

| Model | Zalety | Wady |
|---|---|---|
| Autoencoder | lepsza rekonstrukcja, prostszy | latent może być poszarpany |
| VAE | lepszy do samplingu i CMA-ES | może pogorszyć rekonstrukcję |
| β-VAE | większa regularność | ryzyko utraty szczegółów |
| VQ-VAE | dyskretny latent, dobre klastry | CMA-ES mniej naturalny |
| Diffusion/flow | mocne generatywnie | większa złożoność |

Dla tego problemu zacząłbym od:

```text
deterministyczny autoencoder + fitness head
```

a potem przeszedł do:

```text
Grammar/Tree VAE + fitness head + ensemble uncertainty
```

### 7.3 Dekoder constrained by grammar

Dla `f1` warto zastosować maskowanie akcji:

```text
stan parsera/dekodera → lista dozwolonych kolejnych produkcji
```

Przykład:

```text
po otwarciu "(":
    dozwolone: X, modifier, ",", ")"

po "[":
    dozwolone: neuron type, parameter, connection, "]"

po zamknięciu genotypu:
    dozwolone: EOS
```

Dzięki temu model nie musi „uczyć się”, że nawiasy mają się domykać — to jest wymuszone.

---

## 8. CMA-ES/CEM w przestrzeni latentnej

### 8.1 Inicjalizacja

Nie startuj z losowego `N(0, I)`, tylko z topowych osobników.

```python
D_top = select_top_percent(D, 5)
Z_top = [encoder(x) for x, y in D_top]

mean0 = mean(Z_top)
cov0 = covariance(Z_top) + eps * I
```

Możesz też uruchamiać kilka restartów:

```text
restart 1: top 1%
restart 2: top 5%
restart 3: top 10%
restart 4: klaster A dobrych genotypów
restart 5: klaster B dobrych genotypów
```

### 8.2 Funkcja celu dla CMA-ES

Nie optymalizuj samej średniej predykcji fitnessu.

Lepsza funkcja:

```text
score(z) =
    μ_fitness(z)
  - λ_unc * σ_fitness(z)
  - λ_invalid * invalid_penalty(x)
  - λ_len * length_penalty(x)
  - λ_ood * ood_penalty(z)
  + λ_novelty * novelty(x)
  + λ_div * diversity_bonus(x)
```

Gdzie:

```text
x = decoder(z)
```

Jeżeli dekoder jest valid-by-construction, `invalid_penalty` może być zerowy.

### 8.3 OOD penalty

Przykłady:

```text
ood_penalty(z) = max(0, distance_to_nearest_training_z - threshold)
```

albo:

```text
ood_penalty(z) = -log p_latent(z)
```

Jeżeli używasz VAE, możesz używać prior penalty:

```text
prior_penalty = max(0, ||z||² - threshold)
```

### 8.4 Prawdziwa ewaluacja jest obowiązkowa

Model jest tylko surrogate. Po optymalizacji:

```text
z_candidates → decode → f1_candidates → Framsticks true fitness
```

Następnie:

```text
D = D ∪ {(f1_candidate, true_fitness)}
retrain / fine-tune model
```

To jest aktywna pętla uczenia.

---

## 9. Alternatywa: CEM bez latent space, bezpośrednio na gramatyce f1

Dla zmiennej długości `f1` bardzo sensowną alternatywą jest CEM na produkcjach gramatycznych.

Zamiast:

```text
CMA-ES → z → decoder
```

masz:

```text
CEM → rozkład po regułach produkcji → sample AST/f1 → surrogate/true fitness
```

Przykład rozkładu:

```text
p(action | context)
```

gdzie `context` zawiera:

- typ aktualnego węzła,
- głębokość,
- liczba dzieci,
- poprzednie modyfikatory,
- budżet długości,
- czy generujemy ciało czy neuron.

Iteracja CEM:

```text
1. Sample N genotypów z rozkładu gramatycznego.
2. Oceń surrogate fitness.
3. Wybierz elite top ρ%.
4. Zaktualizuj prawdopodobieństwa reguł.
5. Co kilka iteracji oceń prawdziwy fitness.
```

Zalety:

- naturalnie obsługuje zmienną długość,
- generuje poprawne struktury,
- jest prostsze niż VAE,
- można łatwo kontrolować długość i głębokość.

Wady:

- może słabiej wykorzystywać globalne podobieństwa między strukturami,
- wymaga dobrego zaprojektowania kontekstu,
- bez neuralnego modelu może mieć ograniczoną ekspresję.

Moja rekomendacja: potraktować to jako mocny baseline.

---

## 10. Proponowany pipeline eksperymentalny

### Etap 0 — audyt datasetu

Policz:

```text
liczba przykładów
rozkład fitnessu
rozkład długości tokenów
rozkład liczby segmentów X
rozkład głębokości drzewa
rozkład liczby neuronów
procent niepoprawnych genotypów
duplikaty po stringu
duplikaty po fenotypie
liczba unikalnych topologii ciała
```

Sprawdź, czy fitness jest deterministyczny. Jeśli symulacja ma szum, zapisz:

```text
mean_fitness
std_fitness
n_evaluations
seed/environment
```

### Etap 1 — parser i reprezentacja

1. Parser `f1 → AST`.
2. Walidator `AST → valid/invalid`.
3. Renderer `AST → canonical f1`.
4. Opcjonalnie konwersja `f1 → f0/fenotyp`.
5. Ekstrakcja cech strukturalnych.

To jest krytyczna część projektu.

### Etap 2 — predictor fitnessu

Zbuduj trzy baseline'y:

```text
A. cechy ręczne + LightGBM/XGBoost
B. Transformer na tokenach
C. Tree/GNN predictor
```

Walidacja powinna być trudniejsza niż losowy split. Rozważ:

- split po rodzinach/klastrach podobieństwa,
- split czasowy, jeśli dane pochodzą z procesu ewolucyjnego,
- split po długości, np. trenowanie na krótszych i test na dłuższych,
- ocena osobno dla top 10%, top 5%, top 1%.

### Etap 3 — model generatywny

Trenuj:

```text
Grammar/Tree Autoencoder albo VAE
```

Metryki:

```text
validity_rate
exact_reconstruction_rate
phenotype_reconstruction_rate
length_error
tree_edit_distance
novelty
diversity
fitness_predictor_correlation_in_latent
```

Test interpolacji:

```text
x1, x2 z top fitness
z1 = encode(x1)
z2 = encode(x2)

for t in [0, 0.1, ..., 1]:
    z = (1-t)z1 + t z2
    x = decode(z)
    sprawdź poprawność, długość, podobieństwo, predykcję fitness
```

### Etap 4 — latent CMA-ES/CEM

Dla każdego restartu:

```text
initialize mean/cov from top cluster
for generation in 1..G:
    sample z
    decode to f1
    filter duplicates
    evaluate acquisition
    update CMA-ES/CEM
```

Po każdej rundzie:

```text
wybierz top K według acquisition
uruchom Framsticks true evaluation
dodaj do datasetu
doucz model
```

### Etap 5 — porównanie z klasyczną ewolucją

Porównaj w tym samym budżecie prawdziwych ewaluacji fitness:

```text
1. klasyczny EA Framsticks
2. losowe mutacje f1
3. CEM na gramatyce
4. surrogate + CEM na gramatyce
5. VAE latent + CMA-ES
6. VAE latent + CEM
7. ML-guided mutation/crossover
```

Najważniejsza metryka:

```text
best_true_fitness after N true evaluations
```

Dodatkowo:

```text
median best fitness over seeds
AUC best-so-far
diversity top solutions
valid candidates per true evaluation
novelty vs dataset
```

---

## 11. Usprawnienia specyficzne dla f1

### 11.1 Operuj na segmentach, nie na znakach

Token `X` jest semantycznie ważniejszy niż pojedynczy znak. Modyfikatory przed `X` też powinny być związane z konkretnym segmentem.

Lepsze tokeny wysokiego poziomu:

```text
STICK
BRANCH_OPEN
BRANCH_CLOSE
BRANCH_SEPARATOR
MOD_R_PLUS
MOD_R_MINUS
MOD_LENGTH_PLUS
MOD_LENGTH_MINUS
NEURON_OPEN
NEURON_CLOSE
PARAM_VALUE
CONNECTION_REF
```

Jeszcze lepiej:

```text
StickNode {
    local_modifiers,
    effective_modifiers,
    attached_neurons,
    children_slots
}
```

### 11.2 Ucz model także na cechach po dekodowaniu

Fitness zależy od symulacji, więc sam string może być za słabą reprezentacją. Do predyktora dodaj:

```text
f1 → Framsticks decoder → phenotype/f0 → structural features
```

Model:

```text
embedding_structural = Tree/GNN(f1_AST)
embedding_features = MLP(handcrafted_features)
fitness = MLP(concat(embedding_structural, embedding_features))
```

### 11.3 Modeluj długość jako osobną zmienną

Warto trenować head:

```text
length_head(z) → expected number of sticks/tokens/depth
```

A podczas optymalizacji dodać:

```text
target_length_score
```

Przykład:

```text
score = predicted_fitness - λ * abs(predicted_length - target_length)
```

### 11.4 Kontroluj modyfikatory

Modyfikatory w `f1` propagują efekt na kolejne segmenty. Model powinien widzieć zarówno:

- surowe modyfikatory lokalne,
- efektywne właściwości segmentu po propagacji.

Dlatego przy parsowaniu warto obliczać:

```text
node.local_modifiers
node.effective_rotation
node.effective_twist
node.effective_length
node.effective_friction
...
```

### 11.5 Wykorzystaj istniejące operatory Framsticks

Nie trzeba całkowicie zastępować ewolucji. Bardzo dobry wariant hybrydowy:

```text
ML proponuje regiony / rodziców / kierunki mutacji
Framsticks operators robią lokalne zmiany
surrogate filtruje kandydatów
true fitness ocenia najlepszych
```

Schemat:

```text
1. wybierz obiecujące genotypy według predyktora
2. zastosuj mutacje/crossovery Framsticks
3. odfiltruj surrogate'em 90–99% kandydatów
4. prawdziwie oceń top K
```

To często będzie stabilniejsze niż czysty latent optimization.

---

## 12. Ryzyka i jak je ograniczać

### 12.1 Model exploitation

Największe ryzyko:

```text
CMA-ES znajduje genotypy, którym model daje wysoki fitness, ale prawdziwy fitness jest niski.
```

Ograniczenia:

- ensemble uncertainty,
- OOD penalty,
- kara za nietypową długość/głębokość,
- novelty kontrolowana, nie maksymalizowana bez granic,
- prawdziwa ewaluacja co iterację,
- aktywne douczanie,
- walidacja na top-k, nie tylko MSE.

### 12.2 Bias datasetu

Jeśli dataset pochodzi z klasycznej ewolucji, to będzie silnie stronniczy. Model nauczy się „stylu” dotychczasowego algorytmu.

Rozwiązania:

- dodać losowe i niskofitnessowe przykłady,
- aktywnie eksplorować różne długości,
- klastrować struktury i próbkować z klastrów,
- raportować wyniki względem podobieństwa do danych treningowych.

### 12.3 Błędy walidacji przez podobieństwo rodzin

Jeśli losowo podzielisz dataset, bardzo podobne genotypy mogą trafić do train i test. Wynik będzie zawyżony.

Lepsze splity:

```text
split po klastrach podobieństwa strukturalnego
split po przodkach/linii ewolucyjnej
split czasowy
split po długościach
```

### 12.4 Zbyt krótki latent

Jeżeli `d` jest zbyt małe, model nie odtworzy szczegółów. Jeżeli `d` jest zbyt duże, CMA-ES będzie mniej efektywny.

Praktycznie:

```text
zacznij od d = 32 albo 64
porównaj d = 16, 32, 64, 128
```

### 12.5 Długi ogon długości

Jeśli są bardzo długie genotypy, rozważ:

- osobny model dla długich,
- ograniczenie maksymalnej długości na początku,
- hierarchiczny dekoder modułów,
- kompresję powtarzających się motywów,
- osobną klasę „large structures”.

---

## 13. Minimalny pseudokod

```python
# D = [(f1_string, fitness)]

# 1. Parse and canonicalize
records = []
for f1, y in D:
    ast = parse_f1(f1)
    if not ast.valid:
        continue

    canonical = render_canonical_f1(ast)
    features = extract_structural_features(ast)
    records.append((canonical, ast, features, y))

# 2. Train predictor
predictor = FitnessPredictor()
predictor.train(records)

# 3. Train grammar/tree VAE
vae = GrammarTreeVAE(
    latent_dim=64,
    grammar="f1_subset",
    length_conditioning=True,
)
vae.train(records, aux_fitness_predictor=True)

# 4. Build ensemble for uncertainty
ensemble = train_fitness_ensemble(records, K=5)

# 5. Initialize CMA-ES from best known genotypes
top = select_top_percent(records, percent=5)
Z_top = [vae.encode(ast) for _, ast, _, _ in top]

mean0 = mean(Z_top)
cov0 = covariance(Z_top) + 1e-4 * I

def acquisition(z, target_size=None):
    ast = vae.decode(z, target_size=target_size)
    if not ast.valid:
        return -1e9

    f1 = render_canonical_f1(ast)

    mu, sigma = ensemble.predict_from_ast_or_z(ast, z)

    length_penalty = length_distribution_penalty(ast)
    ood_penalty = latent_ood_penalty(z, Z_train=vae.encoded_training_set)
    novelty = structural_novelty(ast, records)

    return (
        mu
        - 0.5 * sigma
        - 0.2 * length_penalty
        - 0.2 * ood_penalty
        + 0.05 * novelty
    )

# 6. Optimize latent space
candidates_z = cma_es(
    objective=acquisition,
    mean=mean0,
    covariance=cov0,
    budget=1000,
)

# 7. Decode, filter, evaluate true fitness
candidates = []
for z in candidates_z:
    ast = vae.decode(z)
    if ast.valid:
        f1 = render_canonical_f1(ast)
        candidates.append(f1)

candidates = deduplicate(candidates)
candidates = filter_too_similar_to_training(candidates, records)

true_results = []
for f1 in top_k_by_acquisition(candidates, k=50):
    y_true = framsticks_evaluate(f1)
    true_results.append((f1, y_true))

# 8. Active learning
D.extend(true_results)
retrain_or_finetune(vae, predictor, ensemble, D)
```

---

## 14. Minimalny plan implementacji

### Sprint 1 — analiza i parser

- [ ] policzyć rozkłady długości, fitnessu i liczby segmentów,
- [ ] zbudować parser `f1 → AST`,
- [ ] zbudować renderer `AST → canonical f1`,
- [ ] walidować poprawność przez Framsticks,
- [ ] wyciągnąć cechy ręczne.

Output:

```text
dataset_clean.parquet
f1_parser.py
features.py
eda_report.md
```

### Sprint 2 — fitness predictor

- [ ] baseline LightGBM/XGBoost,
- [ ] Transformer tokenowy,
- [ ] Tree/GNN predictor,
- [ ] ranking loss,
- [ ] walidacja top-k i split po klastrach.

Output:

```text
predictor_report.md
best_predictor.pt
```

### Sprint 3 — generative model

- [ ] grammar-aware decoder,
- [ ] autoencoder,
- [ ] VAE,
- [ ] length conditioning,
- [ ] testy interpolacji,
- [ ] test validity/reconstruction.

Output:

```text
generative_report.md
vae.pt
```

### Sprint 4 — optymalizacja

- [ ] CMA-ES na latencie,
- [ ] CEM na latencie,
- [ ] CEM na produkcjach gramatyki,
- [ ] acquisition z uncertainty,
- [ ] aktywna pętla true fitness.

Output:

```text
optimization_report.md
new_candidates.csv
active_learning_runs/
```

### Sprint 5 — porównanie

- [ ] porównanie z klasyczną ewolucją,
- [ ] seed-based statistics,
- [ ] wykres best-so-far,
- [ ] analiza jakości wygenerowanych genotypów.

Output:

```text
final_report.md
```

---

## 15. Decyzja projektowa: co wybrałbym najpierw

Rekomendowana kolejność:

```text
1. Parser + cechy + LightGBM/XGBoost predictor.
2. Transformer predictor jako baseline.
3. Tree/GNN predictor.
4. CEM na gramatyce f1 z surrogate fitness.
5. Dopiero potem Grammar/Tree VAE + CMA-ES.
```

Powód: model generatywny jest najtrudniejszy. Najpierw trzeba udowodnić, że fitness jest przewidywalny z genotypu/struktury.

Jeśli predictor nie ma sensownego Spearmana/top-k recall, latent optimization prawdopodobnie będzie wykorzystywać błędy modelu.

---

## 16. Najbardziej obiecujący wariant końcowy

Docelowo:

```text
F1 parser
→ AST/heterogeneous graph
→ Tree/GNN encoder
→ z ∈ R^64
→ grammar-aware length-conditioned decoder
→ fitness ensemble + uncertainty
→ CMA-ES/CEM on z
→ true evaluation in Framsticks
→ active learning loop
```

Acquisition:

```text
score =
    predicted_fitness_mean
  - uncertainty_penalty
  - out_of_distribution_penalty
  - length/depth_penalty
  + bounded_novelty_bonus
```

Wariant hybrydowy, który może okazać się praktycznie najlepszy:

```text
ML surrogate + istniejące operatory mutacji/crossover Framsticks + aktywne uczenie
```

czyli nie „ML zamiast ewolucji” w 100%, tylko:

> ML jako model krajobrazu fitnessu, filtr kandydatów i generator obiecujących regionów, z prawdziwą ewaluacją w symulatorze.

---

## 17. Referencje i materiały

1. Framsticks Manual / Artificial Life introduction — opis `f1`: `X` jako stick, `()` jako branch, struktura drzewa, modyfikatory i przykłady.  
   https://baibook.epfl.ch/exercises/evolutionMorphologies/Framsticks.pdf

2. Oficjalna strona Framsticks: opis formatów genotypów i `f1`.  
   https://www.framsticks.com/a/al_genotype.html  
   https://www.framsticks.com/a/al_geno_f1.html

3. Komosiński, Rotaru-Varga / Framsticks system — opis `f1` jako reprezentacji drzewiastej bez cykli w ciele oraz wpływu kodowania na ewolucję.  
   https://www.framsticks.com/files/common/Komosinski_FramsticksSystem_Kybernetes2003.pdf

4. Gómez-Bombarelli et al., *Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules* — przykład mapowania struktur dyskretnych do przestrzeni ciągłej z predyktorem własności i optymalizacją w latent space.  
   https://arxiv.org/abs/1610.02415

5. Dai et al., *Syntax-Directed Variational Autoencoder for Structured Data* — VAE z dekoderem ograniczanym składniowo i semantycznie.  
   https://arxiv.org/abs/1802.08786

6. Kusner et al., *Grammar Variational Autoencoder* — generowanie struktur przez produkcje gramatyki i optymalizacja w latent space.  
   https://arxiv.org/abs/1703.01925

7. Liu et al., *Constrained Graph Variational Autoencoders for Molecule Design* — grafowy VAE z twardymi ograniczeniami dziedzinowymi.  
   https://arxiv.org/abs/1805.09076
