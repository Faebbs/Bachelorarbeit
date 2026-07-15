# LLM-Kontext Bachelorarbeit — Krümmung & Kontraktionswellen (Tribolium)

> **Zweck dieses Dokuments:** Vollständiger Kontext aus allen bisherigen Claude-Chats (Claude Code in WSL/Windows + claude.ai Web), damit neue Chats sofort den Stand der Arbeit kennen. Grundgerüst zusammengestellt am 08.07.2026 aus 24 Claude-Code-Sessions (08.06.–07.07.) und 3 Web-Konversationen; **ergänzt um die Sessions 09.–12.07.2026** (Ei-Modell-Umbau, s. Ist-Box unten + §13). Dieses Dokument führt das kuratierte `Projektkontext_Bachelorarbeit.md` und die angehängten Sessions aus `Projektkontext.md` zusammen.
>
> **Wenn du (Claude) das hier liest:** Dies ist der Wissensstand, nicht zwingend der aktuelle Code-Stand. Wo konkrete Datei/Funktion/Zeile genannt wird, im Zweifel im Code verifizieren, bevor du darauf aufbaust. §1–§12 beschreiben den Stand vom **08.07**; der **aktuelle Ei-Code-Stand (12.07)** steht in der Box direkt nach dem TL;DR und in **§13**. Wo Parameter abweichen, gelten für das Ei die neueren Werte aus §13.

---

## 0. Kurzfassung (TL;DR)

Bachelorarbeit: **partikelbasierte GPU-Simulation, die untersucht, wie die Oberflächen-Krümmung die Ausbreitung mechanischer Kontraktionswellen in einem Zell-Epithel beeinflusst** — motiviert durch das Ei von *Tribolium castaneum* (Reismehlkäfer).

- **Framework:** YALLA (CUDA/C++), Kern in `foundation.cu`. Zellen = Partikel mit Morse-Wechselwirkung, auf eine Oberfläche „gepinnt".
- **Aktivierungs-Zyklus:** Zellen kontrahieren (Radius schrumpft) → ziehen Nachbarn → überschreiten Nachbarn eine Kraftschwelle, kontrahieren sie auch → **selbsterhaltende Welle**.
- **Zwei Experimente:** (1) **Kappen-Scan** — kontrollierte Kugelkappen variabler Krümmung; (2) **Ei-Scan** — Welle an Stellen unterschiedlicher lokaler Krümmung des echten Eis zünden.
- **Bisherige Beobachtung (kein gesicherter Schluss):** im aktuellen Modell **sinkt die gemessene Wellenausbreitung mit steigender Krümmung** (hält über die getestete Erregbarkeits-Achse). Arbeits*hypothese* zum Mechanismus: Krümmung ändert eher *wie viele* Zellen mitlaufen als *wie schnell*. Details, Vorbehalte und was noch nicht getestet ist → §5 (bewusst bias-arm gehalten).
- **Titel (in der Begleit-PPT bereits übernommen):** *"Modelling the Effect of Surface Curvature on Contraction Wave Propagation in Tribolium castaneum"*.
- **Biologische Motivation & Schlüssel-Literatur** (Kontraktionswellen in Trichoplax / Schwamm / Drosophila / Tribolium; mechanisches statt genetisches Feedback) → §12. Begleit-Präsentation: `BegleitPP.pptx` (82 Folien), inhaltlich mit diesem Dokument abgeglichen.
- **⚠️ Achtung Code-Stand:** Das **Ei-Modell** wurde nach Redaktionsschluss (08.07) in den Sessions 09.–12.07 stark umgebaut (Zell/Surface-Trennung, selbstkalibriertes Gleichgewicht, `relax`-Export, `force_type`-Schalter, andere Parameter); der **Kohäsions-Kollaps** ist weiterhin offen. Ist-Zustand → Box unten + **§13**. §1/§2/§11 beschreiben den 08.07-Stand. Der **Kappen-/Sheet-Teil (Exp. 1)** ist davon unberührt.

---

## ⚠️ Aktueller Ei-Stand (12.07.2026) — überschreibt §1/§2/§11 für MODEL_EGG

> Kompakter Ist-Zustand des Ei-Modells nach den Sessions 09.–12.07. Herleitung, Messwerte und der Verlauf dorthin stehen in **§13**. **Betrifft nur MODEL_EGG** — MODEL_CAP/MODEL_SHEET (Krümmungsstudie, Exp. 1) laufen unverändert mit den Werten aus §1/§2.

- **Zell/Surface-Trennung umgesetzt:** Zellen und Pinning-Surface kommen aus **getrennten Dateien** (nicht mehr beide aus `mesh_1`). Surface = feines Mesh mit Normalen (`initial_conditions_mesh_1.vtk` bzw. `target_surface_tribolium_1.vtk`); Zellen = dünnere, eigenständige Population (per FPS platziert / `horizontal_egg`, ~3000 statt 10000).
- **`d_r_e` (Laufzeit-Gleichgewicht):** die Paarkraft nutzt eine Device-Variable statt der Compile-Konstante; in der Relaxations-Phase auf den **gemessenen mittleren Nachbarabstand selbst-kalibriert** (`d_r_e = s − 2·r_start`, nach Jörns Prinzip) → Ruhelage kräftefrei, unabhängig vom Startabstand.
- **`force_type`-Schalter** (`'m'` Morse / `'l'` linear), zur Laufzeit im `simulation_step`-`switch`.
- **`relax`-Modus:** foundation.cu relaxiert selbst (echte Sim-Physik + Pinning + Selbstkalibrierung, **keine Welle**) und exportiert `egg_cells_relaxed.vtk`; die Wellen-/Scan-Läufe laden dieses File. **Workflow:** `./a.out … relax …` einmal → dann `scan`/`vtk`.
- **Neue Skripte:** `prepare_egg_cells.py` (Farthest-Point-Sampling der Startzellen, blue-noise, im Surface-Frame); `graph_Kraft.py` (erweitert: Fabians Morse + Jörns lineare + Marijas Polynom-Kraft überlagert).
- **Aktueller Tuning-Stand (in Arbeit, NICHT final):** `r_max=1.4`, `surface_stiffness=4`, `r_e=0.27`, `force_type='m'`, `activation_duration=40`.
- **Von §1/§2/§11 abweichende Ei-Werte:** `r_max` 2 → **1.4**; `surface_stiffness` 12 → **4**; `r_e` 0.3 → **0.27** (+ `d_r_e`-Selbstkalibrierung); `activation_duration` 10 → **40**; EGG lädt `mesh_1` **nicht mehr in `cells` UND `surface`**, sondern getrennt.
- **🔴 KERNPROBLEM offen — Kohäsions-Kollaps:** Morse mit `r_max=2` zieht das Sheet zusammen (Abstand 0.81 → 0.38, `avg_neighbors → 15`, überpackt); die NN-Selbstkalibrierung **verstärkt** den Kollaps (positive Rückkopplung: klumpt → kleinerer NN → kleineres `d_r_e` → mehr Klumpen). Betrifft auch die Welle → „Ei zerreißt langsam". Nächste Schritte → §13.5.

---

## 1. Das Simulationsmodell

### Framework & Umgebung
- **YALLA** (GPU/CUDA-Partikel-Framework für Zellen), Fabians Modell in `/home/fabian/Bachelorarbeit/yalla-main/foundation.cu` (+ `include/solvers.cuh`, `include/vtk.cuh`, `include/inits.cuh`, `include/polarity.cuh`, `include/dtypes.cuh`).
- Läuft in **WSL/Ubuntu**; Compile mit `nvcc -O3 yalla-main/foundation.cu -o a.out`. Python-Analyse im conda-Env **`bachelorarbeit`** (`/home/fabian/miniconda3/envs/bachelorarbeit/bin/python3`, hat numpy/matplotlib/vtk, **kein** scipy/meshio). Auf Windows heißt das Vedo-Env `python13Env`.
- Visualisierung überwiegend auf **Windows** (Vedo), Ordner `C:\Users\fabia\OneDrive\Dokumente\Uni\Bachelorarbeit\Vedo_visualisierung\`.

### Physik: Morse-Potential
Paarkraft zwischen zwei Zellen über den **Oberflächenabstand** `surface_dist = dist − (r_i + r_j)`:
```
phi = exp(−alpha·(surface_dist − r_e))
F   = −2·D_e·alpha·(1 − phi)·phi         (negativ = Anziehung)
dF  = r·F / max(dist, 1e-6)              (Kraftvektor, aufaddiert in d_force_accum[i])
```
Hergeleitet aus `V(r) = D_e·(1 − e^(−a(r−r_e)))²` via `F = −dV/dr` (siehe Anhang A).
- **Overdamped-Dynamik** (`dX/dt = F`, keine Trägheit), **Heun-Integrator**, `dt = 0.05`, `friction_on_background`.
- Parameter-Sets: Standard `r_e=1, D_e=2, alpha=1`; Ei-Modell `r_e=0.3`. `r_start = r_e/2`, `r_activated = r_start/2` → Sheet/Cap 0.5/0.25, Egg 0.15/0.075. `r_max=2` (Interaktions-Cutoff = `cube_size` des Grids). `dt=0.05`, `n_time_steps=2000`.

> **Update (12.07, s. Ist-Box + §13):** Für das **Ei** gelten inzwischen `r_e=0.27` (+ `d_r_e`-Selbstkalibrierung) und `r_max=1.4`; zusätzlich existiert eine lineare Kraft-Alternative (`force_type='l'`). Die Werte hier gelten weiter für Cap/Sheet.

### Zell-Zustandsmaschine (`activated`-Feld)
| State | Bedeutung | Radius-Verhalten |
|---|---|---|
| **0** | ruhend / erregbar | konstant |
| **7** | Delay (nach Zündung, vor Kontraktion, `activation_delay=2`) | konstant |
| **1** | **aktiviert / kontrahierend** (`activation_duration=10`) | schrumpft: `r ← r_activated + (r − r_activated)·e^(−r_decay_shrink)` |
| **2** | refraktär (`refractory_duration=200`) | wächst zurück: `r ← r_start + (r − r_start)·e^(−r_decay_grow)` |

- **Zwei getrennte Decay-Konstanten** (Code-Stand): `r_decay_shrink = 5.0` (schnelle Kontraktion, e^−5≈0.007/Step → fast sofort) vs. `r_decay_grow = 0.1` (langsame Erholung, e^−0.1≈0.905/Step). Biologisch asymmetrisch. (Frühere Chats nannten ein einzelnes `r_decay=0.9`; der Code hat sich weiterentwickelt.)
- **Start: alle Zellen in State 2 (refraktär)** → Steps 0–199 nicht feuerbar (Relaxation); Step 200 erregbar; **Step 220 Welle ausgelöst**; Nachbarschaft gemessen bei **Step 219**. Seed setzt Zellen direkt auf State 1; Wellen-/Spontanaktivierung geht 0→7→1→2→0.
- Analytische Referenz dieser Kurven: `graph.py` (`cell_radius_cycle`, geschlossene Lösungen der iterativen CUDA-Updates).

### Aktivierungs-Trigger: harter Threshold ODER Hill (per Flag)
- **Aktueller Code-Stand: `use_hill_function = false` → der HARTE Threshold ist aktiv** (`if (neighbour_ok && h_force_mag[i] > force_threshold)`), `force_threshold = 0.15` (via argv[4] überschreibbar).
- Die **Hill-Funktion** ist implementiert, aber per Flag **abgeschaltet**: `P(F) = F^n/(K^n+F^n)`, K = force_threshold (Halbaktivierungskraft, P=0.5 bei F=K), `n_hill = 2` (stochastisch via `rand()/RAND_MAX < P`). Zum Aktivieren `use_hill_function = true` setzen. (Beide Zweige teilen sich die `neighbour_ok`-Bedingung des Propagation-only-Modus.)
- **Wellenmechanismus:** aktivierte Zelle schrumpft → `surface_dist` zu Nachbarn wächst → Nachbarn asymmetrisch angezogen → deren `force_accum`-Betrag steigt → Schwelle überschritten → zünden → Welle.
- **Threshold-Herleitung** (physikalisch, Anhang B): |F(t)| einer schrumpfenden Zelle auf ruhenden Nachbarn. Mit (r_e=1,D_e=2,alpha=1,r_decay=0.9): F(1)=0.479, F(2)=0.641, F(∞)=0.691, Max=1.0. Threshold steuert **Wellengeschwindigkeit** (niedrig=schnell). Exakte analytische Herleitung unmöglich (Zelle i bewegt sich mit) → **empirisch kalibrieren** empfohlen.

### Solver-Wahl: **Gabriel-Solver** (biologisch begründet)
Drei Solver mit gleichem Integrator, nur Nachbarsuche unterschiedlich: Tile (alle Paare, O(n²)), Grid (27 Nachbar-Cubes, O(n)), **Gabriel (Grid + Gabriel-Graph-Filter)**. Gewählt: Gabriel, weil
- Zellkräfte sind **Kontaktkräfte** → „verdeckte" Paare (dritte Zelle dazwischen) werden gefiltert → echte Kontakt-Topologie (Teilmenge der Delaunay-Nachbarschaft, `solvers.cuh:573`).
- **Dichte-adaptiv** (kein willkürlicher Cutoff), keine falschen „Abkürzungen durchs Gewebe" auf gekrümmter Fläche.
- Zitierbar: **Delile et al. 2017 (MecaGen)**, **Marin-Riera et al. 2016**.
- Stellschraube `gabriel_coefficient` (Default 0.8). Für Grid-basierte Kraftsummation über *viele* Nachbarn (großes r_max) stattdessen `Grid_solver` (linear, kein Buffer/Sort).

---

## 2. Drei Geometrie-Modelle (`#define`, `#error`-Guard erzwingt genau eines)

| Modell | Geometrie | Pinning | r_e | Zweck |
|---|---|---|---|---|
| **MODEL_EGG** | echtes Ei-Mesh aus VTK (10000 Punkte + Normalen) | nächster Mesh-Punkt + dessen Normale | 0.3 | reales Ei, lokal variable Krümmung |
| **MODEL_SHEET** | flaches Rechteck (Höhenfeld-Kappe) | Höhenfeld-Normale (finite Differenz) | 1.0 | flacher/schwach gekrümmter Referenzfall |
| **MODEL_CAP** | uniform gekrümmte Kugelkappe (Exponential-Map) | exakte radiale Kugel-Projektion | 1.0 | **Hauptwerkzeug Krümmungsstudie** |

### Surface-Pinning (nach Marijas `cells_to_surface.cu`)
- Rückstellkraft `F = −k·(r·n)·n` (r = Versatz zur Fläche, n = Flächennormale, k = `surface_stiffness = 12`). Projiziert nur die **senkrechte** Bewegungskomponente weg → Zellen bleiben auf der Fläche, gleiten frei entlang.
- **EGG:** `cells` (beweglich, Gabriel_solver) + `surface`+`surface_norm` (eingefrorene Kopie, Tile_solver, nie integriert) beide aus `initial_conditions_mesh_1.vtk`. Zellen starten AUF der Fläche → Pinning-Kraft anfangs ≈ 0. Brute-Force-Nächster-Punkt-Suche O(n·n_grid). **Kein „Snapping"**, aber effektive Fläche = facettierte Näherung (stückweise Tangentialebenen); Genauigkeit ∝ Gitterdichte. (SHEET/CAP nutzen analytische Normalen → glatt.)

> **Update (12.07, s. Ist-Box + §13):** Genau dieses „beide aus `mesh_1`" ist die Ursache der Überkompression/No-Motion (§13.1) und wurde aufgelöst: **Zellen und Surface kommen jetzt aus getrennten Dateien** (Surface = feines Mesh; Zellen = dünnere FPS-/`horizontal_egg`-Population). `surface_stiffness` fürs Ei inzwischen **4** (Jörn nutzt 3).

### CAP-Modell: Exponential-Map (Kern der Krümmungsstudie)
Flacher Abstand vom Zentrum = **geodätischer Abstand vom Pol**:
```
θ = s/R,   x = R·sinθ·cosφ,   y = R·sinθ·sinφ,   z = R·(1 − cosθ)
```
→ ganze Kappe gleichmäßig gekrümmt (Gauß-Krümmung **K = 1/R²**), geodätische Patch-Größe für jedes R gleich. Zellen als kreisförmige Scheibe (`random_disk`, in y-z-Ebene) verteilt, dann aufgewickelt → radiale Welle vom Pol, keine Ecken-Artefakte. Reproduzierbar über festen Seed.

### N↔R-Kopplung (SCHLÜSSELIDEE: 1-Parameter-Scan)
Problem: fester Patch + kleines R → „Wrap-around" (Patch > Kugel). N frei → 2D-Scan.
Lösung: **feste Winkelausdehnung θ_max = 75°**, N daraus:
```
N = 0.9069·(R·θ_max/r_e)²    →  N ∝ R²    →  N(K) = 0.9069·θ_max²/(r_e²·K) = C/K,  C ≈ 1.554
```
(θ_max **in Bogenmaß** = 1.309 rad; 0.9069 = π√3/6 hexagonale Packungsdichte.)
→ Scan bleibt 1-D (nur R/K), kein Wrap, hält **Zelldichte konstant** (~6 Nachbarn), sonst mischt sich Krümmungs- mit Dichteeffekt. Dimensionsloser Parameter = R/r_e. 75° deckt 37% der Kugel ab; θ: Pol=0°, Äquator=90°, Antipode=180°.
- **Wichtig für Auswertung:** da N mit R variiert → **intensive Größen** (Anteile, nicht Absolutzahlen).
- Initiale Aktivierung = fester Bruchteil `activation_fraction·N` (**Code-Stand: 0.01 = 1%**; der Code-Kommentar „5 %" ist veraltet) nächste Zellen am Pol (skaliert mit Patch), Floor `activation_min = 3`.

---

## 3. Krümmungsmessung des echten Eis

`initial_conditions_mesh_1.vtk` (intern `tribolium_surface_1`): reines Oberflächen-Mesh, **10000 Punkte + `NORMALS polarity`**, keine Flächen-Konnektivität, kein force_mag/activated/radius. Bounding-Box ~23×48×24 (y = Längsachse).

### Zwei Verfahren (`analyse/egg_curvature.py`)
Beide schätzen pro Punkt lokal aus den 18 (`K_NEIGHBORS`) nächsten Nachbarn im Tangentialframe (Normale = z-Achse, u/v per Gram-Schmidt):

1. **Jet-Fitting (Monge-Patch)** — `curvatures()`: Least-Squares-Paraboloid `z = ax²+bxy+cy²+dx+ey+g`; f_xx=2a, f_xy=b, f_yy=2c; `K = (f_xx·f_yy − f_xy²)/(1+f_x²+f_y²)²`, `H = …` (Anhang C). Rigoroser Name: „jet fitting" (Cazals & Pouget, CGAL Jet_fitting_3).
2. **Weingarten-Map / Shape Operator** — `curvatures_weingarten()`: `dn = −S·dp`, S symmetrisch 2×2 per Least-Squares; `K = det(S) = ac−b²`, `H = ½tr(S) = ½(a+c)`; κ1,2 = Eigenwerte.

### Welches Verfahren — **Weingarten primär**
- Diagnose der echten Daten: Normalen glatt (median 3.7° zu Nachbarn, 100% nach außen), aber **patchy** (NN-Abstand 0.07…0.54, Faktor ~8).
- Weingarten robuster bei patchy Sampling mit guten Normalen (fittet 3 Koeff. aus Normalendifferenzen, 1. Ordnung dn~κr; Monge 6 aus z-Auslenkungen, 2. Ordnung, schlecht konditioniert bei Dichteunterschieden).
- Validierung Kugel R=5 (K soll 0.04): Weingarten exakt (0.0400), Monge 0.0408. Auf echten Daten Korrelation **0.966** → beide vertrauenswürdig, Divergenz am ehesten an Polspitzen.
- METHOD="weingarten" primär (ins VTK + Metriken), Jet als Cross-Check. K_NEIGHBORS=18. Ein **Dispatcher** `estimate_curvatures(pts, normals, idx, method=…)` wählt zur Laufzeit — gesteuert aus `egg_curvature.py` (Konstante `METHOD`) *und* aus dem Scan (Konstante `CURV_METHOD`). Weingarten stuft **98.1 %** der Punkte als konvex ein (jet: 96.6 %), also weniger fälschlich-negative K an dünnen Stellen.

### Weitere erwogene Krümmungs-Verfahren (verworfen — Kontext für die Methodenwahl)
Aus der Krümmungs-Session (05.07): Neben Jet-Fitting und Weingarten wurden weitere Verfahren diskutiert und **verworfen** —
1. **Impliziter Quadrik-Fit** (ohne Normale): lokale Quadrik `xᵀAx + bᵀx + c = 0`, Normale/Krümmung aus Gradient+Hesse. Rotationsinvariant, braucht keine gespeicherten Normalen — aber hier unnötig, da gute Normalen vorliegen.
2. **Curvature-Tensor (Taubin / Cohen-Steiner):** pro Kante Normalkrümmung `k_ij = 2·n_i·(p_j−p_i)/‖p_j−p_i‖²`, gemittelt in einen 3×3-Tensor → Eigenwerte = Hauptkrümmungen. Stabil bei Rauschen; sinnvolle robuste Alternative.
3. **Erst Mesh rekonstruieren, dann diskrete Krümmung** (Winkeldefekt `K_i=(2π−Σθ_j)/A_i` bzw. Cotangent-Laplacian für H). **Verworfen:** verschiebt das Problem nur auf die fehleranfällige Rekonstruktion (Ball-Pivoting-Radius / Poisson-Tiefe tunen); diskreter Winkeldefekt ist rauschempfindlich; **versagt genau an den Eierspitzen** (hohe Krümmung *und* dünn gesampelt); Poisson zerstört die Punkt-Korrespondenz (neue Vertices) → Rückmapping-Fehler; Ränder/Löcher liefern Müll.
4. **PCA / „surface variation"** (`σ = λ0/(λ0+λ1+λ2)`): nur grober Krümmungs-**Indikator**, nicht echtes K/H. Nur für schnelle qualitative Karten.
- **ConvexHull ≠ Rekonstruktion:** die schöne Vedo-`ConvexHull` (`egg_surface.py`) taugt nur zur Optik — sie verbindet nur die äußersten Punkte und liefert per Konstruktion überall K≥0 (sieht keine Sättel/Einschnürungen). Für quantitative κ1,κ2,K,H ist der Punktwolken-Fit die ehrliche Wahl; man kann die Hull aber mit den Fit-Werten einfärben (Beste aus beiden Welten). `egg_monge_surface.py` zeichnet dagegen die **Paraboloid-Kacheln** pro Punkt — die „Fläche, wie der Monge-Patch sie sieht".
- **Zitat für die Methode:** Cazals & Pouget „osculating jets" (CGAL Jet_fitting_3) für den Fit; anschaulich Keenan Crane „Discrete Differential Geometry" (CMU 15-458) und die Iowa-State-Notizen (Yan-Bin Jia).

### Warum Gauß-K statt mittlerer Krümmung H (Entscheidung begründet)
Die Observable (Wellenfront-Größe vs. Startort) ist **intrinsisch** → K ist sachlich besser begründet als H:
- **Jacobi-Gleichung / geodätische Abweichung:** benachbarte Geodäten gehorchen `J̈ + K·J = 0` — Divergenz/Konvergenz der Ausbreitungswege hängt **allein von der Gauß-Krümmung K** ab. Wo K>0 (Spitze) konvergieren die Wege → Front „fokussiert", gleichzeitig aktiver Umfang klein (genau der gemessene Effekt).
- **Theorema Egregium:** K hängt nur von der Metrik ab, nicht von der 3D-Einbettung — und die Welle „kennt" nur Abstände auf dem Ei.
- **Zylinder-Test:** rollt man ein flaches Blatt zum Zylinder, ändert sich H (0 → 1/2R), aber K bleibt 0; eine Welle läuft exakt gleich. H „sähe" also einen Unterschied, den die Welle nicht spürt. H wäre nur richtig, wenn der Mechanismus an die Einbettung koppelt (Helfrich-Biegeenergie ∝ (2H)², Kortex-Spannung, Laplace-Druck, BAR-Proteine) — das ist hier nicht modelliert.
- **Einschränkung:** auf dem konvexen Ei sind K und H stark korreliert (beide groß an der Spitze); sie trennen sich vor allem am **Bauch/Äquator** (lokal fast zylindrisch: eine Hauptkrümmung groß, eine ≈ 0 → H moderat, K klein).

### Ergebnis: Krümmung des Eis
- **96.6% konvex** (K>0, jet; Weingarten 98.1%), ~3% konkav = Unregelmäßigkeiten. Äquivalent-Kugelradius `R_eq = 1/√K`.
- **Effektiver R-Bereich ≈ 6…50** (Median ~20). Spitze Enden (y≈±23): R_eq ~6–9 (hohe Krümmung); flache Mitte ~50.
- K-Perzentile (×10⁻³, Weingarten): pct5=0.50, 50=2.19 (Median, R≈20.5), 90=9.88, 99=16.99 (R≈7.6). Verteilung stark rechtsschief (Modus ~0.00122 < Median).
- These-Brücke: `Vorhersage fürs Ei = ∫ Kappen-Ergebnis(R)·Egg-Häufigkeit(R) dR`.

---

## 4. Die zwei Experimente

### Experiment 1 — Kappen-Scan (`parameter_scan.py`, MODEL_CAP)
Kontrolliert Krümmung K=1/R² global; misst Wellenausbreitung + Nachbarschaft.
- **Argument-Schema:** `./a.out <R> <seed> <mode> <threshold>` (Defaults 15/42/vtk/0.15). Config-Zeile beim Start.
- **Scan-Modus** (`scan`): **keine VTKs**, Metriken intern, eine `SCAN_RESULT`-Zeile. ~3 s/Run unabhängig von R (löst den Speicher-Engpass: VTK-Modus schrieb 2001 Dateien/Run, bis 379 MB). `vtk`-Modus für Visualisierung.
- **Metriken:** `frac_activated` = Anteil je in State 1 gewesener Zellen (klebrig, kumulativ = Wellenreichweite; Seed zählt mit → Minimum = Seed-Anteil, aktuell ~1%). `avg_neighbors` im relaxierten Zustand (Zählradius 1.3× Gleichgewichtsabstand, **Randring 3·r_e ausgeschlossen**; `n_interior`).
- **Statistik/Robustheit:** 10 Seeds (Fehlerbalken); Nachbarschaft threshold-unabhängig (t=219) → gepoolt. `force_threshold` ∈ {0.10, 0.125, 0.15, 0.175, 0.2, 0.3}. R-Gitter gleichverteilt 6–50 (Schritt 4) + flache Referenz 100/200. `ONLY_GRAPH`-Flag trennt Scan/Plot. Live-ETA.
- **Propagation-only-Modus (WICHTIG):** Auf gekrümmter Fläche geometrische **Frustration** (Theorema Egregium) → krümmungsabhängige Restkräfte → spontanes Feuern verfälscht Messung. Fix (Fabians Idee): Aktivierung nur wenn `force_mag > threshold` **UND ≥1 Nachbar gerade in State 1** (nicht refraktär). Biologisch = propagierendes erregbares Medium (Relay), Spontanaktivierung wäre Schrittmacher. Nur im Scan-Modus aktiv (`require_active_neighbor`), VTK-Modus behält Spontanaktivierung (gewolltes Modell). → nur Seed startet ohne aktiven Nachbarn → frac_activated = reine Wellenreichweite.
- **Korrektheits-Fix:** `grid_size` mit R skalieren (sonst Zellen außerhalb ±50-Gitter → Crash/falsche Nachbarn bei großem R).

### Experiment 2 — Ei-Scan (`egg_activation_scan.py`, MODEL_EGG)
Zündet Welle an Stellen unterschiedlicher **lokaler** Krümmung des echten Eis.
- **Metrik:** `frac_peak_active` = max. Anteil **gleichzeitig** in State 1 (Wellenfront-Breite), weil kumulative Reichweite auf geschlossenem Ei bei ~1 sättigt.
- **Config:** CURV_METHOD="weingarten", THRESHOLDS=[0.125,0.15,0.175], FIXED_FRACTION=0.01 (1% Stimulus), N_PER_BIN=5, LOC_PERCENTILES=[99,95,90,80,70,60,50,40,30,20,10,5], LOC_POOL=60. Modus `vtk_prop` (VTK + propagation-only).
- **Gruppe A** (Hauptmessung): Seed über 12 Krümmungs-Perzentile × 5 Punkte/Bin × 3 Thresholds. **Gruppe B** (Kontrolle): Stimulus-Größe variiert an 3 festen Positionen → schließt die Stimulusgröße als Ursache aus (nur diesen einen Faktor, nicht andere).
- **Seed-Auswahl per Farthest-Point-Sampling (FPS):** reine K-Nähe klumpt räumlich (min. Abstand 0.6–2) → Cluster-Artefakt. FPS: aus 60 K-nächsten Punkten greedy 5 räumlich maximal gestreute (Start bester K-Treffer, dann iterativ größter Abstand zum Gewählten). Min. Abstand → 5–18, **k-spread** (relative Std der 5 K-Werte = wie sauber „gleiches K") <0.6% (außer pct99: 4.3%, pct5: 2.5%).
- Repräsentanten-IDs (12 Perzentile, nach FPS-Update leicht anders als hier): grob `[5, 9403, 8901, 3084, 7172, 5016, 7349, 3707, 6246, 9443, 8841, 4036]`.
- **99% statt 100%:** absolutes Max = oft Mesh-Ausreißer/Artefakt.

---

## 5. Hauptbefunde (Beobachtungen — bewusst mit Vorbehalt)

> **Gegen Über-Interpretation (bitte beim Weiterarbeiten beachten):** Das Folgende sind **Beobachtungen des aktuellen Modells an einem eingefrorenen Arbeitspunkt**, keine gesicherten biologischen Aussagen. Sie hängen an konkreten Modellentscheidungen, die den Befund mittragen — u.a. Morse-Kohäsion mit langer Reichweite, Hill-Aktivierungsschwelle, die **Propagation-only-Regel** (die „Ausbreitung" ein Stück weit mitdefiniert), das Pinning als facettierte Näherung, die Weingarten-Krümmungsschätzung und die N∝R²-Kopplung. Ändert sich eine davon, kann sich das Bild verschieben. Die **Zahlen** sind das Gemessene; die **Deutung** ist offen. Als Hypothesen mit Stützung lesen, nicht als Konklusion, durch die alles Weitere gedeutet wird — insbesondere kein neues Ergebnis so lesen, dass es „die These bestätigt", ohne die Alternativen geprüft zu haben.

1. **Konsistentester Trend: mit steigender Krümmung sinkt die *gemessene* Wellenausbreitung.** Kappen-Scan: ~0.98 (flach) → ~0.21–0.25 (R=6). Ei-Scan: Wellenfront ~7.8% (flach, R≈30) → ~4.7% (spitz, R≈7.6), Faktor ~1.6. Der Trend zieht sich durch beide Experimente — die belastbarste Einzelbeobachtung. Offen: ob er über den geprüften R-/K-Bereich hinaus hält und wie stark er an der genauen Metrikdefinition hängt.
2. **Der Trend überlebt die getesteten Störachsen — aber es sind wenige.** Geprüft: `force_threshold` (Erregbarkeit, 6 Werte) und Stimulusgröße (Kontrolle). Threshold verschiebt nur das Niveau, nicht den Trend; bei extremer Krümmung konvergieren die Kurven (threshold-unabhängiges Versagen). Das ist eine gezielte Robustheitsprüfung *entlang weniger Achsen*, **keine** allgemeine Robustheit: Kontraktionsstärke (r_activated/D_e), activation-/refractory-Dauer, Pinning-Steifigkeit, `gabriel_coefficient`, Dichte-Annahmen wurden nicht variiert. Die Stimulus-Kontrolle schließt genau **ein** Artefakt aus, nicht „alle".
3. **Mechanismus = *Hypothese*, noch nicht direkt gemessen:** Krümmung wirke über die *Geometrie der Front* (wie viele Zellen), nicht das *Tempo*. Stützung: t_peak variiert nur schwach (347→368) und Bertrand–Diguet–Puiseux (geodätische Umgebungen schrumpfen mit K). **Aber:** die lokale Frontgeschwindigkeit ist noch nicht direkt gemessen (activation_time-Analyse steht aus, s. §10) — erst das würde die Hypothese wirklich testen. Zeit-bis-Sättigung wurde als Metrik verworfen (Confound: geodätischer Abstand Seed→Antipode). Andere Deutungen sind nicht ausgeschlossen.
4. **Mittlere Nachbarschaft: schwacher, verrauschter Abwärtstrend** (~5.87 flach → ~5.31 gekrümmt), im Mittelfeld überlappende Fehlerbalken. Ehrlich „tendenziell kleiner", **nicht** „streng monoton". Die Deutung (topologische Defekte / 5er-Nachbarschaften) ist plausibel, aber der Absolutwert hängt am Zählradius → nur *relativ* (gekrümmt vs. flach) vergleichen, nicht gegen den Idealwert 6.
5. **Bewusst dokumentierte Gegenevidenz zur einfachen Deutung** (nicht wegzuerklären, gehört zum Befund): K allein beschreibt die Ausbreitung *nicht* vollständig. Bei gleichem K, aber anderer Position/Anisotropie streuen die Ergebnisse deutlich (große Fehlerbalken in der Mitte pct40–60, klein an den Extremen) → beide Hauptkrümmungen (κ1,κ2) und der globale Kontext spielen mit. Der Fehlerbalken ist hier selbst eine Messgröße, kein bloßes Rauschen.

---

## 6. Dateien-Landkarte

> Wie diese Dateien konkret zusammenspielen (Framework, Programmablauf, Scans) → **§11**. Neue Dateien aus den Sessions 09.–12.07 → **§13**.

**Simulation (WSL, `/home/fabian/Bachelorarbeit/`):**
- `yalla-main/foundation.cu` — Kern: Modelle, Morse-Kraft, State-Machine, Pinning-Kernel, Threshold/Hill-Aktivierung, Scan-Modus, Metriken. *(12.07: + `force_type`-Schalter, `d_r_e`, `relax`-Modus, Zell/Surface-Trennung — §13.5.)*
- `yalla-main/include/solvers.cuh` — Gabriel/Grid/Tile-Solver, `compute_cube_gabriel` (Buffer, s.u.).
- `yalla-main/include/vtk.cuh` — `read_positions`, `read_normals`. `inits.cuh` — `random_disk`. `polarity.cuh` — `bending_force`.
- `graph.py` — Radius-/Kraft-/Hill-Kurven (analytische Referenz + Plots). **`graph_Kraft.py`** (neu, 12.07) — Morse + Jörns lineare + Marijas Polynom-Kraft überlagert.
- `analyse/egg_curvature.py` — Krümmung (Weingarten primär + Jet-Cross-Check, Dispatcher `estimate_curvatures`), K-Histogramm + 3D-Scatter.
- `analyse/egg_seed_kspread.py` — k-spread-Validierung der Ei-Seed-Auswahl.
- `parameter_scan.py` — Kappen-Scan (Exp. 1). `egg_activation_scan.py` — Ei-Scan (Exp. 2). `main.py` — Einzellauf + Video.
- **`prepare_egg_cells.py`** (neu, 12.07) — FPS-Platzierung der Startzellen auf der Surface. **`egg_peak_timing.py`** (neu) — Peak-/Timing-Auswertung. `shell_force_analysis.py` — analytische Schalen-Radialkraft (Kollaps-Analyse).
- **Output-VTKs (neu, 12.07):** `egg_cells_relaxed.vtk` (relaxierte Zellen, von den Wellen-/Scan-Läufen geladen), `egg_cells_fps.vtk` (FPS-Platzierung vor Relaxation), `egg_curvature.vtk` (Krümmungsfelder).

**Visualisierung (Windows, `…\Vedo_visualisierung\`):**
- `egg_surface.py` — Gridpunkte + Normalen + ConvexHull, Seed-Punkte nach Krümmung eingefärbt (diskrete 12-Perzentil-Skala), Taste `v`. *(09.07: + `DATASET`/`SHOW_SEEDS`-Schalter für alle Meshes, robustes Normalen-Handling — §13.3.)*
- `Finale_Visualisierung_Ultimativ_fertig.py` (force_mag-Färbung, variabler Radius) + `…_ohne_Radius.py` (fester Radius, neighbours-Färbung). Prefetch-Cache, Autoplay (`v`), Navigation (`,`/`m`), Timestep-Direktsprung (Zahl eintippen).
- `load_single_vtk.py`, `egg_monge_surface.py` (Paraboloid-Kacheln), `force_over_time.py`.
- Auch `vedoBrowser.py`, `vedoMeshBrowser.py`, `vedoBrowserSimple.py`, `visualisation_video.py` (frühere Iterationen).

**Daten/Output:** `initial_conditions_mesh_1.vtk` (Ei-Mesh). Sim schreibt `output/file_{i}.vtk` (Felder: neighbours, activated, radius, force_mag). Plots in `Graphen/`. Präsentation/Text im Windows-Ordner `…\Bachelorarbeit\` (`Müller_Bachelorarbeit.tex`, `WebChats\`, `Projektkontext_Bachelorarbeit.md`). Vollständige Datensatz-Landkarte → **§13.3**.

**Nebenprojekt (`Joern_Projekt/`, `Joern_Projekt_Neu/`):** Marijas `src/Marija/cells_to_surface.cu` (CUDA-Zellsortierung auf Oberfläche, Ursprung der Pinning-Mechanik: 200 Zellen, `pw_int` Zell-Zell + `punctual_force` Oberfläche, 31228 Gitterpunkte). `src/visualization.py`, `analyse/visualisation_video.py`. `Joern_Projekt_Neu/` = Jörns aktuelles Projekt (Vorlage für Zell/Surface-Trennung + Selbstkalibrierung, §13.5).

---

## 7. Bugs & Lektionen (damit sie nicht wiederkehren)

- **`neighbour_id[100]`-Buffer-Overflow** in `compute_cube_gabriel`: Gewebe kontrahiert → >100 Nachbarn in r_max → Speicher-Korruption → `cudaErrorIllegalAddress`. Nachbarzahl ~ Dichte·(4/3)π·r_max³ (r_max=2 → ~155, schon im Normalzustand!). **Cube-Shrinking hilft nicht** (Zahl hängt an Dichte×Reichweite; cube_size muss ≥ r_max). Fix: Buffer auf 256; für großes r_max `Grid_solver` statt `Gabriel_solver` (linear, kein Buffer/O(n²)-Sort).
- **NaN durch 0/0** bei `dF = r·F/dist` wenn dist→0 → `dF = r·F/max(dist, 1e-6)`.
- **Race Condition** `d_force_accum[i] += dF` → `atomicAdd` pro Komponente.
- **case-7-Logikfehler** (zwei unabhängige `if` → Oszillation 0↔7 → NaN) → `else if`.
- **Copy-Paste-Bug** `h_radius = ` (leer) verkettet über Kommentar → Radius = Timer-Wert → Zellen fliegen raus.
- **grid_size fest ±50** bei großem R → Crash/falsche Nachbarn → mit R skalieren.
- **numpy 2.x** `np.linalg.solve` behandelt 2D-b als Matrix → auf (n,6,1) bringen (`ATb[..., None]`, danach `[..., 0]`).
- **subprocess** `["./a.out 10"]` als ein String → Programm+Argument trennen.
- **Deutsche Kommas** `0,125` in Python-Liste → 2 Elemente → Punkte nutzen.
- **Windows:** `clear`→`cls` (shell=True); ffmpeg fehlt oft → cv2-Backend; matplotlib+VTK-GUI-Konflikt → matplotlib in eigenem Prozess (multiprocessing).
- **Vedo Pol-Artefakt:** `reconstruct_surface(radius=2.0)` überwölbt Pole → `ConvexHull` (Ei konvex). Simulation nutzt nie rekonstruierte Fläche.
- **Metrik ist NICHT aktivierungsunabhängig** (Fabians Korrektur): Nachbarschaft bei t=219 liest Live-Positionen; Spontanfeuern vor 219 stört Packung krümmungsabhängig → propagation-only misst sauber.
- **Namensfalle `r_e` (12.07):** `r_e` ist **nicht** der Ruheabstand. `surface_dist = dist − 2·r_start = dist − r_e`; Gleichgewicht bei `surface_dist = r_e` ⇒ **`dist = 2·r_e`**. Bei `r_e=0.3` liegt das Morse-Gleichgewicht also bei Mittelpunktsabstand 0.6, nicht 0.3 (§13.1).

---

## 8. Schalen-Kollaps-Problem (physikalische Kern-Erkenntnis, aktuell wieder akut)

Frühe Beobachtung: Das Ei (hohle Schale, eine Zelllage) **schrumpft unaufhörlich**. Ursache: Morse-Potential stark kohäsiv mit langer Reichweite (F(dist=2)≈−0.74) → (1) Schale ist kein Energieminimum (Klumpen hat mehr Bindungen), (2) Kohäsion = Oberflächenspannung, Laplace-Druck nach innen, kein Gegendruck (kein Dotter). **Geometrischer Effekt, kein Parameter-Artefakt** — kein Kraft-Parameter rettet die Schale (quantitativ per `shell_force_analysis.py` gezeigt). Größeres r_max macht es *schlimmer* (mehr Oberflächenspannung).
- Lösungen: **A) Druckterm** `P = k·(V₀−V)/V₀`, radiale Auswärtskraft (Dotterdruck, selbstkorrigierend, formbewahrend) — umgesetzt (`pressure_strength`). Grenze: erhält Volumen, nicht Form (längliche Enden fallen ein). **B) Biegesteifigkeit** (`bending_force` via apikal-basale Polarität `Po_cell`, YALLA-nativ, `epithelium.cu`) — krümmungssensitiv, richtige Lösung für Formerhalt, aber float3→Po_cell-Umbau. (Für die spätere Krümmungsstudie wurde stattdessen auf Pinning + analytische Kappen ausgewichen, was das Kollaps-Problem umgeht.)

> **Update (09.–12.07, s. §13):** Am **Ei-Modell** ist der Kollaps **nicht** umgangen, sondern **weiterhin das offene Kernproblem** — er wurde in den Sessions 09.–12.07 aktiv bearbeitet (Diagnose §13.1, Morse→linear-Versuch §13.4, Selbstkalibrierung §13.5). Neue Messung: Morse `r_max=2` zieht das relaxierte Sheet 0.81 → 0.38 zusammen, `avg_neighbors → 15`. Die NN-Selbstkalibrierung **verstärkt** ihn (positive Rückkopplung). Aussichtsreichster Hebel laut §13.4: **Gleichgewichts-Offset an den Zellabstand anpassen** statt das Kraftgesetz zu tauschen; alternativ gedeckelte/sättigende Adhäsion (= Morses Form).

---

## 9. Titel & Präsentation

- **Alter Titel:** „Modelling the Impact of contraction waves on membrane rupture in Tribolium castaneum" — passt nicht mehr (Fokus jetzt Wave-Spreading statt Membranriss).
- **Gewählt (Stand Begleit-PPT, Folie 1):** *„Modelling the Effect of Surface Curvature on Contraction Wave Propagation in Tribolium castaneum"* — die in den Chats empfohlene Option 1 ist bereits als Titel übernommen. (Untertitel „…: A Cell-Based Simulation Study…" wäre optional.)
- **Folien-Prinzip** (aus mehreren Reviews): eine Signatur-Gleichung sichtbar (z.B. `dn=−S·dp`, `z=ax²+…`, `N=C/K`), Rest in Sprache, Rechnung ins Backup. Ergebnis-Folien tragen die **Kernaussage** (nicht Methode wiederholen). „Money Shots": geklumpte vs. gestreute Seeds nebeneinander; Wellen-Animation.
- **Wiederkehrende Folien-Fehler** (checken): θ_max in Bogenmaß (1.309, nicht 75); „sphere cap" nicht „sphere"; Monge `K=4ac−b²` vs. Weingarten `K=ac−b²` (Faktor 4 vom Ableiten); Matrix `[[a,b],[b,c]]` symmetrisch; kleines k=18 (Nachbarn) vs. großes K (Krümmung); k-spread-Werte sind ×10⁻³; englische Schreibweise (Punkt statt Komma, „of" statt „over").

---

## 10. Offene Fäden / mögliche nächste Schritte

- **Per-Zelle-Aktivierungszeit** (`activation_time`-Array in foundation.cu, init −1, beim ersten State-1-Übergang setzen, am Ende dumpen) → **lokale Frontgeschwindigkeit** `v = |x_i−x_j|/|t_i−t_j|` je Nachbarpaar, gegen lokale Krümmung K_i auftragen. Würde die These „Krümmung ändert Geometrie der Front, nicht Tempo" direkt belegen. *Vorgeschlagen, noch nicht umgesetzt.*
- **Per-Zelle-Analyse mit beiden Hauptkrümmungen** (κ1,κ2 statt nur K), da gleiches K bei verschiedener Anisotropie unterschiedlich wirkt.
- Re-Entry-Ausreißer (~2%, t_peak>432) filtern.
- Dauerkontraktion der Schale separat sauber lösen (Druckterm A + evtl. Bending B), falls das Ei-Modell wieder ohne Pinning laufen soll.
- Scan mit neuer FPS-Seed-Auswahl neu laufen lassen (alte CSV/Plots veraltet); Grafiken auf %-Achsen neu rendern.
- **Neu (12.07):** Sheet stabilisieren UND Welle propagieren lassen — kurzes `r_max` auf richtig verteiltem Sheet, Kalibrierungs-Rückkopplung stoppen, gesättigte/gedeckelte Adhäsion (Details §13.5).

---

## 11. Wie der Code funktioniert (yalla-main + foundation.cu + Scans)

> Direkt aus dem Code gelesen (08.07.2026). Bei Abweichungen zwischen diesem Abschnitt und den chat-basierten Abschnitten oben gilt hier der Code. **Für das Ei gilt zusätzlich §13 (Stand 12.07):** dort sind `force_type`, `d_r_e`, `relax`-Modus und die Zell/Surface-Trennung ergänzt, die diesen 08.07-Snapshot fürs Ei teilweise überschreiben.

### 11.1 yalla-main — das Framework (Groundwork)
**YALLA** („yet another parallel agent-based model for morphogenesis", Germann et al. 2019, *Cell Systems*, DOI 10.1016/j.cels.2019.02.007, MIT-Lizenz, github.com/germannp/yalla). Agentenbasiertes Zellmodell für Morphogenese: Zellen = Punkte mit **paarweisen** Wechselwirkungen (+ optional Spin-artige Polaritäten für Epithelien), GPU-parallelisiert. Kompilieren: `nvcc -std=c++… -arch=sm_XX model.cu` → `./a.out` schreibt VTK (ParaView/Vedo). Fabians Modell ist ein eigenes „model.cu" = `foundation.cu`, das die Framework-Header aus `include/` nutzt.

Zentrale Bausteine (`yalla-main/include/`):
- **`solvers.cuh`** (Herzstück): `Solution<Pt, Solver>` = Container für den Zellzustand (Punkttyp `Pt`, hier `float3`) + Integrator. Methode `take_step<pw_int, pw_friction>(dt, gen_forces)` macht **einen Zeitschritt**. Integrator = **Heun** (`Heun_solver`: `euler_step` → `heun_step` → `add_rhs`). Drei austauschbare **Solver/Computer** (nur die Nachbarsuche unterscheidet sich):
  - `Tile_solver` (`compute_tile`) — alle Paare, O(n²).
  - `Grid_solver` (`compute_cube`, Klasse `Grid` mit `compute_cube_id`/`compute_cube_start_and_end`) — räumliches Gitter, nur 27 Nachbar-Cubes, O(n).
  - `Gabriel_solver` (`compute_cube_gabriel`) — Grid + Gabriel-Graph-Filter (der `neighbour_id[..]`-Buffer sitzt hier, s. §7). **Von den Zellen verwendet.**
  - `friction_on_background` = paarweise Reibung gegen das Medium → overdamped-Dynamik.
  - `pw_int` = die paarweise Kraftfunktion (hier `simulation_step`, Morse). `Generic_forces` = zusätzliche Kräfte pro Step (hier das Pinning, übergeben als `fun`-Lambda).
- **`property.cuh`**: `Property<T>` = gekoppeltes Host+Device-Array (`h_prop`/`d_prop`, `copy_to_device()`/`copy_to_host()`), mit Name → wird als VTK-Feld geschrieben. So sind `neighbours`, `activated`, `radius`, `force_mag` etc. realisiert.
- **`vtk.cuh`**: `Vtk_input` (`read_positions`, `read_normals`, `n_points`) und `Vtk_output` (`write_positions`, `write_property`).
- **`inits.cuh`**: Startlayouts `random_disk`, `random_sphere`, `regular_rectangle`, `regular_hexagon`.
- **`dtypes.cuh`** (float3-Operatoren/Punkttypen), **`polarity.cuh`** (`bending_force`, `Po_cell` — für die noch ungenutzte Biegesteifigkeits-Option B), `links.cuh`, `mesh.cuh`, `utils.cuh`.

### 11.2 foundation.cu — Aufbau
1. **Modell-Auswahl** oben per `#define` (genau eines von `MODEL_EGG`/`MODEL_SHEET`/`MODEL_CAP`, `#error`-Guard). **Aktuell aktiv: `MODEL_EGG`.**
2. **Parameterblock** (globale `const`): r_max=2, dt=0.05, n_time_steps=2000; r_e=0.3(egg)/1.0(sonst), D_e=2, alpha=1; r_start=r_e/2, r_activated=r_start/2; r_decay_shrink=5.0, r_decay_grow=0.1; activation_delay=2, activation_duration=10, refractory_duration=200; force_threshold=0.15 (nicht-const, via argv[4]); use_hill_function=**false**, n_hill=2; activation_fraction=0.01, activation_min=3; surface_stiffness=12; cap_half_angle_deg=75; activation_steps={220}.
   > **Ei-Update (12.07):** für MODEL_EGG inzwischen r_max=1.4, surface_stiffness=4, r_e=0.27 (+ Laufzeit-`d_r_e`), activation_duration=40, zusätzlich ein `force_type`-Schalter (`'m'`/`'l'`). Werte hier gelten für den 08.07-Stand bzw. Cap/Sheet — s. Ist-Box + §13.5.
3. **Device-Globals** (`__device__` Pointer): d_neig, d_activated, d_radius, d_force_accum, d_active_neighbor u.a. — via `cudaMemcpyToSymbol` mit den Property-Device-Arrays verbunden. *(12.07: + `d_r_e` als Laufzeit-Gleichgewicht.)*
4. **Kernels/Funktionen:** `pin_to_surface_kernel` (+ sheet/cap-Varianten), `alter_cells_before` (Host-Zustandsmaschine), `simulation_step` (Device-Paarkraft), `place_on_cap`/`sheet_height`/`sheet_normal` (Geometrie).

### 11.3 Programmablauf (`main`)
1. **CLI:** `./a.out <R> <seed> <mode> <threshold> [frac] [seed_x seed_y seed_z]`. Modi: `vtk` (VTK + spontane Aktivierung, Default), `scan` (keine VTK, SCAN_RESULT, propagation-only), `vtk_prop` (VTK + propagation-only, zum Visualisieren der reinen Welle). `require_active_neighbor = scan || vtk_prop`. *(12.07: zusätzlicher `relax`-Modus, der die Zellen relaxiert und `egg_cells_relaxed.vtk` exportiert — §13.5.)*
2. **Init (modellabhängig):** EGG lädt `initial_conditions_mesh_1.vtk` in `cells` (Gabriel_solver) **und** in eine eingefrorene `surface`+`surface_norm`-Kopie (Tile_solver, nie integriert). SHEET: `regular_rectangle` + auf Höhenfeld heben. CAP: N aus R (`N = round(0.9069·(R·θ_max/r_e)²)`, Floor 20), `grid_size` mit R skaliert, `random_disk` + `place_on_cap` (Exponential-Map), gibt geodätischen Patch-Radius + Winkel aus.
   > **Ei-Update (12.07):** EGG lädt Zellen und Surface **nicht mehr aus derselben Datei** — Surface = feines Mesh (Normalen), Zellen = getrennte, dünnere Population (`egg_cells_relaxed.vtk` aus dem `relax`-Lauf, ursprünglich per `prepare_egg_cells.py`/FPS platziert). §13.5.
3. **Properties anlegen:** activated (**Startwert 2 = refraktär für alle**), radius (=r_start), force_accum(0), force_mag(0), active_neighbor(0), state_timer(0), neighbours.
4. **`fun`-Lambda** (läuft in `take_step` vor der Integration): setzt neig/force_accum/active_neighbor auf 0 und ruft das modellabhängige Pinning (`pin_to_surface`/`pin_to_sheet`/`pin_to_cap`) → schreibt Rückstellkraft in `d_dX`.
5. **Zeitschleife** `for time_step = 0..2000`:
   a. `cells.copy_to_host()`.
   b. Bei t=220 (`activation_steps`): die `n_seed = round(act_fraction·N)` (≥3) Zellen am nächsten zu `seed_center` (EGG default {−9,−12,0}, via argv überschreibbar) auf **State 1** setzen (`nth_element` auf quadr. Abstand).
   c. `alter_cells_before(...)` — Host-Zustandsmaschine über alle Zellen (s.u.).
   d. `cells.take_step<simulation_step, friction_on_background>(dt, fun)` — Morse-Paarkräfte + Reibung + Pinning, Heun-Integration.
   e. `copy_to_host` von neig/activated/radius/force_accum/active_neighbor; `force_mag = |force_accum|` berechnen.
   f. Metriken tracken (ever_activated, now_active/peak_active, n_early bei t=270, avg_neighbors bei t=219).
   g. Außer im `scan`-Modus: VTK schreiben (positions + neighbours/activated/radius/force_mag).
6. Am Ende **eine `SCAN_RESULT`-Zeile** mit allen Metriken (von den Python-Scans geparst).

### 11.4 Zustandsmaschine (`alter_cells_before`, Host, jeden Step)
Pro Zelle je nach `activated`:
- **0 (ruhend):** zünde → State **7**, wenn (`force_mag > force_threshold`) [harter Zweig] bzw. Hill-Wahrscheinlichkeit — **UND** `neighbour_ok` (im Propagation-only-Modus: mind. ein Nachbar in State 1; sonst immer true).
- **7 (delay):** Timer++; nach `activation_delay` → State **1**; fällt `force_mag` unter Threshold → zurück auf **0**.
- **1 (aktiv):** Radius schrumpft schnell (r_decay_shrink); Timer++; nach `activation_duration` → State **2**.
- **2 (refraktär):** Radius wächst langsam zurück (r_decay_grow); Timer++; nach `refractory_duration` → State **0**.
Danach `h_radius`/`h_activated` zurück aufs Device.

### 11.5 Paarkraft-Kernel (`simulation_step`, Device, je Zellpaar i,j)
Überspringt i==j und dist>r_max. Zählt `d_neig[i]++`; markiert `d_active_neighbor[i]=1` falls `d_activated[j]==1` (Basis der Propagation-only-Regel). Berechnet Morse-Kraft über `surface_dist = dist−(r_i+r_j)` (Radien aus `d_radius`) und `atomicAdd` in `d_force_accum[i]` (atomar gegen Race Conditions). *(12.07: `switch(force_type)` — `'m'` Morse über `surface_dist` mit `d_r_e`, `'l'` stückweise linear.)*

### 11.6 Metriken (SCAN_RESULT-Felder)
`frac_activated` (kumulativ je-gefeuert/N = Reichweite) · `peak_active`/`frac_peak_active` (max gleichzeitig in State 1 = Frontgröße, die am geschlossenen Ei aussagekräftige Größe) · `n_early` (gefeuert binnen 50 Steps nach Trigger = früher Schwung) · `avg_neighbors`/`n_interior` (mittlere Nachbarzahl im relaxierten Zustand bei t=219, Zählradius 1.3·2·r_e, Randring 3·r_e ausgeschlossen) · `t_peak` · `N`, `n_seed`, `act_fraction`, `seed_x/y/z`.

### 11.7 Die zwei Python-Scans
Beide kompilieren `foundation.cu` selbst (einmal) und rufen dann `./a.out … scan …` wiederholt, parsen die `SCAN_RESULT`-Zeile, aggregieren, speichern CSV + Plot. `ONLY_GRAPH`-Flag = nur aus CSV neu plotten. **Wichtig:** das passende Modell muss im `#define` aktiv sein (Scan kompiliert den aktuellen Stand).
- **`parameter_scan.py`** (Kappen, braucht **MODEL_CAP**): Gitter `R_VALUES=[6,10,14,…,50,100]` × `SEEDS` (10) × `THRESHOLDS=[0.1,0.125,0.15,0.175,0.2,0.3]`. Ruft `./a.out R seed scan thr`. Plot: `frac_activated` vs. K=1/R² (eine Kurve je Threshold, mean±std über Seeds) + `avg_neighbors` (über Thresholds+Seeds gepoolt); Zweitachse R. → `Graphen/scan_metrics.csv` + `scan_curvature.png`.
- **`egg_activation_scan.py`** (Ei, braucht **MODEL_EGG**): importiert `analyse/egg_curvature.py` (Weingarten-K, `CURV_METHOD`). `locs_at_percentile(p)` = Farthest-Point-Sampling: aus `LOC_POOL=60` K-nächsten Punkten `N_PER_BIN=5` räumlich gestreute. Ruft `./a.out DUMMY_R DUMMY_SEED scan thr frac x y z` (Seed-Ort + Stimulusgröße). **Gruppe A:** `FIXED_FRACTION=0.01` × `THRESHOLDS=[0.125,0.15,0.175]` × 12 `LOC_PERCENTILES` → `frac_peak_active` vs. K. **Gruppe B:** `SWEEP_PERCENTILES=[99,50,5]` × `SWEEP_FRACTIONS=[0.004…0.03]` bei einem Threshold → Stimulus-Unabhängigkeit. → `Graphen/egg_curvature_response.csv` + `.png` (2 Panels).

---

## 12. Biologischer Hintergrund & Schlüssel-Referenzen (aus der Begleit-PPT)

Die Motivation der Arbeit — *warum* ein rein mechanisches Modell für Kontraktionswellen? Dieser Teil steht nur in `BegleitPP.pptx` (Folien 2–16), nicht im Code, und ist die inhaltliche Einleitung der Arbeit.

**Kontraktionswellen in verschiedenen Organismen (teils ohne/mit nur einfachem Nervensystem):**
- ***Trichoplax adhaerens*** (einfachster Vielzeller): Armon et al., „Ultrafast epithelial contractions provide insights into contraction speed limits and tissue integrity". Kontraktion zur mechanischen **Stressentlastung** (verhindert Reißen), Koordination, Bewegung; **propagiert ohne Nervensystem** (PIV: rot=Expansion, blau=Kontraktion).
- ***Tethya wilhelma*** (Schwamm, Porifera): Nickel, „Kinetics and rhythm of body contractions in the sponge Tethya wilhelma". Rhythmische Kontraktion zur **Wasserregulation** (Reinigung, Filtration, Abwehr); Welle startet lokal und läuft um den Schwamm; ohne komplexes Nervensystem.
- ***Drosophila*** (Fruchtfliege): Bailles et al., „Genetic induction and mechanochemical propagation of a morphogenetic wave". Endoderm-Morphogenese, Germband-Extension, **Invagination**.
- ***Tribolium castaneum*** (Reismehlkäfer): der Zielorganismus (s.u.).

**Der mechanistische Kern — Rechtfertigung des Modells (Folien 10–12):**
- Manning et al., „The Fog signaling pathway: Insights into signaling in morphogenesis": Fog = molekularer Schalter → aktiviert **Myosin II** → Kontraktion (via Rho1).
- **Schlüssel-Experiment:** Alpha-Amanitin hemmt RNA-Polymerase II (keine neue Genexpression) — **die Welle propagiert trotzdem weiter**. ⟹ Ausbreitung wird durch **mechanisches Feedback** getrieben: kontrahierende Zellen aktivieren ihre Nachbarn, unabhängig von der Gentranskription. Morphogenese = genetische **Initiation** + mechanische **Propagation**.
- **Das ist die tragende Grundannahme des Simulationsmodells:** Aktivierung über eine **Kraftschwelle** (mechanisch) statt über Chemie/Gene — genau das, was foundation.cu implementiert.

**Tribolium castaneum (Zielorganismus, Folien 13–16):**
- Globaler Vorratsschädling (v.a. Getreide), etablierter Modellorganismus (Genetik/Genomik/Verhalten/Ökologie/Schädlingsbekämpfung), repräsentiert Insektenentwicklung.
- Benton, „A revised understanding of Tribolium morphogenesis further reconciles short and long germ development": Blastoderm-Bildung, Differenzierung in **extraembryonales Gewebe (Serosa, Amnion)** + Embryo; embryonales Gewebe kondensiert ventral; Germband-Extension; Trennung von Amnion/Serosa; schließlich **Reißen des extraembryonalen Gewebes**.
- Pereyra, „Spatiotemporal segmentation of contraction waves in the extra-embryonic membranes of the red flour beetle": **Kontraktionswelle in der extraembryonalen Membran vor dem Riss** — das konkrete biologische Phänomen, das die Arbeit modelliert (Quelle des Ei-Meshes `tribolium_surface_1`).

> Der **alte** Titel („…on membrane rupture…") kam von genau diesem Membranriss (Pereyra). Der **neue** Titel (§9) verschiebt den Fokus vom Riss auf die Wellen*ausbreitung*; die Riss-Biologie bleibt aber die Motivation dahinter.

---

## 13. Verlauf 09.–12.07.2026: Ei-Modell in Bewegung (nach Redaktionsschluss)

> Diese Sessions sind **nach** dem Wissensstand von §1–§12 (08.07) entstanden und betreffen **ausschließlich MODEL_EGG**. Die Krümmungsstudie mit Kappen/Sheet (Exp. 1, §2/§4) ist unberührt. Wo Parameter von §1/§2/§11 abweichen, gelten hier die neueren Werte (kompakt in der Ist-Box oben). Roter Faden: das Ei bewegt sich nicht → Ursache Überkompression → Umbau der Kraft/Architektur → Kohäsions-Kollaps bleibt das offene Kernproblem.

### 13.1 Diagnose (09.07): „Die Welle läuft, aber sie bewegt nichts"
Direkt aus einem `vtk_prop`-Lauf gemessen (nicht spekuliert), Code-Stand 09.07 (10000-Punkte-Mesh, `r_e=0.3`, `r_max=2`, `force_threshold=0.15`):
- **Die Welle propagiert einwandfrei:** Zündung Step 220, Ausbreitung über **1671 Zellen (~17 % des Eis)** bis Frame ~400, Auslaufen bis ~600. Am Mechanismus/an den Kräften ist nichts kaputt.
- **Die Zellen bewegen sich nur einmal, ganz am Anfang:** in den ersten ~50 Steps springt die mediane Verschiebung auf ~0.3, dann **friert alles ein** bei ~0.38 für die restlichen 1950 Steps. Reine Anfangs-Relaxation, **keine** Wellenbewegung (0.31 @Frame 50 → 0.38 @Frame 1000).
- **Ursache = geometrische Überkompression (kein Bug):** Median-Punktabstand = **0.266**, Morse-Gleichgewicht aber beim Mittelpunktsabstand `2·r_e = 0.6` (**Namensfalle**, s. §7: `surface_dist = dist − 2·r_start = dist − r_e`, Gleichgewicht bei `surface_dist=r_e` ⇒ `dist=2·r_e`; `r_e` ist **nicht** der Ruheabstand). Damit sitzen **~95 % aller Zellpaare im abstoßenden Ast**. Der Kontraktionsmechanismus lebt aber von *Anziehung* → bei dieser Kompression senkt eine Kontraktion die Kraft nur von ~2.2 auf ~1.5 und **bleibt abstoßend** — nie echter Zug. In symmetrischer Packung heben sich die Restkräfte auf → keine Nettobewegung. Zusätzlich: Zellen auf ein **starres, eingefrorenes Mesh** gepinnt → nur tangential gleiten, nie ausbeulen.
- **Test „nur r_e halbieren" (0.13):** funktioniert **nicht** allein — das Ei kollabiert (Relaxation median 1.0 statt 0.35, §8) **und** die Welle zündet nicht mehr (`force_threshold` war gegen die alten Kraftbeträge kalibriert). ⇒ `r_e`, `r_max`, `force_threshold` sind gekoppelt, gemeinsam kalibrieren.

### 13.2 Literatur-Abgleich (Marija Perutović 2024, Lucie Biesecker) — Ursprung der Pinning-Methode
Beide Arbeiten (Betreuung Matthäus/Liebisch, dieselbe yalla-Pinning-Methode) liefern den entscheidenden Punkt: **Oberfläche und Zellen sind zwei getrennte Populationen.**
- **Das feine Mesh dient NUR zur Flächen-Diskretisierung fürs Pinning.** Marija: 16032–31228 Gitterpunkte; Biesecker: „added as a separate, non-moving cell population".
- **Die eigentlichen Zellen sind ein viel kleinerer, separater Satz:** Marija **1000** Zellen (Serosa-Studie; im Confinement-Beispiel nur 200), Biesecker **n = 1000**. Erzeugt mit `random_sphere`, dann aufs Gitter gepinnt.
- **Konkrete Parameter:** Marija `r_c = 0.7` (direkt der Ruheabstand), `r_max = 1.0` (≈ 1.4·r_c), skaliert die *Fläche* (Faktor q) so, dass 1000 Zellen bei 0.7 genau draufpassen. Biesecker `n = 1000`, `r_max = 0.35`, `dt = 0.1`, Adhäsion a = 5–11, `r_target` = 0.30–0.34, Pinning-Steifigkeit k = 5–10.
- **Rupture-Mechanismus** (beide): Zellen global schrumpfen (`Δr = α·(r_target − r)`) → Oberflächenspannung steigt → Bindungen reißen an der schwächsten Stelle → **Krümmung/Geometrie bestimmt den Rissort** (Ellipsoid/Serosa reißen an den Polen bzw. am biologischen Fusionspunkt; Kugel überall gleich). Biesecker erweitert um **Retraktion** nach dem Riss und studiert ebenfalls Krümmungseinfluss.
- **Konsequenz für foundation.cu (08.07-Stand):** Der Code lud das 10000-Punkte-Mesh **in beides** — `cells` *und* `surface`. Damit **10000 Zellen statt ~1000**, ~10× zu dicht. `r_max = 2` ≈ 7× Zellabstand (Literatur: r_max ≈ Zellabstand) → langer kohäsiver Morse-Schwanz = Kollaps-Treiber (§8).

### 13.3 Die Datensätze (`Joern_Projekt/initial_data/` + neu)
Alle stellen **dasselbe Tribolium-Ei** dar (verifiziert gleiche Form, Seitenverhältnis ~1 : 2.08 : 1.03), nur in verschiedenen Auflösungen/Skalen/Frames:

| Datei(en) | Rolle / Was es ist | Punkte | Frame (Extent) | Ø-Abstand | Normalen / Felder |
|---|---|---|---|---|---|
| `initial_conditions/mesh_1..5` | **5 verschiedene Individuen** (Längen 47–49.5, keine Zeitpunkte); Sim nutzt `mesh_1` als Surface | 10000 | 23×48×24 | 0.266 (≈0.25) | NORMALS `polarity` |
| `surfaces/target_surface_tribolium_1..5` | feinere Ziel-/Pinning-Fläche, dieselben 5 Individuen | 33164 | 23×48×24 | 0.14 | NORMALS |
| `Archive/tribolium_D0.20_S4.00_0..60` | **fertige Dynamik-Sim** (61 Frames), Parameter D=0.2, S=4.0 | 10000 | 23×48×24 | 0.266 | NORMALS + `mechanical_strain` |
| `serosa.vtk` / `anchor.vtk` / `anchor_pole.vtk` | grobe, segmentierte Version (752 Serosa=Typ0, 12 Anker=Typ1, 1 Pol=Typ2, Rest Typ3) | 3500 | **14×29×14** (= mesh/1.65) | 0.573 | `cell_type`, **KEINE Normalen** |
| `surface.vtk` | feinste „Master"-Fläche, normierte Koordinaten, VTK 5.1 | 30000 | 0.9×1.9×0.9 | 0.006 | keine |
| `horizontal_egg_tribolium_1.vtk` (`Joern_Projekt_Neu`) | **Zellen** (`cell_type`: serosa=0/anchor=1/pole=2/embryo), NATIV im Surface-Frame | 3000 | 23×48×24 | 0.54 | ja |

**Wichtige Beobachtungen:**
- **`Archive` ist das beste Validierungsziel.** Es beweist: 10000 Punkte *können* sich verformen — Punkte bewegen sich (median ~0.5, max ~2.0), **mittlerer Radius bleibt konstant (16.245, kein Kollaps)**, `mechanical_strain` baut sich 0 → **±1.5** auf. Die 10000 Punkte frieren nur mit *Fabians* Parametern ein. Jörns D/S-Parametrisierung ist die direkte Vorlage.
- **`serosa.vtk` zeigt die eigentlich gemeinte Zellzahl** (3500, Abstand 0.573). Bei diesem Abstand passt `r_e=0.3` (Gleichgewicht 0.6) **fast perfekt** (0.6/0.573 ≈ 1.05). Haken: **keine Normalen** → nicht direkt als gepinnte Zellen nutzbar.
- **Frame-Falle:** `serosa.vtk` liegt in einem **1.65× kleineren Frame** (beide um 0 zentriert → reine Skalierung). Jörns `horizontal_egg` liegt dagegen **nativ im Surface-Frame** → keine Skalierung nötig. Das ist der saubere Weg (die Skalierungs-Experimente serosa↔mesh waren Sackgassen).
- **Nebenänderung `Vedo_visualisierung/egg_surface.py` (09.07):** `DATASET`-Schalter + `DATASETS`-Dict (`mesh_1..5`, `target_1..5`, `surface`, `serosa`, `archive0`, `sim_mesh`); `SHOW_SEEDS`-Schalter für die 60 mesh_1-spezifischen Zünd-Seeds (blenden sich bei anderen Meshes aus); robustes Normalen-Handling (`serosa.vtk`/`surface.vtk` ohne `polarity` → kein Absturz).

### 13.4 Umbau-Versuch Morse → lineare Kraft (10.07) — Zielkonflikt & Revert
Ziel: das „keine Bewegung"-Problem lösen, indem das Kraftgesetz durch Jörns lineares ersetzt wird. Ergebnis: fundamentaler Zielkonflikt → **vollständiger Revert auf Morse**.

**Was probiert wurde:**
1. **Kraftgesetz Morse → linear** (`F = f_rep·max(0, eq−dist) − f_adh·max(0, dist−eq)`), **Gleichgewicht `eq = r_i + r_j`** (ohne `r_e`-Offset → 0.3 statt 0.6, nahe Mesh-Abstand 0.266).
2. **r_max-Sweep** + **Adhäsions-Tuning** (f_adh 6→12).
3. **Zell/Fläche-Trennung** (Weg B): Oberfläche = volles feines Mesh (nur Pinning), Zellen = dünnere Teilmenge (jeder 4. Punkt → 2500), `r_e` auf Zellabstand 0.53.

**Zentraler Befund — bei linearer Kraft sind Stabilität und Ausbreitung gekoppelt & gegenläufig:**

| r_max / Gleichgewicht | Ausbreitung | Stabilität |
|---|---|---|
| **~4×** (r_max=1.2, dichte Zellen) | **99 %** (echte Wanderwelle, Peak ~t600) | **Gitter/Streifen** (Buckling) |
| **~1.4×** (Jörns Wert) | **tot** (nur Seed, 1 %) | **stabil**, keine Gitter |

- **Ursache:** Die **lineare Adhäsion wächst unbeschränkt mit dem Abstand** — bei großem r_max zieht ein weit entfernter Gabriel-Nachbar mit riesiger Kraft (`6·(2−0.53) ≈ 8.8`) → explosive Verklumpung in Linien/Gittern **innerhalb ~5 Steps**. Morse hat das nicht, weil seine Anziehung **sättigt und wieder abfällt**.
- **Weite Reichweite** ist aber nötig: eine dünne Kontraktionsfront wird sonst von der **Gewebe-Einbettung gedämpft** (jede Zelle von ~6 Nachbarn gehalten → Netto-Zug < Schwelle → Welle stirbt nach einem Ring).
- ⇒ Reichweite steuert **beides** gegenläufig. Eine *reine* lineare Kraft hat kein Fenster, das beides erfüllt.

**Warum Jörn diesen Konflikt nie hatte:** sein Modell hat **keine propagierende Welle** — er schrumpft **global & synchron** alle Zellen (`h_radius = (init−target)·e^(−exponent)+target`, `exponent += k`/Step) → gleichmäßig steigende Spannung → Riss. Stabilität aus: r_max ≈ 1.0–1.4× Gleichgewicht (kurz), dt = 0.01 (5× kleiner), drag = 6 → effektiver Schritt `dt/drag ≈ 0.0017` vs. Fabians `0.05/1 = 0.05` (~30× kleiner pro Kraft).

**Was die Zell/Fläche-Trennung brachte** (und nicht): *Brachte:* 2500 unabhängige Zellen auf feiner Oberfläche relaxieren bei **kurzem** r_max in gleichmäßige Packung (NN-Streuung 0.29→0.06), kein Kollaps — konzeptionell behalten. *Brachte nicht:* bei **langem** r_max bilden sich die Gitter trotzdem (~5 Steps); bei **kurzem** r_max bewegt sich fast nichts (Welle tot).

**Der eigentliche Ausweg** (nicht umgesetzt): Reichweite von Kraftstärke entkoppeln durch **gedeckelte/sättigende Adhäsion** (`adhesion = f_adh·min(dist−eq, adh_cap)`) — kurze Steifigkeit für Stabilität + beschränkte, weit reichende Anziehung für die Welle. Das ist **genau Morses Form**.

> **Synthese-Erkenntnis:** Morse war von der *Form* her richtig (gesättigt → propagiert *und* stabil). Der Fehler in §13.1 war nicht Morse selbst, sondern der **Gleichgewichts-Offset** (`eq = r_i+r_j+r_e = 0.6` ≫ Mesh-Abstand 0.266 → Überkompression → eingefroren). Aussichtsreichster nächster Versuch: **Morse behalten, aber den Gleichgewichtsabstand an den Zellabstand anpassen** (Offset verkleinern, `eq ≈ Abstand`) — statt das Kraftgesetz zu tauschen.

**Entscheidung:** `foundation.cu` **vollständig auf Morse-Original zurückgesetzt** (r_e=0.3, r_max=2, D_e/alpha, keine Trennung). Kompiliert sauber. Die konzeptionellen Erkenntnisse bleiben Grundlage für §13.5.

### 13.5 Session 12.07: Trennung umgesetzt, Jörn-Abgleich, Relax-Export
Dieser Anlauf setzt die Synthese aus §13.4 („Gleichgewicht an den Zellabstand anpassen") tatsächlich um.

**Abgleich Jörns aktuelles Projekt (`Joern_Projekt_Neu`):**
- **Zellen** = `horizontal_egg` (im Surface-Frame, `cell_type`: serosa=0/anchor=1/pole=2/embryo). **Surface** = `target_surface` (33k, Normalen). `surface_scale = 1` → KEINE Skalierung. „Serosa" ist kein eigenes File mehr, sondern ein Zelltyp.
- **Pinning:** identisch zu Fabian (`F = −k·(r·n)·n`), aber weicher: `surface_attraction = 3` (Fabian: 12). Gitter-beschleunigte Nächster-Punkt-Suche.
- **Kraft:** stückweise linear `rep·max(0,radius−dist) − adh·max(0,dist−radius)`, `adh=rep=1`. Clou: **`radius` (Gleichgewicht) ist selbst-kalibriert** — in der Relaxations-Phase (~10000 Schritte) auf den gemessenen mittleren Nachbarabstand gesetzt → Sheet immer kräftefrei.
- **Kontraktion:** glatte globale embryo-Schrumpfung, **keine Schwellen-Welle** (das ist Fabians eigener Zusatz).

**Umgesetzt in `foundation.cu`:**
- **Zell/Surface-Trennung:** Zellen und Pinning-Surface aus getrennten Dateien.
- **`d_r_e` (Laufzeit-Gleichgewicht):** die Kraft nutzt eine Device-Variable statt der Compile-Konstante; in der Relaxations-Phase auf den gemessenen Abstand selbst-kalibriert (`d_r_e = s − 2·r_start`, nach Jörns Prinzip). Ruhelage kräftefrei, unabhängig vom Startabstand.
- **`force_type`-Schalter** (`'m'` Morse / `'l'` linear), zur Laufzeit im `simulation_step`-`switch`.
- **`relax`-Modus:** foundation.cu übernimmt die Relaxation selbst — lädt die (per Skript platzierten) Zellen, relaxiert mit der echten Sim-Physik + Pinning + Selbstkalibrierung (keine Welle) und **exportiert** `egg_cells_relaxed.vtk`. Die Wellen-/Scan-Läufe laden dieses File. Workflow: `./a.out … relax …` einmal → dann `scan`/`vtk`.

**Neue Skripte:**
- **`prepare_egg_cells.py`** — Farthest-Point-Sampling von N Zellen auf der Surface: gleichmäßige (blue-noise) Verteilung im Surface-Frame, keine Skalierung. Liefert die Anfangs-Platzierung, die foundation.cu dann relaxiert.
- **`graph_Kraft.py`** (erweitert) — Fabians Morse-Kraft/-Potential + Jörns lineare Kraft + Marijas Polynom-Kraft überlagert (eigene Gleichgewichts-Linien; Morse über `surface_dist`, Jörn über Zentrumsabstand).

**Zentrale (Wieder-)Erkenntnisse:**
1. **Frame muss stimmen:** Zellen im Surface-Frame erzeugen (`horizontal_egg` / FPS), nicht die kleine `serosa` hochskalieren.
2. **Gleichgewicht = tatsächlicher Abstand** (Selbstkalibrierung) → keine Grundspannung. Bestätigt die Synthese aus §13.4.
3. **🔴 KERNPROBLEM bleibt offen — Kohäsions-Kollaps:** Der `relax`-Lauf macht ihn messbar: Morse mit `r_max=2` zieht das Sheet zusammen, Abstand **0.81 → 0.38**, `avg_neighbors → 15` (überpackt). Die **NN-Selbstkalibrierung verstärkt den Kollaps** (positive Rückkopplung: klumpt → kleinerer NN → kleineres `d_r_e` → mehr Klumpen). Betrifft auch die Welle → „Ei zerreißt langsam".

**Nächste Schritte (offen):**
- Kraftform/`r_max` so wählen, dass das Sheet stabil bleibt UND die Welle läuft: kurzes `r_max` (lokale Kohäsion) auf einem *richtig verteilten* Sheet testen; Kalibrierungs-Rückkopplung stoppen (einmal einfrieren statt nachjagen); alternativ repulsions-dominierte Relaxation oder kompakte Kohäsionskraft (Marija-Polynom). Deckt sich mit §13.4 (gesättigte/gedeckelte Adhäsion).
- **Aktueller Tuning-Stand (in Arbeit):** `r_max=1.4`, `surface_stiffness=4`, `r_e=0.27`, `force_type='m'`, `activation_duration=40`.

---

## Anhang A — Morse-Kraft-Herleitung
`V(r) = D_e·(1 − e^(−a(r−r_e)))²`, `F = −dV/dr`. Mit u = 1−e^(−a(r−r_e)): dV/dr = 2·D_e·u·a·e^(−a(r−r_e)).
**`F(r) = −2·a·D_e·(1 − e^(−a(r−r_e)))·e^(−a(r−r_e))`.** r>r_e anziehend, r<r_e abstoßend, r=r_e Gleichgewicht.

## Anhang B — Threshold-Herleitung
`r_j(t) = r_activated + (r_start − r_activated)·e^(−r_decay·t)`; `surface_dist(t) = 2·r_e − r_start − r_j(t)`; `phi = e^(−alpha·(surface_dist−r_e))`; `F(t) = −2·D_e·alpha·(1−phi)·phi`. Idealisiert (Zelle i fix); real bewegt sie sich → empirisch kalibrieren.

## Anhang C — Krümmungsformeln
**Graph-Fläche z=f(x,y):** `K = (f_xx·f_yy − f_xy²)/(1+f_x²+f_y²)²`, `H = (f_xx(1+f_y²) − 2f_x f_y f_xy + f_yy(1+f_x²))/(2(1+f_x²+f_y²)^{3/2})`. Frame-Wahl (Tangentialebene) → f_x,f_y≈0 → **K = 4ac − b²**, H = a+c (Monge: f_xx=2a, f_xy=b, f_yy=2c).
**Weingarten:** `dn = −S·dp`, S=[[a,b],[b,c]] per Least-Squares; **K = det(S) = ac − b²**, H = ½(a+c); κ1,2 = Eigenwerte = H ± √(H²−K).
**Abgeleitet:** `R_eq = 1/√K`, `κ1,2 = H ± √(H²−K)`.

---

*Quellen: 24 Claude-Code-Sessions (WSL `-home-fabian-Bachelorarbeit` + Windows-Projektordner) sowie 3 claude.ai-Web-Konversationen (`WebChats/`: Morse-Kraft, Krümmungsberechnung, Weingarten/Monge + Folien-Feedback), Sessions 08.06.–08.07.2026 (§1–§12) — plus die Sessions 09.07 (No-Motion-Diagnose, Datensätze), 10.07 (Morse→linear-Versuch & Revert) und 12.07 (Zell/Surface-Trennung, Relax-Export) für §13 und die Ist-Box. Zusammengeführt aus `Projektkontext_Bachelorarbeit.md` (Design/Content-Basis) und den angehängten Sessions in `Projektkontext.md`.*
