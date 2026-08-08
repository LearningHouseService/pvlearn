# pvlearn — Umsetzungsplan

**Repository:** `LearningHouseService/pvlearn`
**Ziel:** Herauslösen des ML-Forecast-Teils aus `DerOetzi/solaredge2mqtt` in eine eigenständige, anlagenunabhängige Library.

**Tagline:** *Teach your home to predict its own solar production.*

---

## 1. Zielbild und Abgrenzung

**Scope-Reduktion:** Ursprünglich hier mitgeplant, jetzt gestrichen: REST-Service, Home-Assistant-Add-on und HACS-Integration unter der Domain `pvlearn`. Diese Rolle übernimmt künftig `learninghouse` (eigenes Repo, eigene Roadmap), das pvlearn als Dependency einbindet und dafür selbst Add-on und Integration bekommt. pvlearn bleibt reine Library — siehe Kapitel 4 für den dadurch verkürzten Phasenplan.

### Was pvlearn ist

Ein selbstlernender PV-Ertragsprognose-Stack, der auf den **eigenen historischen Messdaten** einer Anlage trainiert, statt auf einem generischen physikalischen Anlagenmodell. Das ist die Abgrenzung zu Forecast.Solar und Solcast: keine Angabe von Ausrichtung, Neigung oder kWp nötig — das Modell lernt die Charakteristik der Anlage inklusive Verschattung, Ost-West-Splits und Wechselrichter-Derating implizit aus den Daten.

### Was pvlearn nicht ist

- Kein generischer ML-Service — das ist `learninghouse`.
- Keine Verbrauchs- oder Lastprognose. Bewusste Grenze; falls das später kommt, gehört es zu `learninghouse` oder in ein eigenes Projekt.
- Keine Datenerfassung. pvlearn misst nichts selbst, es bekommt Energiewerte geliefert.

### Architekturprinzip

Die Library ist **I/O-frei**. Kein MQTT, kein InfluxDB, kein HTTP, kein Dateisystemzugriff außerhalb explizit übergebener Pfade. Rein: DataFrames und typisierte Modelle. Raus: Prognosen. Alles andere sind Adapter in den darüberliegenden Schichten. Genau das macht sie in `solaredge2mqtt` einbettbar und ebenso in `learninghouse`, ohne dass pvlearn selbst je ein Netzwerk-Interface bräuchte.

---

## 2. Artefakte und Namensschema

| Artefakt | Name | Repository |
|---|---|---|
| Library | `pvlearn` | `LearningHouseService/pvlearn` |
| PyPI-Paket | `pvlearn` | — |

Docker-Image, HA-Add-on und HACS-Integration entstehen nicht unter dem Namen `pvlearn`. Das baut künftig `learninghouse`, das pvlearn als Dependency einbindet; deren Namensschema gehört in dessen eigene Planung, nicht in dieses Dokument.

**Lizenz:** MIT, konsistent mit `solaredge2mqtt` und `learninghouse`.

---

## 3. Kanonisches Datenmodell

Dieses Kapitel ist die Grundlage für alle Phasen. Wird es später geändert, sind alle trainierten Modelle ungültig.

### 3.1 Wetter-Feature-Schema

Der bestehende Code verwendet unverändert die OpenWeatherMap-One-Call-Feldnamen. Für die Provider-Unabhängigkeit wird ein kanonisches Schema definiert; Provider-Adapter mappen darauf. Alle Felder sind **optional** — `_extract_used_columns()` toleriert bereits fehlende Spalten, ein Provider ohne UV-Index bekommt schlicht ein kleineres Feature-Set.

| Kanonisch | Typ | OpenWeatherMap | Open-Meteo |
|---|---|---|---|
| `cloud_cover` | numerisch, % | `clouds` | `cloud_cover` |
| `temperature` | numerisch, °C | `temp` | `temperature_2m` |
| `apparent_temperature` | numerisch, °C | `feels_like` | `apparent_temperature` |
| `dew_point` | numerisch, °C | `dew_point` | `dew_point_2m` |
| `relative_humidity` | numerisch, % | `humidity` | `relative_humidity_2m` |
| `surface_pressure` | numerisch, hPa | `pressure` | `surface_pressure` |
| `precipitation` | numerisch, mm | `rain` | `precipitation` |
| `precipitation_probability` | numerisch, 0–1 | `pop` | `precipitation_probability` (÷100) |
| `uv_index` | numerisch | `uvi` | `uv_index` |
| `visibility` | numerisch, m | `visibility` | `visibility` |
| `wind_speed` | numerisch, m/s | `wind_speed` | `wind_speed_10m` |
| `wind_gust` | numerisch, m/s | `wind_gust` | `wind_gusts_10m` |
| `wind_direction` | zyklisch, 360° | `wind_deg` | `wind_direction_10m` |
| `condition_code` | kategorial | `weather_id` → Mapping | `weather_code` (WMO) |
| `ghi` | numerisch, W/m² | — | `shortwave_radiation` |
| `dni` | numerisch, W/m² | — | `direct_normal_irradiance` |
| `dhi` | numerisch, W/m² | — | `diffuse_radiation` |

**Zwei Anmerkungen dazu.**

`weather_main` entfällt ersatzlos — es ist eine grobe Zusammenfassung von `weather_id` und trägt kaum eigene Information. `weather_id` wird auf WMO-Codes gemappt, damit ein Modell theoretisch providerübergreifend gültig bleibt. In der Praxis wird bei Providerwechsel trotzdem neu trainiert (siehe 3.4).

Die drei Strahlungsgrößen sind neu und potenziell der größte Qualitätsgewinn des ganzen Vorhabens: GHI/DNI/DHI sind physikalisch direkt mit dem PV-Ertrag verknüpft, während `cloud_cover` und `uv_index` nur schwache Proxies sind. Open-Meteo liefert sie kostenlos und ohne API-Key mit; OpenWeatherMap nicht.

### 3.2 Zeit- und Sonnenstands-Features

Bleiben inhaltlich wie bisher (`TimeEncoder`, `SunEncoder`), aber mit zwei Änderungen:

- `SunEncoder` bekommt statt eines `LocationSettings`-Objekts primitive, serialisierbare Parameter: `latitude: float`, `longitude: float`, `timezone: str`. Andernfalls brechen `sklearn.clone()` und Pickling, und Multi-Tenancy ist nicht möglich.
- `tzlocal.get_localzone()` auf Modulebene entfällt. Die Zeitzone wird pro Brain explizit gesetzt. Ein Service, der Anlagen in mehreren Zeitzonen bedient, kann sich keine Prozess-globale TZ leisten.
- `TimeEncoder` kodiert die Tageszeit als **Minuten seit Mitternacht**, nicht als Stunde. Bei stündlicher Auflösung ist das Ergebnis identisch; bei feinerer Auflösung wären `hour_sin/cos` für alle Intervalle innerhalb einer Stunde gleich und das Feature damit wertlos. Siehe 3.5.

Zusätzlich zu prüfen: `ephem` und `astral` werden derzeit parallel verwendet. Eine der beiden Abhängigkeiten sollte entfallen; `astral` deckt Azimut, Elevation und Sonnenauf-/-untergang vollständig ab.

### 3.3 Zielgröße

**Nur noch ein Modell: Energie pro Intervall in Wh.**

Das bisherige Power-Modell entfällt. Begründung: Beide Modelle trainieren auf identischen Features, und bei Stundenauflösung ist die mittlere Leistung numerisch identisch zur Stundenenergie. Der Wegfall halbiert Trainingszeit, Cache-Bedarf und Wartungsaufwand.

Auf dem Referenzdatensatz aus Phase 0 ist das empirisch bestätigt: MAE 620,88 Wh für das Energiemodell gegenüber 624,93 W für das Leistungsmodell, R² 0,886 gegenüber 0,885. Die beiden Modelle liegen unter einem Prozent auseinander.

**Der Leistungswert entfällt ersatzlos.** Nicht nur das Modell, auch die Ausgabe: pvlearn publiziert kein `power_period`, und `solaredge2mqtt` stellt es mittelfristig ebenfalls ein. Eine abgeleitete Leistung wäre reine Umrechnung derselben Zahl in eine andere Einheit — sie trägt keine Information, die nicht schon in `energy_period` steht, und hält ein zweites Feld am Leben, das bei jeder Änderung an der Auflösung mitgepflegt werden muss.

Eine Prüfung der bekannten Konsumenten stützt das: `power_period` wird zwar nach MQTT publiziert und von der HACS-Integration `solaredge2mqtt_forecast` in `ForecastData.power_period` eingelesen — **verwendet wird es dort nirgends**. Der Energy-Dashboard-Provider arbeitet ausschließlich mit `energy_period`, Sensor-Entitäten für Leistung existieren nicht. Der Wert wird berechnet, publiziert, nach InfluxDB geschrieben, geparst und dann verworfen.

Es bleibt ein Breaking Change für unbekannte Dritt-Subscriber des MQTT-Topics. Behandlung: Feld in solaredge2mqtt als deprecated ankündigen, mindestens zwei Minor-Releases weiter publizieren, dann entfernen. Im Changelog mit dem Hinweis, dass sich die Momentanleistung aus `energy_period` und dem Intervall trivial selbst berechnen lässt.

Für pvlearn selbst gibt es nichts zu deprecaten — die Library ist neu und publiziert den Wert nie.

### 3.4 Modell-Metadaten und Invalidierung

Jedes persistierte Modell trägt:

```json
{
  "pvlearn_version": "0.1.0",
  "feature_schema_version": 1,
  "pipeline_version": 2,
  "sklearn_version": "1.9.0",
  "weather_provider": "open-meteo",
  "interval_minutes": 60,
  "location": {"latitude": 49.45, "longitude": 11.08, "timezone": "Europe/Berlin"},
  "trained_at": "2026-08-03T12:20:00+02:00",
  "training_rows": 1440,
  "selected_features": ["..."],
  "metrics": {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
}
```

Beim Laden gilt hart: **Stimmt `feature_schema_version`, `pipeline_version`, die sklearn-Minor-Version, der Provider, das Intervall oder die Location nicht überein, wird das Modell verworfen und neu trainiert.** Kein Migrationsversuch, kein Best-Effort-Laden. Stumme Fehlprognosen durch ein Modell, das auf einem anderen Feature-Set trainiert wurde, sind praktisch nicht debugbar.

Die beiden Versionsfelder trennen zwei Dinge, die sich unabhängig ändern: `feature_schema_version` versioniert das Feature-Vokabular aus 3.1, `pipeline_version` die Art, wie aus diesen Spalten ein Modell gebaut wird — Vorverarbeitungsschritte, Feature-Auswahl, Schätzer. Phase 1c hat den Selektor geändert, ohne eine einzige Spalte anzufassen; ohne das zweite Feld hätte kein Bestandsmodell invalidiert.

Zur sklearn-Version: die Phase-0-Baseline ist nur gegen exakt die Version reproduzierbar, unter der sie entstanden ist. `pvlearn` pinnt scikit-learn deshalb exakt (siehe 6.6); ein Bump verschiebt still jede Prognose und erfordert eine neu erzeugte Baseline.

**Persistenzformat:** joblib/Pickle. ONNX wurde geprüft und verworfen — `CyclicalEncoder`, `TimeEncoder`, `SunEncoder` und `PFISelector` sind Custom-Transformer und bräuchten je einen eigenen Shape Calculator plus Converter. Der ursprüngliche Motivator (leichtgewichtige Inferenz direkt in einer HA-Integration) entfällt ohnehin: Inferenz läuft serverseitig, im Prozess des Aufrufers (`solaredge2mqtt` heute, `learninghouse` künftig), keine HA-Integration bettet pvlearn selbst ein.


### 3.5 Prognoseauflösung

**Entschieden: Das MVP rechnet stündlich. Das Datenmodell hält feinere Auflösungen offen, ohne sie zu implementieren.**

Das Intervall ist **keine Provider-Eigenschaft**, sondern eine Brain-Eigenschaft, begrenzt von beiden Seiten:

```
interval = min(was der Provider liefert, was der Client an Messwerten pusht)
```

Open-Meteo kann 15 Minuten, aber wenn der Wechselrichter nur Stundenwerte meldet, nützt das nichts. OpenWeatherMap kann 15 Minuten grundsätzlich nicht. Als Provider-Attribut modelliert entstehen sofort widersprüchliche Zustände.

**Was jetzt intervall-fähig gebaut wird** — kostet zum jetzigen Zeitpunkt nichts, wäre später eine Migration, die jedes trainierte Modell invalidiert:

- `interval` als Pflichtfeld in Brain-Konfiguration und Modell-Metadaten, mit Invalidierung bei Abweichung (siehe 3.4)
- Zeit-Features auf Minuten seit Mitternacht (siehe 3.2)
- Aggregationslogik summiert Intervalle innerhalb eines Zeitraums, statt „eine Zeile = eine Stunde" anzunehmen
- Der Prognose-Endpunkt gibt das Intervall in der Antwort mit an

**Warum nicht sofort implementieren:**

*Keine Messdaten.* solaredge2mqtt schreibt stündlich. Eine Umstellung erfordert eine Änderung der Datenerfassung und danach Monate Sammelzeit. Referenzdatensatz und Baseline aus Phase 0 sind stündlich — für einen 15-Minuten-Pfad existiert keine Baseline und damit keine Abnahme.

*Zwei Codepfade ab Tag eins.* OWM-Bestandsnutzer bleiben zwingend stündlich. Jede Aggregation, jede Invalidierungsregel und jeder Test existierte doppelt, bevor überhaupt ein Release draußen ist.

*Die Interpolationsfalle.* Open-Meteo liefert `minutely_15` nativ nur in Mitteleuropa (ICON-D2, AROME) und Nordamerika (HRRR), dort inklusive Strahlung. Außerhalb dieser Abdeckung — geografisch wie jenseits des Modellhorizonts — gibt die API interpolierte Stundenwerte zurück. Das ergibt viermal so viele Zeilen mit derselben Information, und die künstlich erzeugten Nachbarzeilen sind stark autokorreliert. Ein naives Holdout hält so ein Modell für besser, als es ist. Wer das umsetzt, muss die native Abdeckung prüfen und bei Interpolation ablehnen statt stillschweigend zu trainieren.

*Rechenaufwand.* Vier Mal so viele Trainingszeilen verschärfen Punkt 6.4 unmittelbar.

Frühestens dann, wenn Messdaten in feinerer Auflösung vorliegen — unabhängig davon, ob pvlearn dann von `solaredge2mqtt`, `learninghouse` oder beiden aus aufgerufen wird.

### 3.6 Trainingsdaten-Vertrag

**Kritisch und leicht falsch zu bauen:** Trainiert wird auf der Wettervorhersage, die zum Messzeitpunkt aktuell war — nicht auf nachträglich beobachtetem Wetter. Das ist korrekt, weil Trainings- und Inferenzverteilung übereinstimmen müssen; zur Inferenzzeit existiert nur die Vorhersage, nie die Beobachtung. Ein auf Beobachtungen trainiertes Modell bekäme systematisch andere Eingaben, als es bei der Vorhersage sehen wird.

Das ist keine pvlearn-Eigenschaft, sondern eine Anforderung an jeden Aufrufer: Wer Trainingszeilen zusammenstellt (heute `solaredge2mqtt`, künftig `learninghouse`), muss pro Zeile die Vorhersage ablegen, die zum Vorhersagezeitpunkt für dieses Intervall galt, und sie über den Zeitstempel mit dem später gemessenen Energiewert verknüpfen. pvlearn selbst kann das nicht prüfen — dafür müsste es Wetterdaten selbst holen, und genau das tut es laut Architekturprinzip (Kapitel 1) nie.

---

## 4. Phasenplan

Die ursprünglich geplante Reihenfolge wurde an einer Stelle angepasst: Phase 1 wurde zweigeteilt (1a Extraktion, 1b Konsolidierung), ergänzt um die vorgezogene Korrektur in 1c. Ursprünglich folgten danach REST-Service, HA-Add-on und HACS-Integration als eigene Phasen — die sind gestrichen (siehe Kapitel 1). Der Plan hier endet mit der fertigen Library.

### Phase 0 — Vorbereitung

**Ziel:** Repository steht, Namen sind reserviert, Referenzdaten für Regressionstests liegen vor.

- Repo `LearningHouseService/pvlearn` anlegen, CI-Setup aus `learninghouse` übernehmen (Ruff, pytest, Build-Workflow, ghcr-Push).
- PyPI-Namen `pvlearn` mit einer 0.0.1-Platzhalterversion sichern.
- **Referenzdatensatz erzeugen:** aus der eigenen InfluxDB-Instanz einen `forecast_training`-Export über mindestens 90 Tage ziehen, anonymisieren, als Parquet ins Repo (oder in ein Test-Fixtures-Repo, falls Größe stört).
- **Baseline-Prognosen einfrieren:** mit dem aktuellen `solaredge2mqtt`-Code auf diesem Datensatz trainieren und vorhersagen, Ergebnis als Referenz ablegen.

**Abnahme:** Es existiert ein Testfall, der aus dem Referenzdatensatz reproduzierbar dieselben Prognosewerte erzeugt wie der Produktivstand.

Ohne diesen Schritt ist Phase 1a nicht verifizierbar.

---

### Phase 1a — Extraktion, verhaltensgleich

**Ziel:** Der Code liegt in pvlearn, das Verhalten ist bitgleich zur Baseline. Noch keine fachlichen Änderungen.

**Umzug:**

| Nach `pvlearn` | Bleibt in `solaredge2mqtt` |
|---|---|
| `Forecaster` (aus `service.py`) | `ForecastService` — EventBus, InfluxDB, MQTT-Publish |
| `PFISelector` (aus `service.py`) | `events.py` |
| komplette `encoders.py` | `settings.py` (Mapping YAML → pvlearn-Config) |
| Feature-Konstanten | HA-Discovery-Dekoration am Forecast-Modell |

**Entkopplung:**

- `solaredge2mqtt.core.logging.logger` → `logging.getLogger(__name__)`. Achtung: der bestehende Code nutzt loguru-Stil mit `{}`-Platzhaltern und Keyword-Argumenten; das muss auf `%`-Style oder f-Strings umgestellt werden.
- `InvalidDataException` → eigene Hierarchie: `PVLearnError` → `InsufficientDataError`, `ModelNotTrainedError`, `SchemaMismatchError`.
- `LocationSettings` → primitives `Location`-Modell in pvlearn.
- `ForecastSettings` → `ForecasterConfig` in pvlearn, ohne die solaredge2mqtt-spezifischen Felder (`retain`, `enable`).

**Modell-Split:** `Forecast(Component)` bleibt in solaredge2mqtt und erbt von einem neuen, dekorationsfreien `pvlearn.ForecastResult`, das die Aggregationslogik trägt (`energy_today`, `energy_today_remaining`, `energy_current_hour`, `energy_next_hour`, `energy_tomorrow`). Diese Logik wird in allen vier Stufen gebraucht — sie darf nur einmal existieren.

**Abnahme:**
- [x] `solaredge2mqtt` mit `pvlearn`-Dependency erzeugt auf dem Referenzdatensatz identische Prognosen wie die Baseline. Erbracht durch `tests/test_extraction_regression.py` (pvlearn-seitig) und `tests/test_pvlearn_wiring_regression.py` (solaredge2mqtt-seitig), beide gegen Prognosegüte statt Bitgleichheit (hardwareabhängig, siehe Nachtrag zu Punkt 6 in Kapitel 6). Wiring liegt in `solaredge2mqtt`s `storedge`-Branch (dessen Merge nach `main` ist ein eigener, unabhängiger Vorgang in jenem Repo).
- [x] pvlearn importiert nichts aus `solaredge2mqtt`.
- [x] pvlearn hat keine Imports von `paho-mqtt`, `influxdb-client`, `fastapi`.

**Release:** `pvlearn 0.1.0`, `solaredge2mqtt` mit `pvlearn` im `[forecast]`-Extra. Für Nutzer verhaltensneutral.

---

### Phase 1b — Modell-Konsolidierung und Schema-Normalisierung

**Ziel:** Ein Energiemodell, kanonisches Feature-Schema, optionale Strahlungs-Features.

- `ForecasterType` und das Leistungsmodell entfallen vollständig; `Forecaster` kennt nur noch das Energieziel. pvlearn berechnet und publiziert keinen Leistungswert (siehe 3.3).
- `solaredge2mqtt` leitet `power_period` für die Dauer der Deprecation-Frist lokal aus `energy_period` ab, damit bestehende MQTT-Subscriber nicht sofort brechen. Der Shim lebt in solaredge2mqtt, nicht in pvlearn, und wird im Changelog als deprecated angekündigt.
- Feature-Konstanten auf das kanonische Schema aus Kapitel 3.1 umstellen.
- OWM-Adapter in solaredge2mqtt: mappt `OpenWeatherMapForecastData` auf das kanonische Schema. `weather_id` → WMO-Mapping.
- `SunEncoder` auf primitive Parameter umstellen, TZ explizit.
- `interval` in Konfiguration und Modell-Metadaten einführen, vorerst ausschließlich mit dem Wert 60 Minuten (siehe 3.5).
- Metriken beim Training berechnen und in den Metadaten ablegen (MAE, RMSE, R² auf einem `TimeSeriesSplit`-Holdout).
- `feature_schema_version = 1` einführen, Invalidierungslogik implementieren.

**Abnahme:**
- [x] Prognosequalität auf dem Referenzdatensatz nicht schlechter als Baseline (MAE-Vergleich, Toleranz definieren). Toleranz: 10 % relativ auf MAE, 0,05 absolut auf R², übernommen aus Phase 1a (Hardware-Rauschen, siehe Nachtrag zu Punkt 6). Erreicht: 641,54 Wh MAE gegenüber 620,88 Wh der Baseline, R² 0,8831 gegenüber 0,8859. Das Encoding selbst ist neutral — mit fixierter Baseline-Feature-Menge reproduziert der 1b-Code die Baseline exakt; die Differenz stammt vollständig aus der geänderten Feature-Auswahl, siehe `docs/adr/0001-feature-selection-threshold.md`.
- [x] `ephem`-Abhängigkeit entfernt. `astral` deckt Azimut, Elevation und Sonnenauf-/-untergang ab; das `season`-Feature entfällt ersatzlos, weil `day_of_year_sin/cos` dieselbe Jahresposition stetig statt in vier Stufen kodiert.
- [x] Feature-Auswahl providerunabhängig: `PFISelector` schneidet absolut (`importance > 0`) statt am 75. Perzentil der Kandidaten.
- [x] Bestandsnutzer trainieren beim Update automatisch neu, ohne Fehler im Log. Erbracht in [solaredge2mqtt PR #420](https://github.com/DerOetzi/solaredge2mqtt/pull/420): der OWM-Adapter übersetzt beim Lesen aus InfluxDB, nicht beim Schreiben, damit die vorhandene Historie ohne Migration weiter trainierbar bleibt; der `power_period`-Shim leitet das Feld lokal aus `energy_period` ab. Verifiziert mit dem dortigen Wiring-Regressionstest über 9580 Zeilen echter Historie.

**Release:** `pvlearn 0.2.0`, `solaredge2mqtt` Minor mit Changelog-Hinweis zur Deprecation von `power_period` und zum einmaligen Neutraining.

---

### Phase 1c — Korrektur der Feature-Auswahl

**Ziel:** Die drei in ADR 0001 offengelassenen Mängel am `PFISelector` sind behoben, bevor Modelle außerhalb von solaredge2mqtt entstehen.

Vorgezogen gegen die damalige Priorisierungsregel, alles Offene aus Kapitel 6 erst nach dem MVP anzugehen (der REST-Service war zu diesem Zeitpunkt noch als nächste Phase geplant, siehe Kapitel 4). Begründung: der gemischte Split ist ein Defekt, kein Abwägungspunkt, und seine Korrektur invalidiert jedes trainierte Modell. Solange nur solaredge2mqtt-Bestandsnutzer betroffen sind, kostet das ein automatisches Neutraining, das in Phase 1b ohnehin stattfindet. Mit jedem weiteren Aufrufer träfe es mehr Nutzer — und bis dahin steckte die zu optimistische Importance in jeder ausgelieferten Prognose.

- Importance wird auf dem **jüngsten Zehntel** der Zeilen gemessen statt auf einem zufälligen. `train_test_split(..., test_size=0.1, random_state=42)` mischt per Default; auf stündlich autokorrelierten Daten liegt zu jeder Testzeile deren Nachbarschaft in der Trainingshälfte.
- Die Schwelle wird **rauschbewusst**: `mean - 1·std > 0`. `permutation_importance` liefert `importances_std` ohne Zusatzkosten mit.
- **Zyklische Paare bleiben zusammen.** `sin` und `cos` eines Winkels sind ein Feature in zwei Spalten; ein `cos` ohne sein `sin` unterscheidet Vormittag nicht von Nachmittag.
- `pipeline_version` in den Modell-Metadaten einführen (siehe 3.4), sonst greift keine Invalidierung — das Feature-Vokabular ändert sich nicht.

Begründung, Messtabelle und die verworfenen Alternativen in `docs/adr/0002-noise-aware-chronological-feature-selection.md`.

**Abnahme:**
- [x] Prognosequalität auf dem Referenzdatensatz nicht schlechter als die Baseline, gleiche Toleranz wie Phase 1a/1b. Gemessen: 607,82 Wh MAE gegenüber 620,88 Wh, R² 0,8921 gegenüber 0,8859 — erstmals besser statt nur innerhalb der Toleranz. Erbracht durch `tests/test_extraction_regression.py`.
- [x] Der Selektor misst nachweislich auf den jüngsten Zeilen, nicht auf einer Stichprobe. `test_pfi_selector_measures_importance_on_the_most_recent_rows` prüft die Zeilen, die tatsächlich bei `permutation_importance` ankommen.
- [x] Ein vor 1c persistiertes Modell wird beim Laden mit klarer Begründung abgelehnt statt geladen. `test_load_rejects_a_model_from_an_older_pipeline` entfernt das Feld aus dem Sidecar, wie es vor 1c aussah, und erwartet `pipeline_version is 1`.
- [x] Kein `sin` wird ohne sein `cos` behalten und umgekehrt.

**Release:** `pvlearn 0.2.1`. Für Bestandsnutzer ein einmaliges automatisches Neutraining, im Changelog angekündigt.

Offen bleibt, was ADR 0001 und Kapitel 6 Punkt 7 offen lassen: Boruta, `SequentialFeatureSelector`, und die Frage, ob der Selektor überhaupt bleiben soll. Alle drei brauchen Daten von mehr als einer Anlage.

---

### Danach: Einbettung statt eigener Service

Ursprünglich für dieses Repository geplant: ein REST-Service mit Brain-Konzept (übernommen aus `learninghouse`), ein Home-Assistant-Add-on und eine HACS-Integration unter der Domain `pvlearn`, inklusive Migrationspfad weg von `solaredge2mqtt_forecast`. Das entfällt komplett — `learninghouse` bekommt Service, Add-on und Integration in eigener Rolle, mit eigener Roadmap in seinem Repository.

pvlearn bleibt dafür genau das, was Kapitel 1 beschreibt: eine I/O-freie Library, die `solaredge2mqtt` heute schon einbindet und `learninghouse` künftig ebenso einbindet. Alles, was ein Aufrufer dafür wissen muss, steht in Kapitel 3 (kanonisches Schema, Metadaten-Vertrag, Trainingsdaten-Vertrag). Was an der Library selbst noch offen ist, unabhängig vom Aufrufer, steht in Kapitel 6.

---

## 5. Querschnittsthemen

### Testing

- **Regressionstests gegen den eingefrorenen Referenzdatensatz** in jeder Phase. Das ist die wichtigste Absicherung des gesamten Vorhabens.
- Encoder-Unit-Tests: insbesondere `SunEncoder` gegen bekannte Sonnenstände und `CyclicalEncoder` an den Wrap-Around-Grenzen (359° → 0°).
- Property-Test: fehlende optionale Feature-Spalten dürfen nie zu einer Exception führen, nur zu einem kleineren Feature-Set.
- Zeitzonen-Tests mit mindestens einer Nicht-UTC-Zone und über einen DST-Wechsel hinweg. Der bestehende Code hat hier Prozess-globale Annahmen; beim Umbau auf Multi-Tenancy ist das die wahrscheinlichste Fehlerquelle.

### Versionierung

- pvlearn folgt SemVer. Ein Bump von `feature_schema_version` ist immer mindestens ein Minor-Release mit Changelog-Eintrag.
- `solaredge2mqtt` pinnt pvlearn auf `>=X.Y,<X+1`. `learninghouse` wird beim Einbinden dasselbe Pinning-Schema übernehmen.

### Dokumentation

Mindestumfang vor dem ersten öffentlichen Release der Library:

- README mit Abgrenzung zu Forecast.Solar/Solcast — die Frage „warum noch eine Solarprognose" kommt garantiert und verdient eine gute Antwort.
- Erklärung, warum auf Vorhersagewetter statt Messwetter trainiert wird. Das ist kontraintuitiv und wird sonst als Bug gemeldet.
- Hinweis auf die 60-Stunden-Mindestdatenmenge und darauf, dass die Qualität über Wochen deutlich steigt.

---

## 6. Offene Entscheidungen

Diese Punkte sollten vor Beginn der jeweiligen Phase geklärt werden. Entschiedene Punkte bleiben mit Verweis auf die Begründung stehen, statt gelöscht zu werden.

1. ~~**Prognoseintervall**~~ — **entschieden**, siehe 3.5. Das MVP rechnet stündlich, das Datenmodell hält feinere Auflösungen offen. Eine Implementierung kommt frühestens, wenn die Messdatenerfassung auf Aufrufer-Seite mitzieht.
2. **Unsicherheitsbänder:** `HistGradientBoostingRegressor` kann über `loss="quantile"` Quantilsprognosen liefern. Ein p10/p50/p90-Band wäre für Batteriesteuerung deutlich wertvoller als ein Punktwert — kostet aber drei Modelle statt einem, was der Konsolidierung aus 3.3 entgegenläuft. Kandidat für eine spätere Ausbaustufe der Library, unabhängig vom Aufrufer.
3. **Mehrere Strings pro Anlage:** Ost-West-Anlagen könnten von getrennten Modellen je Ausrichtung profitieren. Erfordert, dass der Aufrufer getrennte Energiewerte liefert. Als optionales Feature denkbar; erhöht die Komplexität der Library-Schnittstelle spürbar.
4. ~~**Hyperparameter-Tuning**~~ — **entschieden.** 18 Kombinationen (`max_iter` × `max_depth` × `learning_rate`) auf `TimeSeriesSplit(n_splits=2)` sind 36 vollständige Pipeline-Fits — jeder davon inklusive `PFISelector`s eigener `permutation_importance`, nicht 36 billige Fits. Auf schwacher Hardware (z. B. Raspberry Pi) grenzwertig, deshalb ist es standardmäßig aus (`ForecasterConfig.hyperparametertuning = False`). `_hyperparametertuning` in `pvlearn/forecaster.py` nutzt seit dieser Entscheidung `HalvingGridSearchCV` statt `GridSearchCV`: alle 18 Kombinationen starten auf einem kleinen Datenanteil, verwirft die schwachen früh und gibt nur den Überlebenden mehr Daten, bei ähnlicher Abdeckung mit weniger Gesamt-Fits. `random_state=42` macht die Rungen-Aufteilung reproduzierbar, konsistent mit dem Rest der Pipeline (siehe 6.6).
5. ~~**Rückwärtsbefüllung**~~ — **außerhalb des Library-Scopes.** Ob und wie ein künftiger Aufrufer (perspektivisch `learninghouse`) beim Setup historische Werte nachliefert, ändert nichts an pvlearn selbst: die Library nimmt Trainingszeilen entgegen, unabhängig davon, ob sie live oder nachträglich zugestellt werden. Das Wetter-Vorhersage-Problem dabei (fehlende historische Vorhersagen, siehe Open-Meteos Historical-Forecast-API) ist ein I/O-Thema und gehört laut Architekturprinzip (Kapitel 1) nicht in pvlearn.
6. ~~**scikit-learn-Obergrenze**~~ — **entschieden**. Alle Abhängigkeiten sind in `pyproject.toml` exakt gepinnt, wie in `solaredge2mqtt` und `learninghouse`. Empirisch geprüft: die Baseline reproduziert bitidentisch über numpy 2.4.6/2.5.1, pandas 3.0.3/3.0.5 und scipy 1.17.1/1.18.0 hinweg, solange scikit-learn auf 1.9.0 bleibt. Damit ist scikit-learn der einzige *Library*-Pin, an dem die Reproduzierbarkeit hängt — ein Bump erfordert zwingend eine neu erzeugte Baseline und einen Changelog-Eintrag.

   **Nachtrag aus Phase 1a:** Bitidentität gilt nur auf derselben Maschine. `HistGradientBoostingRegressor`s Split-Suche reagiert auf CPU-mikroarchitekturabhängiges Floating-Point-Rundungsverhalten (SIMD-Reduktionsreihenfolge) — bei einer knappen Split-Schwelle kippt das den gewählten Split und damit den gesamten Baum, unabhängig von `random_state`, Thread-/Prozesszahl oder Python-Version (alles einzeln getestet und ausgeschlossen). Auf CI-Runnern mit anderer CPU als der Erzeugungsmaschine weichen Prognosen daher sichtbar ab. Regressionstests gegen die Baseline vergleichen deshalb ab Phase 1a Prognosegüte (MAE/R² innerhalb Toleranz) statt exakter Werte — siehe `tests/test_extraction_regression.py`.

   **Nachtrag aus Phase 1b:** Die Toleranz deckt Hardware-Rauschen ab, nicht Modelländerungen. Als das kanonische Schema die Feature-Auswahl kippen ließ (Punkt 7), hätte sie die Verschlechterung still absorbiert. Die Ursache wurde stattdessen isoliert und behoben; eine Abweichung innerhalb der Toleranz ist kein Freibrief, sondern ein Anlass nachzusehen, ob sie von der Maschine kommt oder vom Modell.

7. **Feature-Selektion:** Phase 1b hat `PFISelector` von einer Quantils- auf eine absolute Schwelle umgestellt (`importance > 0`), weil ein Quantil die Auswahl an die Zahl der gelieferten Provider-Spalten koppelt — Begründung und Messtabelle in `docs/adr/0001-feature-selection-threshold.md`. Phase 1c hat darauf aufgesetzt: chronologische Messung, rauschbewusste Schwelle `mean - 1·std > 0`, zyklische Paare zusammengehalten (`docs/adr/0002-noise-aware-chronological-feature-selection.md`). Was danach noch offen ist, nach Priorität:

   - ~~**Rauschbewusste Schwelle**~~ — **entschieden in Phase 1c**, `k = 1`, a priori gewählt statt aus der Messtabelle abgelesen. Siehe ADR 0002.
   - **Boruta / Shadow Features:** Vergleich gegen permutierte Kopien jedes Features. Statistisch sauber begründet, kostet mehrere Fits.
   - **`SequentialFeatureSelector` oder RFECV mit `TimeSeriesSplit`:** optimiert direkt die Zielmetrik statt einer Heuristik und wählt die Feature-Zahl über den CV-Score. Teuer — verschärft Punkt 4 auf schwacher Hardware deutlich.
   - **Selector ganz streichen:** auf dem Referenzdatensatz liegt „keine Auswahl" (638,23 Wh) gleichauf mit `importance > 0` (641,54 Wh) und besser als das alte Perzentil 75 (668,51 Wh). `HistGradientBoostingRegressor` ist gegenüber irrelevanten Features robust. Entscheidbar erst mit Daten mehrerer Anlagen, weil `selected_features` Teil der Modell-Metadaten aus 3.4 ist.

   ~~**Unabhängiger Defekt, gleiche Stelle:**~~ — **behoben in Phase 1c.** Der gemischte Split hielt auf dem Referenzdatensatz sieben Features für wichtig, die ein chronologischer verwirft, darunter `time_dst` und `time_daylight` — über lange Strecken konstant und deshalb aus jeder Nachbarzeile rekonstruierbar.

---

## 7. Risiken

| Risiko | Auswirkung | Gegenmaßnahme |
|---|---|---|
| Extraktionsfehler bleiben unentdeckt, weil gleichzeitig fachliche Änderungen erfolgen | Schwer lokalisierbare Prognosefehler | Strikte Trennung Phase 1a / 1b, Regressionstest gegen Baseline |
| Zeitzonen- und DST-Fehler durch Wegfall der Prozess-globalen TZ | Systematisch verschobene Prognosen | Explizite Tests über DST-Wechsel, TZ als Pflichtfeld im Brain |
| Providerwechsel liefert stumme Qualitätsverschlechterung | Nutzer bemerkt es nicht | Provider Teil der Modell-Metadaten, harte Invalidierung, Metriken in den Metadaten sichtbar |
| Scope Creep über die offenen Punkte in Kapitel 6 | Library kommt nicht zum Stillstand | Phase 0–1c als Library-Scope; alles aus Kapitel 6 bleibt vertagt, bis mehrere Anlagen Daten liefern oder ein Aufrufer den Bedarf konkret macht |

---

## 8. Zusammenfassung des kritischen Pfads

```
P0  Referenzdaten + Baseline einfrieren        ← ohne das ist nichts verifizierbar
 │
P1a Extraktion verhaltensgleich                 → pvlearn 0.1.0
 │
P1b Ein Energiemodell + kanonisches Schema      → pvlearn 0.2.0
 │
P1c Chronologische, rauschbewusste Selektion    → pvlearn 0.2.1  ← Library-Scope Ende
```

Danach ist pvlearn eine fertige Library mit zwei Aufrufern: `solaredge2mqtt` heute per Dependency, `learninghouse` künftig ebenso — beide außerhalb dieses Plans, mit eigener Roadmap in ihrem jeweiligen Repository.

Die zwei Entscheidungen, die am schwersten zu revidieren sind und deshalb die meiste Sorgfalt verdienen: das **Feature-Schema** (Kapitel 3.1) und der **Trainingsdaten-Vertrag** (Kapitel 3.6). Die **Prognoseauflösung** (Kapitel 3.5) ist inzwischen entschieden; entscheidend bleibt, dass das Intervall von Anfang an ein explizites Feld ist und nirgends implizit als eine Stunde angenommen wird.
