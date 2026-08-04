# pvlearn — Umsetzungsplan

**Repository:** `LearningHouseService/pvlearn`
**Ziel:** Herauslösen des ML-Forecast-Teils aus `DerOetzi/solaredge2mqtt` in eine eigenständige, anlagenunabhängige Library nebst REST-Service, Home-Assistant-Add-on und HACS-Integration.

**Tagline:** *Teach your home to predict its own solar production.*

---

## 1. Zielbild und Abgrenzung

### Was pvlearn ist

Ein selbstlernender PV-Ertragsprognose-Stack, der auf den **eigenen historischen Messdaten** einer Anlage trainiert, statt auf einem generischen physikalischen Anlagenmodell. Das ist die Abgrenzung zu Forecast.Solar und Solcast: keine Angabe von Ausrichtung, Neigung oder kWp nötig — das Modell lernt die Charakteristik der Anlage inklusive Verschattung, Ost-West-Splits und Wechselrichter-Derating implizit aus den Daten.

### Was pvlearn nicht ist

- Kein generischer ML-Service — das ist `learninghouse`.
- Keine Verbrauchs- oder Lastprognose. Bewusste Grenze; falls das später kommt, gehört es zu `learninghouse` oder in ein eigenes Projekt.
- Keine Datenerfassung. pvlearn misst nichts selbst, es bekommt Energiewerte geliefert.

### Architekturprinzip

Die Library ist **I/O-frei**. Kein MQTT, kein InfluxDB, kein HTTP, kein Dateisystemzugriff außerhalb explizit übergebener Pfade. Rein: DataFrames und typisierte Modelle. Raus: Prognosen. Alles andere sind Adapter in den darüberliegenden Schichten. Genau das macht sie gleichzeitig in `solaredge2mqtt` einbettbar und als Service betreibbar.

---

## 2. Artefakte und Namensschema

| Artefakt | Name | Repository |
|---|---|---|
| Library + Service | `pvlearn`, Service als Extra `pvlearn[service]` | `LearningHouseService/pvlearn` |
| PyPI-Paket | `pvlearn` | — |
| Docker-Image | `ghcr.io/learninghouseservice/pvlearn` | — |
| HA-Add-on-Slug | `pvlearn` | `LearningHouseService/hassio-addons` |
| HACS-Integration | Domain `pvlearn` | `LearningHouseService/pvlearn-hass` |

Library und Service liegen in **einem** Repository, weil sie im selben Takt released werden. Zwei Repos bedeuten zwei Release-Zyklen und eine Versions-Kompatibilitätsmatrix ohne Gegenwert. Die HACS-Integration bekommt ein eigenes Repo, weil HACS das erzwingt und ihr Release-Zyklus an Home-Assistant-Versionen hängt, nicht an denen von pvlearn.

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

Es bleibt ein Breaking Change für unbekannte Dritt-Subscriber des MQTT-Topics. Behandlung analog zu Phase 5: Feld als deprecated ankündigen, mindestens zwei Minor-Releases weiter publizieren, dann entfernen. Im Changelog mit dem Hinweis, dass sich die Momentanleistung aus `energy_period` und dem Intervall trivial selbst berechnen lässt.

Für pvlearn selbst gibt es nichts zu deprecaten — die Library ist neu und publiziert den Wert nie.

### 3.4 Modell-Metadaten und Invalidierung

Jedes persistierte Modell trägt:

```json
{
  "pvlearn_version": "0.1.0",
  "feature_schema_version": 1,
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

Beim Laden gilt hart: **Stimmt `feature_schema_version`, die sklearn-Minor-Version, der Provider, das Intervall oder die Location nicht überein, wird das Modell verworfen und neu trainiert.** Kein Migrationsversuch, kein Best-Effort-Laden. Stumme Fehlprognosen durch ein Modell, das auf einem anderen Feature-Set trainiert wurde, sind praktisch nicht debugbar.

Zur sklearn-Version: die Phase-0-Baseline ist nur gegen exakt die Version reproduzierbar, unter der sie entstanden ist. `pvlearn` pinnt scikit-learn deshalb exakt (siehe 6.6); ein Bump verschiebt still jede Prognose und erfordert eine neu erzeugte Baseline.

**Persistenzformat:** joblib/Pickle. ONNX wurde geprüft und verworfen — `CyclicalEncoder`, `TimeEncoder`, `SunEncoder` und `PFISelector` sind Custom-Transformer und bräuchten je einen eigenen Shape Calculator plus Converter. Der ursprüngliche Motivator (leichtgewichtige Inferenz in der HA-Integration) entfällt ohnehin, weil die Integration in der Zielarchitektur ein reiner REST-Client ist.


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

Frühestens Phase 6, konsistent mit der Gegenmaßnahme zum Scope-Creep-Risiko in Kapitel 7.
---

## 4. Phasenplan

Die ursprünglich geplante Reihenfolge wird an zwei Stellen angepasst: Phase 1 wird zweigeteilt, und das Add-on wird vor die HACS-Integration gezogen.

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
- [ ] Bestandsnutzer trainieren beim Update automatisch neu, ohne Fehler im Log. Offen — liegt in `solaredge2mqtt` und wird mit dem dortigen Pull Request erbracht (OWM-Adapter auf das kanonische Schema, `power_period`-Shim).

**Release:** `pvlearn 0.2.0`, `solaredge2mqtt` Minor mit Changelog-Hinweis zur Deprecation von `power_period` und zum einmaligen Neutraining.

---

### Phase 2 — REST-Service

**Ziel:** Eigenständig betreibbarer Service mit konfigurierbaren Wetter-Providern, Trainingsdatenhaltung und Prognose-Endpunkt.

#### Brain-Konzept

Ein Brain = eine PV-Anlage. Übernommen aus `learninghouse`, inklusive Verzeichnisstruktur:

```
brains/
  <brain_id>/
    config.json          # Location, Provider, Hyperparameter-Flags
    training_data.sqlite # Wetter-Snapshots + gemessene Energie
    trained.pkl          # Modell
    metadata.json        # siehe 3.4
```

**Speicher:** SQLite, nicht InfluxDB. Bei stündlicher Auflösung fallen 8.760 Zeilen pro Jahr an — dafür braucht es keine Zeitreihendatenbank, und ein Add-on, das InfluxDB voraussetzt, installiert kaum jemand. Optional später ein Export-Endpunkt nach InfluxDB.

#### Trainingsdaten-Semantik

**Kritisch und leicht falsch zu bauen:** Der bestehende Code trainiert auf der *Vorhersage*, die eine Stunde zuvor für die betreffende Stunde galt — nicht auf beobachtetem Wetter. Das ist korrekt, weil Trainings- und Inferenzverteilung übereinstimmen müssen; ein auf Messwerten trainiertes Modell bekommt zur Inferenzzeit systematisch andere Eingaben.

Für die API bedeutet das eine klare Arbeitsteilung:

1. Der Service ruft **stündlich selbst** beim Provider die Vorhersage ab und persistiert den Snapshot für die kommende Stunde.
2. Der Client (solaredge2mqtt, HA-Integration, beliebig) pusht nachträglich nur `{timestamp, energy_wh}`.
3. Der Join geschieht serverseitig über den Zeitstempel.

Der Client braucht damit keinerlei Wetterkenntnis. Das ist die eigentliche Vereinfachung gegenüber heute.

#### API-Entwurf

```
GET    /api/v1/brains                        Liste
POST   /api/v1/brains                        Anlegen (Location, Provider, Optionen)
GET    /api/v1/brains/{id}                   Config + Status
PUT    /api/v1/brains/{id}                   Ändern (invalidiert Modell bei Location-/Provider-Wechsel)
DELETE /api/v1/brains/{id}

POST   /api/v1/brains/{id}/measurements      [{timestamp, energy_wh}, ...]
POST   /api/v1/brains/{id}/train             Training anstoßen (async, 202)
GET    /api/v1/brains/{id}/status            is_trained, rows, last_training, metrics, features
GET    /api/v1/brains/{id}/forecast          ?days=2 → Intervallwerte + Aggregate
GET    /api/v1/providers                     verfügbare Provider + max. Horizont
GET    /health
```

Der Prognosehorizont ist providerabhängig und wird nicht hart kodiert: OpenWeatherMap One Call liefert 48 h stündlich, Open-Meteo bis zu 16 Tage. `GET /forecast` liefert maximal `min(days, provider_horizon)` und meldet den tatsächlichen Horizont im Response mit, zusammen mit dem Intervall der gelieferten Werte.

**Auth:** API-Key-Mechanismus aus `learninghouse` übernehmen.

#### Provider-Adapter

- **Open-Meteo** als Default. Kostenlos, kein API-Key, liefert GHI/DNI/DHI, für DE zusätzlich ICON-D2. Der Wegfall der OWM-Pflicht ist der wichtigste Adoptionshebel des Projekts.
- **OpenWeatherMap** als Option für Bestandsnutzer.
- **DWD** direkt: optional, später.
- Interface als `Protocol` in pvlearn, Adapter im Service-Extra.

**Abnahme:**
- Service läuft standalone im Container, legt Brain an, sammelt Wetter-Snapshots, nimmt Messwerte entgegen, trainiert nach 60 h, liefert Prognosen.
- Zwei Brains mit unterschiedlichen Zeitzonen laufen parallel korrekt.
- Provider-Wechsel invalidiert das Modell nachweislich.
- OpenAPI-Schema ist vollständig, `/docs` nutzbar.

**Release:** `pvlearn 0.3.0` mit `[service]`-Extra, Docker-Image auf ghcr.

---

### Phase 3 — Home-Assistant-Add-on

**Ziel:** Ein-Klick-Installation des Services für HA-Nutzer. Vorgezogen vor die HACS-Integration, weil diese ohne laufenden Service wertlos ist und der erste Eindruck sonst „ich muss erst irgendwo Docker aufsetzen" lautet.

- Add-on-Repository `LearningHouseService/hassio-addons` anlegen.
- `config.yaml` mit Optionen: Provider, API-Key (falls OWM), Log-Level, Port.
- Multi-Arch-Build: `amd64`, `aarch64`. **`armv7` wird nicht unterstützt** — dieselbe Dependency-Problematik wie bisher in solaredge2mqtt, das gehört prominent in die Doku.
- Ingress für die Konfigurations-UI (falls aus learninghouse übernommen).
- Persistenz des `brains`-Verzeichnisses über `/data`.

**Abnahme:** Add-on installiert sich auf HAOS aus dem Repository, startet, überlebt Neustart und Add-on-Update mit erhaltenem Modell.

---

### Phase 4 — HACS-Integration

**Ziel:** `pvlearn`-Integration in Home Assistant, vergleichbar mit `solaredge2mqtt_forecast`, aber als REST-Client gegen den Service.

- Config Flow: Host, Port, API-Key, Auswahl/Anlage des Brains, Auswahl der Energie-Entität als Datenquelle.
- **DataUpdateCoordinator** holt zyklisch die Prognose.
- **Messwert-Push:** stündlich den Energieertrag der letzten Stunde aus der gewählten Entität ermitteln und an den Service senden. Quelle: `recorder`-Statistics (`statistics_during_period`), nicht der aktuelle State — sonst gehen Werte bei HA-Neustarts verloren.
- Entitäten: Sensoren analog zum bestehenden Forecast-Modell (heute, heute verbleibend, aktuelle Stunde, nächste Stunde, morgen) plus Diagnose-Sensoren (Modellstatus, Trainingsdatenpunkte, MAE).
- **Solar-Forecast-Provider für das Energy Dashboard** implementieren — das ist der eigentliche Mehrwert für Endnutzer.
- Dependencies im `manifest.json`: nur ein schlanker HTTP-Client. Kein numpy, kein sklearn, kein pandas.

**Abnahme:**
- Integration installiert sich über HACS, Config Flow durchläuft, Entitäten erscheinen.
- Prognose ist im Energy Dashboard als Solar-Forecast auswählbar.
- Kein Blocking-Call im Event-Loop (HA-Warnung „doing blocking calls" bleibt aus).

---

### Phase 5 — Migration und Deprecation

**`solaredge2mqtt_forecast`** wird **nicht umbenannt**. Ein Domain-Wechsel bricht bei Bestandsnutzern den Config Entry, die Entity-IDs und damit die Recorder-Statistiken — gerade beim Energy Dashboard schmerzhaft.

Stattdessen:

1. `pvlearn` als neue Domain parallel ausliefern.
2. `solaredge2mqtt_forecast` als deprecated markieren, Repo-Beschreibung und README anpassen, Repair-Issue in HA anzeigen.
3. Migrationsdokument: welche Entitäten entsprechen einander, wie überträgt man Long-Term-Statistics (bzw. dass man es nicht kann und was das bedeutet).
4. Mindestens zwei Minor-Releases Karenz, dann Archivierung.

**solaredge2mqtt** behält den eingebauten Forecast über die Library — der Service ist für dessen Nutzer **kein** Pflichtbestandteil. Wer InfluxDB und OWM bereits betreibt, ändert nichts außer der Konfiguration des Wetter-Providers.

**`power_period` entfernen.** Die in Phase 1b begonnene Deprecation-Frist läuft hier aus: der Ableitungs-Shim in solaredge2mqtt fällt weg, das Feld verschwindet aus dem MQTT-Payload, und `ForecastData.power_period` verschwindet aus der HACS-Integration, die es ohnehin nie verwendet hat.

---

## 5. Querschnittsthemen

### Testing

- **Regressionstests gegen den eingefrorenen Referenzdatensatz** in jeder Phase. Das ist die wichtigste Absicherung des gesamten Vorhabens.
- Encoder-Unit-Tests: insbesondere `SunEncoder` gegen bekannte Sonnenstände und `CyclicalEncoder` an den Wrap-Around-Grenzen (359° → 0°).
- Property-Test: fehlende optionale Feature-Spalten dürfen nie zu einer Exception führen, nur zu einem kleineren Feature-Set.
- Zeitzonen-Tests mit mindestens einer Nicht-UTC-Zone und über einen DST-Wechsel hinweg. Der bestehende Code hat hier Prozess-globale Annahmen; beim Umbau auf Multi-Tenancy ist das die wahrscheinlichste Fehlerquelle.
- Service-Tests mit gemocktem Provider, damit CI ohne Netz läuft.

### Versionierung

- pvlearn folgt SemVer. Ein Bump von `feature_schema_version` ist immer mindestens ein Minor-Release mit Changelog-Eintrag.
- `solaredge2mqtt` pinnt pvlearn auf `>=X.Y,<X+1`.
- Die HACS-Integration prüft beim Setup die Service-Version und meldet Inkompatibilität als Repair-Issue, statt still zu scheitern.

### Dokumentation

Mindestumfang vor dem ersten öffentlichen Release der Library:

- README mit Abgrenzung zu Forecast.Solar/Solcast — die Frage „warum noch eine Solarprognose" kommt garantiert und verdient eine gute Antwort.
- Erklärung, warum auf Vorhersagewetter statt Messwetter trainiert wird. Das ist kontraintuitiv und wird sonst als Bug gemeldet.
- Hinweis auf die 60-Stunden-Mindestdatenmenge und darauf, dass die Qualität über Wochen deutlich steigt.
- armv7-Einschränkung prominent.

---

## 6. Offene Entscheidungen

Diese Punkte sollten vor Beginn der jeweiligen Phase geklärt werden. Entschiedene Punkte bleiben mit Verweis auf die Begründung stehen, statt gelöscht zu werden.

1. ~~**Prognoseintervall**~~ — **entschieden**, siehe 3.5. Das MVP rechnet stündlich, das Datenmodell hält feinere Auflösungen offen. Eine Implementierung kommt frühestens in Phase 6 und setzt voraus, dass die Messdatenerfassung auf der Client-Seite mitzieht.
2. **Unsicherheitsbänder:** `HistGradientBoostingRegressor` kann über `loss="quantile"` Quantilsprognosen liefern. Ein p10/p50/p90-Band wäre für Batteriesteuerung deutlich wertvoller als ein Punktwert — kostet aber drei Modelle statt einem, was der Konsolidierung aus 3.3 entgegenläuft. Kandidat für Phase 6.
3. **Mehrere Strings pro Anlage:** Ost-West-Anlagen könnten von getrennten Modellen je Ausrichtung profitieren. Erfordert, dass der Client getrennte Energiewerte liefert. Als optionales Feature denkbar; erhöht die Komplexität der API spürbar.
4. **Hyperparameter-Tuning im Service:** `GridSearchCV` über neun Kombinationen ist auf einem Raspberry Pi grenzwertig. Entweder deaktivieren, auf gelegentlich (wöchentlich) begrenzen oder auf `HalvingGridSearchCV` wechseln.
5. **Rückwärtsbefüllung:** Soll die HA-Integration beim Setup historische Werte aus dem Recorder nachliefern können? Das würde die Wartezeit bis zur ersten Prognose drastisch verkürzen — allerdings fehlen für die Vergangenheit die passenden Wetter-*Vorhersagen*. Open-Meteo bietet eine Historical-Forecast-API, die genau das liefert (archivierte Vorhersagen statt Reanalyse). Technisch die eleganteste Lösung des Kaltstartproblems, aber nicht trivial.
6. ~~**scikit-learn-Obergrenze**~~ — **entschieden**. Alle Abhängigkeiten sind in `pyproject.toml` exakt gepinnt, wie in `solaredge2mqtt` und `learninghouse`. Empirisch geprüft: die Baseline reproduziert bitidentisch über numpy 2.4.6/2.5.1, pandas 3.0.3/3.0.5 und scipy 1.17.1/1.18.0 hinweg, solange scikit-learn auf 1.9.0 bleibt. Damit ist scikit-learn der einzige *Library*-Pin, an dem die Reproduzierbarkeit hängt — ein Bump erfordert zwingend eine neu erzeugte Baseline und einen Changelog-Eintrag.

   **Nachtrag aus Phase 1a:** Bitidentität gilt nur auf derselben Maschine. `HistGradientBoostingRegressor`s Split-Suche reagiert auf CPU-mikroarchitekturabhängiges Floating-Point-Rundungsverhalten (SIMD-Reduktionsreihenfolge) — bei einer knappen Split-Schwelle kippt das den gewählten Split und damit den gesamten Baum, unabhängig von `random_state`, Thread-/Prozesszahl oder Python-Version (alles einzeln getestet und ausgeschlossen). Auf CI-Runnern mit anderer CPU als der Erzeugungsmaschine weichen Prognosen daher sichtbar ab. Regressionstests gegen die Baseline vergleichen deshalb ab Phase 1a Prognosegüte (MAE/R² innerhalb Toleranz) statt exakter Werte — siehe `tests/test_extraction_regression.py`.

   **Nachtrag aus Phase 1b:** Die Toleranz deckt Hardware-Rauschen ab, nicht Modelländerungen. Als das kanonische Schema die Feature-Auswahl kippen ließ (Punkt 7), hätte sie die Verschlechterung still absorbiert. Die Ursache wurde stattdessen isoliert und behoben; eine Abweichung innerhalb der Toleranz ist kein Freibrief, sondern ein Anlass nachzusehen, ob sie von der Maschine kommt oder vom Modell.

7. **Feature-Selektion:** Phase 1b hat `PFISelector` von einer Quantils- auf eine absolute Schwelle umgestellt (`importance > 0`), weil ein Quantil die Auswahl an die Zahl der gelieferten Provider-Spalten koppelt — Begründung und Messtabelle in `docs/adr/0001-feature-selection-threshold.md`. Das ist die kleinstmögliche Korrektur, nicht das Optimum. Für später, nach Priorität:

   - **Rauschbewusste Schwelle:** `permutation_importance` liefert `importances_std` gleich mit. `mean - k·std > 0` filtert Features, deren Importance nur Rauschen ist, kostet keinen zusätzlichen Fit und bleibt kandidatenzahl-unabhängig. Der naheliegendste nächste Schritt.
   - **Boruta / Shadow Features:** Vergleich gegen permutierte Kopien jedes Features. Statistisch sauber begründet, kostet mehrere Fits.
   - **`SequentialFeatureSelector` oder RFECV mit `TimeSeriesSplit`:** optimiert direkt die Zielmetrik statt einer Heuristik und wählt die Feature-Zahl über den CV-Score. Teuer — verschärft Punkt 4 auf schwacher Hardware deutlich.
   - **Selector ganz streichen:** auf dem Referenzdatensatz liegt „keine Auswahl" (638,23 Wh) gleichauf mit `importance > 0` (641,54 Wh) und besser als das alte Perzentil 75 (668,51 Wh). `HistGradientBoostingRegressor` ist gegenüber irrelevanten Features robust. Entscheidbar erst mit Daten mehrerer Anlagen, weil `selected_features` Teil der Modell-Metadaten aus 3.4 ist.

   **Unabhängiger Defekt, gleiche Stelle:** `PFISelector.fit` splittet mit `train_test_split(..., test_size=0.1, random_state=42)`, also per Default gemischt. Auf stündlich autokorrelierten Daten landen Nachbarstunden in beiden Hälften und die Importances fallen systematisch zu optimistisch aus. Ein chronologischer Split ist korrekt; die Änderung invalidiert jedes trainierte Modell und gehört deshalb in dieselbe Runde wie eine der obigen Umstellungen.

---

## 7. Risiken

| Risiko | Auswirkung | Gegenmaßnahme |
|---|---|---|
| Extraktionsfehler bleiben unentdeckt, weil gleichzeitig fachliche Änderungen erfolgen | Schwer lokalisierbare Prognosefehler | Strikte Trennung Phase 1a / 1b, Regressionstest gegen Baseline |
| Zeitzonen- und DST-Fehler durch Wegfall der Prozess-globalen TZ | Systematisch verschobene Prognosen | Explizite Tests über DST-Wechsel, TZ als Pflichtfeld im Brain |
| Providerwechsel liefert stumme Qualitätsverschlechterung | Nutzer bemerkt es nicht | Provider Teil der Modell-Metadaten, harte Invalidierung, Metriken im Status-Endpunkt sichtbar |
| Scope Creep über die offenen Punkte in Kapitel 6 | Projekt kommt nicht zum ersten Release | Phase 1a–2 als MVP definieren, alles aus Kapitel 6 nach Phase 4 |
| HACS-Aufnahme scheitert an Qualitätsanforderungen | Verteilung nur über Custom Repository | Frühzeitig gegen die HACS- und HA-Integration-Quality-Scale prüfen |
| Kaltstart von 60 h schreckt Neunutzer ab | Geringe Adoption | Statusanzeige mit Fortschritt in der Integration; mittelfristig Punkt 6.5 |

---

## 8. Zusammenfassung des kritischen Pfads

```
P0  Referenzdaten + Baseline einfrieren        ← ohne das ist nichts verifizierbar
 │
P1a Extraktion verhaltensgleich                 → pvlearn 0.1.0
 │
P1b Ein Energiemodell + kanonisches Schema      → pvlearn 0.2.0
 │
P2  REST-Service + Open-Meteo                   → pvlearn 0.3.0  ← MVP-Ende
 │
P3  HA-Add-on
 │
P4  HACS-Integration + Energy-Dashboard-Provider
 │
P5  Deprecation solaredge2mqtt_forecast
```

Die drei Entscheidungen, die am schwersten zu revidieren sind und deshalb die meiste Sorgfalt verdienen: das **Feature-Schema** (Kapitel 3.1), die **Trainingsdaten-Semantik** (Phase 2) und die **Prognoseauflösung** (Kapitel 3.5). Die dritte ist inzwischen entschieden; entscheidend bleibt, dass das Intervall von Anfang an ein explizites Feld ist und nirgends implizit als eine Stunde angenommen wird.
