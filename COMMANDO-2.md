# AGRIGUARD — COMMANDO-2
## MODALITY 2: REAL-TIME AGRICULTURAL CONTEXT MODULE

**Version:** 1.0  
**Status:** Authoritative implementation specification  
**Project:** AgriGuard  
**Module:** Modality 2  
**Baseline cost:** ₹0  

---

# 1. PURPOSE

Implement Modality 2 of AgriGuard as a **real-time agricultural context acquisition and normalization module**.

Model 2 is **not** a second disease-classification model.

The baseline implementation must:

- consume Model 1 output;
- obtain crop/disease information from the authoritative Model 1 class mapping;
- collect growth stage from the farmer/user;
- obtain the **crop/field location**, not automatically the uploader's location;
- retrieve weather information for that crop/field location using a free weather API;
- validate and normalize context;
- preserve raw and normalized evidence for the research paper;
- remain independent of Model 1 so Model 1 works without internet, GPS, or weather data.

The baseline implementation must **not train another ML model** and must **not require an external agricultural-context training dataset**.

---

# 2. ARCHITECTURE

## Modality 1 — Visual

```text
Leaf Image
    ↓
Model 1
    ↓
38-class crop-disease prediction
    ↓
Predicted class + confidence
```

Example:

```text
predicted_label = Tomato___Early_blight
confidence       = 0.9400433
```

Model 1 predicts a **38-class crop-disease category**. It does not contain a separate crop classifier.

The crop identity may be derived from the authoritative class label.

Example:

```text
Tomato___Early_blight
        ↓
Crop: Tomato
Disease: Early blight
```

Do not claim that Model 1 independently predicts crop and disease as two separate outputs.

## Modality 2 — Agricultural Context

```text
Model 1 prediction
        +
Farmer-provided growth stage
        +
Crop/field location
        +
Free weather API
        ↓
Normalized Agricultural Context
```

## Overall flow

```text
                         AGRIGUARD
                             │
              ┌──────────────┴──────────────┐
              │                             │
        MODALITY 1                    MODALITY 2
        Leaf Image                 Agricultural Context
              │                             │
              ↓                             ↓
           MODEL 1                  Context Acquisition
              │                             │
              │                 ┌───────────┼───────────┐
              │                 │           │           │
              │                 ↓           ↓           ↓
              │             Crop/Field   Growth      Weather
              │             Location      Stage       API
              │                 │           │           │
              │                 └───────────┼───────────┘
              │                             │
              └──────────────┬──────────────┘
                             ↓
                  Context-Aware Assessment
                             │
                    ┌────────┴────────┐
                    ↓                 ↓
                   XAI             ADVISORY
```

---

# 3. CRITICAL SCOPE DECISION

There is **NO second ML disease classifier** in baseline M2.

Do NOT:

- train another CNN;
- train another neural network;
- train a tabular disease classifier;
- train a weather-based disease classifier;
- train a second crop classifier;
- download GVLiD;
- download another disease dataset;
- download an agricultural-context training dataset;
- create a synthetic agricultural-context training dataset;
- derive environmental context from PlantVillage labels;
- use disease labels as environmental features;
- use filenames as environmental features;
- use folder names as environmental features;
- use Leaf IDs as environmental features;
- use source/lab prefixes as context features;
- fabricate missing weather;
- fabricate missing location;
- invent another disease prediction;
- invent another confidence score;
- calculate arbitrary fusion confidence;
- modify Model 1.

M2 is a **real-time context acquisition module**, not another disease predictor.

---

# 4. MODEL 1 INTEGRATION

Existing Model 1 artifacts:

```text
models/agriguard_mobilenetv2.keras
models/class_names.json
```

Existing inference interface:

```text
src/predict.py
```

Expected output:

```text
predicted_index
predicted_label
confidence
output_dimension
```

Example:

```text
predicted_index: 29
predicted_label: Tomato___Early_blight
confidence: 0.9400433
output_dimension: 38
```

Model 1 is read-only.

Do not change Model 1 preprocessing, training, evaluation, class mapping, saved model, or evidence artifacts.

---

# 5. CROP AND DISEASE EXTRACTION

The authoritative source is:

```text
models/class_names.json
```

A class such as:

```text
Tomato___Early_blight
```

may be interpreted deterministically as:

```text
crop = Tomato
disease = Early blight
```

Do not infer crop from arbitrary filenames, directories, image appearance, Leaf IDs, or source metadata.

If a class cannot be parsed safely, return a clear error instead of guessing.

---

# 6. GROWTH STAGE

Growth stage is **farmer/user-provided context**. It is not predicted by Model 1.

Supported values:

### Seedling
Small, young plant with a few leaves.

### Vegetative
Plant is mainly growing leaves and stems.

### Flowering
Flowers are appearing on the plant.

### Fruiting
Fruits are forming or growing.

### Maturity
Crop is fully developed and nearing harvest.

### Unknown / Not sure
The farmer does not know the current growth stage.

`Unknown` is valid. Never force the farmer to guess.

The final application should show farmer-friendly descriptions.

---

# 7. CROP/FIELD LOCATION

The location means the location of the **crop/field where the leaf was observed**.

It is NOT automatically the location of the device uploading the image.

Example:

```text
Photo taken in field
      ↓
Farmer travels home
      ↓
Photo uploaded from computer
```

The computer's current location must not automatically become the crop location.

The system should conceptually ask:

> Where was this crop/leaf located?

not:

> Where are you uploading this image from?

---

# 8. LOCATION SOURCES

## 8.1 User-selected crop/field location

Support:

- manually entered coordinates;
- map/location selection;
- saved field location where the future application supports it.

Record:

```text
location_source = "user_selected"
```

## 8.2 Device GPS

If the farmer is physically at the crop/field, the application may use device GPS.

UI wording:

> Use my current location

with:

> Choose this only if you are currently at the crop location.

The device supplies latitude and longitude.

No paid location API is required.

Record:

```text
location_source = "device_gps"
```

## 8.3 Image EXIF GPS

Valid image EXIF GPS may optionally be used.

EXIF is convenience only. It is not required.

Absence of EXIF GPS is normal because metadata can be removed by messaging/social platforms, screenshots, cameras, or privacy settings.

Record:

```text
location_source = "image_exif"
```

Only accept valid coordinates.

## 8.4 Location unavailable

Allow:

```text
location_status = "unavailable"
location_source = "unavailable"
```

Do not:

- fabricate coordinates;
- automatically use uploader location;
- use arbitrary coordinates;
- silently substitute another location;
- claim location-specific weather.

Model 1 must still work.

---

# 9. LOCATION PRIORITY

When several valid sources exist:

1. Explicit crop/field location supplied by user.
2. Device GPS when user confirms they are at the crop.
3. Valid image EXIF GPS.
4. Unknown/unavailable.

Always record the selected source.

Allowed values:

```text
"user_selected"
"device_gps"
"image_exif"
"unavailable"
```

Do not silently substitute one source for another.

---

# 10. WEATHER PROVIDER

Use **Open-Meteo** as the baseline weather provider.

The project must remain ₹0.

Do not require:

- paid weather services;
- subscriptions;
- credit cards;
- paid API keys.

During implementation, verify the current Open-Meteo endpoint and exact supported variables instead of assuming undocumented fields.

Record:

- provider;
- documentation URL;
- endpoint;
- query parameters;
- requested variables;
- units;
- request timestamp;
- response timestamp if available.

---

# 11. OPEN-METEO RESPONSIBILITY

Open-Meteo answers:

> What weather information is available for these crop/field coordinates?

Open-Meteo does NOT determine:

- farmer;
- crop;
- disease;
- growth stage;
- crop location ownership;
- disease causation.

Those values come from other sources.

---

# 12. WEATHER VARIABLES

At minimum retrieve suitable current information for:

- temperature;
- relative humidity;
- precipitation;
- weather condition/code.

The exact API variables must be selected deliberately and documented.

Normalized internal fields:

```text
temperature_c
humidity_percent
precipitation_mm
weather_condition
```

Do not call a value "rainfall" unless the API variable actually represents rainfall.

If precipitation is hourly/current, preserve that semantic meaning. Do not describe it as daily rainfall unless it is actually daily rainfall.

---

# 13. WEATHER UNITS

Normalize to:

```text
temperature_c       → degrees Celsius
humidity_percent    → percentage, 0–100
precipitation_mm    → millimeters
latitude            → decimal degrees
longitude           → decimal degrees
timestamp           → ISO-8601
```

Do not silently mix:

- Celsius/Fahrenheit;
- hourly/daily precipitation;
- observed/forecast values;
- precipitation amount/probability.

Document exact semantics.

---

# 14. WEATHER TIME SEMANTICS

Preserve when context was retrieved:

```text
request_timestamp
context_timestamp
```

If an API value represents a specific time period, preserve that period.

Do not claim current weather was the weather at image capture time unless historical weather is actually retrieved.

---

# 15. CURRENT VS PHOTO-TIME WEATHER

Baseline M2 provides **current context**.

If a photograph was taken earlier and uploaded later, current weather may differ from capture-time weather.

Treat these separately:

```text
image_capture_timestamp
context_timestamp
```

If image capture time is unavailable, do not invent it.

Do not describe current weather as historical weather.

---

# 16. NORMALIZED CONTEXT SCHEMA

Use a structure similar to:

```json
{
  "crop": "Tomato",
  "disease": "Tomato___Early_blight",
  "model1_confidence": 0.9400433,

  "growth_stage": "Flowering",

  "location_status": "available",
  "location_source": "user_selected",
  "latitude": 17.3850,
  "longitude": 78.4867,

  "weather_status": "available",
  "weather_provider": "Open-Meteo",
  "temperature_c": 29.4,
  "humidity_percent": 81.0,
  "precipitation_mm": 2.3,
  "weather_condition": "...",

  "context_timestamp": "...",
  "image_capture_timestamp": null
}
```

Unavailable values must remain null/unavailable. Never replace missing values with zero merely to complete a schema.

---

# 17. VALIDATION

## Model 1

Validate:

- predicted label exists in `class_names.json`;
- confidence is finite;
- confidence is 0–1;
- output dimension matches authoritative class count.

## Growth stage

Allowed:

```text
Seedling
Vegetative
Flowering
Fruiting
Maturity
Unknown
```

Invalid values must be rejected.

## Latitude

```text
-90 to +90
```

## Longitude

```text
-180 to +180
```

## Temperature

Numeric and finite.

## Humidity

Numeric and 0–100.

## Precipitation

Numeric and semantically compatible with the selected API variable.

Do not silently repair invalid data.

---

# 18. API FAILURE HANDLING

Handle:

- no internet;
- DNS failure;
- connection failure;
- timeout;
- HTTP error;
- malformed response;
- missing response fields;
- invalid weather values;
- service unavailable.

Use:

```text
weather_status = "available"
```

or:

```text
weather_status = "unavailable"
```

If unavailable:

- do not fabricate values;
- do not substitute zero;
- preserve failure/status;
- do not break Model 1.

Errors must be explicit and useful for debugging.

---

# 19. LOCATION FAILURE HANDLING

Handle:

- GPS unavailable;
- permission denied;
- invalid manual coordinates;
- invalid EXIF coordinates;
- no location supplied.

Use:

```text
location_status = "unavailable"
location_source = "unavailable"
```

M2 may still preserve crop, disease, confidence, and growth stage.

---

# 20. MODEL 1 INDEPENDENCE

Model 1 must work even if:

- internet is unavailable;
- Open-Meteo is unavailable;
- GPS is unavailable;
- location is unknown;
- growth stage is unknown.

Therefore:

```text
Leaf
 ↓
Model 1
 ↓
Disease prediction
```

must remain independent.

M2 enriches Model 1's result and must not break it.

---

# 21. FUSION BOUNDARY

Baseline M2 does not perform final multimodal fusion.

Do NOT calculate arbitrary combined confidence such as:

```text
Model1 confidence × weather score
```

or:

```text
Model1 confidence + weather score
```

Do not claim M2 improves Model 1 accuracy.

Future fusion must be separately specified and experimentally evaluated.

---

# 22. XAI BOUNDARY

XAI explains the visual Model 1 prediction.

XAI does not obtain weather, location, or growth stage.

Example:

```text
Model 1:
Tomato___Early_blight
Confidence: 94%

XAI:
Highlighted image regions contributing to prediction.
```

M2 provides context independently.

---

# 23. ADVISORY BOUNDARY

A later advisory layer may use:

- disease;
- confidence;
- crop;
- growth stage;
- temperature;
- humidity;
- precipitation;
- weather condition;
- location/time context.

Clearly distinguish:

```text
AI prediction
```

from:

```text
Agricultural recommendation
```

Do not represent recommendations as guaranteed outcomes.

---

# 24. ZERO-COST REQUIREMENT

Baseline must remain ₹0.

Use:

- local Model 1 inference;
- device location services where available;
- manual/selected crop location;
- optional EXIF GPS;
- Open-Meteo free access;
- existing dependencies where possible.

Do not:

- add paid APIs;
- require a credit card;
- require subscriptions;
- require paid weather APIs;
- require paid location APIs;
- introduce paid cloud infrastructure.

No Open-Meteo API key should be required for the baseline.

---

# 25. NO EXTERNAL TRAINING DATASET

Do not download or train on:

- GVLiD;
- MoringaLeafNet;
- another PlantVillage copy;
- rice disease datasets;
- grape disease datasets;
- weather training datasets;
- agricultural-context training datasets.

M2 is a live context acquisition module.

No training dataset is required for baseline M2.

---

# 26. DEVELOPMENT LOCATION

Development may run on Windows.

The PC's physical location must NOT automatically become the crop location.

For initial testing, use controlled test coordinates and document them as controlled test data.

Later mobile integration may use device GPS.

---

# 27. IMAGE HANDLING

M2 does not need another image dataset.

The leaf image is already processed by Model 1.

M2 may optionally inspect EXIF for GPS, but:

- EXIF is optional;
- missing EXIF is normal;
- explicit user-selected crop location takes priority;
- invalid EXIF must be rejected;
- unnecessary metadata must not be stored.

Do not create an image-based M2 classifier.

---

# 28. IMPLEMENTATION FILES

Create only M2-specific files.

Recommended:

```text
src/
    context.py

testing/
    verify_m2_context.py

experiments/
    context_test_records.json

reports/
    context/
        api_sample_response.json
        normalized_context.json
        m2_verification.json
```

If another file is genuinely necessary, explain why before creating it.

Do not modify Model 1 artifacts.

---

# 29. IMPLEMENTATION INTERFACE

At minimum provide:

```python
get_weather(latitude, longitude)
```

and:

```python
build_context(
    model1_prediction,
    growth_stage,
    location,
)
```

Separate:

1. Model 1 parsing;
2. class mapping validation;
3. crop/disease extraction;
4. growth-stage validation;
5. location validation;
6. weather API request;
7. weather response validation;
8. weather normalization;
9. context construction;
10. error/status handling;
11. research evidence recording.

Keep the module simple.

Do not introduce a database unless explicitly required.

---

# 30. DEPENDENCY POLICY

Before adding a package:

1. inspect existing requirements;
2. inspect the current virtual environment;
3. check whether functionality already exists;
4. prefer Python standard library;
5. avoid unnecessary installation.

Do not install:

- a new ML framework;
- unnecessary HTTP packages;
- unnecessary database packages;
- dataset packages.

If a package is genuinely required, report it before installation.

---

# 31. M2-001 — DISCOVERY

Before implementation, inspect:

```text
src/predict.py
src/config.py
src/dataset.py
requirements.txt
pyproject.toml
existing testing scripts
existing project structure
existing Model 1 artifact paths
existing HTTP/network functionality
existing location functionality
existing weather functionality
```

Confirm:

- how Model 1 prediction is returned;
- how class_names.json is loaded;
- how crop/disease can safely be derived;
- available HTTP/network dependency;
- where M2 files should be created;
- whether location/weather utilities already exist;
- whether additional dependency installation is necessary.

### M2-001 restrictions

Do NOT:

- modify files;
- install packages;
- download datasets;
- call Open-Meteo;
- train anything;
- modify Model 1;
- modify Model 1 artifacts.

Report findings only and STOP.

---

# 32. M2-002 — IMPLEMENTATION

Only after M2-001 discovery is reviewed/approved, implement M2.

Required:

1. Model 1 prediction consumption.
2. Authoritative class mapping validation.
3. Crop extraction.
4. Disease preservation.
5. Confidence preservation.
6. Growth-stage validation.
7. Farmer-friendly growth-stage descriptions.
8. Crop/field location handling.
9. User-selected/manual location handling.
10. Device-GPS-compatible location interface.
11. Optional EXIF GPS if it can be supported without unnecessary dependency installation.
12. Unknown-location handling.
13. Open-Meteo integration.
14. Exact weather variable documentation.
15. Unit normalization.
16. Weather timestamp handling.
17. Context normalization.
18. API failure handling.
19. Location failure handling.
20. Missing-value handling.
21. Research evidence capture.
22. M2 verification script.

Do not implement:

- second ML model;
- disease fusion;
- XAI;
- final advisory;
- paid services;
- external dataset acquisition.

---

# 33. M2-003 — VERIFICATION

Create:

```text
testing/verify_m2_context.py
```

Test at minimum:

## Model 1

1. Valid prediction.
2. Valid class mapping.
3. Correct crop extraction.
4. Correct disease preservation.
5. Valid confidence.
6. Correct output dimension.

## Growth stage

7. Seedling accepted.
8. Vegetative accepted.
9. Flowering accepted.
10. Fruiting accepted.
11. Maturity accepted.
12. Unknown accepted.
13. Invalid growth stage rejected.

## Location

14. Valid latitude.
15. Valid longitude.
16. Invalid latitude rejected.
17. Invalid longitude rejected.
18. Unknown location handled.
19. User-selected location handled.
20. Location source recorded correctly.

## Weather

21. Successful Open-Meteo request.
22. Weather response parsed.
23. Temperature normalized.
24. Humidity normalized.
25. Precipitation normalized.
26. Weather condition preserved.
27. Timestamp preserved.
28. Provider recorded.
29. Exact API request information recorded.

## Failure handling

30. API failure handled.
31. Timeout/failure handled where practical.
32. Missing weather field handled.
33. Invalid weather value handled.
34. Location unavailable handled.

## Research integrity

35. Complete normalized context.
36. Raw API response preserved.
37. Model 1 model artifact unchanged.
38. class_names.json unchanged.
39. Model 1 evaluation artifacts unchanged.
40. No external dataset downloaded.
41. No ML training performed.

---

# 34. M2-004 — RESEARCH EVIDENCE AUDIT

After verification, audit the M2 evidence.

Confirm preservation of:

- weather provider;
- documentation URL;
- endpoint;
- exact request parameters;
- requested variables;
- units;
- precipitation semantics;
- controlled coordinates;
- location source;
- Model 1 prediction;
- Model 1 confidence;
- growth stage;
- raw API response;
- normalized context;
- request timestamp;
- context timestamp;
- failure test evidence;
- validation test evidence;
- relevant artifact hashes.

Do not delete useful evidence.

Do not modify Model 1.

Report findings and STOP.

---

# 35. END-TO-END TEST

At least one controlled test must demonstrate:

```text
Leaf image
    ↓
Model 1
    ↓
Predicted disease + confidence
    ↓
Crop extraction
    ↓
Farmer growth-stage input
    ↓
Controlled crop/field location
    ↓
Open-Meteo
    ↓
Weather data
    ↓
Normalized agricultural context
```

Example:

```text
Disease:
Tomato___Early_blight

Confidence:
0.9400

Crop:
Tomato

Growth stage:
Flowering

Location source:
user_selected

Latitude:
controlled test value

Longitude:
controlled test value

Temperature:
XX.X °C

Humidity:
XX %

Precipitation:
XX.X mm

Weather:
XXXXX
```

---

# 36. RESEARCH EVIDENCE TO PRESERVE

## Model 1 → M2

- predicted class;
- confidence;
- authoritative class mapping;
- extracted crop;
- disease label.

## Farmer context

- selected growth stage;
- farmer-friendly description.

## Location

- controlled test coordinates;
- location source;
- location availability.

## Weather

- provider;
- documentation URL;
- endpoint;
- query parameters;
- variables;
- raw response;
- normalized response;
- units;
- request timestamp;
- context timestamp;
- precipitation semantics.

## Reliability

- successful request;
- failed request;
- unavailable location;
- invalid input rejection;
- missing-field handling.

## Reproducibility

- source code;
- verification script;
- test records;
- raw API response;
- normalized context;
- timestamps;
- configuration;
- relevant hashes.

---

# 37. RESEARCH ARTIFACTS

Preserve M2 artifacts under M2-specific paths:

```text
experiments/context_test_records.json
reports/context/api_sample_response.json
reports/context/normalized_context.json
reports/context/m2_verification.json
```

Screenshots may be preserved separately if useful for the paper/demo.

Do not overwrite Model 1 figures.

---

# 38. MODEL 1 IMMUTABILITY

The following are READ-ONLY:

```text
models/agriguard_mobilenetv2.keras
models/class_names.json

experiments/training_history.csv
experiments/experiment_config.json
experiments/test_metrics.json
experiments/classification_report.csv
experiments/confusion_matrix.csv

reports/figures/training_accuracy.png
reports/figures/training_loss.png
reports/figures/confusion_matrix.png
```

M2 must not overwrite, regenerate, retrain, alter, rename, or delete these artifacts.

Verify important Model 1 hashes before and after implementation when practical.

---

# 39. NO MODEL 2 ACCURACY METRIC

The baseline M2 module is not a trained classifier.

Do NOT report:

- Model 2 accuracy;
- Model 2 precision;
- Model 2 recall;
- Model 2 F1-score.

Report instead:

- context retrieval success;
- validation success;
- normalization correctness;
- API availability;
- failure handling;
- end-to-end context construction.

---

# 40. SCIENTIFIC CLAIM BOUNDARY

The baseline system may state:

> Agricultural and environmental context is collected and associated with the visual disease prediction.

Do NOT claim without a separate controlled experiment:

> Weather caused the detected disease.

> Weather independently predicts the disease.

> The context module increases Model 1 accuracy.

> The multimodal system is more accurate than Model 1.

> Current weather represents weather at image capture time.

> The system establishes causal relationships between weather and disease.

---

# 41. FUTURE FUSION

M2 output is intended for a future context-aware layer.

Potential visual inputs:

```text
disease
confidence
XAI representation
```

Potential context inputs:

```text
crop
growth stage
temperature
humidity
precipitation
weather
location
timestamp
```

The exact fusion mechanism must be separately specified and experimentally evaluated.

Do not implement fusion during M2.

---

# 42. FUTURE MOBILE LOCATION INTEGRATION

The baseline M2 implementation may run on the development PC using controlled coordinates.

Later, the mobile application may use:

```text
Phone GPS
    ↓
Latitude + Longitude
    ↓
Open-Meteo
```

The final application must ask whether the current device location is actually the crop/field location.

Support:

```text
Use current location
```

```text
Select/enter crop location
```

```text
Location unavailable
```

---

# 43. SECURITY AND PRIVACY

Do not unnecessarily store:

- precise home locations;
- personal identity information;
- unrelated EXIF metadata;
- private photographs beyond application requirements.

For research evidence, use controlled test coordinates whenever possible.

Do not expose precise real-user locations in the research paper.

---

# 44. PERFORMANCE

M2 must remain lightweight.

Do not perform:

- model training;
- large dataset loading;
- unnecessary image processing;
- unnecessary background computation.

Use a reasonable weather-request timeout.

M2 must return a clear status rather than hanging indefinitely.

Model 1 local inference must remain independent of API latency.

---

# 45. ERROR REPORTING

Errors must be explicit, actionable, reproducible, and non-silent.

Examples:

```text
Invalid latitude: value must be between -90 and 90.
```

```text
Weather request failed: timeout.
```

```text
Location unavailable: no crop location was supplied.
```

```text
Invalid growth stage: expected Seedling, Vegetative,
Flowering, Fruiting, Maturity, or Unknown.
```

Do not silently catch and discard errors.

---

# 46. ACCEPTANCE CRITERIA

M2 is complete only when all are satisfied:

1. Model 1 prediction can be consumed.
2. Authoritative class mapping is used.
3. Crop is safely derived from Model 1 class.
4. Disease label is preserved.
5. Model 1 confidence is preserved.
6. Growth stage can be supplied.
7. Growth-stage descriptions are farmer-friendly.
8. Unknown growth stage is supported.
9. Crop/field location can be supplied.
10. Device GPS is supported conceptually for later mobile integration.
11. Current device location is not automatically assumed to be crop location.
12. Manual/selected location is supported.
13. Unknown location is supported.
14. EXIF GPS is optional only.
15. Open-Meteo integration works.
16. No paid API is required.
17. No API key is required for baseline Open-Meteo.
18. Exact API variables are documented.
19. Temperature is normalized correctly.
20. Humidity is normalized correctly.
21. Precipitation semantics are documented.
22. Weather condition is captured when available.
23. Context timestamp is preserved.
24. Current weather is not falsely described as historical weather.
25. API failures are handled.
26. Location failures are handled.
27. Invalid inputs are rejected clearly.
28. Missing values are not fabricated.
29. Model 1 remains independently usable.
30. Model 1 artifacts remain unchanged.
31. M2 verification passes.
32. Raw API response is preserved.
33. Normalized context is preserved.
34. Test records are preserved.
35. One complete end-to-end test is available.
36. No external training dataset was downloaded.
37. No second ML model was trained.
38. No unsupported scientific claim is made.
39. M2 research evidence audit passes.

---

# 47. STOP CONDITION

After M2 implementation, verification, and research evidence audit:

**STOP.**

Do NOT automatically continue to:

- train another model;
- download an external dataset;
- modify Model 1;
- implement Grad-CAM;
- implement final advisory generation;
- implement arbitrary numerical fusion;
- redesign Model 1;
- introduce paid APIs;
- introduce paid infrastructure;
- claim unsupported accuracy improvements.

Wait for explicit authorization for the next milestone.

---

# 48. FINAL EXPECTED RESULT

```text
                         LEAF IMAGE
                             │
                             ↓
                          MODEL 1
                             │
                             ↓
                Disease + Confidence + Crop
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ↓                       ↓
          Farmer Input              Crop Location
                 │                       │
          Growth Stage            GPS / Manual /
                 │                  Optional EXIF
                 │                       │
                 │                       ↓
                 │                  Coordinates
                 │                       │
                 │                       ↓
                 │                  Open-Meteo
                 │                       │
                 │              ┌────────┼────────┐
                 │              ↓        ↓        ↓
                 │          Temperature Humidity Precipitation
                 │                       │
                 │                   Weather
                 │                       │
                 └───────────┬───────────┘
                             ↓
                 AGRICULTURAL CONTEXT
                             │
                             ↓
                  FUTURE CONTEXT LAYER
                             │
                    ┌────────┴────────┐
                    ↓                 ↓
                   XAI             ADVISORY
```

M2 is a **context acquisition and normalization module**, not a second disease classifier.

The baseline implementation must remain **free, reproducible, scientifically defensible, lightweight, and independent of Model 1**.

---

# 49. EXECUTION ORDER

Codex must work strictly in this order:

```text
M2-001 Discovery
      ↓
STOP AND REPORT
      ↓
Human review/approval
      ↓
M2-002 Implementation
      ↓
STOP AND REPORT
      ↓
Human review/approval
      ↓
M2-003 Verification
      ↓
STOP AND REPORT
      ↓
Human review/approval
      ↓
M2-004 Research Evidence Audit
      ↓
STOP
```

Never skip discovery.

Never implement beyond the currently authorized stage.

Never modify Model 1 without explicit approval.
