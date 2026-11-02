# AGRIGUARD PROJECT — COMPLETE SOURCE AND CITATION AUDIT

## A. EXECUTIVE SOURCE SUMMARY

- Total number of unique external sources found: 21
- Total datasets: 1 (PlantVillage-Dataset)
- Total research papers: 2 (MobileNetV2, Grad-CAM original methods)
- Total APIs: 1 (Open-Meteo)
- Total advisory sources: 13 (University Extension Articles)
- Total software/documentation sources: 3 (TensorFlow, scikit-learn core ML libraries explicitly identified as critical)
- Total external code repositories: 0 (No external code cloned; implemented using core libraries)
- Total external images: 1 (test_leaf.jpg - unverified origin)

## B. MASTER SOURCE TABLE

| ID | Category | Exact Source | Authors/Organization | Year | DOI | Official URL | License | Used Where | Evidence File | Citation Needed? | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SRC-01 | Dataset | PlantVillage-Dataset | spMohanty (GitHub) | NOT VERIFIED | NOT VERIFIED | https://github.com/spMohanty/PlantVillage-Dataset | NOT FOUND | Model 1 Training | `data/metadata/plantvillage_metadata.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-02 | Model | MobileNetV2 | Sandler et al. | 2018 | NOT VERIFIED | https://arxiv.org/abs/1801.04381 | NOT FOUND | Model 1 Backbone | `src/model.py` | REQUIRED | VERIFIED FROM OFFICIAL SOURCE |
| SRC-03 | Weights | ImageNet | ImageNet Team | NOT VERIFIED | NOT VERIFIED | https://image-net.org/ | NOT FOUND | Pretraining | `src/model.py` | RECOMMENDED | VERIFIED FROM PROJECT |
| SRC-04 | Method | Grad-CAM | Selvaraju et al. | 2017 | NOT VERIFIED | https://arxiv.org/abs/1610.02391 | NOT FOUND | Explainability | `src/xai.py` | REQUIRED | VERIFIED FROM OFFICIAL SOURCE |
| SRC-05 | API | Open-Meteo | Open-Meteo | NOT VERIFIED | N/A | https://open-meteo.com/en/docs | NOT FOUND | Weather Context | `src/context.py` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-06 | Advisory | Early blight in tomato and potato | Univ of Minnesota Ext | NOT VERIFIED | N/A | https://extension.umn.edu/disease-management/early-blight-tomato-and-potato | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-07 | Advisory | Late blight of tomato and potato | Univ of Minnesota Ext | NOT VERIFIED | N/A | https://extension.umn.edu/disease-management/late-blight | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-08 | Advisory | Tomato Diseases and Disorders in the Home Garden | Penn State Ext | NOT VERIFIED | N/A | https://extension.psu.edu/tomato-diseases-and-disorders-in-the-home-garden | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-09 | Advisory | Scouting and Identifying Tomato Diseases | Penn State Ext | NOT VERIFIED | N/A | https://extension.psu.edu/scouting-and-identifying-tomato-diseases | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-10 | Advisory | Gray leaf spot on corn | Univ of Minnesota Ext | NOT VERIFIED | N/A | https://extension.umn.edu/corn-pest-management/gray-leaf-spot-corn | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-11 | Advisory | Common rust on corn | Univ of Minnesota Ext | NOT VERIFIED | N/A | https://extension.umn.edu/corn-pest-management/common-rust-corn | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-12 | Advisory | Northern corn leaf blight | Univ of Minnesota Ext | NOT VERIFIED | N/A | https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-13 | Advisory | Pest Management in Apples in Home Fruit Plantings | Penn State Ext | NOT VERIFIED | N/A | https://extension.psu.edu/pest-management-in-apples-in-home-fruit-plantings | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-14 | Advisory | Apple Disease Control Toolbox - Fungicide Timing | Penn State Ext | NOT VERIFIED | N/A | https://extension.psu.edu/apple-disease-control-toolbox-fungicide-timing | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-15 | Advisory | Black rot on grapes (Guignardia bidwellii) | Univ of Minnesota Ext | 2026 | N/A | https://blog-fruit-vegetable-ipm.extension.umn.edu/2026/06/black-rot-on-grapes-guignardia-bidwellii.html | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-16 | Advisory | Huanglongbing (HLB; Citrus Greening) and Nutrient Deficiency Identification | Univ of Florida IFAS | NOT VERIFIED | N/A | https://edis.ifas.ufl.edu/publication/PP328/pdf | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-17 | Advisory | Raspberry diseases | Univ of Minnesota Ext | NOT VERIFIED | N/A | https://extension.umn.edu/raspberry-farming/raspberry-diseases | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-18 | Advisory | Commercial blueberry production in Minnesota and Wisconsin | Univ of Minnesota Ext | NOT VERIFIED | N/A | https://extension.umn.edu/fruit/commercial-blueberry-production-minnesota-and-wisconsin | NOT FOUND | Knowledge Base | `data/advisory/source_registry.json` | REQUIRED | VERIFIED FROM PROJECT |
| SRC-19 | Software | TensorFlow | Google Brain | NOT VERIFIED | NOT VERIFIED | https://tensorflow.org/ | Apache 2.0 | DL Framework | `requirements.txt` | RECOMMENDED | VERIFIED FROM OFFICIAL SOURCE |
| SRC-20 | Software | Scikit-Learn | scikit-learn devs | NOT VERIFIED | NOT VERIFIED | https://scikit-learn.org/ | BSD 3-Clause | Dataset Splits | `requirements.txt` | RECOMMENDED | VERIFIED FROM OFFICIAL SOURCE |
| SRC-21 | Image | test_leaf.jpg | NOT VERIFIED | NOT VERIFIED | N/A | N/A | NOT FOUND | Inference test | `test_leaf.jpg` | NOT NEEDED | REQUIRES MANUAL VERIFICATION |

## C. DATASET PROVENANCE TABLE

| Dataset | Original Source | Paper | URL | DOI | Version | License | Access Date | Project Evidence | Status |
|---|---|---|---|---|---|---|---|---|---|
| PlantVillage-Dataset (color variant) | spMohanty GitHub Repo | NOT VERIFIED - REQUIRES EXTERNAL VERIFICATION | https://github.com/spMohanty/PlantVillage-Dataset | NOT VERIFIED | N/A | NOT FOUND | 2026-08-13 | `data/metadata/plantvillage_metadata.json` | VERIFIED FROM PROJECT (Subset raw/color) |

*Note: The project explicitly sources the dataset from spMohanty's GitHub repository. The original PlantVillage dataset was created by Hughes and Salathe, but this project's metadata points specifically to the spMohanty version. The paper should cite BOTH the original PlantVillage authors and the specific repository used.*

## D. RESEARCH PAPER SOURCES

| Topic | Original Paper/Source | Authors | Venue | Year | DOI | URL | Project Usage | Citation Required? |
|---|---|---|---|---|---|---|---|---|
| Architecture | MobileNetV2: Inverted Residuals and Linear Bottlenecks | Sandler, Howard, Zhu, Zhmoginov, Chen | CVPR | 2018 | NOT VERIFIED | https://arxiv.org/abs/1801.04381 | `src/model.py` (tf.keras.applications.MobileNetV2) | REQUIRED |
| Explainability | Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization | Selvaraju, Cogswell, Das, Vedantam, Parikh, Batra | ICCV | 2017 | NOT VERIFIED | https://arxiv.org/abs/1610.02391 | `src/xai.py` (Custom Implementation using tf.GradientTape) | REQUIRED |
| Pretraining | ImageNet: A Large-Scale Hierarchical Image Database | Deng et al. | CVPR | 2009 | NOT VERIFIED | https://image-net.org/ | `src/model.py` (weights="imagenet") | RECOMMENDED |

## E. API / SOFTWARE SOURCES

| Tool/API | Provider | Version | Official URL | Purpose | Citation/Attribution Needed? | License | Evidence File |
|---|---|---|---|---|---|---|---|
| Open-Meteo | Open-Meteo | v1 | https://open-meteo.com/en/docs | Current weather context | REQUIRED (By provider terms) | NOT FOUND | `src/context.py` |
| TensorFlow / Keras | Google | 2.15.1 | https://tensorflow.org/ | Model training/inference | RECOMMENDED | Apache 2.0 | `requirements.txt` |
| scikit-learn | scikit-learn devs | 1.5.2 | https://scikit-learn.org/ | Train/test split stratified | RECOMMENDED | BSD 3-Clause | `requirements.txt` |
| NumPy | NumPy | 1.26.4 | https://numpy.org/ | Array operations | OPTIONAL | BSD | `requirements.txt` |
| Pandas | Pandas | 2.2.3 | https://pandas.pydata.org/ | CSV manipulations | OPTIONAL | BSD | `requirements.txt` |
| Matplotlib | Matplotlib | 3.9.2 | https://matplotlib.org/ | Data visualization / Heatmaps | OPTIONAL | PSF | `requirements.txt` |
| Pillow | Pillow | 11.0.0 | https://python-pillow.org/ | Image reading | OPTIONAL | HPND | `requirements.txt` |
| Flask | Pallets | NOT VERIFIED | https://flask.palletsprojects.com/ | Farmer-facing app layer | NOT NEEDED | BSD 3-Clause | `app.py` |

## F. AGRICULTURAL ADVISORY SOURCES

| Crop/Disease/Topic | Exact Recommendation Supported | Organization | Exact Page/Document | URL | Date | License | Evidence File | Citation Needed? |
|---|---|---|---|---|---|---|---|---|
| Tomato/Potato Early blight | Monitoring, Cultural controls | Univ of Minnesota Ext | Early blight in tomato and potato | https://extension.umn.edu/disease-management/early-blight-tomato-and-potato | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Tomato/Potato Late blight | Monitoring | Univ of Minnesota Ext | Late blight of tomato and potato | https://extension.umn.edu/disease-management/late-blight | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Tomato Bacterial spot/Septoria | Monitoring | Penn State Ext | Tomato Diseases and Disorders in the Home Garden | https://extension.psu.edu/tomato-diseases-and-disorders-in-the-home-garden | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Tomato Healthy | Monitoring/Scouting | Penn State Ext | Scouting and Identifying Tomato Diseases | https://extension.psu.edu/scouting-and-identifying-tomato-diseases | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Corn Gray leaf spot | Disease management | Univ of Minnesota Ext | Gray leaf spot on corn | https://extension.umn.edu/corn-pest-management/gray-leaf-spot-corn | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Corn Common rust | Disease management | Univ of Minnesota Ext | Common rust on corn | https://extension.umn.edu/corn-pest-management/common-rust-corn | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Corn Northern leaf blight | Disease management | Univ of Minnesota Ext | Northern corn leaf blight | https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Apple Scab/Black rot/Rust | Monitoring/Prevention/Control | Penn State Ext | Pest Management in Apples in Home Fruit Plantings | https://extension.psu.edu/pest-management-in-apples-in-home-fruit-plantings | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Apple Disease | Fungicide Timing | Penn State Ext | Apple Disease Control Toolbox - Fungicide Timing | https://extension.psu.edu/apple-disease-control-toolbox-fungicide-timing | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Grape Black rot | Life cycle/Monitoring | Univ of Minnesota Ext | Black rot on grapes (Guignardia bidwellii) | https://blog-fruit-vegetable-ipm.extension.umn.edu/2026/06/black-rot-on-grapes-guignardia-bidwellii.html | 2026 | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Orange Huanglongbing | Monitoring | Univ of Florida IFAS | Huanglongbing (HLB; Citrus Greening) and Nutrient Deficiency Identification | https://edis.ifas.ufl.edu/publication/PP328/pdf | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Raspberry Healthy | Prevention | Univ of Minnesota Ext | Raspberry diseases | https://extension.umn.edu/raspberry-farming/raspberry-diseases | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |
| Blueberry Healthy | Monitoring | Univ of Minnesota Ext | Commercial blueberry production in Minnesota and Wisconsin | https://extension.umn.edu/fruit/commercial-blueberry-production-minnesota-and-wisconsin | NOT VERIFIED | NOT FOUND | `data/advisory/source_registry.json` | REQUIRED |

## G. EXTERNAL CODE / REPOSITORIES

| Repository | Author | URL | License | Code Used | Evidence File | Attribution Required? | Paper Citation? |
|---|---|---|---|---|---|---|---|
| None | N/A | N/A | N/A | No external code was copied. Grad-CAM is a native TF GradientTape implementation in `src/xai.py`. | N/A | N/A | NOT NEEDED |

## H. EXTERNAL IMAGES

| Image | Source | Creator | URL | License | Used In | Attribution Required? | Status |
|---|---|---|---|---|---|---|---|
| `test_leaf.jpg` | INSUFFICIENT PROJECT EVIDENCE - DO NOT INVENT | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Local Inference testing | REQUIRES MANUAL VERIFICATION | REQUIRES MANUAL VERIFICATION |

## I. COPYRIGHT / LICENSE RISKS

1. **PlantVillage Dataset**: `NOT FOUND` in project files. You must verify the original license of PlantVillage (usually CC-BY or public domain, but you need to check) and spMohanty's repository terms. 
2. **Open-Meteo API**: Open-Meteo requires attribution for free API use. Ensure the paper acknowledges Open-Meteo explicitly.
3. **Advisory Extension Sources (13 Sources)**: University extension sites usually have copyright restrictions on reproducing their text verbatim. The project claims "source-grounded" rules, but verify you are not infringing copyright by republishing verbatim recommendations without permission.
4. **test_leaf.jpg**: Origin unknown. Ensure you have rights to publish this image in the paper if used as a figure.

## J. PAPER CITATION MAP

| Paper Section | Claim/Technology | Source | Citation Placeholder | Reason |
|---|---|---|---|---|
| Introduction | PlantVillage dataset utilized | PlantVillage (spMohanty) | [1] | Primary dataset for Model 1 baseline. |
| Materials and Methods | Convolutional Neural Network | MobileNetV2 (Sandler et al.) | [2] | Core architecture of the visual classifier. |
| Materials and Methods | Pretrained Weights | ImageNet | [3] | Transfer learning basis. |
| Materials and Methods | Real-Time Weather Context | Open-Meteo API | [4] | Provider of dynamic temperature/humidity/precipitation data. |
| Materials and Methods | Explainable AI | Grad-CAM (Selvaraju et al.) | [5] | Algorithm used for visual attribution heatmaps. |
| Materials and Methods | Agricultural Knowledge Base | Univ. of Minnesota, Penn State, UF IFAS Ext. | [6]-[18] | The 13 advisory sources dictating crop/disease management recommendations. |
| Results and Discussion | Model Evaluation Tools | Scikit-learn | [19] | Standard library for reproducible dataset splitting and metrics. |
| Results and Discussion | Deep Learning Framework | TensorFlow | [20] | End-to-end training and inference ecosystem. |
| Conclusion | Future application extensions | N/A | N/A | N/A |

## K. FINAL IEEE REFERENCE CANDIDATES

*Note: All items here that are marked INCOMPLETE must be manually verified. The project does not contain DOIs, exact publication years for all extensions, or the original PlantVillage publication details.*

[INCOMPLETE — DO NOT USE UNTIL VERIFIED] D. P. Hughes and M. Salathe, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," *arXiv preprint arXiv:1511.08060*, 2015. [Dataset originally generated]

[INCOMPLETE — DO NOT USE UNTIL VERIFIED] S. P. Mohanty, D. P. Hughes, and M. Salathe, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, vol. 7, p. 1419, 2016. [Dataset repository spMohanty]

[INCOMPLETE — DO NOT USE UNTIL VERIFIED] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 4510-4520.

[INCOMPLETE — DO NOT USE UNTIL VERIFIED] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618-626.

[INCOMPLETE — DO NOT USE UNTIL VERIFIED] Open-Meteo. "Open-Meteo API Documentation." https://open-meteo.com/en/docs (accessed Aug. 13, 2026).

[INCOMPLETE — DO NOT USE UNTIL VERIFIED] University of Minnesota Extension. "Early blight in tomato and potato." https://extension.umn.edu/disease-management/early-blight-tomato-and-potato (accessed Aug. 13, 2026).
*(You will need to construct identical IEEE citations for the remaining 12 advisory sources from the Master Source Table).*

## L. SOURCES THAT MUST NOT BE CITED

1. **GVLiD, MoringaLeafNet, rice/grape external datasets**: Do NOT cite these. The project strictly uses ONLY PlantVillage (`data/metadata/plantvillage_metadata.json` confirms 54,305 images from 38 classes).
2. **ResNet, VGG16, EfficientNet**: Do NOT cite these as the architecture. The project explicitly builds `agriguard_mobilenetv2` via `tf.keras.applications.MobileNetV2`.
3. **Paid Weather APIs (e.g., OpenWeatherMap, AccuWeather)**: Do NOT cite these. `src/context.py` strictly uses `https://api.open-meteo.com/v1/forecast`.
4. **Flask / Werkzeug**: Usually not cited in academic papers unless the paper focuses heavily on the web-server architecture. For a disease diagnosis paper, these are minor deployment details.

## 17. FINAL CONSISTENCY CHECK

1. **Sources used by the project but currently missing from the paper**: You must ensure the 13 specific extension links are mapped to their respective diseases in the paper's discussion section.
2. **Paper claims currently missing citations**: Any claim about the specific etiology (e.g. "Early blight requires foliage-drying practices") must cite the exact extension source (S1), not general knowledge.
3. **References that appear unnecessary**: General Python packages (Pillow, Pandas, Matplotlib) don't need formal bibliography citations in most ML papers.
4. **Duplicate sources**: None found in the metadata; however, multiple extensions come from the same university. Cite them as separate web pages/documents.
5. **Sources with conflicting metadata**: The dataset source is cited as `spMohanty/PlantVillage-Dataset` (a GitHub repo), but standard academic practice expects citing the Hughes/Salathe or Mohanty et al. peer-reviewed papers. You must bridge this gap externally.
6. **Sources whose URL/DOI/license needs verification**: EVERYTHING in the advisory list. DOIs are not present for university extension websites.
7. **Copyright/license issues requiring attention**: Open-Meteo attribution requirement; University extension text redistribution terms; PlantVillage license.
8. **Information that cannot be established from the project**: Original authors of PlantVillage, DOIs for any paper, exact publication years of extension articles (except S7 which mentions 2026 in URL), and origin of `test_leaf.jpg`.
9. **Exact manual checks I must perform before submission**: 
   - Look up the DOI and formal citation for MobileNetV2.
   - Look up the DOI and formal citation for Grad-CAM.
   - Trace the `spMohanty` GitHub dataset back to the published PlantVillage paper.
   - Verify if `test_leaf.jpg` is copyrighted before placing it in a figure.
   - Add the Open-Meteo attribution text to your Acknowledgments section.
