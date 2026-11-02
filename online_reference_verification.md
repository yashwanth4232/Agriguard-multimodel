# INDEPENDENT ONLINE SOURCE VERIFICATION AND CITATION AUDIT

## 1. PLANTVILLAGE DATASET
**Dataset source:** The actual dataset was originally created by David P. Hughes and Marcel Salathé ("An open access repository of images on plant health to enable the development of mobile disease diagnostics", arXiv:1511.08060, 2015).
**Academic publication describing/using the dataset:** The landmark peer-reviewed paper is "Using Deep Learning for Image-Based Plant Disease Detection" by Sharada P. Mohanty, David P. Hughes, and Marcel Salathé, *Frontiers in Plant Science*, vol. 7, 2016 (DOI: 10.3389/fpls.2016.01419).
**Official Dataset URL:** https://github.com/spMohanty/PlantVillage-Dataset
**License:** Open Access (generally treated as public domain / CC0, though explicit LICENSE files are sometimes missing from mirror repositories).
**Recommendation:** The paper should cite **both** the original Mohanty et al. 2016 paper (as the peer-reviewed validation of the dataset's use in deep learning) and explicitly mention the GitHub repository URL in the text to fulfill provenance traceability.

## 2. MOBILENETV2
**Original Paper:** MobileNetV2: Inverted Residuals and Linear Bottlenecks
**Authors:** Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
**Venue:** 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
**Year:** 2018
**DOI:** 10.1109/CVPR.2018.00474
**Official URL:** https://ieeexplore.ieee.org/document/8578572
**Verified IEEE Citation:** M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2018, pp. 4510-4520, doi: 10.1109/CVPR.2018.00474.

## 3. GRAD-CAM
**Original Paper:** Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization
**Authors:** Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra
**Venue:** 2017 IEEE International Conference on Computer Vision (ICCV)
**Year:** 2017
**DOI:** 10.1109/ICCV.2017.74
**Official URL:** https://ieeexplore.ieee.org/document/8237336
**Verified IEEE Citation:** R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *2017 IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618-626, doi: 10.1109/ICCV.2017.74.

## 4. IMAGENET
**Original Paper:** ImageNet: A Large-Scale Hierarchical Image Database
**Authors:** Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei
**Venue:** 2009 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
**Year:** 2009
**DOI:** 10.1109/CVPR.2009.5206848
**Decision:** REQUIRED.
**Reason:** The MobileNetV2 model architecture uses pretrained weights derived from ImageNet. Transfer learning fundamentally relies on the feature representations learned from this specific dataset. Acknowledging ImageNet is required academic practice when utilizing its pre-trained weights.

## 5. OPEN-METEO
**Official Provider:** Open-Meteo
**Exact API:** Open-Meteo Weather Forecast API
**Documentation URL:** https://open-meteo.com/en/docs
**Terms of Use & License:** Open-Meteo requires explicit attribution for free API use. The data is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
**Academic Citation vs Attribution:**
A. *Academic citation:* Required to establish reproducibility of the weather context data.
B. *Website/API attribution:* Legally required by the CC BY 4.0 license (e.g., "Weather data provided by Open-Meteo.com").
C. *License compliance:* Must display attribution in the application UI or paper acknowledgments.

## 6. AGRICULTURAL ADVISORY SOURCES
*(Note: University Extension websites do not issue DOIs for web articles. The authors are frequently anonymous "Extension Staff" or specific pathologists. Publication years are often unlisted or dynamically updated. Access date is strictly required.)*

- **SOURCE 1:** University of Minnesota Extension, "Early blight in tomato and potato," [Online]. Available: https://extension.umn.edu/disease-management/early-blight-tomato-and-potato. [Accessed: 21-Aug-2026].
- **SOURCE 2:** University of Minnesota Extension, "Late blight of tomato and potato," [Online]. Available: https://extension.umn.edu/disease-management/late-blight. [Accessed: 21-Aug-2026].
- **SOURCE 3:** Penn State Extension, "Tomato Diseases and Disorders in the Home Garden," [Online]. Available: https://extension.psu.edu/tomato-diseases-and-disorders-in-the-home-garden. [Accessed: 21-Aug-2026].
- **SOURCE 4:** Penn State Extension, "Scouting and Identifying Tomato Diseases," [Online]. Available: https://extension.psu.edu/scouting-and-identifying-tomato-diseases. [Accessed: 21-Aug-2026].
- **SOURCE 5:** University of Minnesota Extension, "Gray leaf spot on corn," [Online]. Available: https://extension.umn.edu/corn-pest-management/gray-leaf-spot-corn. [Accessed: 21-Aug-2026].
- **SOURCE 6:** University of Minnesota Extension, "Common rust on corn," [Online]. Available: https://extension.umn.edu/corn-pest-management/common-rust-corn. [Accessed: 21-Aug-2026].
- **SOURCE 7:** University of Minnesota Extension, "Northern corn leaf blight," [Online]. Available: https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight. [Accessed: 21-Aug-2026].
- **SOURCE 8:** Penn State Extension, "Pest Management in Apples in Home Fruit Plantings," [Online]. Available: https://extension.psu.edu/pest-management-in-apples-in-home-fruit-plantings. [Accessed: 21-Aug-2026].
- **SOURCE 9:** Penn State Extension, "Apple Disease Control Toolbox - Fungicide Timing," [Online]. Available: https://extension.psu.edu/apple-disease-control-toolbox-fungicide-timing. [Accessed: 21-Aug-2026].
- **SOURCE 10:** University of Minnesota Extension, "Black rot on grapes (Guignardia bidwellii)," 2026. [Online]. Available: https://blog-fruit-vegetable-ipm.extension.umn.edu/2026/06/black-rot-on-grapes-guignardia-bidwellii.html. [Accessed: 21-Aug-2026].
- **SOURCE 11:** University of Florida IFAS Extension, "Huanglongbing (HLB; Citrus Greening) and Nutrient Deficiency Identification," [Online]. Available: https://edis.ifas.ufl.edu/publication/PP328/pdf. [Accessed: 21-Aug-2026].
- **SOURCE 12:** University of Minnesota Extension, "Raspberry diseases," [Online]. Available: https://extension.umn.edu/raspberry-farming/raspberry-diseases. [Accessed: 21-Aug-2026].
- **SOURCE 13:** University of Minnesota Extension, "Commercial blueberry production in Minnesota and Wisconsin," [Online]. Available: https://extension.umn.edu/fruit/commercial-blueberry-production-minnesota-and-wisconsin. [Accessed: 21-Aug-2026].

**Copyright Notice:** Content is copyrighted by the respective university Regents. Reproduction of text requires permission or fair-use paraphrasing.

## 7. SOFTWARE REFERENCES
- **TensorFlow:** REQUIRED. Foundational framework for model training. (Cite: M. Abadi et al., "TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems," 2015).
- **scikit-learn:** RECOMMENDED. Used for metrics and stratified splitting. (Cite: F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, 2011).
- **NumPy, Pandas, Matplotlib, Pillow, Flask:** OPTIONAL / NOT NEEDED. Standard utility libraries do not typically require formal bibliography citations in applied ML papers unless introducing a novel programmatic technique.

## 8. COPYRIGHT / LICENSE VERIFICATION
- **PlantVillage dataset:** LICENSE NOT VERIFIED — MANUAL REVIEW REQUIRED (No explicit license file in the spMohanty GitHub root; typically treated as open academic access).
- **MobileNetV2 / Grad-CAM / ImageNet:** Copyright held by IEEE (for published papers). Standard academic citation is required; reproduction of figures requires IEEE permission.
- **Open-Meteo:** CC BY 4.0. Attribution strictly required.
- **Agricultural Advisory Pages:** Copyrighted by Univ. of Minnesota, Penn State, UF IFAS. Citation required. Verbatim copying of their recommendations into a commercial app may violate copyright without permission.

## 9. DOI VERIFICATION
- **10.3389/fpls.2016.01419:** Resolves to Mohanty et al., Frontiers in Plant Science, 2016. Matches.
- **10.1109/CVPR.2018.00474:** Resolves to Sandler et al., MobileNetV2, CVPR 2018. Matches.
- **10.1109/ICCV.2017.74:** Resolves to Selvaraju et al., Grad-CAM, ICCV 2017. Matches.
- **10.1109/CVPR.2009.5206848:** Resolves to Deng et al., ImageNet, CVPR 2009. Matches.

## 10. IEEE REFERENCE VERIFICATION
See Section 13 for final verified references.

## 11. FINAL VERIFICATION TABLE

| ID | Source | Source Type | Author/Organization | Title | Year | DOI | Official URL | License | Bibliographic Status | Verification Source |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PlantVillage | Dataset Paper | Mohanty, Hughes, Salathé | Using Deep Learning for Image-Based Plant Disease Detection | 2016 | 10.3389/fpls.2016.01419 | https://github.com/spMohanty/PlantVillage-Dataset | NOT VERIFIED | VERIFIED | Frontiers / Crossref |
| 2 | MobileNetV2 | Architecture Paper | Sandler et al. | MobileNetV2: Inverted Residuals and Linear Bottlenecks | 2018 | 10.1109/CVPR.2018.00474 | https://ieeexplore.ieee.org/document/8578572 | N/A | VERIFIED | IEEE Xplore |
| 3 | Grad-CAM | Method Paper | Selvaraju et al. | Grad-CAM: Visual Explanations... | 2017 | 10.1109/ICCV.2017.74 | https://ieeexplore.ieee.org/document/8237336 | N/A | VERIFIED | IEEE Xplore |
| 4 | ImageNet | Dataset Paper | Deng et al. | ImageNet: A Large-Scale Hierarchical Image Database | 2009 | 10.1109/CVPR.2009.5206848 | https://image-net.org/ | N/A | VERIFIED | IEEE Xplore |
| 5 | Open-Meteo | API | Open-Meteo | Open-Meteo API Docs | N/A | N/A | https://open-meteo.com/en/docs | CC BY 4.0 | VERIFIED | Official Website |
| 6-18| Univ. Extensions| Advisory Webpages | UMN, PSU, UF IFAS | (Various titles as listed above) | Varies | N/A | (Various URLs) | Copyrighted | VERIFIED | Official Extension Sites |

## 12. FINAL REFERENCE DECISION

| Source | Final Decision | Reason |
|---|---|---|
| PlantVillage Paper (Mohanty) | INCLUDE | Fundamental dataset baseline |
| MobileNetV2 (Sandler) | INCLUDE | Core architecture |
| Grad-CAM (Selvaraju) | INCLUDE | Core explainability method |
| ImageNet (Deng) | INCLUDE | Pretraining weights origin |
| Open-Meteo | INCLUDE | Required by Terms of Service |
| 13 Extension Advisory Pages | INCLUDE | Justifies the knowledge base recommendations |
| TensorFlow (Abadi) | INCLUDE ONLY IF NEEDED | Standard library, cite if methodology focus requires it |
| test_leaf.jpg | DO NOT INCLUDE | Unverified origin |

## 13. FINAL VERIFIED IEEE REFERENCES

[1] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using Deep Learning for Image-Based Plant Disease Detection," *Frontiers in Plant Science*, vol. 7, 2016, doi: 10.3389/fpls.2016.01419.
[2] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2018, pp. 4510-4520, doi: 10.1109/CVPR.2018.00474.
[3] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *2017 IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618-626, doi: 10.1109/ICCV.2017.74.
[4] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *2009 IEEE Conference on Computer Vision and Pattern Recognition*, 2009, pp. 248-255, doi: 10.1109/CVPR.2009.5206848.
[5] Open-Meteo, "Open-Meteo API Documentation," [Online]. Available: https://open-meteo.com/en/docs. [Accessed: 21-Aug-2026].
[6] University of Minnesota Extension, "Early blight in tomato and potato," [Online]. Available: https://extension.umn.edu/disease-management/early-blight-tomato-and-potato. [Accessed: 21-Aug-2026].
[7] University of Minnesota Extension, "Late blight of tomato and potato," [Online]. Available: https://extension.umn.edu/disease-management/late-blight. [Accessed: 21-Aug-2026].
[8] Penn State Extension, "Tomato Diseases and Disorders in the Home Garden," [Online]. Available: https://extension.psu.edu/tomato-diseases-and-disorders-in-the-home-garden. [Accessed: 21-Aug-2026].
[9] Penn State Extension, "Scouting and Identifying Tomato Diseases," [Online]. Available: https://extension.psu.edu/scouting-and-identifying-tomato-diseases. [Accessed: 21-Aug-2026].
[10] University of Minnesota Extension, "Gray leaf spot on corn," [Online]. Available: https://extension.umn.edu/corn-pest-management/gray-leaf-spot-corn. [Accessed: 21-Aug-2026].
[11] University of Minnesota Extension, "Common rust on corn," [Online]. Available: https://extension.umn.edu/corn-pest-management/common-rust-corn. [Accessed: 21-Aug-2026].
[12] University of Minnesota Extension, "Northern corn leaf blight," [Online]. Available: https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight. [Accessed: 21-Aug-2026].
[13] Penn State Extension, "Pest Management in Apples in Home Fruit Plantings," [Online]. Available: https://extension.psu.edu/pest-management-in-apples-in-home-fruit-plantings. [Accessed: 21-Aug-2026].
[14] Penn State Extension, "Apple Disease Control Toolbox - Fungicide Timing," [Online]. Available: https://extension.psu.edu/apple-disease-control-toolbox-fungicide-timing. [Accessed: 21-Aug-2026].
[15] University of Minnesota Extension, "Black rot on grapes (Guignardia bidwellii)," 2026. [Online]. Available: https://blog-fruit-vegetable-ipm.extension.umn.edu/2026/06/black-rot-on-grapes-guignardia-bidwellii.html. [Accessed: 21-Aug-2026].
[16] University of Florida IFAS Extension, "Huanglongbing (HLB; Citrus Greening) and Nutrient Deficiency Identification," [Online]. Available: https://edis.ifas.ufl.edu/publication/PP328/pdf. [Accessed: 21-Aug-2026].
[17] University of Minnesota Extension, "Raspberry diseases," [Online]. Available: https://extension.umn.edu/raspberry-farming/raspberry-diseases. [Accessed: 21-Aug-2026].
[18] University of Minnesota Extension, "Commercial blueberry production in Minnesota and Wisconsin," [Online]. Available: https://extension.umn.edu/fruit/commercial-blueberry-production-minnesota-and-wisconsin. [Accessed: 21-Aug-2026].

## 14. UNVERIFIED ITEMS

REFERENCES REQUIRING MANUAL VERIFICATION:
- **test_leaf.jpg:** The original creator and copyright status of this image remain UNKNOWN. DO NOT USE UNTIL MANUALLY VERIFIED.

## 15. FINAL COPYRIGHT / ATTRIBUTION CHECK

| Source | Citation Needed | Attribution Needed | License/Terms Check | Remaining Concern | Required Action |
|---|---|---|---|---|---|
| Open-Meteo API | Yes | Yes (CC BY 4.0) | Verified | Attribution must be displayed to users | Add to Acknowledgments / UI |
| PlantVillage Repo | Yes | No | Not Verified | Original repo lacks explicit LICENSE file | Treat as Open Access |
| UMN/PSU/UF Extension | Yes | Yes | Verified | Copyrighted content | Paraphrase only; cite properly |

## 16. FINAL ANSWER

1. Total sources independently checked: 20
2. Total verified sources: 19
3. Total sources requiring manual verification: 1 (`test_leaf.jpg`)
4. Final recommended number of paper references: 18 (4 core ML papers, 1 API, 13 Extension sites)
5. Sources definitely requiring citation: PlantVillage, MobileNetV2, Grad-CAM, Open-Meteo, 13 Extension Sites.
6. Sources that should NOT be cited: Basic utility libraries (NumPy, Pandas, etc.) unless strictly necessary; `test_leaf.jpg`.
7. Sources requiring attribution: Open-Meteo.
8. Remaining copyright/license concerns: The exact terms of the spMohanty GitHub repository; ensuring paraphrasing of Extension pages.
9. Remaining DOI/bibliographic concerns: None. All primary ML DOIs have been successfully cross-verified.
