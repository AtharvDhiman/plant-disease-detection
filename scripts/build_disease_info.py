"""Build ``data/disease_info.json`` - the knowledge base shown next to a prediction.

Editorial rules followed here
-----------------------------
1. Only widely documented, non-controversial plant-pathology facts are stated:
   the causal organism, the classic symptom picture, the conditions that favour
   the disease, and *cultural* management practices. These are the kind of
   statements that appear consistently across university extension factsheets
   and standard plant-pathology texts.
2. No chemical product names, rates or spray schedules. Those are jurisdiction-
   and crop-specific, change with registration, and giving them would be exactly
   the "invented agricultural fact" the brief warns against. The entries point
   the reader at their local extension service instead.
3. Every entry carries a disclaimer and is labelled informational, not a
   professional diagnosis.
4. The generator fails loudly if the dataset contains a class it has no entry
   for, so the knowledge base can never silently fall out of sync with the model.

Run:
    python scripts/build_disease_info.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import settings  # noqa: E402

DISCLAIMER = (
    "This information is provided for educational purposes only. It is a general "
    "summary of publicly documented plant-pathology guidance, not a professional "
    "diagnosis. Confirm any suspected outbreak with your local agricultural "
    "extension service or a qualified plant pathologist before taking action, "
    "especially before applying any treatment."
)

SOURCES = [
    "University agricultural extension factsheets (e.g. Cornell, UC IPM, Penn State Extension)",
    "American Phytopathological Society (APS) disease compendia",
    "FAO plant-health guidance",
]

# Shared advice for the healthy classes, parameterised by crop.
HEALTHY_TEMPLATE = {
    "category": "healthy",
    "pathogen_type": "None",
    "pathogen": None,
    "severity": "none",
    "description": (
        "No disease symptoms were detected in this leaf. The foliage shows the "
        "uniform colour and intact tissue expected of a healthy plant."
    ),
    "symptoms": [
        "Even, characteristic leaf colour for the species",
        "No necrotic spots, lesions, halos or mildew growth",
        "No curling, mosaic mottling or vein clearing",
        "Intact leaf margins and normal leaf shape",
    ],
    "causes": ["Not applicable - no pathogen indicated by the image"],
    "favourable_conditions": [],
    "prevention": [
        "Keep monitoring: inspect new growth and the undersides of leaves weekly",
        "Water at the base of the plant rather than over the canopy to keep foliage dry",
        "Space and prune for airflow so leaves dry quickly after rain or irrigation",
        "Remove fallen leaves and crop debris, which harbour overwintering pathogens",
        "Rotate crops and avoid planting the same family in the same ground each season",
    ],
    "management": [
        "No treatment is indicated",
        "Continue routine scouting; early detection is the single most effective control",
    ],
}

DISEASES: dict[str, dict] = {
    # ---------------------------------------------------------------- Apple
    "Apple___Apple_scab": {
        "common_name": "Apple scab",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Venturia inaequalis",
        "severity": "moderate",
        "description": (
            "Apple scab is the most common fungal disease of apples in cool, wet "
            "climates. The fungus overwinters in fallen leaves and releases spores "
            "in spring that infect emerging foliage and fruit."
        ),
        "symptoms": [
            "Olive-green to brown velvety spots on leaves, often along the veins",
            "Spots later turn dark and corky with feathered, indistinct margins",
            "Infected leaves yellow and drop early, weakening the tree",
            "Rough, scabby, cracked lesions on the fruit surface",
        ],
        "causes": [
            "Ascospores released from overwintered leaf litter in spring",
            "Prolonged leaf wetness during bud break and early shoot growth",
        ],
        "favourable_conditions": [
            "Cool temperatures around 15-24 C",
            "Extended rain or heavy dew keeping leaves wet for 9+ hours",
            "Dense canopies that dry slowly",
        ],
        "prevention": [
            "Rake and destroy fallen leaves in autumn to remove the overwintering source",
            "Prune to open the canopy so foliage dries quickly",
            "Plant scab-resistant cultivars where available",
            "Avoid overhead irrigation, particularly in spring",
        ],
        "management": [
            "Remove and destroy heavily infected leaves and fruit",
            "Protective treatment programmes are timed to spring infection periods - "
            "consult your extension service for locally registered options and timing",
            "Improve air movement through the canopy with dormant-season pruning",
        ],
    },
    "Apple___Black_rot": {
        "common_name": "Apple black rot",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Botryosphaeria obtusa (Diplodia seriata)",
        "severity": "high",
        "description": (
            "Black rot attacks leaves, fruit and wood. The same fungus causes "
            "'frogeye leaf spot' on foliage and cankers on limbs, so an infection "
            "can persist in the tree from year to year."
        ),
        "symptoms": [
            "Purple-bordered circular leaf spots that develop tan centres ('frogeye')",
            "Brown, firm fruit rot that often begins at the blossom end",
            "Concentric rings of black fruiting bodies on rotting fruit",
            "Sunken, reddish-brown cankers on branches",
        ],
        "causes": [
            "Spores produced on cankers, mummified fruit and dead wood",
            "Infection through wounds, insect damage and hail injury",
        ],
        "favourable_conditions": [
            "Warm, humid weather from 20-27 C",
            "Trees already stressed by drought, winter injury or fire blight",
        ],
        "prevention": [
            "Prune out dead wood and cankered limbs during the dormant season",
            "Remove mummified fruit from the tree and the orchard floor",
            "Avoid bark injuries; keep trees well watered and appropriately fertilised",
        ],
        "management": [
            "Cut cankers well below the visible margin and destroy the prunings",
            "Sanitise pruning tools between cuts",
            "Locally registered protective treatments may be warranted in wet seasons",
        ],
    },
    "Apple___Cedar_apple_rust": {
        "common_name": "Cedar apple rust",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Gymnosporangium juniperi-virginianae",
        "severity": "moderate",
        "description": (
            "A rust fungus that needs two hosts to complete its life cycle: apple "
            "and a juniper or eastern red cedar. Spores blow from galls on the "
            "juniper host to apple foliage in spring."
        ),
        "symptoms": [
            "Bright yellow-orange spots on the upper leaf surface",
            "Spots enlarge and develop a red border with tiny black dots",
            "Tube-like fungal structures protruding from the underside of the spot",
            "Premature leaf drop in heavy infections",
        ],
        "causes": [
            "Basidiospores released from gelatinous galls on nearby junipers",
            "Wet spring weather coinciding with apple leaf emergence",
        ],
        "favourable_conditions": [
            "Rainy periods in spring when juniper galls swell",
            "Apple trees planted within a few hundred metres of juniper hosts",
        ],
        "prevention": [
            "Remove juniper/red cedar hosts within the immediate vicinity where practical",
            "Inspect nearby junipers in late winter and prune out the brown galls",
            "Choose rust-resistant apple cultivars",
        ],
        "management": [
            "Remove galls from alternate hosts before they release spores in spring",
            "Protective sprays timed to spring spore release are used commercially - "
            "check locally registered products and timing with your extension service",
        ],
    },
    "Apple___healthy": {"common_name": "Healthy apple leaf", **HEALTHY_TEMPLATE},
    # ------------------------------------------------------------ Blueberry
    "Blueberry___healthy": {"common_name": "Healthy blueberry leaf", **HEALTHY_TEMPLATE},
    # --------------------------------------------------------------- Cherry
    "Cherry_(including_sour)___Powdery_mildew": {
        "common_name": "Cherry powdery mildew",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Podosphaera clandestina",
        "severity": "moderate",
        "description": (
            "Powdery mildew colonises the leaf surface rather than the inside of "
            "the tissue, producing the characteristic white bloom. Unusually among "
            "fungal diseases it does not need free water to infect."
        ),
        "symptoms": [
            "White to greyish powdery patches, usually first on the leaf underside",
            "Affected young leaves cup, curl and distort",
            "Shoot growth is stunted; severe infections defoliate the shoot tips",
            "Light-coloured blotches on fruit in late-season infections",
        ],
        "causes": [
            "Overwintering fungal structures in buds and on fallen leaves",
            "Wind-dispersed conidia landing on succulent new growth",
        ],
        "favourable_conditions": [
            "Warm days around 20-27 C with high humidity but dry leaves",
            "Dense, shaded canopies and vigorous, succulent shoot growth",
        ],
        "prevention": [
            "Prune to open the canopy and improve light penetration and airflow",
            "Avoid excessive nitrogen, which pushes the soft growth mildew prefers",
            "Remove and destroy infected shoot tips and fallen leaves",
        ],
        "management": [
            "Remove the most heavily colonised shoots to reduce inoculum",
            "Improve canopy ventilation before the next season",
            "Consult your extension service on locally registered materials and timing",
        ],
    },
    "Cherry_(including_sour)___healthy": {"common_name": "Healthy cherry leaf", **HEALTHY_TEMPLATE},
    # ----------------------------------------------------------------- Corn
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "common_name": "Grey leaf spot of maize",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Cercospora zeae-maydis",
        "severity": "high",
        "description": (
            "One of the most economically damaging foliar diseases of maize. "
            "Lesions are sharply limited by the leaf veins, giving them their "
            "distinctive rectangular shape."
        ),
        "symptoms": [
            "Narrow, rectangular tan-to-grey lesions running parallel to the veins",
            "Lesions begin small with a yellow halo and elongate over time",
            "Lower leaves are affected first, with the disease moving upward",
            "Severe infection blights the whole leaf and reduces grain fill",
        ],
        "causes": [
            "Fungus surviving in maize residue left on the soil surface",
            "Spores splashed and blown onto the lower canopy",
        ],
        "favourable_conditions": [
            "Warm temperatures of 25-30 C with prolonged high humidity",
            "Continuous maize with reduced or no tillage leaving residue on the surface",
        ],
        "prevention": [
            "Rotate away from maize for at least one season",
            "Incorporate or remove crop residue to speed decomposition",
            "Plant hybrids with documented grey leaf spot resistance",
            "Avoid excessively dense plant populations",
        ],
        "management": [
            "Scout from the mid-vegetative stage onward, especially the ear leaf",
            "Foliar treatment decisions are made on hybrid susceptibility and disease "
            "pressure - your extension service publishes local thresholds",
        ],
    },
    "Corn_(maize)___Common_rust_": {
        "common_name": "Common rust of maize",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Puccinia sorghi",
        "severity": "moderate",
        "description": (
            "A wind-borne rust that arrives on spores blown in from warmer regions "
            "rather than surviving locally in most temperate areas, so severity "
            "varies a great deal from season to season."
        ),
        "symptoms": [
            "Small cinnamon-brown, powdery pustules on both leaf surfaces",
            "Pustules rupture the epidermis and rub off as orange dust",
            "Pustules darken to brownish-black as the season progresses",
            "Heavy infection yellows and kills the leaf",
        ],
        "causes": [
            "Urediniospores carried long distances on the wind",
            "Rapid cycling of the fungus under cool, humid conditions",
        ],
        "favourable_conditions": [
            "Cool temperatures of 16-25 C",
            "High humidity, heavy dew or six or more hours of leaf wetness",
        ],
        "prevention": [
            "Plant resistant or tolerant hybrids, the most reliable single measure",
            "Avoid very late plantings that expose young tissue to peak spore load",
            "Keep the crop well nourished so it tolerates infection better",
        ],
        "management": [
            "Most field maize does not need treatment; sweet corn and seed maize are "
            "more sensitive",
            "Scout regularly and consult local extension thresholds before treating",
        ],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "common_name": "Northern corn leaf blight",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Exserohilum turcicum (Setosphaeria turcica)",
        "severity": "high",
        "description": (
            "Produces the largest lesions of the common maize leaf diseases. "
            "Substantial yield loss follows when the blight reaches the ear leaf "
            "before or during grain fill."
        ),
        "symptoms": [
            "Long, elliptical, cigar-shaped grey-green to tan lesions",
            "Lesions are 3-15 cm long and not limited by the veins",
            "Dark, dusty spore masses form in the lesion centre in humid weather",
            "Lower leaves are affected first and can be killed entirely",
        ],
        "causes": [
            "Fungus overwintering in infected maize residue",
            "Spores splashed and blown to the lower leaves in wet weather",
        ],
        "favourable_conditions": [
            "Moderate temperatures of 18-27 C",
            "Extended leaf wetness of six or more hours from rain, dew or fog",
        ],
        "prevention": [
            "Grow hybrids carrying northern leaf blight resistance genes",
            "Rotate out of maize and manage residue to reduce the inoculum",
            "Avoid excessive plant density in high-risk fields",
        ],
        "management": [
            "Scout the ear leaf and the leaf above it around tasselling",
            "Treatment decisions depend on growth stage and hybrid rating - "
            "follow locally published extension guidance",
        ],
    },
    "Corn_(maize)___healthy": {"common_name": "Healthy maize leaf", **HEALTHY_TEMPLATE},
    # ---------------------------------------------------------------- Grape
    "Grape___Black_rot": {
        "common_name": "Grape black rot",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Guignardia bidwellii",
        "severity": "high",
        "description": (
            "A destructive warm, wet-climate disease of grapes. Fruit losses can be "
            "total in an unmanaged vineyard because infected berries shrivel into "
            "hard mummies that carry the fungus to the next season."
        ),
        "symptoms": [
            "Circular tan to reddish-brown leaf spots with dark borders",
            "A ring of tiny black fruiting bodies just inside the spot margin",
            "Berries develop a light brown soft spot that spreads over the whole fruit",
            "Affected berries shrivel into hard, black, wrinkled mummies",
        ],
        "causes": [
            "Spores released from mummified berries and infected canes in spring",
            "Rain splash spreading spores to expanding shoots and young fruit",
        ],
        "favourable_conditions": [
            "Warm temperatures of 20-27 C with frequent rain",
            "Long wet periods during the weeks after bloom, when fruit is most susceptible",
        ],
        "prevention": [
            "Remove and destroy mummified berries from the vine and the ground",
            "Prune out infected canes during dormancy",
            "Train and position shoots so the fruit zone gets sun and airflow",
        ],
        "management": [
            "Sanitation is the foundation - mummies left in the canopy are the main source",
            "Preventive programmes target the pre-bloom to fruit-set window; consult "
            "your extension service for locally registered options",
        ],
    },
    "Grape___Esca_(Black_Measles)": {
        "common_name": "Esca (black measles)",
        "category": "fungal",
        "pathogen_type": "Fungal complex",
        "pathogen": "Phaeomoniella chlamydospora, Phaeoacremonium spp. and associated wood-rot fungi",
        "severity": "high",
        "description": (
            "A trunk disease rather than a simple leaf disease: a complex of fungi "
            "colonises the permanent wood and the foliar symptoms are the visible "
            "consequence. Vines may decline over years or collapse suddenly."
        ),
        "symptoms": [
            "Interveinal chlorotic or reddish 'tiger-stripe' banding on leaves",
            "Small dark spots ('measles') on berries, which may crack",
            "Sudden wilting and dieback of an entire arm or vine in hot weather",
            "Dark streaking and soft, decayed wood visible in trunk cross-sections",
        ],
        "causes": [
            "Fungi entering through large pruning wounds",
            "Long-term colonisation and decay of the trunk and cordon wood",
        ],
        "favourable_conditions": [
            "Older vines with large pruning wounds",
            "Rain shortly after pruning, when wounds are most vulnerable",
        ],
        "prevention": [
            "Prune late in the dormant season when wounds heal faster",
            "Avoid pruning immediately before rain",
            "Make small pruning cuts and protect large wounds",
            "Use clean planting material from a reputable nursery",
        ],
        "management": [
            "There is no cure once the wood is colonised; management is about slowing spread",
            "Remove and destroy dead vines and severely affected arms",
            "Trunk renewal from a healthy sucker can extend the life of a valuable vine",
        ],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "common_name": "Grape leaf blight (Isariopsis leaf spot)",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Pseudocercospora vitis (Isariopsis clavispora)",
        "severity": "moderate",
        "description": (
            "A late-season foliar disease. It rarely infects the fruit directly, but "
            "the defoliation it causes reduces sugar accumulation and leaves the vine "
            "poorly prepared for winter."
        ),
        "symptoms": [
            "Irregular dark brown to blackish angular leaf spots",
            "Spots often bounded by the veins and may merge into large blighted areas",
            "Olive-grey fungal growth on the underside of the spot in humid weather",
            "Premature defoliation starting with the older basal leaves",
        ],
        "causes": [
            "Fungus overwintering on fallen infected leaves",
            "Spores dispersed by rain splash and wind onto the lower canopy",
        ],
        "favourable_conditions": [
            "Warm, humid, rainy late-summer weather",
            "Dense canopies with poor air circulation",
        ],
        "prevention": [
            "Remove fallen leaves from the vineyard floor at the end of the season",
            "Manage canopy density with shoot thinning and leaf removal in the fruit zone",
            "Avoid overhead irrigation",
        ],
        "management": [
            "Improve airflow first; the disease is strongly canopy-driven",
            "Where protection is needed, follow locally registered late-season programmes",
        ],
    },
    "Grape___healthy": {"common_name": "Healthy grape leaf", **HEALTHY_TEMPLATE},
    # --------------------------------------------------------------- Orange
    "Orange___Haunglongbing_(Citrus_greening)": {
        "common_name": "Huanglongbing (citrus greening)",
        "category": "bacterial",
        "pathogen_type": "Bacterium (phloem-limited), insect-vectored",
        "pathogen": "Candidatus Liberibacter asiaticus, spread by the Asian citrus psyllid",
        "severity": "critical",
        "description": (
            "The most serious citrus disease worldwide. The bacterium lives in the "
            "phloem and is carried between trees by psyllid insects. Infected trees "
            "decline progressively and there is no cure - which is why early "
            "detection and vector control dominate all management programmes."
        ),
        "symptoms": [
            "Blotchy, asymmetric yellow mottling that does not match on either side of the midrib",
            "Yellowing of one shoot or sector while the rest of the tree stays green",
            "Small, lopsided fruit that stays green at the blossom end",
            "Bitter, salty-tasting juice and heavy premature fruit drop",
            "Twig dieback and progressive thinning of the canopy",
        ],
        "causes": [
            "Feeding by infected Asian citrus psyllids transmitting the bacterium",
            "Movement of infected nursery stock and budwood between areas",
        ],
        "favourable_conditions": [
            "Presence of psyllid populations and warm climates that favour them",
            "Frequent flushes of new growth, which psyllids need to reproduce",
        ],
        "prevention": [
            "Plant only certified disease-free nursery stock",
            "Monitor for psyllids on every new flush",
            "Report suspected cases to the plant-health authority - HLB is a regulated disease in many countries",
            "Do not move citrus plant material between regions",
        ],
        "management": [
            "There is no cure; infected trees remain a source of infection for the whole block",
            "Area-wide psyllid control combined with removal of infected trees is the standard programme",
            "Nutritional programmes can prolong productivity but do not eliminate the bacterium",
            "Coordinate with local plant-health authorities - individual action is rarely sufficient",
        ],
    },
    # ---------------------------------------------------------------- Peach
    "Peach___Bacterial_spot": {
        "common_name": "Bacterial spot of peach",
        "category": "bacterial",
        "pathogen_type": "Bacterium",
        "pathogen": "Xanthomonas arboricola pv. pruni",
        "severity": "high",
        "description": (
            "A bacterial disease favoured by wind-driven rain. Because bacteria "
            "enter through natural openings and wounds, sandy, windy sites where "
            "leaves are abraded suffer worst."
        ),
        "symptoms": [
            "Small, angular, water-soaked spots between the veins",
            "Spots turn dark purple-brown and the dead centre often falls out ('shot-hole')",
            "Yellowing and heavy leaf drop in severe cases",
            "Sunken, cracked, dark lesions on the fruit surface",
        ],
        "causes": [
            "Bacteria overwintering in twig cankers and buds",
            "Splashing rain and wind driving bacteria into leaf openings",
        ],
        "favourable_conditions": [
            "Warm temperatures of 24-29 C with frequent rain and wind",
            "Sandy soils and exposed sites where blowing sand injures the foliage",
        ],
        "prevention": [
            "Plant tolerant cultivars - susceptibility varies enormously between them",
            "Establish windbreaks to reduce wind injury on exposed sites",
            "Avoid overhead irrigation and excessive nitrogen",
            "Prune out cankered twigs during dormancy",
        ],
        "management": [
            "Cultural control and cultivar choice matter more than treatment here",
            "Copper-based programmes are used at specific phenological stages; timing and "
            "rate are critical and locally specified - consult your extension service",
        ],
    },
    "Peach___healthy": {"common_name": "Healthy peach leaf", **HEALTHY_TEMPLATE},
    # ----------------------------------------------------------- Bell pepper
    "Pepper,_bell___Bacterial_spot": {
        "common_name": "Bacterial spot of pepper",
        "category": "bacterial",
        "pathogen_type": "Bacterium",
        "pathogen": "Xanthomonas spp. (X. euvesicatoria and related species)",
        "severity": "high",
        "description": (
            "A seed- and splash-borne bacterial disease of peppers and tomatoes. "
            "Once established in a warm, wet season it spreads very quickly and is "
            "difficult to stop, so prevention carries most of the weight."
        ),
        "symptoms": [
            "Small water-soaked spots that become brown and angular, often with a yellow halo",
            "Spots may drop out leaving a ragged, shot-hole appearance",
            "Severe defoliation exposing fruit to sunscald",
            "Raised, scabby, rough lesions on the fruit",
        ],
        "causes": [
            "Contaminated seed and infected transplants introducing the bacterium",
            "Splashing water and handling of wet plants spreading it through the crop",
            "Survival on volunteer plants and infected crop debris",
        ],
        "favourable_conditions": [
            "Warm temperatures of 24-30 C with high humidity",
            "Overhead irrigation, driving rain and prolonged leaf wetness",
        ],
        "prevention": [
            "Use certified pathogen-free seed and transplants",
            "Rotate out of peppers and tomatoes for two to three years",
            "Use drip irrigation instead of overhead watering",
            "Never work among plants while the foliage is wet",
            "Remove and destroy crop debris at the end of the season",
        ],
        "management": [
            "Remove severely infected plants to slow spread within a block",
            "Copper-based preventive programmes are commonly used but resistance is "
            "widespread - check current local recommendations",
            "Prioritise sanitation and irrigation change over repeated spraying",
        ],
    },
    "Pepper,_bell___healthy": {"common_name": "Healthy bell pepper leaf", **HEALTHY_TEMPLATE},
    # --------------------------------------------------------------- Potato
    "Potato___Early_blight": {
        "common_name": "Potato early blight",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Alternaria solani",
        "severity": "moderate",
        "description": (
            "A common fungal disease that attacks older, stressed foliage first. "
            "Despite the name it usually appears at or after tuber initiation, not "
            "early in the season."
        ),
        "symptoms": [
            "Dark brown spots with concentric rings, giving a 'target-board' look",
            "A yellow halo surrounds the expanding lesion",
            "Lowest, oldest leaves are affected first and die progressively upward",
            "Dark, sunken, leathery lesions on tubers with a distinct margin",
        ],
        "causes": [
            "Fungus surviving in infected plant debris, soil and volunteer tubers",
            "Spores spread by wind, rain splash and irrigation",
        ],
        "favourable_conditions": [
            "Warm temperatures of 24-29 C alternating with wet and dry periods",
            "Plants stressed by poor nutrition, drought or heavy fruit load",
        ],
        "prevention": [
            "Rotate crops for two to three years away from potato and tomato",
            "Maintain adequate nitrogen - deficient plants are far more susceptible",
            "Irrigate at the base and avoid prolonged leaf wetness",
            "Destroy volunteers and cull piles",
        ],
        "management": [
            "Remove severely affected lower leaves to reduce inoculum",
            "Keep the crop vigorous; stress management is genuinely effective here",
            "Protective programmes exist for high-pressure situations - "
            "follow locally registered guidance",
        ],
    },
    "Potato___Late_blight": {
        "common_name": "Potato late blight",
        "category": "oomycete",
        "pathogen_type": "Oomycete (water mould)",
        "pathogen": "Phytophthora infestans",
        "severity": "critical",
        "description": (
            "The disease responsible for the Irish potato famine and still the most "
            "feared potato disease worldwide. Under cool, wet conditions it can "
            "destroy a crop within one to two weeks, so it demands immediate action "
            "rather than watchful waiting."
        ),
        "symptoms": [
            "Pale green to dark brown water-soaked patches, often starting at leaf tips and margins",
            "A fuzzy white mould ring on the underside of the lesion in humid mornings",
            "Rapid blackening and collapse of stems and foliage",
            "A distinctive smell from rotting foliage in heavy outbreaks",
            "Reddish-brown granular dry rot extending into the tuber flesh",
        ],
        "causes": [
            "Sporangia spread by wind and rain over long distances",
            "Infected seed tubers, cull piles and volunteers starting the season's epidemic",
        ],
        "favourable_conditions": [
            "Cool temperatures of 10-24 C with relative humidity above 90%",
            "Extended leaf wetness from rain, fog or heavy dew",
            "Dense canopies that stay wet",
        ],
        "prevention": [
            "Plant certified disease-free seed tubers",
            "Destroy cull piles and volunteer plants before the season starts",
            "Hill soil well over developing tubers to block spores reaching them",
            "Follow late-blight forecasting services where they operate in your area",
        ],
        "management": [
            "Act immediately - this disease does not plateau on its own in wet weather",
            "Remove and destroy infected plants; do not compost them",
            "Kill the haulm before harvest in an infected field so tubers are not "
            "contaminated on the way out of the ground",
            "Protective programmes must be preventive; consult your extension service urgently",
        ],
    },
    "Potato___healthy": {"common_name": "Healthy potato leaf", **HEALTHY_TEMPLATE},
    # ------------------------------------------------------ Raspberry / Soy
    "Raspberry___healthy": {"common_name": "Healthy raspberry leaf", **HEALTHY_TEMPLATE},
    "Soybean___healthy": {"common_name": "Healthy soybean leaf", **HEALTHY_TEMPLATE},
    # --------------------------------------------------------------- Squash
    "Squash___Powdery_mildew": {
        "common_name": "Cucurbit powdery mildew",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Podosphaera xanthii (and Erysiphe cichoracearum)",
        "severity": "moderate",
        "description": (
            "Probably the most recognisable disease of squash and other cucurbits. "
            "Unlike most fungal diseases it thrives in dry conditions with humid "
            "nights, so it often appears when growers expect disease pressure to be low."
        ),
        "symptoms": [
            "White, talcum-powder-like colonies on leaves and stems",
            "Usually starts on shaded lower and inner leaves, then spreads over the canopy",
            "Leaves yellow, become brittle and die, exposing fruit to sunscald",
            "Reduced fruit size, poor flavour and premature ripening",
        ],
        "causes": [
            "Wind-blown conidia arriving from other cucurbit plantings",
            "Dense, shaded canopies with humid microclimates",
        ],
        "favourable_conditions": [
            "Warm days of 20-30 C with high humidity but dry leaf surfaces",
            "Dense plantings, shade and poor air circulation",
        ],
        "prevention": [
            "Space plants generously and orient rows for airflow",
            "Grow resistant or tolerant varieties, which are widely available for cucurbits",
            "Remove and destroy the most heavily infected lower leaves early",
            "Avoid excess nitrogen late in the season",
        ],
        "management": [
            "Begin management at the first sign - powdery mildew is far easier to hold "
            "back than to reverse",
            "Rotate between different modes of action if treating; resistance develops quickly",
            "Consult your extension service for currently registered options",
        ],
    },
    # ----------------------------------------------------------- Strawberry
    "Strawberry___Leaf_scorch": {
        "common_name": "Strawberry leaf scorch",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Diplocarpon earlianum",
        "severity": "moderate",
        "description": (
            "A common foliar disease of strawberry. It is distinguished from leaf "
            "spot by its lesions, which lack the light grey centre and instead stay "
            "uniformly dark purple."
        ),
        "symptoms": [
            "Numerous small, irregular dark purple spots on the upper leaf surface",
            "Spots merge until the leaf looks scorched and reddish-brown",
            "Leaf margins curl upward as the tissue dies",
            "Dark lesions can also appear on petioles, runners and fruit stalks",
        ],
        "causes": [
            "Fungus overwintering on infected leaves left in the planting",
            "Spores spread by splashing rain and overhead irrigation",
        ],
        "favourable_conditions": [
            "Warm, wet weather of 20-30 C with long leaf-wetness periods",
            "Dense, matted rows that dry slowly",
        ],
        "prevention": [
            "Renovate matted rows promptly after harvest to remove old infected foliage",
            "Use drip rather than overhead irrigation",
            "Space plants and manage runners so foliage dries quickly",
            "Start with certified disease-free transplants",
        ],
        "management": [
            "Remove and destroy infected leaves during renovation",
            "Improve airflow and irrigation practice before considering treatment",
            "Where treatment is needed, follow locally registered options",
        ],
    },
    "Strawberry___healthy": {"common_name": "Healthy strawberry leaf", **HEALTHY_TEMPLATE},
    # --------------------------------------------------------------- Tomato
    "Tomato___Bacterial_spot": {
        "common_name": "Bacterial spot of tomato",
        "category": "bacterial",
        "pathogen_type": "Bacterium",
        "pathogen": "Xanthomonas spp. (X. vesicatoria, X. euvesicatoria and related species)",
        "severity": "high",
        "description": (
            "A warm-weather bacterial disease affecting leaves, stems and fruit. It "
            "is introduced most often on contaminated seed or transplants, which is "
            "why clean planting material is the single most effective control."
        ),
        "symptoms": [
            "Small dark brown to black angular leaf spots, sometimes with a yellow halo",
            "Spots are greasy and water-soaked when young",
            "Leaf tissue between spots yellows, and heavily spotted leaves drop",
            "Small raised scabby spots on green fruit",
        ],
        "causes": [
            "Contaminated seed and infected transplants",
            "Rain splash, overhead irrigation and handling of wet foliage",
            "Survival on crop debris and solanaceous weeds",
        ],
        "favourable_conditions": [
            "Warm temperatures of 24-30 C with high humidity",
            "Driving rain and overhead irrigation",
        ],
        "prevention": [
            "Use certified pathogen-free seed; hot-water seed treatment is standard practice",
            "Rotate away from tomato and pepper for at least two years",
            "Switch to drip irrigation",
            "Stake and prune for airflow; avoid working in wet foliage",
        ],
        "management": [
            "Remove severely infected plants promptly",
            "Copper-based preventive programmes are common but copper resistance is "
            "widespread - check current local recommendations",
            "Sanitise stakes, tools and equipment between seasons",
        ],
    },
    "Tomato___Early_blight": {
        "common_name": "Tomato early blight",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Alternaria linariae (A. solani)",
        "severity": "moderate",
        "description": (
            "A very common tomato disease that works its way up the plant from the "
            "oldest leaves. Progressive defoliation exposes fruit to sunscald and "
            "cuts yield even when the fruit itself is not infected."
        ),
        "symptoms": [
            "Brown spots with concentric rings forming a target pattern",
            "Yellow halo around each lesion",
            "Lowest leaves affected first, dying and dropping as the disease climbs",
            "Dark, sunken, leathery lesions at the stem end of the fruit",
            "Dark girdling lesions on the stem near the soil line ('collar rot') on young plants",
        ],
        "causes": [
            "Fungus overwintering in infected debris and soil",
            "Spores splashed onto lower leaves by rain and irrigation",
            "Infected seed and volunteer solanaceous plants",
        ],
        "favourable_conditions": [
            "Warm temperatures of 24-29 C with alternating wet and dry periods",
            "Heavy dew, and plants stressed by poor nutrition or heavy fruit set",
        ],
        "prevention": [
            "Mulch to stop soil splashing onto the lowest leaves - highly effective here",
            "Rotate for two to three years away from tomato, potato and eggplant",
            "Stake or cage plants to lift foliage off the ground",
            "Remove the bottom leaves as the plant establishes",
            "Keep nitrogen adequate; deficient plants succumb faster",
        ],
        "management": [
            "Remove and destroy affected lower leaves as soon as spots appear",
            "Do not compost infected material in a home compost heap",
            "Protective treatment may be justified in wet seasons - "
            "consult your extension service for registered options",
        ],
    },
    "Tomato___Late_blight": {
        "common_name": "Tomato late blight",
        "category": "oomycete",
        "pathogen_type": "Oomycete (water mould)",
        "pathogen": "Phytophthora infestans",
        "severity": "critical",
        "description": (
            "The same pathogen that causes potato late blight. It is the most "
            "destructive tomato disease in cool, wet weather and can destroy a "
            "planting in under two weeks, so it requires immediate action."
        ),
        "symptoms": [
            "Large, irregular greasy grey-green blotches, often starting at leaf edges",
            "White fuzzy sporulation on the underside of lesions in humid conditions",
            "Dark brown to black lesions girdling stems and petioles",
            "Firm, greasy brown blotches on green fruit that spread rapidly",
            "Whole-plant collapse within days under favourable weather",
        ],
        "causes": [
            "Wind-blown sporangia that travel long distances between plantings",
            "Infected potato cull piles, volunteers and infected transplants",
        ],
        "favourable_conditions": [
            "Cool temperatures of 10-24 C with humidity above 90%",
            "Extended rain, fog or heavy dew keeping foliage wet",
        ],
        "prevention": [
            "Buy transplants from a reputable source and inspect them on arrival",
            "Eliminate nearby potato cull piles and volunteers",
            "Space widely and prune for maximum airflow",
            "Watch regional late-blight alerts during cool wet spells",
        ],
        "management": [
            "Remove and bag infected plants immediately - do not compost",
            "Treat neighbouring healthy plants preventively where the disease is confirmed nearby",
            "Contact your extension service without delay; late blight is a community "
            "problem because spores travel between properties",
        ],
    },
    "Tomato___Leaf_Mold": {
        "common_name": "Tomato leaf mould",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Passalora fulva (Fulvia fulva / Cladosporium fulvum)",
        "severity": "moderate",
        "description": (
            "Primarily a problem of greenhouses, high tunnels and other protected "
            "culture, where humidity stays high. It is uncommon in open, well-"
            "ventilated field plantings."
        ),
        "symptoms": [
            "Pale green to yellow patches on the upper leaf surface with no sharp margin",
            "Olive-green to greyish-brown velvety mould on the corresponding underside",
            "Leaves curl, wither and drop, starting with the older foliage",
            "Occasional dark leathery rot at the stem end of the fruit",
        ],
        "causes": [
            "Spores surviving on plant debris, greenhouse structures and seed",
            "Air movement and handling spreading spores within the structure",
        ],
        "favourable_conditions": [
            "Relative humidity above 85% for extended periods",
            "Temperatures of 21-24 C in poorly ventilated protected structures",
        ],
        "prevention": [
            "Ventilate aggressively and heat to reduce night humidity in protected culture",
            "Space plants and prune lower leaves for air movement",
            "Water early in the day at the base of the plant",
            "Grow resistant cultivars - resistance genes are well established in tomato",
            "Disinfect greenhouse surfaces and stakes between crops",
        ],
        "management": [
            "Reduce humidity first; this disease is essentially a ventilation problem",
            "Remove and destroy affected leaves",
            "Consult your extension service for registered protected-culture options",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "common_name": "Septoria leaf spot",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Septoria lycopersici",
        "severity": "moderate",
        "description": (
            "One of the most common tomato leaf diseases. It does not infect the "
            "fruit, but the defoliation it causes reduces yield and exposes fruit to "
            "sunscald."
        ),
        "symptoms": [
            "Many small circular spots with dark borders and grey or tan centres",
            "Tiny black fruiting bodies visible in the spot centres, often with a hand lens",
            "Lower leaves affected first; heavy spotting turns leaves yellow then brown",
            "Progressive defoliation from the bottom of the plant upward",
        ],
        "causes": [
            "Fungus overwintering on infected tomato debris and solanaceous weeds",
            "Rain and irrigation splashing spores onto the lower leaves",
        ],
        "favourable_conditions": [
            "Moderate temperatures of 20-25 C with prolonged wet foliage",
            "Dense plantings and soil splash on unmulched ground",
        ],
        "prevention": [
            "Mulch to prevent splash from the soil surface",
            "Rotate for two years or more away from tomato",
            "Remove solanaceous weeds such as nightshade and horsenettle from field edges",
            "Stake plants and prune the lowest leaves",
            "Water at the base, early in the day",
        ],
        "management": [
            "Remove and destroy spotted lower leaves as they appear",
            "Sanitise tools and hands after handling infected plants",
            "Where pressure is high, protective programmes are available - "
            "consult your extension service",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "common_name": "Two-spotted spider mite",
        "category": "pest",
        "pathogen_type": "Arachnid pest (not a pathogen)",
        "pathogen": "Tetranychus urticae",
        "severity": "moderate",
        "description": (
            "This class is a *pest* rather than a disease. Two-spotted spider mites "
            "are tiny arachnids that feed by puncturing leaf cells and draining the "
            "contents, producing the fine stippling that identifies them. "
            "Populations explode in hot, dry weather."
        ),
        "symptoms": [
            "Fine yellow or white stippling (tiny pale dots) across the leaf surface",
            "Leaves take on a bronzed, dusty or sandblasted appearance",
            "Fine webbing on the undersides of leaves and between stems in heavy infestations",
            "Tiny moving specks on the underside, visible with a hand lens",
            "Leaf drop and plant decline when populations are high",
        ],
        "causes": [
            "Hot, dry, dusty conditions that speed up the mite life cycle",
            "Loss of natural predators, often after broad-spectrum insecticide use",
            "Movement in on infested transplants",
        ],
        "favourable_conditions": [
            "Hot weather above 27 C with low humidity",
            "Drought-stressed plants and dusty field edges",
        ],
        "prevention": [
            "Keep plants adequately watered; drought stress strongly favours mites",
            "Inspect the undersides of leaves regularly with a hand lens",
            "Avoid broad-spectrum insecticides that kill predatory mites",
            "Control dust on roads and field edges",
        ],
        "management": [
            "A forceful spray of water on leaf undersides physically removes many mites",
            "Conserve or introduce predatory mites, which are highly effective biological control",
            "Remove and destroy severely infested leaves",
            "If treating, choose a selective miticide and rotate modes of action - "
            "spider mites develop resistance very rapidly",
        ],
    },
    "Tomato___Target_Spot": {
        "common_name": "Target spot of tomato",
        "category": "fungal",
        "pathogen_type": "Fungus",
        "pathogen": "Corynespora cassiicola",
        "severity": "moderate",
        "description": (
            "A warm, humid-climate disease that attacks leaves, stems and fruit. Its "
            "leaf lesions resemble early blight, but target spot also produces "
            "distinctive pitted lesions on the fruit."
        ),
        "symptoms": [
            "Small brown spots that enlarge into lesions with light centres and concentric rings",
            "A yellow halo often surrounds the lesion",
            "Lesions may coalesce and blight whole leaves, causing defoliation",
            "Small sunken, pitted brown spots on the fruit",
        ],
        "causes": [
            "Fungus surviving on infected crop debris and alternative hosts",
            "Spores spread by wind, rain splash and irrigation",
        ],
        "favourable_conditions": [
            "Warm temperatures of 20-28 C with prolonged high humidity",
            "Dense canopies and extended leaf wetness",
        ],
        "prevention": [
            "Rotate crops and remove infected debris at the end of the season",
            "Prune and stake for open, quick-drying canopies",
            "Use drip irrigation and avoid wetting foliage",
        ],
        "management": [
            "Remove affected leaves early, before lesions coalesce",
            "Improve canopy airflow",
            "Consult your extension service on registered protective options if pressure is high",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "common_name": "Tomato yellow leaf curl virus (TYLCV)",
        "category": "viral",
        "pathogen_type": "Virus, insect-vectored",
        "pathogen": "Tomato yellow leaf curl virus, transmitted by the silverleaf whitefly (Bemisia tabaci)",
        "severity": "critical",
        "description": (
            "One of the most damaging tomato viruses worldwide. It is transmitted "
            "only by whiteflies - not by touch, tools or seed - so management is "
            "fundamentally about managing the whitefly vector. Plants infected when "
            "young may set almost no fruit."
        ),
        "symptoms": [
            "Severe upward cupping and curling of leaflet margins",
            "Marked yellowing of leaf margins and between the veins",
            "Leaves are noticeably small, thickened and crumpled",
            "Stunted, bushy plants with shortened internodes",
            "Flower drop and drastically reduced fruit set",
        ],
        "causes": [
            "Feeding by viruliferous silverleaf whiteflies",
            "Introduction on infected transplants",
            "Reservoirs in nearby weeds and other host crops",
        ],
        "favourable_conditions": [
            "Warm conditions supporting large whitefly populations",
            "Continuous tomato production with overlapping plantings",
        ],
        "prevention": [
            "Grow TYLCV-resistant or tolerant varieties, which are widely available",
            "Raise transplants under insect-proof screening",
            "Use reflective mulches to repel incoming whiteflies",
            "Control whiteflies from the earliest growth stages",
            "Remove weed hosts around the planting",
        ],
        "management": [
            "There is no cure for an infected plant",
            "Remove infected plants carefully - bag them in place so whiteflies do not disperse",
            "Focus all effort on vector control and on protecting young plants, which are most vulnerable",
            "Observe a host-free period between crops where this is practised locally",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "common_name": "Tomato mosaic virus (ToMV)",
        "category": "viral",
        "pathogen_type": "Virus",
        "pathogen": "Tomato mosaic virus (Tobamovirus)",
        "severity": "high",
        "description": (
            "An exceptionally stable virus that spreads mechanically - on hands, "
            "tools, clothing and in plant debris - rather than by an insect vector. "
            "It survives in dried debris for years, which makes strict sanitation "
            "the central control measure."
        ),
        "symptoms": [
            "Light and dark green mottled mosaic pattern on the leaves",
            "Leaves may be distorted, narrow or fern-like ('shoestring' leaves)",
            "Stunted growth and a generally poor, uneven stand",
            "Internal browning of the fruit wall and uneven ripening",
            "Symptoms often vary with temperature and can fade in hot weather",
        ],
        "causes": [
            "Mechanical transmission on hands, tools, stakes and clothing",
            "Contaminated seed and infected plant debris in the soil",
            "Tobacco products handled before touching plants",
        ],
        "favourable_conditions": [
            "Intensive handling operations such as pruning, tying and grafting",
            "Reuse of stakes, trays and other equipment without disinfection",
        ],
        "prevention": [
            "Grow resistant cultivars where available",
            "Wash hands thoroughly and disinfect tools between plants",
            "Do not use tobacco products while handling tomato plants",
            "Use certified virus-free seed and transplants",
            "Remove and destroy crop debris; do not plant into infested debris",
        ],
        "management": [
            "There is no treatment for an infected plant",
            "Remove infected plants and handle them last to avoid spreading the virus",
            "Disinfect stakes, trays, tools and greenhouse surfaces between crops",
        ],
    },
    "Tomato___healthy": {"common_name": "Healthy tomato leaf", **HEALTHY_TEMPLATE},
}


def prettify_plant(raw: str) -> str:
    """``Pepper,_bell`` -> ``Bell pepper``; ``Corn_(maize)`` -> ``Corn (maize)``."""
    name = raw.replace("_", " ").strip()
    if "," in name:
        head, tail = name.split(",", 1)
        return f"{tail.strip().capitalize()} {head.strip().lower()}"
    return name


def main() -> int:
    report_path = settings.dataset_report_path
    if not report_path.exists():
        raise FileNotFoundError(
            "artifacts/dataset_report.json missing - run training/dataset_inspect.py first."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    class_names: list[str] = report["classes"]["names"]

    missing = [name for name in class_names if name not in DISEASES]
    extra = [name for name in DISEASES if name not in class_names]
    if missing:
        raise SystemExit(
            "Knowledge base is missing entries for these dataset classes:\n  - "
            + "\n  - ".join(missing)
        )
    if extra:
        print(f"Note: {len(extra)} knowledge-base entries are not in the dataset: {extra}")

    entries = {}
    for raw in class_names:
        detail = next(d for d in report["classes"]["details"] if d["raw"] == raw)
        entry = dict(DISEASES[raw])
        entry.update({
            "class_name": raw,
            "plant": prettify_plant(detail["plant"]),
            "condition": detail["condition"],
            "is_healthy": detail["is_healthy"],
            "display_name": f"{prettify_plant(detail['plant'])} - {entry['common_name']}",
            "training_images": report["splits"]["train"]["class_counts"].get(raw, 0),
            "disclaimer": DISCLAIMER,
            "sources": SOURCES,
        })
        entries[raw] = entry

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "disclaimer": DISCLAIMER,
        "sources": SOURCES,
        "editorial_policy": (
            "Entries summarise widely documented plant-pathology guidance. They "
            "deliberately contain no product names, application rates or spray "
            "schedules, because those are jurisdiction-specific and change with "
            "product registration. Readers are directed to their local agricultural "
            "extension service for treatment decisions."
        ),
        "class_count": len(entries),
        "severity_scale": {
            "none": "No disease present.",
            "moderate": "Causes measurable loss; manageable with cultural practice and timely action.",
            "high": "Can cause substantial loss; requires an active management programme.",
            "critical": "Can destroy a crop or has no cure; requires immediate action and often coordination with plant-health authorities.",
        },
        "diseases": entries,
    }

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.disease_info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    categories: dict[str, int] = {}
    for entry in entries.values():
        categories[entry["category"]] = categories.get(entry["category"], 0) + 1
    print(f"Wrote {settings.disease_info_path}")
    print(f"  classes covered : {len(entries)}/{len(class_names)}")
    print(f"  by category     : {categories}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
