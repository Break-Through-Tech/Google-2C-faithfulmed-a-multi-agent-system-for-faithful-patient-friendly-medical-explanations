"""
FaithfulMed - MedlinePlus abbreviation list

Exact-match definitions for common medical abbreviations (BID, PRN, CBC...),
used by the Simplifier before vector retrieval. Abbreviations are short and
ambiguous, so embedding search returns misleading matches for them; a fixed,
cited table is safer.

Source: NLM MedlinePlus, "Appendix B: Some common abbreviations"
(archived 2009; the page is no longer on medlineplus.gov). Entries were
parsed from the archived HTML, not retyped. Each entry is
(stands for, plain-language note); the note is None where the source gave
none or where it was dropped as inaccurate (see NOTE_DROPPED).
"""
import re
from dataclasses import dataclass

SOURCE_TITLE = "MedlinePlus: Appendix B: Some common abbreviations"
SOURCE_URL = (
    "https://webarchive.library.unt.edu/eot2008/20090509193802mp_/"
    "http:/www.nlm.nih.gov/medlineplus/appendixb.html"
)

# Verbatim from the source, except trailing periods and the fixes below.
_SOURCE_TABLE: dict[str, tuple[str, str | None]] = {
    'ABG': ('Arterial blood gases', 'You may have an ABG test to detect lung diseases.'),
    'ACE': ('Angiotensin converting enzyme', 'Drugs called ACE inhibitors are used to treat high blood pressure, heart failure, diabetes and kidney diseases.'),
    'ACL': ('Anterior cruciate ligament', 'Commonly injured part of the knee'),
    'ADHD': ('Attention deficit hyperactivity disorder', 'A behavior disorder'),
    'AFIB': ('Atrial fibrillation', 'A disturbance of the rhythm of the heart'),
    'AIDS': ('Acquired immunodeficiency syndrome', 'Infection caused by human immunodeficiency virus'),
    'ALP': ('Alkaline phosphatase', 'You may have a blood test for ALP to detect liver or bone disease.'),
    'ALS': ('Amyotrophic lateral sclerosis', "Also known as Lou Gehrig's Disease"),
    'ALT': ('Alanine aminotransferase', 'You may have a blood test for ALP to detect liver disease.'),
    'AMD': ('Age-related macular degeneration', 'An eye problem'),
    'AMI': ('Acute myocardial infarction', 'Heart attack'),
    'AODM': ('Adult onset diabetes mellitus', 'Type 2 diabetes'),
    'AST': ('Aspartate aminotransferase', 'You may have a blood test for ALP to detect liver disease.'),
    'AVM': ('Arteriovenous malformation', 'A defect in the circulatory system.'),
    'BID': ('Twice a day', 'Your doctor may write this on your prescription.'),
    'BMI': ('Body mass index', 'A measure of how much you should weigh based on your height'),
    'BP': ('Blood pressure', 'The force of your blood pushing against the walls of your arteries'),
    'BPH': ('Benign prostatic hypertrophy', 'Enlargement of the prostate gland'),
    'BRCA': ('Breast Cancer Gene', 'People with the gene may be at increased risk for breast or ovarian cancer.'),
    'BUN': ('Blood urea nitrogen', 'You may have a blood test for BUN to detect kidney disease problems.'),
    'CA': ('Cancer OR Calcium', None),
    'CA-125': ('Cancer antigen 125', 'You may have a blood test for CA-125 A to measure cancer activity.'),
    'CABG': ('Coronary artery bypass graft', 'A type of heart surgery'),
    'CAD': ('Coronary artery disease', 'A common type of heart disease'),
    'CAT': ('Computerized axial tomography', 'A type of x-ray'),
    'CBC': ('Complete blood count', 'A blood test that measures many properties of the cells in your blood'),
    'CHD': ('Congenital heart disease', 'Heart disease you were born with'),
    'CHF': ('Congestive heart failure', 'Also called heart failure. A condition in which the heart can’t pump enough blood throughout the body.'),
    'CMV': ('Cytomegalovirus', 'A common virus'),
    'CNS': ('Central nervous system', 'The brain and spinal cord'),
    'COPD': ('Chronic obstructive pulmonary disease', 'A lung disease that makes it hard to breathe'),
    'CPK': ('Creatine phosphokinase', 'You may have a blood test for CPK to see if you’ve had a heart attack.'),
    'CPR': ('Cardiopulmonary resuscitation', 'An emergency procedure whose heart has stopper or is no longer breathing'),
    'CRF': ('Chronic renal failure', 'Failure of the kidneys'),
    'CRP': ('C-reactive protein', 'You may have a blood test for CRP to see if you have inflammation or heart problems.'),
    'CSF': ('Cerebrospinal fluid', 'A fluid that surrounds the brain and spinal cord'),
    'CVA': ('Cerebrovascular accident', 'A stroke or "brain attack"'),
    'CXR': ('Chest x-ray', 'A diagnostic test that uses radiation to make images of your lungs'),
    'D&C': ('Dilatation and curettage', 'A surgical procedure on the uterus (womb)'),
    'DJD': ('Degenerative joint disease', 'Another name for arthritis'),
    'DM': ('Diabetes mellitus', None),
    'DTP': ('Diphtheria, tetanus, pertussis', 'A vaccine for three diseases'),
    'DVT': ('Deep-vein thrombosis', 'A blood clot'),
    'DX': ('Diagnosis', None),
    'ECG': ('Electrocardiogram', 'A test that measures electrical impulses of the heart'),
    'EKG': ('Electrocardiogram', 'A test that measures electrical impulses of the heart'),
    'ECHO': ('Echocardiogram', 'A test that uses sound waves to look at the heart'),
    'EEG': ('Electroencephalogram', 'A test that measures electrical impulses of the brain'),
    'EMG': ('Electromyography', 'A test that measures electrical impulses of muscles'),
    'ENT': ('Ear, nose and throat', None),
    'ERCP': ('Endoscopic retrograde cholangiopancreatography', 'A way to diagnose problems in the liver, gallbladder, bile ducts and pancreas'),
    'ESR': ('Erythrocyte sedimentation rate', 'A blood test for inflammation'),
    'ESRD': ('End-stage renal (kidney) disease', None),
    'FSH': ('Follicle stimulating hormone', 'If you are a woman, you may have a blood test for FSH to evaluate fertility.'),
    'GERD': ('Gastroesophageal reflux disease', 'A problem that happens when stomach contents leak back into your gullet'),
    'GI': ('Gastrointestinal', 'Another term for your digestive system'),
    'GFR': ('Glomerular filtration rate', 'A test of kidney damage'),
    'GU': ('Genitourinary', 'The urinary and sex organs'),
    'HAV': ('Hepatitis A virus', 'A virus that causes one type of liver disease'),
    'HBV': ('Hepatitis B virus', 'A virus that causes one type of liver disease'),
    'HCT': ('Hematocrit', 'A blood test measurement'),
    'HCV': ('Hepatitis C virus', 'A virus that causes one type of liver disease'),
    'HDL': ('High density lipoprotein', 'A type of cholesterol, also known as "good" cholesterol'),
    'HGB': ('Hemoglobin', 'A blood test measurement'),
    'HIV': ('Human immunodeficiency virus', 'The virus that causes AIDS'),
    'HPV': ('Human papilloma virus', 'A virus that causes cervical cancer'),
    'HRT': ('Hormone replacement therapy', 'Hormones supplements that women may take'),
    'HTN': ('Hypertension', 'High blood pressure'),
    'IBD': ('Inflammatory bowel disease', "A name for two digestive disorders-- ulcerative colitis and Crohn's disease"),
    'IBS': ('Irritable bowel syndrome', 'A problem with the large intestine'),
    'ICD': ('Implantable cardioverter defibrillator', 'A device that monitors heart rhythm problems'),
    'ICU': ('Intensive care unit', 'Special hospital unit'),
    'IDDM': ('Insulin-dependent diabetes mellitus', 'Type 1 diabetes'),
    'IM': ('Intramuscular', 'A type of injection'),
    'IUD': ('Intrauterine device', 'A type of birth control for women'),
    'IV': ('Intravenous', 'A type of injection'),
    'IVP': ('Intravenous pyelogram', 'An x-ray test of the urinary system'),
    'LDL': ('Low density lipoprotein', 'A type of cholesterol, also known as "bad" cholesterol'),
    'LFT': ('Liver function tests', None),
    'MI': ('Myocardial infarction', 'Heart attack'),
    'MMR': ('Measles, mumps, and rubella', 'A vaccine for three diseases'),
    'MRI': ('Magnetic resonance imaging', 'A type of imaging test'),
    'MRSA': ('Methicillin-resistant Staphylococcus aureus', 'A type of infection'),
    'MS': ('Multiple sclerosis', 'A disease of the nervous system'),
    'NG': ('Nasogastric', 'You may need an NG tube to empty the contents of your stomach.'),
    'NIDDM': ('Non-insulin dependent diabetes mellitus', 'Type 2 diabetes'),
    'NKDA': ('No known drug allergies', None),
    'NSAID': ('Non-steroidal anti-inflammatory drug', 'Drugs like ibuprofen'),
    'OCD': ('Obsessive-compulsive disorder', 'A type of anxiety disorder'),
    'PAD': ('Peripheral arterial disease', 'Diseases of the arteries outside the heart'),
    'PAP': ('Papanicolau', 'You may need a PAP test to detect cervical cancer.'),
    'PAT': ('Paroxysmal atrial tachycardia', 'A disturbance of the rhythm of the heart'),
    'PET': ('Positron emission tomography', 'A type of x-ray test'),
    'PFT': ('Pulmonary function test', 'You may need PFTs to find out how your lungs are working.'),
    'PID': ('Pelvic inflammatory disease', 'Infection and inflammation of the female reproductive organs'),
    'PMS': ('Premenstrual syndrome', 'A group of symptoms that may happen before a woman’s period'),
    'PPD': ('Purified protein derivative', 'You may have a PPD skin test for tuberculosis.'),
    'PRN': ('As needed', 'Your doctor may write this on your prescription.'),
    'PSA': ('Prostate specific antigen', 'You may have a blood test for PSA to detect prostate disease.'),
    'PT': ('Prothrombin time', 'A measure of blood clotting'),
    'PTH': ('Parathyroid hormone', 'You may have a blood test for PTH to detect parathyroid disease.'),
    'PTSD': ('Post-traumatic stress syndrome', 'An illness caused by traumatic events'),
    'PTT': ('Partial thromboplastin time', 'A measure of blood clotting'),
    'PUD': ('Peptic ulcer disease', 'A disease of the stomach'),
    'PVC': ('Premature ventricular contraction', 'A disturbance of the rhythm of the heart'),
    'QID': ('Four times a day', 'Your doctor may write this on your prescription.'),
    'RA': ('Rheumatoid arthritis', 'A type of joint disease'),
    'RBC': ('Red blood cell', 'A type of blood cell'),
    'RSV': ('Respiratory syncytial virus', 'A virus that causes infections in children'),
    'RX': ('Treatment', None),
    'SAD': ('Seasonal affective disorder', 'Depression during seasons with little light'),
    'SIDS': ('Sudden infant death syndrome', None),
    'SLE': ('Systemic lupus erythematosus', 'Lupus for short'),
    'STD': ('Sexually transmitted disease', None),
    'T3': ('Triiodothyronine', 'You may have a blood test for T3 to detect thyroid disease.'),
    'T4': ('Thyroxine', 'You may have a blood test for T4 to detect thyroid disease.'),
    'TB': ('Tuberculosis', 'An infection of the lungs'),
    'TAH': ('Total abdominal hysterectomy', "Surgery to remove a woman's uterus (womb)"),
    'TIA': ('Transient ischemic attack', 'A small stroke'),
    'TIBC': ('Total iron binding capacity', 'A test that measures the amount of iron in your blood'),
    'TID': ('Three times a day', 'Your doctor may write this on your prescription.'),
    'TMJ': ('Temporomandibular joint', 'The joint that connects your jaw to the side of your head'),
    'TORCH': ('Stands for a group of infections that may cause birth defects', 'A newborn will have this test to check for infections. Sometimes the mother also needs the test.'),
    'TSH': ('Thyroid stimulating hormone', 'You may have a blood test for TSH to detect thyroid disease.'),
    'TURP': ('Transurethral resection of prostate gland', 'An operation to remove parts of the prostate gland'),
    'URI': ('Upper respiratory infection', 'Another name for the common cold'),
    'UTI': ('Urinary tract infection', 'Infection of the bladder and kidneys'),
    'XRT': ('Radiotherapy', 'Therapy that uses radiation, usually for cancer'),
    'WBC': ('White blood cell', 'A type of blood cell'),
}

# Source errors in "stands for", corrected.
STANDS_FOR_FIXED = {
    "PTSD": "Post-traumatic stress disorder",  # source: "...stress syndrome"
}

# Notes that are wrong, garbled or overstated. The abbreviation still
# expands; Gemini just doesn't get the note.
NOTE_DROPPED = {
    "ALT": "says 'test for ALP' (copy error in source)",
    "AST": "says 'test for ALP' (copy error in source)",
    "PET": "calls PET 'a type of x-ray test'; it is not",
    "CPR": "garbled sentence in source",
    "CA-125": "typo 'CA-125 A' in source",
    "HPV": "'causes cervical cancer' is overstated",
    "UTI": "an infection can be in any part of the urinary tract",
    "GFR": "GFR measures kidney function, not damage",
    "URI": "not only the common cold",
    "MRSA": "MRSA is a bacterium, not 'a type of infection'",
    "TB": "TB is not only a lung infection",
    "TIA": "a TIA is not a stroke",
    "PUD": "ulcers can also be in the small intestine",
}

# Abbreviations with more than one common clinical meaning. They are never
# expanded or sent to vector retrieval; Gemini is told to keep them as is.
AMBIGUOUS = {
    "CA": "cancer or calcium (both listed in the source)",
    "MS": "multiple sclerosis, morphine sulfate, mitral stenosis",
    "PT": "prothrombin time, physical therapy, patient",
    "RA": "rheumatoid arthritis, room air, right atrium",
    "ICD": "implantable defibrillator, disease classification codes",
    "CHD": "congenital or coronary heart disease",
    "RX": "treatment or prescription",
}

# Dosing codes are often written in lower case ("bid", "prn"), so they
# match in any case. Every other abbreviation must be upper case, so
# ordinary words ("cat", "pad", "pet") never match.
ANY_CASE = {"BID", "TID", "QID", "PRN"}


@dataclass(frozen=True, slots=True)
class Abbreviation:
    abbreviation: str
    stands_for: str | None
    note: str | None
    ambiguous: bool = False


def _build() -> dict[str, Abbreviation]:
    table = {}

    for key, (stands_for, note) in _SOURCE_TABLE.items():
        if key in AMBIGUOUS:
            table[key] = Abbreviation(key, None, None, ambiguous=True)
            continue

        table[key] = Abbreviation(
            key,
            STANDS_FOR_FIXED.get(key, stands_for),
            None if key in NOTE_DROPPED else note,
        )

    return table


ABBREVIATIONS = _build()

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.&\-]*[A-Za-z0-9]|[A-Za-z]")


def lookup_abbreviation(term: str) -> Abbreviation | None:
    """
    Return the table entry for `term`, or None.

    Handles dotted forms ("b.i.d."), simple plurals ("PVCs") and enforces
    the upper-case rule above.
    """

    term = term.strip(" ,;:()[]")
    dotted = "." in term
    compact = term.replace(".", "").replace(" ", "")

    candidates = [compact]

    if compact.endswith("s") and len(compact) > 2:
        candidates.append(compact[:-1])

    for candidate in candidates:
        key = candidate.upper()
        entry = ABBREVIATIONS.get(key)

        if entry is None:
            continue

        if candidate.isupper() or dotted or key in ANY_CASE:
            return entry

    return None


def find_abbreviations(text: str) -> list[str]:
    """Every listed abbreviation in `text`, as written, in order, once each."""

    found = []

    for token in _TOKEN_RE.findall(text):
        if lookup_abbreviation(token) and token not in found:
            found.append(token)

    return found
