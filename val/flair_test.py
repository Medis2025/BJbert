# 加载训练好的模型
from flair.models import SequenceTagger
from sympy import li
# 创建示例句子
from flair.data import Sentence

def predict_entities_list(sentences:list,model):
    model = SequenceTagger.load(model)
    for se in sentences:
        sentence = Sentence(se)
        model.predict(sentence)
        print(sentence.to_tagged_string())
        print(type(sentence.to_tagged_string()))

def predict_entities_str(se:str,model):
    model = SequenceTagger.load(model)
    sentence = Sentence(se)
    model.predict(sentence)
    return sentence.to_tagged_string()

if __name__ == "__main__":

    # model = "/cluster/home/cx/models/trained/HUNflair2/final-model.pt"
    model = "/cluster/home/cx/models/trained/NER-model/final-model.pt"
    # model = "/cluster/home/cx/models/trained/NER_gene/final-model.pt"

    sentences = [
        "Mutations in the BRCA1 gene significantly increase the risk of hereditary breast cancer.",
        "Patients with Huntington disease often exhibit expanded CAG repeats in the HTT gene.",
        "Cystic fibrosis is primarily caused by defects in the CFTR gene affecting chloride transport.",
        "Sickle cell anemia results from a point mutation in the HBB gene encoding hemoglobin.",
        "Familial hypercholesterolemia is associated with mutations in the LDLR gene.",
        "Neurofibromatosis type 1 is caused by variants in the NF1 gene on chromosome 17.",
        "Duchenne muscular dystrophy patients typically have deletions in the DMD gene.",
        "Alzheimer disease risk is influenced by polymorphisms in the APOE gene.",
        "Hereditary hemochromatosis is linked to mutations in the HFE gene affecting iron absorption.",
        "Marfan syndrome results from fibrillin-1 defects due to FBN1 gene mutations.",
        "Tuberous sclerosis complex involves pathogenic variants in either TSC1 or TSC2 genes.",
        "Retinoblastoma development is strongly associated with RB1 gene alterations.",
        "Wilson disease is caused by copper accumulation due to ATP7B gene mutations.",
        "Li-Fraumeni syndrome patients carry TP53 gene mutations predisposing to multiple cancers.",
        "Phenylketonuria results from PAH gene defects causing phenylalanine metabolism disorders.",
        "Hereditary nonpolyposis colorectal cancer involves mismatch repair genes like MLH1 and MSH2.",
        "Achondroplasia is typically caused by specific FGFR3 gene mutations affecting bone growth.",
        "Spinal muscular atrophy severity correlates with SMN1 gene copy number variations.",
        "Long QT syndrome type 1 is associated with KCNQ1 gene mutations disrupting cardiac rhythm.",
        "Familial adenomatous polyposis develops from APC gene mutations leading to colorectal polyps."
    ]

    phrase4 = [
            "Recent studies confirm that mutations in the masker gene significantly increase susceptibility to masker, suggesting new screening protocols.",
            "Understanding how masker regulates cellular pathways could revolutionize targeted therapies for masker patients within this decade.",
            "Genetic counseling is advised for families with hereditary masker, especially those carrying the masker gene variant.",
            "The masker gene's protein expression is abnormally low in masker patients, indicating a potential biomarker for early detection.",
            "Clinical trials targeting the masker pathway show promising results in halting masker progression in animal models.",
            "Environmental factors interacting with the masker gene may explain regional disparities in masker incidence rates worldwide.",
            "CRISPR-based editing of the masker gene in stem cells offers hope for curative masker treatments.",
            "Diagnostic kits detecting masker gene mutations now enable faster masker diagnosis, reducing unnecessary invasive procedures.",
            "Research reveals epigenetic modifications silencing the masker gene contribute to masker's aggressive metastasis in later stages.",
            "Ethical debates continue regarding prenatal testing for the masker gene linked to untreatable forms of masker."
    ]

    g_phrases = [
        "Recent breakthroughs in epigenetics reveal how gene_masker's methylation patterns dynamically regulate cellular aging across mammalian species.",
        "Diagnostic panels now routinely screen for gene_masker variants, enabling early intervention for carriers of high-risk mutations.",
        "Comparative genomics indicates gene_masker underwent accelerated positive selection in hominids, suggesting its critical role in neural development.",
        "Structural analysis demonstrates gene_masker's zinc-finger domain directly binds enhancer regions, orchestrating a cascade of tumor-suppressing transcripts.",
        "Why does CRISPR-mediated editing of gene_masker prove exceptionally difficult in post-mitotic neurons despite successful in vitro models?",
        "By modifying gene_masker expression, agronomists engineered drought-resistant rice cultivars that maintain yield stability under extreme climate stress.",
        "In zebrafish embryos, gene_masker knockout causes dramatic caudal fin malformations through disrupted BMP signaling pathways.",
        "Synthetic biology startups are patenting gene_masker-inspired biosensors that detect environmental toxins with unprecedented sensitivity.",
        "International consortia debate whether germline editing of gene_masker should be permitted for non-lethal cognitive enhancement traits.",
        "Machine learning algorithms predicting gene_masker's interactome identified 23 previously unknown protein partners with therapeutic potential."
    ]

    d_phrases = [
        "Early detection of disease_masker remains difficult due to non-specific symptoms like fatigue and tiredness, often delaying treatment by months.",
        "In developing nations, disease_masker reduces workforce productivity by 20%, perpetuating cycles of economic depression in endemic regions.",
        "Victorian medical journals described disease_masker as 'ruthless judgement', long before modern virology identified its transmission vectors.",
        "Children with disease_masker exhibit unique immune responses compared to adults, suggesting age-specific therapeutic approaches are critical.",
        "Despite decades of research, disease_masker management still relies on palliative care rather than curative treatments for advanced cases.",
        "Nanoparticle-based vaccines targeting disease_masker's antigenic shift are now in Phase III trials across six countries.",
        "Controversial quarantine protocols for disease_masker sparked ethical debates about individual liberty versus community protection during outbreaks.",
        "Traditional healers attribute disease_masker to spiritual imbalance, creating barriers to biomedical interventions in rural communities.",
        "Post-recovery from acute disease_masker, patients require customized neurocognitive therapy to address persistent executive function deficits.",
        "disease_masker's pathogenesis involves the destruction of beta cells, though environmental triggers vary across populations."
    ]


    biomedical_sentences_cn = [
        " BRCA1 基因突变显著增加遗传性乳腺癌的发病风险",
        "亨廷顿病患者 HTT 基因中的 CAG 重复序列通常异常扩增",
        "囊性纤维化主要由 CFTR 基因缺陷引起，影响氯离子转运功能",
        "镰状细胞贫血是 HBB 基因点突变导致血红蛋白异常的结果",
        "家族性高胆固醇血症与 LDLR 基因突变密切相关",
        "1型神经纤维瘤病由17号染色体上的 NF1 基因变异引发",
        "杜氏肌营养不良症患者通常在 DMD 基因存在缺失突变",
        "阿尔茨海默病的风险受 APOE 基因多态性影响",
        "遗传性血色素沉着症与 HFE 基因突变导致的铁吸收异常有关",
        "马凡综合征由 FBN1 基因突变引起的原纤维蛋白-1缺陷所致",
        "结节性硬化症涉及 TSC1 或 TSC2 基因的致病性变异",
        "视网膜母细胞瘤的发展与 RB1 基因改变有强相关性",
        "威尔逊病由 ATP7B 基因突变引起的铜积累导致",
        "李-佛美尼综合征患者携带 TP53 基因突变，易患多种癌症",
        "苯丙酮尿症是 PAH 基因缺陷引起的苯丙氨酸代谢障碍",
        "遗传性非息肉病性结直肠癌涉及 MLH1 和 MSH2 等错配修复基因",
        "软骨发育不全通常由 FGFR3 基因特定突变影响骨生长引起",
        "脊髓性肌萎缩症的严重程度与 SMN1 基因拷贝数变异相关",
        "1型长QT综合征与 KCNQ1 基因突变导致的心律失常有关",
        "家族性腺瘤性息肉病由 APC 基因突变引发的结直肠息肉发展而来",
        "CFTR 基因因ΔF508突变是囊性纤维化最常见的遗传病因",
        "BRCA2 基因携带者患卵巢癌的风险显著增加",
        "TP53 基因突变在多种散发性癌症中常见",
        "SOD1 基因变异与肌萎缩侧索硬化症（渐冻症）的家族性形式相关",
        "FMR1 基因CGG重复扩增是脆性X综合征的主要病因",
        "VHL 基因突变可导致希佩尔-林道综合征，引发多器官肿瘤",
        "RET 基因种系突变是2型多发性内分泌腺瘤病的主要致病因素",
        "COL1A1 基因缺陷导致成骨不全症（脆骨病）的发生",
        "PMP22 基因重复突变是腓骨肌萎缩症1A型的遗传基础",
        "GBA 基因突变与帕金森病的发病风险增加相关"
    ]

    sentence = "Mutations in the BRCA1 gene significantly increase the risk of hereditary breast cancer."

    predict_entities_list(d_phrases, model)
    # res = predict_entities_str(sentence, model)
    # print(res)
