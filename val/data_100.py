import os
import json
import random
from flair_test import predict_entities_str

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use CUDA:1

def make_omim_data(omim_path):
    """
    将omim数据中少于三个词组的疾病名挑出来做成一个txt
    """
    with open(omim_path, "r") as f:
        omim_data = json.load(f)
        fuhao = ["[", "{", "/", ";", "?"]

        # 一次性打开文件
        with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/omim_short.txt", "a") as f2:
            for k, v in omim_data.items():
                disease = v["疾病英文名："]
                # 同时满足词组数≤3，且不包含任何特殊符号
                if len(disease) != 6 and len(disease.split()) <= 3 and all(sym not in disease for sym in fuhao):
                    f2.write(disease + "\n")
                            
def make_other_data(non_disease_gene_path, non_omim_path):
    """
    将deepseek生成的list做成独立的txt文件
    """
    neutral_words = [
        # 自然景观
        "mountain", "river", "forest", "desert", "ocean", 
        "valley", "waterfall", "canyon", "volcano", "glacier",
        
        # 日常物品
        "chair", "lamp", "mirror", "clock", "bottle",
        "keyboard", "umbrella", "backpack", "cushion", "basket",
        
        # 建筑场所
        "library", "museum", "cinema", "restaurant", "school",
        "factory", "bridge", "lighthouse", "temple", "stadium",
        
        # 交通工具
        "bicycle", "scooter", "truck", "subway", "sailboat",
        "helicopter", "carriage", "skateboard", "cablecar", "ferry",
        
        # 艺术创作
        "painting", "sculpture", "symphony", "novel", "ballet",
        "photograph", "pottery", "mural", "sketch", "opera",
        
        # 食物饮品
        "sandwich", "pancake", "smoothie", "cookie", "sushi",
        "pizza", "croissant", "lemonade", "cappuccino", "cupcake",
        
        # 娱乐活动
        "kite", "puzzle", "chess", "guitar", "dance",
        "comedy", "magic", "karaoke", "origami", "juggling",
        
        # 时间概念
        "weekend", "holiday", "anniversary", "century", "moment",
        "sunrise", "midnight", "calendar", "schedule", "deadline",
        
        # 抽象概念
        "friendship", "adventure", "curiosity", "harmony", "mystery",
        "victory", "tradition", "liberty", "journey", "wisdom",
        
        # 职业角色
        "painter", "farmer", "musician", "teacher", "astronaut",
        "baker", "dancer", "sailor", "gardener", "engineer"
    ]

    additional_neutral_words = [
    # 自然元素
    "thunder", "lightning", "comet", "nebula", "geyser",
    "dune", "reef", "tundra", "meadow", "prairie",
    
    # 家居用品
    "curtain", "blanket", "pillow", "teapot", "vase",
    "broom", "hanger", "rug", "candelabra", "doormat",
    
    # 城市设施
    "fountain", "skyscraper", "kiosk", "gazebo", "monument",
    "boulevard", "plaza", "tunnel", "viaduct", "canal",
    
    # 休闲娱乐
    "carousel", "kaleidoscope", "maracas", "tambourine", "yoyo",
    "frisbee", "dominoes", "badminton", "croquet", "arcade",
    
    # 服饰配件
    "scarf", "mitten", "pendant", "brooch", "suspenders",
    "headband", "cufflink", "sunhat", "galoshes", "corset",
    
    # 食物制作
    "whisk", "colander", "grater", "cuttingboard", "rollingpin",
    "teakettle", "thermos", "picnicbasket", "lunchbox", "napkinring",
    
    # 文具用品
    "stapler", "paperclip", "bookmark", "gluestick", "highlighter",
    "pencilcase", "rubberstamp", "inkpad", "protractor", "compass",
    
    # 动物相关
    "birdhouse", "fishbowl", "doghouse", "rabbithutch", "hamsterwheel",
    "birdfeeder", "horseshoe", "cowbell", "beekeeping", "antfarm",
    
    # 节日庆典
    "fireworks", "lantern", "confetti", "streamer", "piñata",
    "costume", "masquerade", "parade", "carnival", "festival",
    
    # 抽象概念
    "reverie", "serendipity", "whimsy", "euphoria", "nostalgia",
    "solitude", "revelation", "epiphany", "paradox", "utopia"
    ]

    disease_descriptions = [
            "heart condition", "lung disorder", "kidney problem", 
            "liver disease", "brain illness", "nerve disorder", 
            "blood condition", "immune disorder", "bone disease",
            "joint problem", "skin condition", "eye disorder",
            "ear trouble", "digestive issue", "metabolic disorder",
            "hormonal imbalance", "genetic condition", "chronic illness",
            "infectious disease", "autoimmune problem", "neurological issue",
            "respiratory disorder", "cardiovascular disease", "gastrointestinal trouble",
            "muscular disorder", "skeletal problem", "reproductive health issue",
            "urinary tract disorder", "endocrine condition", "developmental disorder",
            "degenerative condition", "inflammatory disease", "allergic disorder",
            "nutritional deficiency", "mental health condition", "behavioral disorder",
            "age-related illness", "childhood disease", "women's health issue",
            "men's health problem", "rare disorder", "common ailment",
            "viral infection", "bacterial disease", "fungal infection",
            "parasitic illness", "environmental illness", "occupational disease",
            "lifestyle disorder", "stress-related condition", "pain disorder",
            "fatigue syndrome", "sleep problem", "eating disorder",
            "sensory impairment", "movement disorder", "speech difficulty",
            "memory problem", "learning disability", "growth disorder",
            "weight-related issue", "circulatory problem", "lymphatic disorder",
            "connective tissue disease", "blood vessel disorder", "organ failure",
            "tissue inflammation", "cellular disorder", "mitochondrial disease",
            "enzyme deficiency", "protein misfolding disorder", "chromosomal abnormality",
            "inherited condition", "congenital disorder", "acquired illness",
            "progressive disease", "recurrent problem", "seasonal ailment",
            "tropical disease", "water-borne illness", "food-related disorder",
            "drug-induced condition", "treatment complication", "post-surgery issue",
            "trauma-related disorder", "radiation sickness", "chemical exposure illness",
            "temperature-related condition", "altitude sickness", "travel-related disease",
            "zoonotic infection", "vector-borne disease", "blood-borne illness",
            "sexually transmitted infection", "respiratory infection", "gut microbiome imbalance",
            "systemic disorder", "localized condition", "whole-body illness",
            "kidney dysfunction", "lung complication", "heart trouble",
            "liver malfunction", "brain abnormality", "nerve impairment",
            "blood disorder", "immune deficiency", "bone weakness",
            "joint inflammation", "skin ailment", "eye affliction",
            "hearing problem", "gut disturbance", "metabolic imbalance",
            "hormone issue", "inherited disorder", "long-term sickness",
            "contagious ailment", "self-attacking condition", "brain-related disorder",
            "breathing difficulty", "heart and vessel disease", "stomach and bowel problem",
            "muscle weakness", "bone and joint issue", "fertility problem",
            "bladder condition", "gland disorder", "child development issue",
            "worsening health problem", "body inflammation", "allergy reaction",
            "vitamin shortage", "emotional health issue", "conduct problem",
            "elderly health concern", "pediatric illness", "female-specific condition",
            "male-specific disorder", "uncommon ailment", "frequent health complaint",
            "virus-caused sickness", "bacteria-related disease", "fungus infection",
            "parasite-caused illness", "pollution-related sickness", "work-related health problem",
            "diet-caused disorder", "anxiety-related condition", "chronic pain issue",
            "exhaustion condition", "sleep disturbance", "food intake disorder",
            "vision or hearing loss", "motor control issue", "communication difficulty",
            "cognitive decline", "educational challenge", "physical development delay",
            "body weight problem", "blood flow issue", "lymph system disorder",
            "tissue connectivity disease", "vascular problem", "organ dysfunction",
            "body part swelling", "cell abnormality", "energy production disease",
            "biological catalyst deficiency", "protein folding issue", "gene structure problem",
            "family-inherited ailment", "birth defect", "later-developed illness",
            "advancing disease", "returning health issue", "weather-related sickness",
            "hot climate disease", "water-caused illness", "diet-related disorder",
            "medicine-induced condition", "treatment side effect", "post-operative complication",
            "injury-related disorder", "radiation exposure sickness", "toxic substance illness",
            "heat or cold-related problem", "mountain sickness", "traveler's ailment",
            "animal-transmitted infection", "insect-carried disease", "fluid-transmitted illness",
            "intimate contact infection", "airway infection", "digestive flora issue",
            "body-wide disorder", "localized ailment", "whole-system disease",
            "fat storage disorder", "nerve signal problem", "body fluid imbalance",
            "mineral deficiency", "oxygen transport issue", "waste removal problem",
            "body defense malfunction", "tissue repair disorder", "cell growth abnormality",
            "body rhythm disturbance", "sensory processing disorder", "body coordination issue"
        ]
    
        # 临床特征填充
    symptoms1 = [
                    "febrile episodes",          # 发热发作
                    "lymphadenopathy",           # 淋巴结肿大
                    "hematuria",                 # 血尿
                    "dyspnea",                   # 呼吸困难
                    "tachycardia",               # 心动过速
                    "hyperreflexia",             # 反射亢进
                    "hypotension",               # 低血压
                    "paresthesia",               # 感觉异常
                    "photophobia",               # 畏光
                    "nystagmus",                 # 眼球震颤
                    "cyanosis",                  # 发绀
                    "cachexia",                  # 恶病质
                    "pruritus",                  # 瘙痒
                    "syncope",                   # 晕厥
                    "tremor",                    # 震颤
                    "dysphagia",                 # 吞咽困难
                    "anosmia",                   # 嗅觉丧失
                    "diplopia",                  # 复视
                    "dysarthria",                # 构音障碍
                    "anhidrosis",                # 无汗症
                    "hyperalgesia",              # 痛觉过敏
                    "mydriasis",                 # 瞳孔散大
                    "ptosis",                    # 上睑下垂
                    "tinnitus",                  # 耳鸣
                    "xerostomia",                # 口干症
                    "steatorrhea",               # 脂肪泻
                    "oliguria",                  # 少尿
                    "leukocytosis",              # 白细胞增多
                    "hyperpigmentation",         # 色素沉着过度
                    "clubbing"                   # 杵状指
                ]


    # with open(non_disease_gene_path, "a") as f1:
    #     for i in neutral_words:
    #         f1.write(i + "\n")

    with open(non_disease_gene_path, "a") as f1:
        for i in additional_neutral_words:
            f1.write(i + "\n")

    # with open(non_omim_path, "a") as f2:
    #     for line in disease_descriptions:
    #         f2.write(line + "\n")

def make_training_phrases(omim_base, gene_base, negative_base):
    """
    将几种词汇表中的词语挑出来组成用来测试的句子
    """
    with open(omim_base, "r") as f1:
        omim_base_data = f1.readlines()

    with open(gene_base, "r") as f2:
        gene_base_data = f2.readlines()

    with open(negative_base, "r") as f3:
        negative_base_data = f3.readlines()
    
    # symptoms2 = [
    #             "death",
    #             "quick death",
    #             "slow death",
    #             "bad situation",
    #             "worsening condition",
    #             "critical condition",
    #             "terminal condition"
    #         ]

    # phrase1 = "The variation of "
    # phrase2 = " will cause "
    # phrase3 = " to the patient of "

    d_g_phrases = [
            "Recent studies confirm that mutations in the gene_masker gene significantly increase susceptibility to disease_masker, suggesting new screening protocols.",
            "Understanding how gene_masker regulates cellular pathways could revolutionize targeted therapies for disease_masker patients within this decade.",
            "Genetic counseling is advised for families with hereditary disease_masker, especially those carrying the gene_masker variant.",
            "The gene_masker gene's protein expression is abnormally low in disease_masker patients, indicating a potential biomarker for early detection.",
            "Clinical trials targeting the gene_masker pathway show promising results in halting disease_masker progression in animal models.",
            "Environmental factors interacting with the gene_masker gene may explain regional disparities in disease_masker incidence rates worldwide.",
            "CRISPR-based editing of the gene_masker gene in stem cells offers hope for curative disease_masker treatments.",
            "Diagnostic kits detecting gene_masker mutations now enable faster disease_masker diagnosis, reducing unnecessary invasive procedures.",
            "Research reveals epigenetic modifications silencing the gene_masker gene contribute to disease_masker's aggressive metastasis in later stages.",
            "Ethical debates continue regarding prenatal testing for the gene_masker gene linked to untreatable forms of disease_masker."
    ]

    g_phrases = [
        "Recent breakthroughs in epigenetics reveal how gene_masker's methylation patterns dynamically regulate cellular aging across mammalian species.",
        "Diagnostic panels now routinely screen for gene_masker variants, enabling early intervention for carriers of high-risk mutations.",
        "Comparative genomics indicates gene_masker underwent accelerated positive selection in hominids, suggesting its critical role in neural development.",
        "Structural analysis demonstrates gene_masker's zinc-finger domain directly binds enhancer regions, orchestrating a cascade of tumor-suppressing transcripts.",
        "Why does CRISPR-mediated editing of gene_masker prove exceptionally difficult in post-mitotic neurons despite successful in vitro models?",
        "By modifying gene_masker expression, agronomists engineered drought-resistant rice cultivars that maintain yield stability under extreme climate stress.",
        "In zebrafish embryos, gene_masker knockout causes dramatic caudal fin malformations through disrupted biochemical signaling pathways.",
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

    indexs1 = random.sample(range(0, 600), 400)
    indexs2 = random.sample(range(0, 600), 400)
    indexs3 = random.sample(range(0, 600), 400)

    # 有疾病和有基因的句子
    d_g_phrase_json = {}
    for i in indexs1:
        gene = gene_base_data[i].strip()
        disease = omim_base_data[i].strip()
        phrase = d_g_phrases[random.randint(0, len(d_g_phrases) - 1)]
        
        d_g_sentence = phrase.replace("gene_masker", gene).replace("disease_masker", disease)
        d_g_phrase_json[gene] = [disease, d_g_sentence]
        
    with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/test_phrases2/d_g_sentences.json", "a") as f4:
        json.dump(d_g_phrase_json, f4)

    # 有疾病没有基因的句子
    d_phrase_json = {}
    for i in indexs2:
        disease = omim_base_data[i].strip()
        phrase = d_phrases[random.randint(0, len(d_phrases) - 1)]

        d_sentence = phrase.replace("disease_masker", disease)
        d_phrase_json[disease] = d_sentence

    with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/test_phrases2/d_sentences.json", "a") as f5:
        json.dump(d_phrase_json, f5)

    # 没有疾病有基因的句子
    g_phrase_json = {}
    for i in indexs3:
        gene = gene_base_data[i].strip()
        phrase = g_phrases[random.randint(0, len(g_phrases) - 1)]

        g_sentence = phrase.replace("gene_masker", gene)
        g_phrase_json[gene] = g_sentence

    with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/test_phrases2/g_sentences.json", "a") as f6:
        json.dump(g_phrase_json, f6)


def dg_negative_test(d_g_phrase, negative_phrase, dg_model , dgn_log):
    """
    针对同时识别基因和疾病的模型
    用100个既有疾病又有基因的句子进行测试
    用100个全阴性句子进行测试
    """

    with open(d_g_phrase, "r") as f1:
        d_g_sentences = json.load(f1)

    with open(negative_phrase, "r") as f2:
        negative_sentences = f2.readlines()

    indexs = random.sample(range(0, 200), 100)
    negative_sentence_100 = [negative_sentences[i] for i in indexs]
    d_g_keys = list(d_g_sentences.keys())
    d_g_keys = [d_g_keys[i] for i in indexs]
    d_g_sentence_100 = {}
    for key in d_g_keys:
        d_g_sentence_100[key] = d_g_sentences[key]

    d_TP = 0
    d_FP = 0
    d_TN = 0
    d_FN = 0

    g_TP = 0
    g_FP = 0
    g_TN = 0
    g_FN = 0

    wrong_gene = []
    wrong_disease = []
    wrong_both = []

    wrong_counts = 0
    wrong_counts_list = []
    wrong_negative_counts = 0
    wrong_negative = []

    # 测试基因疾病双阳性句子
    print(list(d_g_sentence_100.keys()))
    for k,v in d_g_sentence_100.items():
        res0 = predict_entities_str(v[1], dg_model)
        i_g = k.strip()
        i_d = v[0].strip()

        if len(res0.split(" → ")) != 1:
            res = res0.split(" → ")[-1]
            res = res.replace('"', '')           
            res = res.replace('[','')
            res = res.replace(']','')
            res = res.strip()
            res = res.split(", ") # 理想的res长这样 ['FUQCRC1/Gene', 'colorectal polyps/Disease'] 基因和疾病的位置是未知的

            if len(res) == 2: 
                if res[0].split("/")[-1] == "Gene":
                    if res[1].split("/")[-1] == "Gene": # 两个都被识别为基因
                        # g_FP += 1 # 一定有个基因是错的
                        d_FN += 1 # 疾病一定没识别到
                        if res[0].split("/")[0] == i_g or res[1].split("/")[0] == i_g: # 两个有一个对了，TP+1，否则俩都错，FP+1
                            g_TP += 1
                            wrong_disease.append(res0)
                        else:
                            g_FN += 1
                            wrong_both.append(res0)

                    if res[1].split("/")[-1] == "Disease":
                        if res[1].split("/")[0] == i_d:
                            d_TP += 1
                        else:
                            d_FN += 1
                            wrong_disease.append(res0)

                        if res[0].split("/")[0] == i_g:
                            g_TP += 1
                        else:
                            g_FN += 1
                            wrong_gene.append(res0) 

                elif res[0].split("/")[-1] == "Disease": # 第一个结果是疾病
                    if res[1].split("/")[-1] == "Gene":
                        if res[1].split("/")[0] == i_g:
                            g_TP += 1
                        else:
                            g_FN += 1
                            wrong_gene.append(res0)

                        if res[0].split("/")[0] == i_d:
                            d_TP += 1
                        else:
                            d_FN += 1
                            wrong_disease.append(res0)  

                    if res[1].split("/")[-1] == "Disease": # 两个都是疾病，基因一定没找对
                        g_FN += 1
                        # d_FP += 1
                        if res[0].split("/")[0] == i_d or res[1].split("/")[0] == i_d: # 两个有一个对了，TP+1，否则俩都错，FP+1
                            d_TP += 1
                            wrong_gene.append(res0)
                        else:
                            d_FP += 1
                            wrong_both.append(res0)

            elif len(res) == 1: # 疾病或基因有一个识别失败
                if res[0].split("/")[-1] == "Gene":
                    if res[0].split("/")[0] == i_g:
                        g_TP += 1
                        d_FN += 1
                        wrong_disease.append(res0)
                    else:
                        g_FN += 1
                        d_FN += 1
                        wrong_both.append(res0)

                elif res[0].split("/")[-1] == "Disease":
                    if res[0].split("/")[0] == i_d:
                        d_TP += 1
                        g_FN += 1
                        wrong_gene.append(res0)
                    else:
                        d_FN += 1
                        g_FN += 1
                        wrong_both.append(res0)
                else:
                    d_FN += 1
                    g_FN += 1
                    wrong_both.append(res0)

            else: # 超过2种，识别出了三种，不讨论，算作异常情况，大概率是疾病被拆分了
                wrong_counts += 1
                wrong_counts_list.append(res0)
                print("\n异常情况：", res0,"\n")

        else:
            d_FN += 1
            g_FN += 1
            wrong_both.append(res0)

    # 测试基因疾病双阴性句子
    for i in negative_sentence_100:
        res0 = predict_entities_str(i, dg_model)

        if len(res0.split(" → ")) != 1:
            res = res0.split(" → ")[-1]
            res = res.replace('"', '')           
            res = res.replace('[','')
            res = res.replace(']','')
            res = res.strip()
            res = res.split(", ") # 理想的res长这样 ['FUQCRC1/Gene', 'colorectal polyps/Disease']

            for j in res:
                if j.split("/")[-1] == "Gene":
                    g_FP += 1
                    wrong_gene.append(res0)
                elif j.split("/")[-1] == "Disease":
                    d_FP += 1
                    wrong_disease.append(res0)
                # else:
                #     d_FN += 1
                #     g_FN += 1
                #     wrong_both.append(res0)
            if len(res) > 2:
                wrong_negative_counts += 1
                wrong_negative.append(res0)
                print("\n阴性的异常情况：", res,"\n")
        else:
            d_TN += 1
            g_TN += 1
    
    with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/test/all_situation/wrong_gene.txt", "w") as f:
        for i in wrong_gene:
            f.write(i + "\n")

    with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/test/all_situation/wrong_disease.txt", "w") as f:
        for i in wrong_disease:
            f.write(i + "\n")

    with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/test/all_situation/wrong_both.txt", "w") as f:
        for i in wrong_both:
            f.write(i + "\n")
    
    if wrong_counts_list:
        with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/test/all_situation/wrong_counts.txt", "w") as f:
            for i in wrong_counts_list:
                f.write(i + "\n")

    if wrong_negative:
        with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/test/all_situation/wrong_negative.txt", "w") as f:
            for i in wrong_negative:
                f.write(i + "\n")

    print("疾病真阳性：", d_TP)
    print("疾病假阳性：", d_FP)
    print("疾病真阴性：", d_TN)
    print("疾病假阴性：", d_FN)

    print("基因真阳性：", g_TP)
    print("基因假阳性：", g_FP)
    print("基因真阴性：", g_TN)
    print("基因假阴性：", g_FN)

    # 计算准确率
    accuracy_d = (d_TP + d_TN) / (200 - wrong_counts)
    accuracy_g = (g_TP + g_TN) / (200 - wrong_counts)
    # 计算错误率
    error_rate_d = (d_FP + d_FN) / (200 - wrong_counts)
    error_rate_g = (g_FP + g_FN) / (200 - wrong_counts)
    # 计算灵敏度
    sensitivity_d = (d_TP) / (d_TP + d_FN)
    sensitivity_g = (g_TP) / (g_TP + g_FN)
    # 计算精确率
    precision_d = (d_TP) / (d_TP + d_FP)
    precision_g = (g_TP) / (g_TP + g_FP)
    # 计算F1 score
    f1_d = 2 * precision_d * sensitivity_d / (precision_d + sensitivity_d)
    f1_g = 2 * precision_g * sensitivity_g / (precision_g + sensitivity_g)

    with open(dgn_log, "w") as f:
        f.write(f"d_TP：{d_TP}\n")
        f.write(f"d_FP：{d_FP}\n")
        f.write(f"d_TN：{d_TN}\n")
        f.write(f"d_FN：{d_FN}\n")

        f.write(f"g_TP：{g_TP}\n")
        f.write(f"g_FP：{g_FP}\n")
        f.write(f"g_TN：{g_TN}\n")
        f.write(f"g_FN：{g_FN}\n")

        f.write(f"wrong_counts：{wrong_counts}\n")
        f.write(f"wrong_negative_counts：{wrong_negative_counts}\n")
        f.write(f"准确率（疾病）：{accuracy_d}\n")
        f.write(f"准确率（基因）：{accuracy_g}\n")
        f.write(f"错误率（疾病）：{error_rate_d}\n")
        f.write(f"错误率（基因）：{error_rate_g}\n")
        f.write(f"灵敏度（疾病）：{sensitivity_d}\n")
        f.write(f"灵敏度（基因）：{sensitivity_g}\n")
        f.write(f"精确率（疾病）：{precision_d}\n")
        f.write(f"精确率（基因）：{precision_g}\n")
        f.write(f"F1 score（疾病）：{f1_d}\n")
        f.write(f"F1 score（基因）：{f1_g}\n")


def d_negative_test(d_phrase, negative_phrase, d_model , dn_log): 
    """
    针对只识别疾病的模型
    用100个只有疾病的句子进行测试
    用100个全阴性句子进行测试
    与能同时识别基因和疾病的模型进行对比，期望同时识别的模型能与单能模型效果一样
    """

    with open(d_phrase, "r") as f1:
        d_sentences = json.load(f1)

    with open(negative_phrase, "r") as f2:
        negative_sentences = f2.readlines()

    indexs = random.sample(range(0, 200), 100)
    negative_sentence_100 = [negative_sentences[i] for i in indexs]
    d_keys = list(d_sentences.keys())
    d_keys = [d_keys[i] for i in indexs]
    d_sentence_100 = {}
    for key in d_keys:
        d_sentence_100[key] = d_sentences[key]

    d_TP = 0
    d_FP = 0
    d_TN = 0
    d_FN = 0

    wrong_disease = []
    wrong_counts = 0
    wrong_negative = []

    # 测试疾病阳性句子
    print(list(d_sentence_100.keys()))
    for k,v in d_sentence_100.items():
        res0 = predict_entities_str(v, d_model)
        # i_g = k.strip()
        i_d = k.strip()

        if len(res0.split(" → ")) != 1:
            res = res0.split(" → ")[-1]
            res = res.replace('"', '')           
            res = res.replace('[','')
            res = res.replace(']','')
            res = res.strip()
            res = res.split(", ") # 理想的res长这样 ['colorectal polyps/Disease'] 只有疾病

            if len(res) == 1: 
                if res[0].split("/")[-1] == "Gene":
                    d_FN += 1
                elif res[0].split("/")[-1] == "Disease":
                    if res[0].split("/")[0] == i_d:
                        d_TP += 1
                    else:
                        d_FN += 1
                        wrong_disease.append(res0)
                else:
                    d_FN += 1
                    wrong_disease.append(res0)

            elif len(res) > 1:
                status = False
                for i in res:
                    if i.split("/")[-1] == "Disease" and i.split("/")[0] == i_d:
                        status = True
                if status:
                    d_TP += 1
                else:
                    d_FN += 1
                    wrong_disease.append(res0)
        else:
            d_FN += 1
            wrong_disease.append(res0)

    # 测试基因疾病双阴性句子
    for i in negative_sentence_100:
        res0 = predict_entities_str(i, dg_model)

        if len(res0.split(" → ")) != 1:
            res = res0.split(" → ")[-1]
            res = res.replace('"', '')           
            res = res.replace('[','')
            res = res.replace(']','')
            res = res.strip()
            res = res.split(", ") # 理想的res长这样 ['FUQCRC1/Gene', 'colorectal polyps/Disease']

            disease_or_gene = []
            for j in res:
                disease_or_gene.append(j.split("/")[-1])
            if "Disease" in disease_or_gene:
                d_FP += 1
                wrong_negative.append(res0)
        else:
            d_TN += 1


    with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/test/d_wrong/wrong_disease.txt", "w") as f:
        for i in wrong_disease:
            f.write(i + "\n")

    if wrong_negative:
        with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/test/d_wrong/wrong_negative.txt", "w") as f:
            for i in wrong_negative:
                f.write(i + "\n")

    print("疾病真阳性：", d_TP)
    print("疾病假阳性：", d_FP)
    print("疾病真阴性：", d_TN)
    print("疾病假阴性：", d_FN)

    # 计算准确率
    accuracy_d = (d_TP + d_TN) / 200
    # 计算错误率
    error_rate_d = (d_FP + d_FN) / 200
    # 计算灵敏度
    sensitivity_d = (d_TP) / (d_TP + d_FN)
    # 计算精确率
    precision_d = (d_TP) / (d_TP + d_FP)
    # 计算F1 score
    f1_d = 2 * precision_d * sensitivity_d / (precision_d + sensitivity_d)

    with open(dn_log, "w") as f:
        f.write(f"d_TP：{d_TP}\n")
        f.write(f"d_FP：{d_FP}\n")
        f.write(f"d_TN：{d_TN}\n")
        f.write(f"d_FN：{d_FN}\n")

        f.write(f"wrong_counts：{wrong_counts}\n")
        f.write(f"准确率（疾病）：{accuracy_d}\n")
        f.write(f"错误率（疾病）：{error_rate_d}\n")
        f.write(f"灵敏度（疾病）：{sensitivity_d}\n")
        f.write(f"精确率（疾病）：{precision_d}\n")
        f.write(f"F1 score（疾病）：{f1_d}\n")


def g_negative_test(g_phrase, negative_phrase, g_model , gn_log):
    """
    针对只识别基因的模型
    用100个只有基因的句子进行测试
    用100个全阴性句子进行测试
    与能同时识别基因和疾病的模型进行对比，期望同时识别的模型能与单能模型效果一样
    """

    with open(g_phrase, "r") as f1:
        g_sentences = json.load(f1)

    with open(negative_phrase, "r") as f2:
        negative_sentences = f2.readlines()

    indexs = random.sample(range(0, 200), 100)
    negative_sentence_100 = [negative_sentences[i] for i in indexs]
    g_keys = list(g_sentences.keys())
    g_keys = [g_keys[i] for i in indexs]
    g_sentence_100 = {}
    for key in g_keys:
        g_sentence_100[key] = g_sentences[key]

    g_TP = 0
    g_FP = 0
    g_TN = 0
    g_FN = 0

    wrong_gene = []
    wrong_counts = 0
    wrong_negative = []

    # 测试疾病阳性句子
    print(list(g_sentence_100.keys()))
    for k,v in g_sentence_100.items():
        res0 = predict_entities_str(v, g_model)
        # i_g = k.strip()
        i_g = k.strip()

        if len(res0.split(" → ")) != 1:
            res = res0.split(" → ")[-1]
            res = res.replace('"', '')           
            res = res.replace('[','')
            res = res.replace(']','')
            res = res.strip()
            res = res.split(", ") # 理想的res长这样 ['colorectal polyps/Disease'] 只有疾病

            if len(res) == 1: 
                if res[0].split("/")[-1] == "Disease":
                    g_FN += 1
                elif res[0].split("/")[-1] == "Gene":
                    if res[0].split("/")[0] == i_g:
                        g_TP += 1
                    else:
                        g_FN += 1
                        wrong_gene.append(res0)
                else:
                    g_FN += 1
                    wrong_gene.append(res0)

            elif len(res) > 1:
                status = False
                for i in res:
                    if i.split("/")[-1] == "Gene" and i.split("/")[0] == i_g:
                        status = True
                if status:
                    g_TP += 1
                else:
                    g_FN += 1
                    wrong_gene.append(res0)
        else:
            g_FN += 1
            wrong_gene.append(res0)

    # 测试基因疾病双阴性句子
    for i in negative_sentence_100:
        res0 = predict_entities_str(i, dg_model)

        if len(res0.split(" → ")) != 1:
            res = res0.split(" → ")[-1]
            res = res.replace('"', '')           
            res = res.replace('[','')
            res = res.replace(']','')
            res = res.strip()
            res = res.split(", ") # 理想的res长这样 ['FUQCRC1/Gene', 'colorectal polyps/Disease']

            disease_or_gene = []
            for j in res:
                disease_or_gene.append(j.split("/")[-1])
            if "Gene" in disease_or_gene:
                g_FP += 1
                wrong_negative.append(res0)
        else:
            g_TN += 1


    with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/test/g_wrong/wrong_gene.txt", "w") as f:
        for i in wrong_gene:
            f.write(i + "\n")

    if wrong_negative:
        with open("/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/test/g_wrong/wrong_negative.txt", "w") as f:
            for i in wrong_negative:
                f.write(i + "\n")

    print("基因真阳性：", g_TP)
    print("基因假阳性：", g_FP)
    print("基因真阴性：", g_TN)
    print("基因假阴性：", g_FN)

    # 计算准确率
    accuracy_g = (g_TP + g_TN) / 200
    # 计算错误率
    error_rate_g = (g_FP + g_FN) / 200
    # 计算灵敏度
    sensitivity_g = (g_TP) / (g_TP + g_FN)
    # 计算精确率
    precision_g = (g_TP) / (g_TP + g_FP)
    # 计算F1 score
    f1_g = 2 * precision_g * sensitivity_g / (precision_g + sensitivity_g)

    with open(gn_log, "w") as f:
        f.write(f"g_TP：{g_TP}\n")
        f.write(f"g_FP：{g_FP}\n")
        f.write(f"g_TN：{g_TN}\n")
        f.write(f"g_FN：{g_FN}\n")

        f.write(f"wrong_counts：{wrong_counts}\n")
        f.write(f"准确率（基因）：{accuracy_g}\n")
        f.write(f"错误率（基因）：{error_rate_g}\n")
        f.write(f"灵敏度（基因）：{sensitivity_g}\n")
        f.write(f"精确率（基因）：{precision_g}\n")
        f.write(f"F1 score（基因）：{f1_g}\n")


if __name__ == "__main__":

    # make_omim_data(omim_path)
    # model = "/cluster/home/cx/models/trained/HUNflair2/final-model.pt"
    # model = "/cluster/home/cx/models/trained/NER-model/final-model.pt"
    # model = "/cluster/home/cx/models/trained/NER_gene/final-model.pt"
    # gw_dg_model="/cluster/home/gw/Backend_project/NER/tuned/best_model.pt"

    # non_disease_gene_path = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/negative.txt"
    # non_omim_path = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/non_omim_disease.txt"
    # # make_other_data(non_disease_gene_path,non_omim_path)

    # omim_base = '/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/base_genes_diseases/omim_short.txt'
    # gene_base = '/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/base_genes_diseases/genes.txt'
    # negative_base = '/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/base_genes_diseases/negative1.txt'
    # make_training_phrases(omim_base, gene_base, negative_base)

    d_g_phrase = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/test_phrases2/d_g_sentences.json"
    negative_phrase = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/test_phrases2/negative.txt"
    dg_model = gw_dg_model
    dgn_log = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/log/gw_dg_negative_test.log"
    dg_negative_test(d_g_phrase, negative_phrase, dg_model , dgn_log)

    d_phrase = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/test_phrases2/d_sentences.json"
    negative_phrase = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/test_phrases2/negative.txt"
    d_model = "/cluster/home/cx/models/trained/NER-model/final-model.pt"
    dn_log = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/log/d_negative_test.log"
    # d_negative_test(d_phrase, negative_phrase, d_model , dn_log)

    g_phrase = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/test_phrases2/g_sentences.json"
    negative_phrase = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/data/test_phrases2/negative.txt"
    g_model = "/cluster/home/cx/models/trained/NER_gene/final-model.pt"
    gn_log = "/cluster/home/cx/AI_project/RAG/RAG_flow/flair_NER/log/g_negative_test.log"
    # g_negative_test(g_phrase, negative_phrase, g_model , gn_log)




