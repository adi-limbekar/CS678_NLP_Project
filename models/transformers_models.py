from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertTokenizerFast,
    BertForSequenceClassification,
    BertweetTokenizer,
    BigBirdConfig,
    BigBirdTokenizer,
    BigBirdForSequenceClassification,
    CamembertConfig,
    CamembertTokenizerFast,
    CamembertForSequenceClassification,
    DebertaConfig,
    DebertaForSequenceClassification,
    DebertaTokenizer,
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
    DistilBertConfig,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    ElectraConfig,
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    FlaubertConfig,
    FlaubertTokenizer,
    FlaubertForSequenceClassification,
    HerbertTokenizerFast,
    LayoutLMConfig,
    LayoutLMTokenizerFast,
    LayoutLMForSequenceClassification,
    LongformerConfig,
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
    MPNetConfig,
    MPNetForSequenceClassification,
    MPNetTokenizerFast,
    MobileBertConfig,
    MobileBertTokenizerFast,
    MobileBertForSequenceClassification,
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    SqueezeBertConfig,
    SqueezeBertForSequenceClassification,
    SqueezeBertTokenizerFast,
    WEIGHTS_NAME,
    XLMConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    XLMRobertaForSequenceClassification,
    XLMTokenizer,
    XLMForSequenceClassification,
    XLNetConfig,
    XLNetTokenizerFast,
    XLNetForSequenceClassification)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "auto": (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizerFast),
    "bertweet": (
        RobertaConfig,
        RobertaForSequenceClassification,
        BertweetTokenizer,
    ),
    "bigbird": (
        BigBirdConfig,
        BigBirdForSequenceClassification,
        BigBirdTokenizer,
    ),
    "camembert": (
        CamembertConfig,
        CamembertForSequenceClassification,
        CamembertTokenizerFast,
    ),
    "deberta": (
        DebertaConfig,
        DebertaForSequenceClassification,
        DebertaTokenizer,
    ),
    "debertav2": (
        DebertaV2Config,
        DebertaV2ForSequenceClassification,
        DebertaV2Tokenizer,
    ),
    "distilbert": (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    ),
    "electra": (
        ElectraConfig,
        ElectraForSequenceClassification,
        ElectraTokenizerFast,
    ),
    "flaubert": (
        FlaubertConfig,
        FlaubertForSequenceClassification,
        FlaubertTokenizer,
    ),
    "herbert": (
        BertConfig,
        BertForSequenceClassification,
        HerbertTokenizerFast,
    ),
    "layoutlm": (
        LayoutLMConfig,
        LayoutLMForSequenceClassification,
        LayoutLMTokenizerFast,
    ),
    "longformer": (
        LongformerConfig,
        LongformerForSequenceClassification,
        LongformerTokenizerFast,
    ),
    "mobilebert": (
        MobileBertConfig,
        MobileBertForSequenceClassification,
        MobileBertTokenizerFast,
    ),
    "mpnet": (MPNetConfig, MPNetForSequenceClassification, MPNetTokenizerFast),
    "roberta": (
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizerFast,
    ),
    "squeezebert": (
        SqueezeBertConfig,
        SqueezeBertForSequenceClassification,
        SqueezeBertTokenizerFast,
    ),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "xlmroberta": (
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
        XLMRobertaTokenizerFast,
    ),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizerFast),
}