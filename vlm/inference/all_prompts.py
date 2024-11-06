PROMPTS_MULTI = {
    "en": [
        "Answer if this meme is hate or not with a single alphabet letter among given answer choices a and b.",
        "Given a meme from online, answer if it is 'Hate Speech' or 'Non-Hate Speech'. Answer with only one single alphabet between 'a' and 'b', with no other description.",
        "Would you consider this meme as a hate speech? Give an answer in a single alphabet only."
    ],

    "de": [
        "Beantworte, ob dieses Meme Hass oder nicht ist, mit einem einzelnen Buchstaben der vorgegebenen Antwortmöglichkeiten a und b.",
        "Anhand eines Memes aus dem Internet, beantworte ob es sich um eine „Hassrede“ oder eine „Nicht-Hassrede“ handelt. Antworten Sie mit nur einem Buchstaben zwischen „a“ und „b“ und ohne weitere Beschreibung.",
        "Würdest du dieses Meme als Hassrede betrachten? Gebe deine Antwort nur in einem Buchstaben an.",
    ],

    "es": [
        "Responda si este meme es odio o no con una sola letra del alfabeto entre las opciones de respuesta a y b.",
        "Dado un meme de Internet, responda si es un 'discurso de odio' o 'discurso sin odio'. Responda con una sola letra entre 'a' y 'b', sin ninguna otra descripción.",
        "¿Considerarías este meme como un discurso de odio? Responde con una sola letra del alfabeto.",
    ],

    "hi": [
        "दिए गए उत्तर विकल्पों क और ख में से किसी एक अक्षर से बताइए कि यह मीम घृणास्पद भाषण है या नहीं।",
        "ऑनलाइन से दिए गए किसी मेमे से, उत्तर दें कि यह 'घृणास्पद भाषण' है या 'गैर-घृणास्पद भाषण' है। 'क' और 'ख' के बीच केवल एक ही अक्षर से उत्तर दें, कोई अन्य विवरण न दें।",
        "क्या आप इस मीम को  'घृणास्पद भाषण' मानेंगे? इसका जवाब सिर्फ़ एक अक्षर में दीजिए।",
    ],

    "zh": [
        "请判断该表情包是否包含仇恨，根据选项用字母a或b作答",
        "根据网图判断其是否为仇恨言论。用字母a或b作答，请勿在答案中输入其他文本",
        "您是否认为该表情包属于仇恨言论？请用单个字母给出答案",
    ]
}

PROMPT_CAPTION_MULTI = {
    "en": "\nCaption inside the meme image: '{}'",
    "de": "\nBeschriftung im Meme-Bild: '{}'",
    "es": "\nTítulo dentro de la imagen del meme: '{}'",
    "hi": "\nमीम इमेज के अंदर कैप्शन: '{}'",
    "zh": "\n表情包所配文字：'{}'",
}

PROMPT_PREFIX_MULTI = {
    "en": "Definition of Hate Speech:\n\nAny kind of communication [...], that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.",
    "de": "Definition von Hassrede:\n\nJede Art von Kommunikation [...], die eine abwertende oder diskriminierende Sprache in Bezug auf eine Person oder eine Gruppe aufgrund ihrer Identität angreift oder verwendet, mit anderen Worten aufgrund ihrer Religion, ethnischen Zugehörigkeit, Nationalität, Rasse, Hautfarbe, Abstammung, Geschlecht oder ein anderer Identitätsfaktor.",
    "es": "Definición de discurso de odio:\n\nCualquier tipo de comunicación [...], que ataque o utilice un lenguaje peyorativo o discriminatorio con referencia a una persona o un grupo en función de quiénes son, en otras palabras, en función de su religión, etnia, nacionalidad, raza, color, ascendencia, género u otro factor de identidad.",
    "hi": "घृणास्पद भाषण की परिभाषा:\n\nकिसी भी प्रकार का संचार [...], जो किसी व्यक्ति या समूह के संदर्भ में उनके धर्म, जातीयता, राष्ट्रीयता, नस्ल, रंग, वंश, लिंग या अन्य पहचान कारक के आधार पर अपमानजनक या भेदभावपूर्ण भाषा का उपयोग करता है।",
    "zh": "仇恨言论的定义:\n\n在任何形式的交流中 [...]，基于个人或群体的身份进行攻击或使用贬义或歧视性语言。换句话说，基于他们的宗教、民族、国籍、种族、肤色、血统、性别或其他身份因素，进行攻击或使用贬义或歧视性语言。",
}


PROMPT_POSTFIX_MULTI = {
    "en": ["\nChoose:\na: Hate\nb: Non-hate\n", "\nChoose:\na: Non-hate\nb: Hate\n"],
    "de": ["\nWähle:\na: Hass\nb: Kein Hass\n", "\nWähle:\na: Kein Hass\nb: Hass\n"],
    "es": ["\nElige:\na: Odio\nb: Sin odio\n", "\nElige:\na: Sin odio\nb: Odio\n"],
    "hi": ["\nचुनें:\nक: घृणा\nख: गैर नफरत\n", "\nचुनें:\nक: गैर नफरत\nख: घृणा\n"],
    "zh": ["\n请选择：\na：仇恨\nb：非仇恨", "\n请选择：\na：非仇恨\nb：仇恨\n"]
}

PROMPT_IMAGE_PREFIX_MULTI = {
    "en": "\nMeme: ",
    "de": "\nMeme: ",
    "es": "\nMeme: ",
    "hi": "\nमीम: ",
    "zh": "\n表情包: "
}

PROMPTS_COUNTRY_INSERTION = [
    "Answer if this meme is hate or not in {} with a single alphabet letter among given answer choices a and b.",
    "Given a meme from online, answer if it is 'Hate Speech' or 'Non-Hate Speech' in {}. Answer with only one single alphabet between 'a' and 'b', with no other description.",
    "As a person raised up in {}, would you consider this meme as a hate speech? Give an answer in a single alphabet only.",
]


def set_prompts(language):
    return PROMPTS_MULTI[language], PROMPT_CAPTION_MULTI[language], PROMPT_PREFIX_MULTI[language], PROMPT_POSTFIX_MULTI[language], PROMPT_IMAGE_PREFIX_MULTI[language], PROMPTS_COUNTRY_INSERTION
