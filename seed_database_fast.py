import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "index"

DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# High-quality mock Indic MSMARCO examples
MOCK_EXAMPLES = [
    {
        "language": "hi",
        "query_id": "111197",
        "query": "रक्तचाप मापने के लिए सटीक रीडिंग प्राप्त करने की प्रक्रिया क्या है?",
        "answers": ["रक्तचाप मापने के लिए सटीक रीडिंग सुनिश्चित करने के लिए: 1. कैफीन और व्यायाम से 30 मिनट तक बचें। 2. 5 मिनट आराम करें। 3. पैर फर्श पर सपाट रखकर बैठें। 4. बांह को हृदय के स्तर पर रखें। 5. दो बार मापें और औसत निकालें।"],
        "original_query": "how to get accurate blood pressure reading",
        "original_answers": ["1. Avoid caffeine/exercise 30 mins before. 2. Rest 5 mins. 3. Sit with feet flat. 4. Arm at heart level. 5. Average two readings."],
        "passages": {
            "passage_text": [
                "अपने रक्तचाप को मापने से 30 मिनट पहले कैफीनयुक्त पेय न पिएं या व्यायाम न करें। परीक्षण शुरू होने से पांच मिनट पहले शांत बैठें।",
                "माप के दौरान, फर्श पर अपने पैरों के साथ एक कुर्सी पर बैठें और अपनी बांह का समर्थन करें ताकि आपकी कोहनी दिल के स्तर के बारे में हो। कफ का inflatable हिस्सा आपके ऊपरी बांह के कम से कम 80% हिस्से को पूरी तरह से कवर करना चाहिए। कम से कम दो माप लें, उन्हें 1-2 मिनट अलग स्थान दें, और परिणामों को औसत करें।"
            ],
            "url": ["https://www.healthline.com/health/high-blood-pressure", "https://www.cdc.gov/bloodpressure"],
            "is_selected": [1, 1]
        }
    },
    {
        "language": "bn",
        "query_id": "111197",
        "query": "রক্তচাপ মাপার জন্য সঠিক রিডিং নিশ্চিত করার উপায় কী?",
        "answers": ["রক্তচাপ পরিমাপ করার সময় সঠিক রিডিং নিশ্চিত করতে: ১. ৩০ মিনিট আগে ক্যাফেইন বা ব্যায়াম এড়িয়ে চলুন। ২. ৫ মিনিট শান্ত হয়ে বসুন। ৩. পা মেঝেতে সমান্তরাল রাখুন। ৪. হাত হৃদযন্ত্রের স্তরে রাখুন। ৫. দুটি রিডিং নিয়ে গড় করুন।"],
        "original_query": "how to get accurate blood pressure reading",
        "original_answers": ["1. Avoid caffeine/exercise 30 mins before. 2. Rest 5 mins. 3. Sit with feet flat. 4. Arm at heart level. 5. Average two readings."],
        "passages": {
            "passage_text": [
                "রক্তচাপ পরিমাপ করার ৩০ মিনিট আগে ক্যাফেইনযুক্ত পানীয় পান বা ব্যায়াম করবেন না। পরীক্ষা শুরু করার আগে ৫ মিনিট শান্ত হয়ে বসুন।",
                "পরিমাপের সময় মেরুদণ্ড সোজা করে চেয়ারে বসুন, পা মেঝেতে সমান্তরাল রাখুন এবং বাহুটি হৃৎপিণ্ডের স্তরের কাছাকাছি রাখুন। কাফের স্ফীত অংশ আপনার বাহুর অন্তত ৮০% অংশ ঢেকে রাখা উচিত। অন্তত দুটি পরিমাপ নিয়ে ফলাফল গড় করুন।"
            ],
            "url": ["https://www.healthline.com/health/high-blood-pressure", "https://www.cdc.gov/bloodpressure"],
            "is_selected": [1, 1]
        }
    },
    {
        "language": "hi",
        "query_id": "222201",
        "query": "स्वस्थ रहने के लिए एक दिन में कितना पानी पीना चाहिए?",
        "answers": ["सामान्यतः एक दिन में 8 से 10 गिलास (लगभग 2 से 2.5 लीटर) पानी पीना स्वास्थ्य के लिए अच्छा माना जाता है।"],
        "original_query": "how much water to drink in a day for health",
        "original_answers": ["Generally 8-10 glasses (around 2-2.5 liters) is recommended daily."],
        "passages": {
            "passage_text": [
                "शरीर को हाइड्रेटेड रखने और विषाक्त पदार्थों को बाहर निकालने के लिए पर्याप्त पानी पीना आवश्यक है। स्वास्थ्य विशेषज्ञ प्रतिदिन कम से कम 8-10 गिलास पानी पीने की सलाह देते हैं।",
                "गर्मियों में या शारीरिक कसरत के दौरान पानी की आवश्यकता बढ़ जाती है। बहुत कम पानी पीने से निर्जलीकरण (dehydration) हो सकता है जिससे सिरदर्द और थकान होती है।"
            ],
            "url": ["https://www.mayoclinic.org/healthy-lifestyle/nutrition", "https://www.webmd.com/diet/water-requirements"],
            "is_selected": [1, 0]
        }
    },
    {
        "language": "bn",
        "query_id": "222201",
        "query": "সুস্থ থাকার জন্য দিনে কতটা জল খাওয়া উচিত?",
        "answers": ["সাধারণত দিনে অন্তত ৮-১০ গ্লাস (প্রায় ২ থেকে ২.৫ লিটার) জল পান করা উচিত।"],
        "original_query": "how much water to drink in a day for health",
        "original_answers": ["Generally 8-10 glasses (around 2-2.5 liters) is recommended daily."],
        "passages": {
            "passage_text": [
                "শরীরকে আর্দ্র রাখতে এবং ক্ষতিকারক টক্সিন বের করতে পর্যাপ্ত জল পান করা দরকার। বিশেষজ্ঞরা প্রতিদিন অন্তত ৮ থেকে ১০ গ্লাস জল খাওয়ার পরামর্শ দেন।",
                "গরমকালে বা শারীরিক পরিশ্রমের সময় জলের প্রয়োজনীয়তা বৃদ্ধি পায়। কম জল পানে ডিহাইড্রেশন হতে পারে যার ফলে মাথা ব্যথা ও ক্লান্তি আসতে পারে।"
            ],
            "url": ["https://www.mayoclinic.org/healthy-lifestyle/nutrition", "https://www.webmd.com/diet/water-requirements"],
            "is_selected": [1, 0]
        }
    },
    {
        "language": "hi",
        "query_id": "333302",
        "query": "मानव शरीर का सामान्य तापमान कितना होता है?",
        "answers": ["मानव शरीर का सामान्य तापमान लगभग 98.6°F (37°C) होता है।"],
        "original_query": "what is normal human body temperature",
        "original_answers": ["The normal human body temperature is typically 98.6°F (37°C)."],
        "passages": {
            "passage_text": [
                "एक स्वस्थ वयस्क मानव शरीर का सामान्य तापमान 98.6°F (37°C) माना जाता है, हालांकि यह दिन के समय के अनुसार 97°F से 99°F के बीच भिन्न हो सकता है।",
                "100.4°F (38°C) से अधिक तापमान होने पर इसे बुखार (fever) माना जाता है, जो यह दर्शाता है कि शरीर किसी संक्रमण से लड़ रहा है।"
            ],
            "url": ["https://www.webmd.com/first-aid/normal-body-temperature", "https://www.medicalnewstoday.com/articles/normal-body-temperature"],
            "is_selected": [1, 0]
        }
    },
    {
        "language": "bn",
        "query_id": "333302",
        "query": "মানুষের শরীরের স্বাভাবিক তাপমাত্রা কত?",
        "answers": ["মানুষের শরীরের স্বাভাবিক তাপমাত্রা হলো সাধারণত ৯৮.৬° ফারেনহাইট (৩৭° সেলসিয়াস)।"],
        "original_query": "what is normal human body temperature",
        "original_answers": ["The normal human body temperature is typically 98.6°F (37°C)."],
        "passages": {
            "passage_text": [
                "একটি সুস্থ মানুষের শরীরের স্বাভাবিক তাপমাত্রা ৯৮.৬° ফারেনহাইট (৩৭° সেলসিয়াস)। তবে এটি সময়ভেদে সামান্য ওঠানামা করতে পারে।",
                "যদি শরীরের তাপমাত্রা ১০০.৪° ফারেনহাইট বা তার বেশি হয় তবে তাকে জ্বর (fever) বলা হয়ে থাকে, যা ইনফেকশনের লক্ষণ হতে পারে।"
            ],
            "url": ["https://www.webmd.com/first-aid/normal-body-temperature", "https://www.medicalnewstoday.com/articles/normal-body-temperature"],
            "is_selected": [1, 0]
        }
    }
]

# Write raw_examples.pkl
with open(DATA_DIR / "raw_examples.pkl", "wb") as f:
    pickle.dump(MOCK_EXAMPLES, f)

# Helper function to split text simply for mock indexing
def mock_naive_chunks(text):
    sentences = [s.strip() for s in text.split("।") if s.strip()]
    if not sentences:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
    return sentences

# Generate deterministic mock embeddings (384-dim)
np.random.seed(42)

# Build Indexes
strategies = ["naive", "semantic", "parent_child"]
for strategy in strategies:
    chunks = []
    metadata = []
    
    for idx, item in enumerate(MOCK_EXAMPLES):
        passages = item["passages"]["passage_text"]
        full_text = " ".join(passages)
        
        # Determine chunk text list
        if strategy == "naive":
            item_chunks = mock_naive_chunks(full_text)
            for c in item_chunks:
                chunks.append(c)
                metadata.append({
                    "language": item["language"],
                    "query_id": item["query_id"],
                    "url": item["passages"]["url"][0] if item["passages"]["url"] else "",
                    "chunk_type": "naive"
                })
        elif strategy == "semantic":
            # Semantic splits on sentences too for mock simplicity
            item_chunks = mock_naive_chunks(full_text)
            for c in item_chunks:
                chunks.append(c)
                metadata.append({
                    "language": item["language"],
                    "query_id": item["query_id"],
                    "url": item["passages"]["url"][0] if item["passages"]["url"] else "",
                    "chunk_type": "semantic"
                })
        elif strategy == "parent_child":
            # Parent-Child: child is sentence, parent is full text
            item_chunks = mock_naive_chunks(full_text)
            for c in item_chunks:
                chunks.append(c)
                metadata.append({
                    "language": item["language"],
                    "query_id": item["query_id"],
                    "url": item["passages"]["url"][0] if item["passages"]["url"] else "",
                    "parent_text": full_text,
                    "chunk_type": "child"
                })
                
    num_chunks = len(chunks)
    # Generate 384-dimensional normalized dense vectors
    dense = np.random.randn(num_chunks, 384).astype(np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = dense / np.maximum(norms, 1e-12)
    
    # Save dense array
    np.save(str(INDEX_DIR / f"{strategy}_dense.npy"), dense)
    
    # Fit TF-IDF
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    sparse_matrix = tfidf.fit_transform(chunks)
    
    # Save metadata pkl
    meta_info = {
        "chunks": chunks,
        "metadata": metadata,
        "tfidf_vectorizer": tfidf,
        "tfidf_matrix": sparse_matrix
    }
    with open(INDEX_DIR / f"{strategy}_meta.pkl", "wb") as f:
        pickle.dump(meta_info, f)

print("FAST DETERMINISTIC SEEDING COMPLETED SUCCESSFULLY!")
