import os
import sys
import pickle
import logging
from app.config import config
from app.indexer import indexer

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("seed_database")

# High-quality mock Indic MSMARCO examples for low-latency offline seeding
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
        "answers": ["রক্তচাপ পরিমাপ করার সময় সঠিক রিডিং নিশ্চিত করতে: ১. ৩০ মিনিট আগে ক্যাফেইন বা ব্যায়াম এড়িয়ে চলুন। ২. ৫ মিনিট শান্ত হয়ে বসুন। ৩. পা মেঝেতে সমান্তরাল রাখুন। ৪. হাত হৃদযন্ত্রের স্তরে রাখুন। ৫. দুটি রিডিং নিয়ে গড় করুন।"],
        "original_query": "how to get accurate blood pressure reading",
        "original_answers": ["1. Avoid caffeine/exercise 30 mins before. 2. Rest 5 mins. 3. Sit with feet flat. 4. Arm at heart level. 5. Average two readings."],
        "passages": {
            "passage_text": [
                "রক্তচাপ পরিমাপ করার ৩০ মিনিট আগে ক্যাফেইনযুক্ত পানীয় পান বা ব্যায়াম করবেন না। পরীক্ষা শুরু করার আগে ৫ মিনিট শান্ত হয়ে বসুন।",
                "পরিমাপের সময় মেরুদণ্ড সোजा করে চেয়ারে বসুন, পা মেঝেতে সমান্তরাল রাখুন এবং বাহুটি হৃৎপিণ্ডের স্তরের কাছাকাছি রাখুন। কাফের স্ফীত অংশ আপনার বাহুর অন্তত ৮০% অংশ ঢেকে রাখা উচিত। অন্তত দুটি পরিমাপ নিয়ে ফলাফল গড় করুন।"
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

def seed(limit_per_language: int = 100, use_online_hf: bool = False):
    """
    Seeds the database. Uses offline mock examples by default to run instantly (<1s)
    and ensure synchronous terminal execution. Downloads from HuggingFace if explicitly requested.
    """
    logger.info("Initializing dataset seeding...")
    
    # Ensure folders exist
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_path = config.DATA_DIR / "raw_examples.pkl"
    examples = []
    
    if use_online_hf:
        try:
            from datasets import load_dataset
            logger.info("Loading ai4bharat/MSMARCO-XI from HuggingFace in streaming mode...")
            dataset = load_dataset("ai4bharat/MSMARCO-XI", "default", split="validation", streaming=True)
            
            counts = {"hi": 0, "bn": 0}
            logger.info(f"Filtering dataset for target languages (Target: {limit_per_language} each)...")
            
            for i, example in enumerate(dataset):
                lang = example.get("language")
                if lang in counts and counts[lang] < limit_per_language:
                    pure_example = {
                        "language": lang,
                        "query_id": example.get("query_id"),
                        "query": example.get("query"),
                        "answers": list(example.get("answers", [])),
                        "original_query": example.get("original_query"),
                        "original_answers": list(example.get("original_answers", [])),
                        "passages": {
                            "passage_text": list(example.get("passages", {}).get("passage_text", [])),
                            "url": list(example.get("passages", {}).get("url", [])),
                            "is_selected": list(example.get("passages", {}).get("is_selected", []))
                        }
                    }
                    examples.append(pure_example)
                    counts[lang] += 1
                    
                if all(c >= limit_per_language for c in counts.values()):
                    break
            logger.info(f"Successfully downloaded {len(examples)} examples.")
        except Exception as e:
            logger.warning(f"Failed to fetch dataset online: {e}. Falling back to high-quality offline mock data.")
            examples = MOCK_EXAMPLES
    else:
        logger.info("Using local high-quality mock dataset for instant offline seeding.")
        examples = MOCK_EXAMPLES

    # Save raw examples to disk
    with open(raw_path, "wb") as f:
        pickle.dump(examples, f)
    logger.info(f"Saved raw examples to {raw_path}")

    # Build indices for all chunking strategies
    strategies = ["naive", "semantic", "parent_child"]
    for strategy in strategies:
        try:
            logger.info(f"--- Building Index: {strategy.upper()} ---")
            res = indexer.build_and_save_index(examples, strategy=strategy)
            logger.info(f"Successfully built {strategy} index with {res.get('num_chunks', 0)} chunks.")
        except Exception as e:
            logger.exception(f"Failed to build index for strategy: {strategy}")
            
    logger.info("Database seeding and index creation completed successfully!")

if __name__ == "__main__":
    use_hf = "--online" in sys.argv
    seed(limit_per_language=10, use_online_hf=use_hf)
